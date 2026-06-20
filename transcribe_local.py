import argparse
import json
import os
import sys
import time
from pathlib import Path

# AMD ROCm RDNA3/3.5 iGPU/APU 互換性（Radeon 890M / 780M など）のための環境変数設定
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
# Windows上での Hugging Face Hub シンボリックリンク警告を抑止
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# OpenMP ランタイムの競合回避（Windows環境）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Windows コンソール出力時のエンコーディングエラー（UnicodeEncodeError）を防止
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# グローバルな環境へのインストールなしで ffmpeg を利用可能にする
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

# 対象とする音声・動画ファイルの拡張子
MEDIA_EXTENSIONS = {".mp3", ".m4a", ".wav", ".opus", ".flac", ".mp4", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数の解析器を構築します。"""
    parser = argparse.ArgumentParser(
        description="Local media transcription script optimized for AMD ROCm GPU."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="downloads",
        help="Audio/Video file path or directory containing media files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="transcripts",
        help="Directory where text and JSON transcripts will be saved.",
    )
    parser.add_argument(
        "--model",
        default="turbo",
        choices=("tiny", "base", "small", "medium", "large", "turbo"),
        help="Whisper model size.",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="Language code (e.g. ja, en). Use 'auto' for auto-detection.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (auto, cuda, cpu).",
    )
    parser.add_argument(
        "--compute-type",
        choices=("auto", "int8", "float16", "int8_float16", "float32"),
        default="auto",
        help="Inference compute type.",
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Whisper task type.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search size.",
    )
    return parser


def load_cache(output_dir: Path) -> set[str]:
    """指定ディレクトリから完了済みファイルのキャッシュリストをロードします。"""
    cache_path = output_dir / "transcribe_cache.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return set(data.get("completed_files", []))
        except Exception:
            pass
    return set()


def save_cache(output_dir: Path, completed_files: set[str]) -> None:
    """完了済みファイルのキャッシュリストを JSON に保存します。"""
    cache_path = output_dir / "transcribe_cache.json"
    data = {"completed_files": sorted(list(completed_files))}
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_input_media(input_path: Path) -> list[Path]:
    """入力パスを解決し、処理対象となるメディアファイルのリストを作成日時(古い順)でソートして返します。"""
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    candidates = [
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No supported media files found in: {input_path}")

    # 作成日時 (st_ctime) の古い順にソート
    return sorted(candidates, key=lambda path: path.stat().st_ctime)


def select_device(device: str) -> str:
    """デバイスが auto の場合に、GPUが利用可能なら cuda を選択し、そうでなければ cpu を選択します。"""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        return "cpu"
    return device


def build_output_paths(output_dir: Path, media_path: Path) -> tuple[Path, Path]:
    """出力先のテキストファイルと JSON ファイルのパスを構築します。"""
    base_name = media_path.stem
    return output_dir / f"{base_name}.txt", output_dir / f"{base_name}.json"


def format_seconds(seconds: float) -> str:
    """秒数を HH:MM:SS 形式の文字列にフォーマットします。"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_outputs(text_path: Path, json_path: Path, result: dict) -> None:
    """タイムスタンプ付きのテキスト出力ファイルおよび詳細情報付きの JSON ファイルを書き出します。"""
    lines = []
    for seg in result.get("segments", []):
        start_str = format_seconds(seg.get("start", 0.0))
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{start_str}] {text}")

    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "text": result["text"].strip(),
        "language": result.get("language"),
        "segments": result.get("segments", []),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def transcribe_file(media_path: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """
    指定された単一のメディアファイルを OpenAI Whisper で文字起こし処理します。
    無音/BGM区間のハルシネーション（幻覚）防止のために簡易 VAD ポストプロセスを適用します。
    """
    import torch
    import whisper
    import numpy as np

    active_device = select_device(args.device)
    print(f"Loading Whisper model '{args.model}' on {active_device}...", flush=True)
    model = whisper.load_model(args.model, device=active_device)

    # 推論設定の構築
    decode_options = {
        "task": args.task,
        "beam_size": args.beam_size,
        "verbose": True,
        "condition_on_previous_text": False,  # 幻覚ループの伝染を防ぐ
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
    }
    if args.language.lower() != "auto":
        decode_options["language"] = args.language

    # CPUとGPUでの浮動小数点精度の選択
    decode_options["fp16"] = False if active_device == "cpu" else True

    # FFmpeg dynamic パイプ処理でのハング・クラッシュを避けるために事前に一括メモリロード
    print("Loading audio file into memory...", flush=True)
    audio = whisper.load_audio(str(media_path))
    print(f"Audio loaded. Duration: {audio.shape[0]/16000:.2f} seconds", flush=True)

    # 簡易 VAD (Voice Activity Detection): 30秒ウィンドウの音量 (RMS) が 0.018 未満のチャンクを無音と判定
    sr = 16000
    chunk_size = sr * 30
    num_chunks = int(np.ceil(len(audio) / chunk_size))
    silent_chunks = set()
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio))
        chunk = audio[start_idx:end_idx]
        if len(chunk) > 0:
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms < 0.018:
                silent_chunks.add(i)
    print(f"VAD detected {len(silent_chunks)} of {num_chunks} chunks as silence (RMS < 0.018).", flush=True)

    # AMD ROCm での Flash/Memory-Efficient Attention のフリーズを防ぐため math バックエンドを強制
    print("Starting Whisper transcription...", flush=True)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        raw_result = model.transcribe(audio, **decode_options)

    # VAD に基づく無音区間のテキストセグメント除外（ハルシネーションポストフィルタ）
    segments = []
    filtered_count = 0
    for i, seg in enumerate(raw_result.get("segments", [])):
        start_time = seg.get("start", 0.0)
        chunk_idx = int(start_time // 30)
        if chunk_idx in silent_chunks:
            filtered_count += 1
            continue
        segments.append({
            "id": len(segments),
            "start": start_time,
            "end": seg.get("end"),
            "text": seg.get("text", "").strip(),
        })
    print(f"VAD post-process: filtered out {filtered_count} segments in silent zones.", flush=True)

    full_text = " ".join([seg["text"] for seg in segments])
    result = {
        "text": full_text.strip(),
        "language": raw_result.get("language", args.language),
        "segments": segments
    }

    # 出力ファイルパスの定義と書き出し
    text_path, json_path = build_output_paths(output_dir, media_path)
    write_outputs(text_path, json_path, result)

    active_compute_type = "float16" if decode_options.get("fp16") else "float32"
    print(f"Success! output={text_path} (device={active_device}, compute={active_compute_type})", flush=True)
    return 0


def run_transcribe_worker(args: argparse.Namespace) -> int:
    """
    対象となるすべてのメディアファイルをキャッシュ判定を伴うバッチループで順次文字起こしします。
    """
    media_paths = resolve_input_media(Path(args.input_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_files = load_cache(output_dir)
    total = len(media_paths)
    skipped = 0
    processed = 0

    print(f"Found {total} media files. (Skipped: {len(completed_files & {p.name for p in media_paths})})", flush=True)

    for i, path in enumerate(media_paths, 1):
        file_key = path.name
        if file_key in completed_files:
            print(f"[{i}/{total}] Skipping already transcribed file: {file_key}", flush=True)
            skipped += 1
            continue

        print(f"[{i}/{total}] Processing: {file_key}", flush=True)
        try:
            exit_code = transcribe_file(path, output_dir, args)
            if exit_code == 0:
                completed_files.add(file_key)
                save_cache(output_dir, completed_files)
                processed += 1
            else:
                print(f"[{i}/{total}] Failed transcribing: {file_key} (exit={exit_code})", flush=True)
                return exit_code
        except Exception as e:
            print(f"[{i}/{total}] Exception during transcription of {file_key}: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            return 1

    print(f"Batch transcription completed. Total: {total}, Processed: {processed}, Skipped: {skipped}", flush=True)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_transcribe_worker(args)


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)