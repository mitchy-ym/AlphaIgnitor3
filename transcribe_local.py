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
        default="float16",
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
        if "#" in input_path.name:
            new_path = input_path.with_name(input_path.name.replace("#", "_"))
            print(f"Renaming unsafe file (contains '#') to: {new_path.name}", flush=True)
            input_path.rename(new_path)
            input_path = new_path
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # '#' が含まれるファイルがあれば自動でリネーム（PyAVが '#' を含むパスをオープンできない問題への対策）
    for path in input_path.iterdir():
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS and "#" in path.name:
            new_path = path.with_name(path.name.replace("#", "_"))
            print(f"Renaming unsafe file (contains '#') to: {new_path.name}", flush=True)
            try:
                path.rename(new_path)
            except Exception as e:
                print(f"Failed to rename {path.name}: {e}", file=sys.stderr, flush=True)

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


def transcribe_file(media_path: Path, output_dir: Path, model: "WhisperModel", args: argparse.Namespace) -> int:
    """
    指定された単一のメディアファイルを faster-whisper で文字起こし処理します。
    """
    active_device = select_device(args.device)

    # 推論設定の構築
    decode_options = {
        "task": args.task,
        "beam_size": args.beam_size,
        "condition_on_previous_text": False,  # 幻覚ループの伝染を防ぐ
        "vad_filter": True,  # 内蔵の Silero VAD を使用して無音区間をフィルタリング
        "batch_size": 8,  # バッチサイズを 8 に設定して推論を高速化
    }
    if args.language.lower() != "auto":
        decode_options["language"] = args.language

    print("Starting Whisper transcription...", flush=True)
    segments, info = model.transcribe(str(media_path), **decode_options)

    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}", flush=True)

    result_segments = []
    for segment in segments:
        # 進捗表示用に出力
        print(f"[{format_seconds(segment.start)} -> {format_seconds(segment.end)}] {segment.text}", flush=True)
        result_segments.append({
            "id": len(result_segments),
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })

    full_text = " ".join([seg["text"] for seg in result_segments])
    result = {
        "text": full_text.strip(),
        "language": info.language,
        "segments": result_segments
    }

    # 出力ファイルパスの定義と書き出し
    text_path, json_path = build_output_paths(output_dir, media_path)
    write_outputs(text_path, json_path, result)

    print(f"Success! output={text_path} (device={active_device}, compute={args.compute_type})", flush=True)
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

    # 実際に処理を行うファイルがある場合のみモデルをロードする
    target_paths = [p for p in media_paths if p.name not in completed_files]
    if not target_paths:
        print(f"Batch transcription completed. Total: {total}, Processed: 0, Skipped: {total}", flush=True)
        return 0

    from faster_whisper import WhisperModel
    active_device = select_device(args.device)
    print(f"Loading faster-whisper model '{args.model}' on {active_device} (compute_type={args.compute_type})...", flush=True)
    model = WhisperModel(
        args.model,
        device=active_device,
        compute_type=args.compute_type
    )

    for i, path in enumerate(media_paths, 1):
        file_key = path.name
        if file_key in completed_files:
            print(f"[{i}/{total}] Skipping already transcribed file: {file_key}", flush=True)
            skipped += 1
            continue

        print(f"[{i}/{total}] Processing: {file_key}", flush=True)
        try:
            exit_code = transcribe_file(path, output_dir, model, args)
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