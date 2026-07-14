import json
import queue
import sys
import threading
import time
from pathlib import Path
import numpy as np
import torch
import argparse
import subprocess

from . import utils
from .utils import (
    MEDIA_EXTENSIONS,
    log_info,
    log_warn,
    log_error,
    format_seconds,
    get_media_duration
)

def load_transcribe_cache(output_dir: Path) -> set[str]:
    """文字起こし済みのファイル名キャッシュリストを読み込みます。"""
    cache_path = output_dir / "transcribe_cache.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return set(data.get("completed_files", []))
        except Exception:
            pass
    return set()

def save_transcribe_cache(output_dir: Path, completed_files: set[str]) -> None:
    """文字起こし済みのファイル名キャッシュリストを保存します。"""
    cache_path = output_dir / "transcribe_cache.json"
    data = {"completed_files": sorted(list(completed_files))}
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def resolve_input_media(input_path: Path) -> list[Path]:
    """入力メディアファイルのパスを解決し、作成日時の古い順にソートしたリストを返します。"""
    if input_path.is_file():
        if "#" in input_path.name:
            new_path = input_path.with_name(input_path.name.replace("#", "_"))
            print(f"Renaming unsafe file (contains '#') to: {new_path.name}", flush=True)
            input_path.rename(new_path)
            input_path = new_path
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # PyAV 互換性向上のため、ファイル名に含まれる '#' を自動でアンダースコアにリネームします
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

    return sorted(candidates, key=lambda path: path.stat().st_ctime)

def select_device(device: str) -> str:
    """デバイス設定が auto の場合に GPU (CUDA) が利用可能なら cuda、不可なら cpu を返します。"""
    if device == "auto":
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        return "cpu"
    return device

def load_whisper_model(model_path: str, device: str, compute_type: str) -> "BatchedInferencePipeline":
    """Whisper モデルを読み込み、BatchedInferencePipeline でラップして返します。"""
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    active_device = select_device(device)
    
    if "/" in model_path and not Path(model_path).exists():
        try:
            from huggingface_hub import snapshot_download
            print(f"Downloading model '{model_path}' from Hugging Face Hub...", flush=True)
            model_path = snapshot_download(repo_id=model_path)
        except Exception as e:
            print(f"Warning: Failed to pre-download model using huggingface_hub: {e}. Falling back to default loader.", file=sys.stderr, flush=True)
            
    print(f"Loading faster-whisper model '{model_path}' on {active_device} (compute_type={compute_type})...", flush=True)
    model_raw = WhisperModel(
        model_path,
        device=active_device,
        compute_type=compute_type
    )
    return BatchedInferencePipeline(model=model_raw)

def build_decode_options(args: argparse.Namespace) -> dict:
    """Namespace の設定から faster-whisper の推論デコードオプション辞書を構築します。"""
    decode_options = {
        "task": args.task,
        "beam_size": args.beam_size,
        "condition_on_previous_text": False,  # 幻覚ループ防止用
        "vad_filter": True,
    }
    if args.language.lower() != "auto":
        decode_options["language"] = args.language
    if args.initial_prompt:
        decode_options["initial_prompt"] = args.initial_prompt

    vad_params = {}
    if args.vad_threshold is not None:
        vad_params["threshold"] = args.vad_threshold
    if vad_params:
        decode_options["vad_parameters"] = vad_params
        
    return decode_options



def build_output_paths(output_dir: Path, media_path: Path) -> tuple[Path, Path]:
    """出力するテキストと JSON の保存パスを構築します。"""
    base_name = media_path.stem
    return output_dir / f"{base_name}.txt", output_dir / f"{base_name}.json"

def write_outputs(text_path: Path, json_path: Path, result: dict) -> None:
    """文字起こし結果をテキストおよび JSON 形式で出力します。"""
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

def transcribe_file(
    media_path: Path,
    output_dir: Path,
    model: "BatchedInferencePipeline",
    args: argparse.Namespace,
    preloaded_audio: np.ndarray | None = None
) -> int:
    """指定された単一のメディアファイルを Whisper で文字起こし（無音区間のフィルタリング含む）処理します。"""
    active_device = select_device(args.device)
    start_time_all = time.time()

    # GPU実行時のみ非同期プレロード・分割処理を有効にします
    use_async = (active_device == "cuda")
    total_duration = None

    if use_async:
        if preloaded_audio is not None:
            total_duration = len(preloaded_audio) / 16000
        else:
            total_duration = get_media_duration(media_path)
            if total_duration is None:
                print(f"Warning: Could not determine total duration of {media_path.name}. Falling back to sequential mode.", file=sys.stderr, flush=True)
                use_async = False

    decode_options = build_decode_options(args)
    result_segments = []
    filtered_count = 0
    detected_language = None

    if use_async:
        chunk_duration = getattr(args, "chunk_duration", 600.0)
        chunks = []
        curr_start = 0.0
        chunk_idx = 0
        while curr_start < total_duration:
            dur = min(chunk_duration, total_duration - curr_start)
            chunks.append((chunk_idx, curr_start, dur))
            curr_start += chunk_duration
            chunk_idx += 1

        print(f"Asynchronous processing enabled: splitting {media_path.name} into {len(chunks)} chunks of max {chunk_duration}s.", flush=True)

        class AudioChunk:
            def __init__(self, index: int, start_time: float, duration: float, audio_data: np.ndarray):
                self.index = index
                self.start_time = start_time
                self.duration = duration
                self.audio_data = audio_data

        class ChunkError:
            def __init__(self, exception: Exception):
                self.exception = exception

        chunk_queue = queue.Queue(maxsize=2)

        def audio_loader_worker():
            try:
                if preloaded_audio is not None:
                    sr = 16000
                    for idx, start_time, duration in chunks:
                        start_idx = int(start_time * sr)
                        end_idx = int((start_time + duration) * sr)
                        chunk_audio = preloaded_audio[start_idx:end_idx]
                        chunk_queue.put(AudioChunk(idx, start_time, duration, chunk_audio))
                else:
                    for idx, start_time, duration in chunks:
                        cmd = [
                            "ffmpeg", "-y", "-nostdin",
                            "-ss", f"{start_time:.3f}",
                            "-t", f"{duration:.3f}",
                            "-i", str(media_path),
                            "-f", "s16le", "-ac", "1", "-ar", "16000", "-"
                        ]
                        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if process.returncode != 0:
                            raise RuntimeError(
                                f"FFmpeg failed decoding chunk starting at {start_time}s: "
                                f"{process.stderr.decode('utf-8', errors='replace')}"
                            )
                        audio_data = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                        chunk_queue.put(AudioChunk(idx, start_time, duration, audio_data))
                chunk_queue.put(None)
            except Exception as ex:
                chunk_queue.put(ChunkError(ex))

        loader_thread = threading.Thread(target=audio_loader_worker, daemon=True)
        loader_thread.start()

        print(f"Starting batched Whisper transcription (batch_size={args.batch_size})...", flush=True)

        while True:
            item = chunk_queue.get()
            if item is None:
                break
            if isinstance(item, ChunkError):
                raise item.exception

            chunk = item

            chunk_decode_options = decode_options.copy()
            if args.language.lower() == "auto" and detected_language is not None:
                chunk_decode_options["language"] = detected_language

            segments, info = model.transcribe(chunk.audio_data, batch_size=args.batch_size, **chunk_decode_options)

            if args.language.lower() == "auto" and detected_language is None:
                detected_language = info.language
                print(f"Detected language '{detected_language}' with probability {info.language_probability:.2f} on first chunk.", flush=True)

            for segment in segments:
                absolute_start = chunk.start_time + segment.start
                absolute_end = chunk.start_time + segment.end

                # セグメント単位のRMS無音足切り
                min_rms = getattr(args, "min_rms", 0.003)
                if min_rms > 0.0:
                    start_sample = int(segment.start * 16000)
                    end_sample = int(segment.end * 16000)
                    segment_audio = chunk.audio_data[start_sample:end_sample]
                    if len(segment_audio) > 0:
                        rms = np.sqrt(np.mean(segment_audio ** 2))
                        if rms < min_rms:
                            filtered_count += 1
                            continue

                text = segment.text.strip()
                if not text:
                    continue

                print(f"[{format_seconds(absolute_start)} -> {format_seconds(absolute_end)}] {text}", flush=True)
                result_segments.append({
                    "id": len(result_segments),
                    "start": absolute_start,
                    "end": absolute_end,
                    "text": text,
                })

        print(f"Filtered out {filtered_count} hallucinated segments in silent zones.", flush=True)

        full_text = " ".join([seg["text"] for seg in result_segments])
        result = {
            "text": full_text.strip(),
            "language": detected_language or args.language,
            "segments": result_segments
        }

    else:
        # シーケンシャル処理（従来フロー）
        if preloaded_audio is not None:
            audio = preloaded_audio
            print(f"Audio already loaded. Shape: {audio.shape}", flush=True)
        else:
            from faster_whisper.audio import decode_audio
            print(f"Loading audio file: {media_path}", flush=True)
            start_load = time.time()
            audio = decode_audio(str(media_path), sampling_rate=16000)
            print(f"Audio loaded in {time.time() - start_load:.2f} seconds. Shape: {audio.shape}", flush=True)

        print(f"Starting batched Whisper transcription (batch_size={args.batch_size})...", flush=True)
        segments, info = model.transcribe(audio, batch_size=args.batch_size, **decode_options)

        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}", flush=True)

        for segment in segments:
            # セグメント単位のRMS無音足切り
            min_rms = getattr(args, "min_rms", 0.003)
            if min_rms > 0.0:
                start_sample = int(segment.start * 16000)
                end_sample = int(segment.end * 16000)
                segment_audio = audio[start_sample:end_sample]
                if len(segment_audio) > 0:
                    rms = np.sqrt(np.mean(segment_audio ** 2))
                    if rms < min_rms:
                        filtered_count += 1
                        continue

            text = segment.text.strip()
            if not text:
                continue

            print(f"[{format_seconds(segment.start)} -> {format_seconds(segment.end)}] {text}", flush=True)
            result_segments.append({
                "id": len(result_segments),
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })

        print(f"Filtered out {filtered_count} hallucinated segments in silent zones.", flush=True)

        full_text = " ".join([seg["text"] for seg in result_segments])
        result = {
            "text": full_text.strip(),
            "language": info.language,
            "segments": result_segments
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    text_path, json_path = build_output_paths(output_dir, media_path)
    write_outputs(text_path, json_path, result)

    elapsed_time = time.time() - start_time_all
    print(f"Success! output={text_path} (device={active_device}, compute={args.compute_type}) - Time taken: {elapsed_time:.2f} seconds", flush=True)
    return 0

def run_transcribe_worker(args: argparse.Namespace) -> int:
    """対象となるすべてのメディアファイルを順次文字起こしするCLI向けのメイン実行ハンドラーです。"""
    media_paths = resolve_input_media(Path(args.input_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_files = load_transcribe_cache(output_dir)
    total = len(media_paths)
    skipped = 0
    processed = 0

    print(f"Found {total} media files. (Skipped: {len(completed_files & {p.name for p in media_paths})})", flush=True)

    target_paths = [p for p in media_paths if p.name not in completed_files]
    if not target_paths:
        if args.delete_audio:
            for path in media_paths:
                try:
                    print(f"Deleting already transcribed media file: {path}", flush=True)
                    path.unlink()
                except Exception as de:
                    print(f"Failed to delete {path}: {de}", file=sys.stderr, flush=True)
        print(f"Batch transcription completed. Total: {total}, Processed: 0, Skipped: {total}", flush=True)
        return 0

    model = load_whisper_model(args.model, args.device, args.compute_type)

    for i, path in enumerate(media_paths, 1):
        file_key = path.name
        if file_key in completed_files:
            print(f"[{i}/{total}] Skipping already transcribed file: {file_key}", flush=True)
            skipped += 1
            if args.delete_audio:
                try:
                    print(f"Deleting already transcribed media file: {path}", flush=True)
                    path.unlink()
                except Exception as de:
                    print(f"Failed to delete {path}: {de}", file=sys.stderr, flush=True)
            continue

        print(f"[{i}/{total}] Processing: {file_key}", flush=True)
        try:
            exit_code = transcribe_file(path, output_dir, model, args)
            if exit_code == 0:
                completed_files.add(file_key)
                save_transcribe_cache(output_dir, completed_files)
                processed += 1
                if args.delete_audio:
                    try:
                        print(f"Deleting transcribed media file: {path}", flush=True)
                        path.unlink()
                    except Exception as de:
                        print(f"Failed to delete {path}: {de}", file=sys.stderr, flush=True)
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
