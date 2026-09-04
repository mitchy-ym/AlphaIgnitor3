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

from .utils import format_seconds, get_media_duration, log_info, log_success, log_warn


class AudioChunk:
    def __init__(self, index: int, start_time: float, duration: float, audio_data: np.ndarray):
        self.index = index
        self.start_time = start_time
        self.duration = duration
        self.audio_data = audio_data


class ChunkError:
    def __init__(self, exception: Exception):
        self.exception = exception

def select_device(device: str) -> str:
    """GPU 前提でデバイスを決定し、GPU が使えなければ例外を送出します。"""
    if device == "cpu":
        raise ValueError("CPU execution is disabled. Use GPU (cuda) only.")

    if device not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:
        raise RuntimeError(f"Failed to detect GPU availability: {exc}") from exc

    raise RuntimeError("GPU is not available. CPU fallback is disabled.")

def load_whisper_model(model_path: str, device: str, compute_type: str) -> "BatchedInferencePipeline":
    """Whisper モデルを読み込み、BatchedInferencePipeline でラップして返します。"""
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    active_device = select_device(device)
    
    if "/" in model_path and not Path(model_path).exists():
        try:
            from huggingface_hub import snapshot_download
            log_info(f"Downloading model '{model_path}' from Hugging Face Hub...")
            model_path = snapshot_download(repo_id=model_path)
        except Exception as e:
            log_warn(f"Failed to pre-download model using huggingface_hub: {e}. Falling back to default loader.")
            
    log_info(f"Loading faster-whisper model '{model_path}' on {active_device} (compute_type={compute_type})...")
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



def build_output_paths(output_dir: Path, media_path: Path, output_json: bool = False) -> tuple[Path, Path | None]:
    """出力するテキストと（必要なら）JSONの保存パスを構築します。"""
    base_name = media_path.stem
    json_path = output_dir / f"{base_name}.json" if output_json else None
    return output_dir / f"{base_name}.txt", json_path

def write_outputs(text_path: Path, json_path: Path | None, result: dict) -> None:
    """文字起こし結果をテキストおよび（指定時のみ）JSON形式で出力します。"""
    lines = []
    for seg in result.get("segments", []):
        start_str = format_seconds(seg.get("start", 0.0))
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{start_str}] {text}")

    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_path is not None:
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
    active_compute_type = args.compute_type
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
                log_warn(f"Could not determine total duration of {media_path.name}. Falling back to sequential mode.")
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

        log_info(f"Asynchronous processing enabled: splitting {media_path.name} into {len(chunks)} chunks of max {chunk_duration}s.")

        chunk_queue: queue.Queue[AudioChunk | ChunkError | None] = queue.Queue(maxsize=2)

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

        log_info(f"Starting batched Whisper transcription (batch_size={args.batch_size})...")

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
    text_path, json_path = build_output_paths(output_dir, media_path, output_json=getattr(args, "json", False))
    write_outputs(text_path, json_path, result)

    elapsed_time = time.time() - start_time_all
    log_success(f"文字起こし完了: {text_path.name} (device={active_device}, compute={active_compute_type}, 所要時間: {elapsed_time:.2f}秒)")
    return 0
