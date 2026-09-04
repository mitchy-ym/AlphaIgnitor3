from __future__ import annotations

import argparse
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from . import utils
from .downloader import download_and_extract, get_live_videos, get_videos_from_url, load_download_cache, save_download_cache
from .enricher import enrich_transcript_file
from .merge import merge_directory
from .transcriber import load_whisper_model, transcribe_file
from .utils import (
    MEDIA_EXTENSIONS,
    ProgressPositionManager,
    log_error,
    log_info,
    log_warn,
    sanitize_channel_title,
    sanitize_cookie_file,
)


def _resolve_channel_dir(base_dir: Path, safe_channel_title: str) -> Path:
    if base_dir.name == safe_channel_title:
        return base_dir
    return base_dir / safe_channel_title


def _existing_transcript_video_ids(videos: list[dict], transcript_dir: Path) -> set[str]:
    completed_ids: set[str] = set()
    if not transcript_dir.exists():
        return completed_ids
    existing_names = {path.name for path in transcript_dir.iterdir() if path.is_file() and path.suffix == ".txt"}
    for video in videos:
        video_id = video["id"]
        if any(f"_{video_id}" in name for name in existing_names):
            completed_ids.add(video_id)
    return completed_ids


def _find_pending_media_files(download_dir: Path, transcript_dir: Path) -> list[Path]:
    pending: list[Path] = []
    if not download_dir.exists():
        return pending
    for path in sorted(download_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        transcript_path = transcript_dir / f"{path.stem}.txt"
        if not transcript_path.exists():
            pending.append(path)
    return pending


def _find_incomplete_enrich_transcripts(transcript_dir: Path, enrich_output_dir: Path) -> list[Path]:
    incomplete: list[Path] = []
    if not transcript_dir.exists():
        return incomplete
    for transcript_path in sorted(transcript_dir.glob("*.txt")):
        channel_dir = enrich_output_dir / transcript_dir.name
        clean_path = channel_dir / "clean" / f"{transcript_path.stem}.txt"
        summary_path = channel_dir / "summary" / f"{transcript_path.stem}.md"
        chapters_path = channel_dir / "chapters" / f"{transcript_path.stem}.md"
        if not (clean_path.exists() and summary_path.exists() and chapters_path.exists()):
            incomplete.append(transcript_path)
    return incomplete


def _build_transcribe_args(args: argparse.Namespace, transcript_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(transcript_dir),
        model=args.transcribe_model,
        language=args.transcribe_language,
        device=args.transcribe_device,
        compute_type=args.transcribe_compute_type,
        task=args.transcribe_task,
        beam_size=args.transcribe_beam_size,
        batch_size=args.transcribe_batch_size,
        json=args.transcribe_json,
        delete_audio=False,
        initial_prompt=args.transcribe_initial_prompt,
        vad_threshold=args.transcribe_vad_threshold,
        chunk_duration=args.transcribe_chunk_duration,
        min_rms=args.transcribe_min_rms,
    )


def _build_enrich_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
        llm_max_tokens=args.llm_max_tokens,
        llm_api_key_env=args.llm_api_key_env,
        max_chars=args.max_chars,
        force=args.force,
    )


def _process_media_file(
    media_path: Path,
    transcript_dir: Path,
    enrich_output_dir: Path,
    transcribe_model,
    transcribe_args: argparse.Namespace,
    enrich_args: argparse.Namespace,
    keep_audio: bool,
    enrich_enabled: bool,
) -> None:
    log_info(f"文字起こしを開始します: {media_path.name}")
    exit_code = transcribe_file(media_path, transcript_dir, transcribe_model, transcribe_args)
    if exit_code != 0:
        raise RuntimeError(f"Transcription failed with exit code {exit_code}: {media_path.name}")

    transcript_path = transcript_dir / f"{media_path.stem}.txt"
    if enrich_enabled:
        enrich_transcript_file(transcript_path, enrich_output_dir, enrich_args)

    if not keep_audio and media_path.exists():
        media_path.unlink()


def _enrich_existing_transcripts(transcript_dir: Path, enrich_output_dir: Path, enrich_args: argparse.Namespace) -> None:
    if not transcript_dir.exists():
        return
    for transcript_path in sorted(transcript_dir.glob("*.txt")):
        enrich_transcript_file(transcript_path, enrich_output_dir, enrich_args)


def run_pipeline(args: argparse.Namespace) -> int:
    """ダウンロード -> 文字起こし -> Ollama enrich の多段パイプラインを実行します。"""
    utils.SHOW_PROGRESS_TEXT = getattr(args, "verbose_progress", False)

    cookie_file = getattr(args, "cookies", None)
    if not cookie_file and Path("cookies/cookies.txt").is_file():
        cookie_file = "cookies/cookies.txt"
    if cookie_file:
        sanitize_cookie_file(cookie_file)

    try:
        if getattr(args, "video_url", None):
            if getattr(args, "channel_handle", None):
                log_warn("--video-url が指定されているため channel_handle は無視されます。")
            videos, channel_title = get_videos_from_url(args.video_url, args.cookies_from_browser, cookie_file=cookie_file)
        else:
            if not getattr(args, "channel_handle", None):
                log_error("channel_handle または --video-url のいずれかを指定してください。")
                return 1
            videos, channel_title = get_live_videos(args.channel_handle, args.cookies_from_browser, cookie_file=cookie_file)
    except Exception as exc:
        log_error(f"エラーが発生したため処理を中断します: {exc}")
        if getattr(args, "debug", False):
            import traceback

            traceback.print_exc()
        return 1

    safe_channel_title = sanitize_channel_title(channel_title, args.channel_handle or args.video_url)
    download_dir = Path(args.output) if args.output else Path("downloads") / safe_channel_title
    download_dir.mkdir(parents=True, exist_ok=True)

    transcript_dir = _resolve_channel_dir(Path(args.transcribe_output_dir), safe_channel_title)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    enrich_output_dir = Path(args.enrich_output_dir)

    safe_handle_source = args.channel_handle or args.video_url or "unknown"
    safe_handle_filename = "".join(c for c in safe_handle_source if c.isalnum() or c in ("_", "-"))
    cache_file = download_dir / f"download_cache_{safe_handle_filename}.json"
    cache_data = load_download_cache(cache_file, safe_handle_source)
    downloaded_ids = set(cache_data.get("downloaded_ids", []))
    transcript_ids = _existing_transcript_video_ids(videos, transcript_dir)

    target_videos = [v for v in videos if v["id"] not in downloaded_ids and v["id"] not in transcript_ids]
    skipped_count = len(videos) - len(target_videos)
    if args.max_downloads is not None:
        target_videos = target_videos[: args.max_downloads]

    log_info(f"設定 - フォーマット: {args.format}, ビットレート: {args.bitrate}, 保存先: {download_dir}")
    log_info(f"ライブ動画を {len(videos)} 本検出しました。（うち {skipped_count} 本はキャッシュまたは transcript によりスキップ）")
    log_info(f"今回の処理対象: {len(target_videos)} 本")

    transcribe_args = _build_transcribe_args(args, transcript_dir)
    enrich_args = _build_enrich_args(args)

    media_queue: queue.Queue[Path | None] = queue.Queue()
    failures: list[tuple[str, str]] = []
    state_lock = threading.Lock()

    def media_worker() -> None:
        model = None
        while True:
            media_path = media_queue.get()
            try:
                if media_path is None:
                    return
                if model is None:
                    model = load_whisper_model(
                        transcribe_args.model,
                        transcribe_args.device,
                        transcribe_args.compute_type,
                    )
                _process_media_file(
                    media_path=media_path,
                    transcript_dir=transcript_dir,
                    enrich_output_dir=enrich_output_dir,
                    transcribe_model=model,
                    transcribe_args=transcribe_args,
                    enrich_args=enrich_args,
                    keep_audio=args.transcribe_keep_audio,
                    enrich_enabled=args.enrich,
                )
            except Exception as exc:
                with state_lock:
                    failures.append((str(media_path) if media_path else "<sentinel>", str(exc)))
                log_warn(f"パイプライン処理に失敗しました: {exc}")
            finally:
                media_queue.task_done()

    worker_thread = threading.Thread(target=media_worker, daemon=True)
    worker_thread.start()

    for pending_path in _find_pending_media_files(download_dir, transcript_dir):
        log_info(f"未文字起こしの既存音声をキューへ追加します: {pending_path.name}")
        media_queue.put(pending_path)

    success_count = 0
    fail_count = 0
    cache_lock = threading.Lock()
    pos_manager = ProgressPositionManager(args.max_workers)
    overall_pbar = None
    if not args.verbose_progress:
        overall_pbar = tqdm(total=len(target_videos), desc="Overall Progress", position=0, leave=True)

    def process_video(video: dict, index: int) -> None:
        nonlocal success_count, fail_count
        if index > 1:
            stagger_time = random.uniform(1.0, 4.0)
            log_info(f"スレッド [{index}]: リクエスト集中防止のため {stagger_time:.1f} 秒待機してから開始します...")
            time.sleep(stagger_time)

        pbar_pos = None
        if not args.verbose_progress:
            pbar_pos = pos_manager.acquire()

        try:
            success, is_unrecoverable, downloaded_file = download_and_extract(
                video=video,
                output_dir=download_dir,
                audio_format=args.format,
                bitrate=args.bitrate,
                cookies_browser=args.cookies_from_browser,
                ffmpeg_location=args.ffmpeg_location,
                current_index=index,
                total_count=len(target_videos),
                pbar_position=pbar_pos,
                debug=args.debug,
                cookie_file=cookie_file,
            )
        finally:
            if pbar_pos is not None:
                pos_manager.release(pbar_pos)

        with cache_lock:
            if success:
                success_count += 1
                if video["id"] not in cache_data["downloaded_ids"]:
                    cache_data["downloaded_ids"].append(video["id"])
                cache_data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                save_download_cache(cache_file, cache_data)
                if downloaded_file is not None:
                    media_queue.put(downloaded_file)
            else:
                fail_count += 1
                if is_unrecoverable and video["id"] not in cache_data["downloaded_ids"]:
                    cache_data["downloaded_ids"].append(video["id"])
                    cache_data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    save_download_cache(cache_file, cache_data)

            if overall_pbar is not None:
                overall_pbar.update(1)

    if target_videos:
        log_info(f"並行ダウンロードを開始します (スレッド数: {args.max_workers})...")
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(process_video, video, index) for index, video in enumerate(target_videos, start=1)]
                for future in futures:
                    future.result()
        finally:
            if overall_pbar is not None:
                overall_pbar.close()
    else:
        log_info("ダウンロード対象の新規動画はありません。既存音声/文字起こしの後続処理だけを行います。")
        if overall_pbar is not None:
            overall_pbar.close()

    log_info(f"ダウンロード処理が完了しました。成功: {success_count} 本, 失敗: {fail_count} 本")

    media_queue.join()
    media_queue.put(None)
    media_queue.join()
    worker_thread.join(timeout=2.0)

    if args.enrich:
        try:
            _enrich_existing_transcripts(transcript_dir, enrich_output_dir, enrich_args)
        except Exception as exc:
            failures.append((str(transcript_dir), str(exc)))
            log_warn(f"既存 transcript の enrich に失敗しました: {exc}")

    if getattr(args, "merge", True):
        try:
            mode = getattr(args, "merge_mode", "yearly")
            strip_ts = getattr(args, "merge_strip_timestamps", False)

            # 1. transcripts 配下のテキスト結合
            if transcript_dir.exists():
                log_info(f"文字起こしテキストの結合（{mode}）を実行します: {transcript_dir}")
                merge_directory(input_dir=transcript_dir, mode=mode, strip_timestamps=strip_ts)

            # 2. enrich (clean) 配下のテキスト結合
            clean_dir = enrich_output_dir / safe_channel_title / "clean"
            if args.enrich and clean_dir.exists():
                log_info(f"enrich(clean) テキストの結合（{mode}）を実行します: {clean_dir}")
                merge_directory(input_dir=clean_dir, mode=mode, strip_timestamps=strip_ts)
        except Exception as exc:
            failures.append((str(safe_channel_title), str(exc)))
            log_warn(f"テキスト結合処理に失敗しました: {exc}")

    if fail_count > 0:
        failures.append((str(download_dir), f"download stage failed for {fail_count} video(s)"))

    pending_media = _find_pending_media_files(download_dir, transcript_dir)
    if pending_media:
        log_warn(f"まだ文字起こしされていない音声ファイルがあります: {len(pending_media)} 件")
        for path in pending_media:
            log_warn(f"  未処理音声: {path}")

    if args.enrich:
        incomplete_enrich = _find_incomplete_enrich_transcripts(transcript_dir, enrich_output_dir)
        if incomplete_enrich:
            log_warn(f"まだenrichされていない transcript があります: {len(incomplete_enrich)} 件")
            for path in incomplete_enrich:
                log_warn(f"  未enrich transcript: {path}")

    if failures:
        for path, message in failures:
            log_error(f"失敗: {path} - {message}")
        return 1

    return 0
