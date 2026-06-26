import queue
import threading
import time
import sys
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from . import utils
from .utils import (
    ProgressPositionManager,
    log_info,
    log_warn,
    log_error,
    log_success,
    sanitize_cookie_file
)
from .downloader import (
    get_live_videos,
    download_and_extract,
    load_download_cache,
    save_download_cache
)
from .transcriber import (
    load_transcribe_cache,
    load_whisper_model,
    transcribe_file
)
from .consistency import check_consistency_report

def run_pipeline(args: argparse.Namespace) -> int:
    """ダウンロード -> デコード -> 文字起こしの3ステージ非同期パイプラインを実行します。"""
    utils.SHOW_PROGRESS_TEXT = getattr(args, "verbose_progress", True)

    # Cookieファイルの決定
    cookie_file = getattr(args, "cookies", None)
    if not cookie_file and not getattr(args, "cookies_from_browser", None):
        default_cookies = Path("cookies/cookies.txt")
        if default_cookies.exists():
            cookie_file = str(default_cookies)
            
    if cookie_file:
        sanitize_cookie_file(cookie_file)

    # 1. 動画一覧の取得
    try:
        videos, channel_title = get_live_videos(args.channel_handle, args.cookies_from_browser, cookie_file=cookie_file)
    except Exception as e:
        log_error(0, 0, f"エラーが発生したため処理を中断します: {e}")
        if getattr(args, "debug", False):
            import traceback
            traceback.print_exc()
        return 1

    # 2. 保存フォルダの決定
    safe_channel_title = "".join(c for c in channel_title if c.isalnum() or c in (" ", "_", "-")).strip()
    if not safe_channel_title:
        safe_channel_title = args.channel_handle.replace("@", "")

    if getattr(args, "output", None):
        output_dir = Path(args.output)
    else:
        output_dir = Path("downloads") / safe_channel_title

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2.5. トランスクリプト出力フォルダの解決（配信者ごとにフォルダを分ける）
    t_out_base = Path(args.transcribe_output_dir)
    if t_out_base.name != safe_channel_title:
        t_out_dir_path = t_out_base / safe_channel_title
    else:
        t_out_dir_path = t_out_base
    args.transcribe_output_dir = str(t_out_dir_path)
    
    # 3. キャッシュファイルの読み込み
    safe_handle_filename = "".join(c for c in args.channel_handle if c.isalnum() or c in ("_", "-"))
    cache_file = output_dir / f"download_cache_{safe_handle_filename}.json"
    cache_data = load_download_cache(cache_file, args.channel_handle)

    downloaded_ids = set(cache_data.get("downloaded_ids", []))
    
    is_retry_run = False
    target_videos = []

    # 整合性確認オプションの処理
    if getattr(args, "check_consistency", False):
        t_out_dir = Path(args.transcribe_output_dir)
        exit_code, retry_video_ids = check_consistency_report(args, videos, channel_title, downloaded_ids, output_dir, t_out_dir)
        if exit_code != 0:
            return exit_code
        if retry_video_ids:
            target_videos = [v for v in videos if v["id"] in retry_video_ids]
            is_retry_run = True
        else:
            return 0

    if not is_retry_run:
        # すでに文字起こし結果（.txt または .json）が出力先フォルダに存在する動画IDを検出する
        t_out_dir = Path(args.transcribe_output_dir)
        completed_video_ids = set()
        if t_out_dir.exists():
            existing_files = {p.name for p in t_out_dir.iterdir() if p.is_file() and p.suffix in (".txt", ".json")}
            for v in videos:
                v_id = v["id"]
                for fname in existing_files:
                    if f"_{v_id}" in fname:
                        completed_video_ids.add(v_id)
                        break

        # すでに処理済みの動画をフィルタリング（ダウンロードキャッシュにある、またはすでに文字起こしテキストが存在する）
        for v in videos:
            v_id = v["id"]
            if v_id in downloaded_ids or v_id in completed_video_ids:
                continue
            target_videos.append(v)
        
        skipped_count = len(videos) - len(target_videos)
        if getattr(args, "max_downloads", None) is not None:
            target_videos = target_videos[:args.max_downloads]

        log_info(f"設定 - フォーマット: {args.format}, ビットレート: {args.bitrate}, 保存先: {output_dir}")
        log_info(f"ライブ動画を {len(videos)} 本検出しました。（うち {skipped_count} 本は処理済みのキャッシュによりスキップ）")
        log_info(f"今回の処理対象: {len(target_videos)} 本")
    else:
        log_info(f"設定 - フォーマット: {args.format}, ビットレート: {args.bitrate}, 保存先: {output_dir}")
        log_info(f"【再試行モード】キャッシュ済・文字起こし無の動画 {len(target_videos)} 本を再処理します。")

    # キューと同期オブジェクトの初期化
    decode_queue = queue.Queue()
    transcribe_queue = queue.Queue(maxsize=args.max_workers * 2)  # メモリ制限のため、デコード済み音声のバッファ数を制限
    
    def decode_worker():
        from faster_whisper.audio import decode_audio
        while True:
            item = decode_queue.get()
            if item is None:
                break
            media_path, t_out_dir_path, transcribe_args = item
            try:
                log_info(f"オーディオのデコードを開始します: {media_path.name}")
                start_decode = time.time()
                audio_data = decode_audio(str(media_path), sampling_rate=16000)
                log_info(f"オーディオのデコード完了: {media_path.name} (所要時間: {time.time() - start_decode:.2f}秒, 形状: {audio_data.shape})")
                transcribe_queue.put((media_path, t_out_dir_path, transcribe_args, audio_data))
            except Exception as e:
                log_error(0, 0, f"オーディオのデコード中にエラーが発生しました ({media_path.name}): {e}")
                transcribe_queue.put((media_path, t_out_dir_path, transcribe_args, None))
            finally:
                decode_queue.task_done()
    
    def transcribe_worker():
        model = None
        while True:
            item = transcribe_queue.get()
            if item is None:
                break
            media_path, t_out_dir_path, transcribe_args, preloaded_audio = item
            try:
                if model is None:
                    model = load_whisper_model(
                        transcribe_args.transcribe_model,
                        transcribe_args.transcribe_device,
                        transcribe_args.transcribe_compute_type
                    )

                import argparse as ap
                t_args = ap.Namespace(
                    input_path=str(media_path),
                    output_dir=t_out_dir_path,
                    model=transcribe_args.transcribe_model,
                    language=transcribe_args.transcribe_language,
                    device=transcribe_args.transcribe_device,
                    compute_type=transcribe_args.transcribe_compute_type,
                    task=transcribe_args.transcribe_task,
                    beam_size=transcribe_args.transcribe_beam_size,
                    batch_size=transcribe_args.transcribe_batch_size,
                    delete_audio=not getattr(transcribe_args, "transcribe_keep_audio", False),
                    initial_prompt=transcribe_args.transcribe_initial_prompt,
                    vad_threshold=transcribe_args.transcribe_vad_threshold,
                    chunk_duration=transcribe_args.transcribe_chunk_duration
                )
                log_info(f"文字起こしを開始します: {media_path.name}")
                transcribe_file(media_path, t_out_dir_path, model, t_args, preloaded_audio=preloaded_audio)
                if not getattr(transcribe_args, "transcribe_keep_audio", False):
                    try:
                        log_info(f"文字起こしが完了したため、音声ファイルを削除します: {media_path.name}")
                        media_path.unlink()
                    except Exception as de:
                        log_warn(f"音声ファイルの削除に失敗しました: {de}")
            except Exception as ex:
                log_error(0, 0, f"文字起こし処理中に例外が発生しました ({media_path.name}): {ex}")
                import traceback
                traceback.print_exc()
            finally:
                transcribe_queue.task_done()

    num_threads = args.max_workers
    transcribe_threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=transcribe_worker, daemon=True)
        t.start()
        transcribe_threads.append(t)
    
    decode_threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=decode_worker, daemon=True)
        t.start()
        decode_threads.append(t)

    # 既存ファイルのうち、未文字起こしのファイルを検出してデコードキューへ
    try:
        t_out_dir = Path(args.transcribe_output_dir)
        completed_transcripts = load_transcribe_cache(t_out_dir)
        
        from .utils import MEDIA_EXTENSIONS
        existing_files = []
        if output_dir.exists():
            for path in output_dir.iterdir():
                if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
                    if path.name not in completed_transcripts:
                        existing_files.append(path)
        
        if existing_files:
            log_info(f"未文字起こしの既存ファイルを {len(existing_files)} 件検出しました。デコードキューに追加します。")
            for path in existing_files:
                decode_queue.put((path, t_out_dir, args))
    except Exception as e:
        log_warn(f"未文字起こしの既存ファイルの検出中にエラーが発生しました: {e}")

    if not target_videos:
        log_info("ダウンロード対象の新規動画はありません。")
        log_info("未完了のデコード処理と文字起こし処理の完了を待機しています...")
        decode_queue.join()
        for _ in range(num_threads):
            decode_queue.put(None)
        for t in decode_threads:
            t.join()
        
        transcribe_queue.join()
        for _ in range(num_threads):
            transcribe_queue.put(None)
        for t in transcribe_threads:
            t.join()
        return 0

    success_count = 0
    fail_count = 0
    cache_lock = threading.Lock()

    max_workers = args.max_workers
    pos_manager = ProgressPositionManager(max_workers)

    overall_pbar = None
    if not getattr(args, "verbose_progress", True):
        overall_pbar = tqdm(
            total=len(target_videos),
            desc="Overall Progress",
            position=0,
            leave=True
        )

    def process_video(video, idx):
        nonlocal success_count, fail_count
        video_id = video["id"]
        
        # ディスク容量節約のための流量制限:
        # ディスク上の未処理メディアファイルが制限に達している場合は待機
        max_active_files = args.max_workers * 2
        from .utils import MEDIA_EXTENSIONS
        wait_logged = False
        while True:
            active_media_files = []
            if output_dir.exists():
                for p in output_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS:
                        txt_path = Path(args.transcribe_output_dir) / f"{p.stem}.txt"
                        json_path = Path(args.transcribe_output_dir) / f"{p.stem}.json"
                        if not (txt_path.exists() or json_path.exists()):
                            active_media_files.append(p)
            if len(active_media_files) < max_active_files:
                break
            if not wait_logged:
                log_info(f"スレッド [{idx}]: ディスク上の未処理ファイルが上限（{max_active_files}本）に達しているため、処理完了を待機しています...")
                wait_logged = True
            time.sleep(2.0)
        
        # YouTubeのBAN対策: stagger
        if idx > 1:
            stagger_time = random.uniform(1.0, 4.0)
            log_info(f"スレッド [{idx}]: リクエスト集中防止のため {stagger_time:.1f} 秒待機してから開始します...")
            time.sleep(stagger_time)

        pbar_pos = None
        if not getattr(args, "verbose_progress", True):
            pbar_pos = pos_manager.acquire()

        try:
            # ダウンロード実行
            success, is_unrecoverable, downloaded_file = download_and_extract(
                video=video,
                output_dir=output_dir,
                audio_format=args.format,
                bitrate=args.bitrate,
                cookies_browser=args.cookies_from_browser,
                ffmpeg_location=args.ffmpeg_location,
                current_index=idx,
                total_count=len(target_videos),
                pbar_position=pbar_pos,
                debug=getattr(args, "debug", False),
                cookie_file=cookie_file
            )
        finally:
            if pbar_pos is not None:
                pos_manager.release(pbar_pos)

        with cache_lock:
            if success:
                success_count += 1
                cache_data["downloaded_ids"].append(video_id)
                cache_data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                save_download_cache(cache_file, cache_data)
                
                if downloaded_file is not None:
                    log_info(f"ダウンロード完了ファイルをデコードキューに追加します: {downloaded_file.name}")
                    decode_queue.put((downloaded_file, Path(args.transcribe_output_dir), args))
            else:
                fail_count += 1
                if is_unrecoverable:
                    log_info(f"動画 (ID: {video_id}) は回復不能なエラーのため、次回以降はスキップします。")
                    cache_data["downloaded_ids"].append(video_id)
                    cache_data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    save_download_cache(cache_file, cache_data)
            
            if overall_pbar is not None:
                overall_pbar.update(1)

    log_info(f"並行ダウンロードを開始します (スレッド数: {max_workers})...")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_video, video, idx) for idx, video in enumerate(target_videos, start=1)]
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        log_info("処理がユーザーによって中断されました。キャッシュを保存して終了します。")
        sys.exit(0)
    finally:
        if overall_pbar is not None:
            overall_pbar.close()

    log_info(f"すべてのダウンロード処理が完了しました。合計: {success_count}本成功、{fail_count}本失敗。")

    log_info("デコードおよび文字起こし処理の完了を待機しています...")
    decode_queue.join()
    for _ in range(num_threads):
        decode_queue.put(None)
    for t in decode_threads:
        t.join()
    
    transcribe_queue.join()
    for _ in range(num_threads):
        transcribe_queue.put(None)
    for t in transcribe_threads:
        t.join()
    log_info("すべての文字起こし処理が完了しました。")
    return 0
