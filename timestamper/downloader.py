import json
import random
import time
from pathlib import Path
import yt_dlp
from tqdm import tqdm
import argparse

from . import utils
from .utils import log_info, log_progress, log_success, log_error, log_warn, get_timestamp, sanitize_cookie_file

def load_download_cache(cache_file: Path, channel_handle: str) -> dict:
    """ダウンロード済みの動画ID履歴キャッシュを読み込みます。"""
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "downloaded_ids" not in data:
                    data["downloaded_ids"] = []
                return data
        except Exception as e:
            log_warn(f"キャッシュファイルの読み込みに失敗しました ({e})。新規作成します。")
    return {
        "channel_handle": channel_handle,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "downloaded_ids": []
    }

def save_download_cache(cache_file: Path, cache_data: dict):
    """ダウンロード済みの動画ID履歴キャッシュを保存します。"""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        temp_file.replace(cache_file)
    except Exception as e:
        log_warn(f"キャッシュファイルの保存に失敗しました: {e}")

def get_live_videos(channel_handle: str, cookies_browser: str | None, cookie_file: str | Path | None = None) -> tuple[list[dict], str]:
    """指定されたチャンネル識別子から、アーカイブ配信完了したライブ動画のリストとチャンネルタイトルを取得します。"""
    handle = channel_handle if channel_handle.startswith("@") else f"@{channel_handle}"
    url = f"https://www.youtube.com/{handle}/streams"

    log_info(f"チャンネル {handle} の解析を開始します (URL: {url})...")

    ydl_opts = {
        "extract_flat": True,
        "playlistend": None,
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)
    elif cookie_file:
        ydl_opts["cookiefile"] = str(cookie_file)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            log_error(0, 0, f"チャンネルの解析に失敗しました。チャンネル名やネットワーク接続を確認してください: {e}")
            raise e

        channel_title = info.get("title", handle)
        entries = info.get("entries", [])
        
        valid_videos = []
        for entry in entries:
            if not entry:
                continue
            
            video_id = entry.get("id")
            title = entry.get("title")
            
            if not video_id or not title:
                continue

            live_status = entry.get("live_status")
            is_live = entry.get("is_live")
            duration = entry.get("duration")

            # 配信中、配信予定、およびアーカイブが完了していない動画を除外
            if live_status in ("is_live", "live", "is_upcoming", "upcoming") or is_live is True or duration is None or duration == 0:
                continue

            valid_videos.append({
                "id": video_id,
                "title": title,
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            })
            
        return valid_videos, channel_title

def make_progress_hook(current_idx: int, total_cnt: int, video_title: str, pbar=None):
    """yt-dlp のダウンロード進捗フックを構築します。"""
    last_percent = -10
    
    def hook(d):
        nonlocal last_percent
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            
            if pbar is not None:
                if total is not None and pbar.total != total:
                    pbar.total = total
                pbar.update(downloaded - pbar.n)
            
            if utils.SHOW_PROGRESS_TEXT and total:
                percent = int((downloaded / total) * 100)
                displayed_percent = (percent // 10) * 10
                if displayed_percent > last_percent:
                    last_percent = displayed_percent
                    eta = d.get('eta')
                    eta_str = f"残り時間: {int(eta//60):02d}:{int(eta%60):02d}" if eta is not None else "残り時間: 不明"
                    print(f"[PROGRESS] {get_timestamp()} - [{current_idx}/{total_cnt}] 進捗: {displayed_percent}% ({eta_str})", flush=True)
                    
        elif d['status'] == 'finished':
            if pbar is not None:
                pbar.desc = f"[{current_idx}/{total_cnt}] {video_title[:30]} (Extracting...)"
                if pbar.total is not None:
                    pbar.update(pbar.total - pbar.n)
    return hook

def download_and_extract(
    video: dict,
    output_dir: Path,
    audio_format: str,
    bitrate: str,
    cookies_browser: str | None,
    ffmpeg_location: str | None,
    current_index: int,
    total_count: int,
    pbar_position: int | None = None,
    debug: bool = False,
    cookie_file: str | Path | None = None
) -> tuple[bool, bool | None, Path | None]:
    """単一の動画をダウンロードし、指定の形式で音声を抽出します。"""
    video_id = video["id"]
    title = video["title"]
    url = video["url"]
    quality = bitrate.rstrip("kK").rstrip("bpsBPS")
    outtmpl_str = str(output_dir / "%(upload_date)s_%(title).120s_%(id)s.%(ext)s")

    pbar = None
    if pbar_position is not None:
        pbar = tqdm(
            total=None,
            desc=f"[{current_index}/{total_count}] {title[:30]}",
            position=pbar_position,
            leave=False,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl_str,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "skip_unavailable_fragments": True,
        "concurrent_fragment_downloads": 5,
        "continuedl": True,
        "no_part": False,
        "quiet": not debug,
        "no_warnings": not debug,
        "noprogress": not debug,
        "progress_hooks": [make_progress_hook(current_index, total_count, title, pbar)] if not debug else [],
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": quality,
        }],
    }

    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)
    elif cookie_file:
        ydl_opts["cookiefile"] = str(cookie_file)
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    log_progress(current_index, total_count, f"ダウンロード中: \"{title}\" (ID: {video_id})")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # PyAV/FFmpeg のバグ回避（ファイル名に '#' が含まれるとオープンできない）用のリネーム
        for path in output_dir.iterdir():
            if path.is_file() and video_id in path.name and "#" in path.name:
                new_name = path.name.replace("#", "_")
                new_path = path.with_name(new_name)
                try:
                    path.rename(new_path)
                    log_info(f"ファイル名を安全化しました: \"{new_name}\"")
                except Exception as rename_err:
                    log_warn(f"ファイル名のリネームに失敗しました: {rename_err}")

        log_success(current_index, total_count, f"音声抽出完了: \"{title}\" ({audio_format})")
        
        downloaded_file = None
        for path in output_dir.iterdir():
            if path.is_file() and video_id in path.name:
                downloaded_file = path
                break
        return True, None, downloaded_file
    except Exception as e:
        error_msg = str(e)
        unrecoverable_keywords = [
            "unavailable", "private", "removed", "deleted", "copyright", 
            "members-only", "sign in", "confirm your age", "age-restricted",
            "blocked", "not available", "403: Forbidden"
        ]
        is_unrecoverable = any(kw in error_msg.lower() for kw in unrecoverable_keywords)
        log_error(current_index, total_count, f"ダウンロード失敗: \"{title}\" (理由: {error_msg})")
        return False, is_unrecoverable, None
    finally:
        if pbar is not None:
            pbar.close()

def run_downloader(args: argparse.Namespace) -> int:
    """YouTubeの動画を取得し、並行ダウンロード処理を行うCLI向けのメイン実行ハンドラーです。"""
    utils.SHOW_PROGRESS_TEXT = args.verbose_progress

    # Cookieファイルの決定
    cookie_file = getattr(args, "cookies", None)
    
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

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("downloads") / safe_channel_title

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. キャッシュファイルの読み込み
    safe_handle_filename = "".join(c for c in args.channel_handle if c.isalnum() or c in ("_", "-"))
    cache_file = output_dir / f"download_cache_{safe_handle_filename}.json"
    cache_data = load_download_cache(cache_file, args.channel_handle)

    downloaded_ids = set(cache_data.get("downloaded_ids", []))
    
    # すでに処理済みの動画をフィルタリング
    target_videos = [v for v in videos if v["id"] not in downloaded_ids]
    skipped_count = len(videos) - len(target_videos)
    if getattr(args, "max_downloads", None) is not None:
        target_videos = target_videos[:args.max_downloads]

    log_info(f"設定 - フォーマット: {args.format}, ビットレート: {args.bitrate}, 保存先: {output_dir}")
    log_info(f"ライブ動画を {len(videos)} 本検出しました。（うち {skipped_count} 本は処理済みのキャッシュによりスキップ）")
    log_info(f"今回の処理対象: {len(target_videos)} 本")

    if not target_videos:
        log_info("処理対象の動画はありません。処理を終了します。")
        return 0

    import threading
    from concurrent.futures import ThreadPoolExecutor
    from .utils import ProgressPositionManager

    success_count = 0
    fail_count = 0
    cache_lock = threading.Lock()

    max_workers = args.max_workers
    pos_manager = ProgressPositionManager(max_workers)

    overall_pbar = None
    if not args.verbose_progress:
        overall_pbar = tqdm(
            total=len(target_videos),
            desc="Overall Progress",
            position=0,
            leave=True
        )

    def process_video(video, idx):
        nonlocal success_count, fail_count
        video_id = video["id"]
        
        if idx > 1:
            stagger_time = random.uniform(1.0, 4.0)
            log_info(f"スレッド [{idx}]: リクエスト集中防止のため {stagger_time:.1f} 秒待機してから開始します...")
            time.sleep(stagger_time)

        pbar_pos = None
        if not args.verbose_progress:
            pbar_pos = pos_manager.acquire()

        try:
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
                debug=args.debug,
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
    return 0
