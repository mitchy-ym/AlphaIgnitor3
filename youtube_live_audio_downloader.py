import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path
import yt_dlp


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(message: str):
    print(f"[INFO] {get_timestamp()} - {message}", flush=True)


def log_progress(current: int, total: int, message: str):
    print(f"[PROGRESS] {get_timestamp()} - [{current}/{total}] {message}", flush=True)


def log_success(current: int, total: int, message: str):
    print(f"[SUCCESS] {get_timestamp()} - [{current}/{total}] {message}", flush=True)


def log_error(current: int, total: int, message: str):
    print(f"[ERROR] {get_timestamp()} - [{current}/{total}] {message}", flush=True)


def log_warn(message: str):
    print(f"[WARN] {get_timestamp()} - {message}", flush=True)


def load_cache(cache_file: Path, channel_handle: str) -> dict:
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
        "last_updated": datetime.datetime.now().isoformat(),
        "downloaded_ids": []
    }


def save_cache(cache_file: Path, cache_data: dict):
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # 一時ファイルに書いてからリネームする安全な書き込み
        temp_file = cache_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        temp_file.replace(cache_file)
    except Exception as e:
        log_warn(f"キャッシュファイルの保存に失敗しました: {e}")


def get_live_videos(channel_handle: str, cookies_browser: str | None) -> list[dict]:
    # @から始まるハンドルであるか確認し、URLを構築
    # YouTubeの配信アーカイブ一覧ページ: https://www.youtube.com/@ChannelHandle/streams
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

            # 配信中および配信予定の動画は除外
            live_status = entry.get("live_status")
            is_live = entry.get("is_live")
            duration = entry.get("duration")

            if live_status in ("is_live", "live", "is_upcoming", "upcoming") or is_live is True or duration is None or duration == 0:
                # アーカイブ完了していないものはスキップ
                continue

            valid_videos.append({
                "id": video_id,
                "title": title,
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            })
            
        return valid_videos, channel_title


def make_progress_hook(current_idx: int, total_cnt: int):
    last_percent = -10  # 10%刻みで出力するための状態保持用
    
    def hook(d):
        nonlocal last_percent
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            if total:
                percent = int((downloaded / total) * 100)
                # 10%刻みの境界
                displayed_percent = (percent // 10) * 10
                if displayed_percent > last_percent:
                    last_percent = displayed_percent
                    eta = d.get('eta')
                    if eta is not None:
                        mins, secs = divmod(eta, 60)
                        eta_str = f"残り時間: {mins:02d}:{secs:02d}"
                    else:
                        eta_str = "残り時間: 不明"
                    
                    print(f"[PROGRESS] {get_timestamp()} - [{current_idx}/{total_cnt}] 進捗: {displayed_percent}% ({eta_str})", flush=True)
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
    debug: bool = False
) -> bool:
    video_id = video["id"]
    title = video["title"]
    url = video["url"]

    # 128kなどのビットレート文字列から数値だけを取り出す
    quality = bitrate.rstrip("kK").rstrip("bpsBPS")

    # 出力ファイル名テンプレート: [アップロード日]_[タイトル]_[動画ID].[拡張子]
    # yt-dlpのouttmplで指定する
    outtmpl_str = str(output_dir / "%(upload_date)s_%(title).120s_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl_str,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "skip_unavailable_fragments": True,
        "continuedl": True,
        "no_part": False,
        "quiet": not debug,
        "no_warnings": not debug,
        "noprogress": not debug,
        "progress_hooks": [make_progress_hook(current_index, total_count)] if not debug else [],
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": quality,
        }],
    }

    if cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser,)
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    log_progress(current_index, total_count, f"ダウンロード中: \"{title}\" (ID: {video_id})")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # 成功
        log_success(current_index, total_count, f"音声抽出完了: \"{title}\" ({audio_format})")
        return True, None
    except Exception as e:
        error_msg = str(e)
        # 回復不能と思われるエラーを判定
        unrecoverable_keywords = [
            "unavailable", "private", "removed", "deleted", "copyright", 
            "members-only", "sign in", "confirm your age", "age-restricted",
            "blocked", "not available", "403: Forbidden"
        ]
        
        is_unrecoverable = False
        for kw in unrecoverable_keywords:
            if kw in error_msg.lower():
                is_unrecoverable = True
                break

        log_error(current_index, total_count, f"ダウンロード失敗: \"{title}\" (理由: {error_msg})")
        return False, is_unrecoverable


def main():
    # Windows環境等でのUnicodeEncodeError対策
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="YouTubeの指定チャンネルのライブ配信アーカイブから音声を一括ダウンロードします。"
    )
    parser.add_argument(
        "channel_handle",
        help="@から始まるチャンネル識別子（例: @Google）"
    )
    parser.add_argument(
        "-f", "--format",
        default="mp3",
        choices=["mp3", "m4a", "wav", "opus", "flac"],
        help="出力する音声ファイルのフォーマット (デフォルト: mp3)"
    )
    parser.add_argument(
        "-b", "--bitrate",
        default="128k",
        help="音声のビットレート (デフォルト: 128k)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="音声ファイルを保存するフォルダパス (デフォルト: ./downloads/[チャンネル名]/)"
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="年齢制限などを回避するためにブラウザからCookieを読み込む (例: chrome, edge, firefox)"
    )
    parser.add_argument(
        "--ffmpeg-location",
        default=None,
        help="ffmpegバイナリのパス (システム環境変数に通っていない場合に使用)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="yt-dlpの詳細なログ出力を有効にする"
    )

    args = parser.parse_args()

    # 1. 動画一覧の取得
    try:
        videos, channel_title = get_live_videos(args.channel_handle, args.cookies_from_browser)
    except Exception as e:
        log_error(0, 0, f"エラーが発生したため処理を中断します: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # 2. 保存フォルダの決定
    # チャンネル名に使えない文字などを置換する（yt-dlpがフォルダ名に使用する可能性に備える）
    safe_channel_title = "".join(c for c in channel_title if c.isalnum() or c in (" ", "_", "-")).strip()
    if not safe_channel_title:
        safe_channel_title = args.channel_handle.replace("@", "")

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("downloads") / safe_channel_title

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. キャッシュファイルの読み込み
    # チャンネルハンドルをファイル名に安全に使えるようにする
    safe_handle_filename = "".join(c for c in args.channel_handle if c.isalnum() or c in ("_", "-"))
    cache_file = output_dir / f"download_cache_{safe_handle_filename}.json"
    cache_data = load_cache(cache_file, args.channel_handle)

    downloaded_ids = set(cache_data.get("downloaded_ids", []))
    
    # すでに処理済みの動画をフィルタリング
    target_videos = [v for v in videos if v["id"] not in downloaded_ids]
    skipped_count = len(videos) - len(target_videos)

    log_info(f"設定 - フォーマット: {args.format}, ビットレート: {args.bitrate}, 保存先: {output_dir}")
    log_info(f"ライブ動画を {len(videos)} 本検出しました。（うち {skipped_count} 本は処理済みのキャッシュによりスキップ）")
    log_info(f"今回の処理対象: {len(target_videos)} 本")

    if not target_videos:
        log_info("処理対象の動画はありません。処理を終了します。")
        return

    success_count = 0
    fail_count = 0
    
    try:
        for idx, video in enumerate(target_videos, start=1):
            video_id = video["id"]
            
            # ダウンロード実行
            success, is_unrecoverable = download_and_extract(
                video=video,
                output_dir=output_dir,
                audio_format=args.format,
                bitrate=args.bitrate,
                cookies_browser=args.cookies_from_browser,
                ffmpeg_location=args.ffmpeg_location,
                current_index=idx,
                total_count=len(target_videos),
                debug=args.debug
            )

            if success:
                success_count += 1
                # キャッシュに即座に保存
                cache_data["downloaded_ids"].append(video_id)
                cache_data["last_updated"] = datetime.datetime.now().isoformat()
                save_cache(cache_file, cache_data)
            else:
                fail_count += 1
                # 回復不能エラーの場合はキャッシュに保存し、次回からスキップ
                if is_unrecoverable:
                    log_info(f"動画 (ID: {video_id}) は回復不能なエラーのため、次回以降はスキップします。")
                    cache_data["downloaded_ids"].append(video_id)
                    cache_data["last_updated"] = datetime.datetime.now().isoformat()
                    save_cache(cache_file, cache_data)

            # 最後の動画でなければ、3〜7秒のランダム待機を挿入
            if idx < len(target_videos):
                wait_time = random.randint(3, 7)
                log_info(f"{wait_time}秒間待機します...")
                time.sleep(wait_time)

    except KeyboardInterrupt:
        log_info("処理がユーザーによって中断されました。キャッシュを保存して終了します。")
        sys.exit(0)

    log_info(f"すべての処理が完了しました。合計: {success_count}本成功、{fail_count}本失敗。")


if __name__ == "__main__":
    main()
