#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# ローカルモジュールへのインポートを可能にする
sys.path.append(str(Path(__file__).parent))

# バックワードコンパティビティ（後方互換性）のためにモジュールレベルで関数を公開
from timestamper.downloader import (
    get_live_videos,
    download_and_extract,
    load_download_cache as load_cache,
    save_download_cache as save_cache,
    run_downloader
)
from timestamper.utils import (
    ProgressPositionManager,
    log_info,
    log_warn,
    log_error,
    log_success,
    SHOW_PROGRESS_TEXT
)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="YouTubeの指定チャンネルのライブ配信アーカイブから音声を一括ダウンロードします。"
    )
    add_download_arguments(parser)
    
    # 文字起こし（トランスクリプション）連携用の引数
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="ダウンロード完了後に自動的に文字起こし（音声認識）を実行します。"
    )
    add_transcribe_arguments(parser, prefix="transcribe-")

    args = parser.parse_args()
    sys.exit(run_downloader(args))

if __name__ == "__main__":
    from timestamper.cli import add_download_arguments, add_transcribe_arguments
    main()
