import argparse
import sys
from pathlib import Path

# ローカルモジュールへのインポートを可能にする
sys.path.append(str(Path(__file__).parent))

from timestamper.cli import add_download_arguments, add_transcribe_arguments

def setup_parsers() -> argparse.ArgumentParser:
    """TimeStamper の統一コマンドラインパーサーを構築します。"""
    parser = argparse.ArgumentParser(
        description="TimeStamper: YouTubeライブ音声の自動ダウンロードと文字起こし（音声認識）を行う統合ツール"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="実行するコマンドサブタイプ")

    # ---- 1. download サブコマンド ----
    download_parser = subparsers.add_parser(
        "download",
        help="YouTubeの指定チャンネルの配信アーカイブから音声をダウンロードします（並行処理対応）。"
    )
    add_download_arguments(download_parser)

    # ---- 2. transcribe サブコマンド ----
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="ローカルの音声/動画ファイルを読み込み、Whisperで高速に文字起こし処理を行います。"
    )
    add_transcribe_arguments(transcribe_parser)

    # ---- 3. pipeline サブコマンド ----
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="ダウンロード -> デコード -> GPU文字起こしを、並行かつインメモリの3ステージ非同期パイプラインで一括実行します。"
    )
    add_download_arguments(pipeline_parser)
    add_transcribe_arguments(pipeline_parser, prefix="transcribe-")

    return parser

def main():
    # Windows環境等のエンコーディング対策（常に行バッファリングを有効にしてデッドロックを防ぐ）
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

    parser = setup_parsers()
    args = parser.parse_args()

    import os
    if args.command == "download":
        from timestamper.downloader import run_downloader
        os._exit(run_downloader(args))
    elif args.command == "transcribe":
        from timestamper.transcriber import run_transcribe_worker
        os._exit(run_transcribe_worker(args))
    elif args.command == "pipeline":
        args.transcribe = True  # パイプライン処理のフラグを有効化
        from timestamper.pipeline import run_pipeline
        os._exit(run_pipeline(args))

if __name__ == "__main__":
    main()
