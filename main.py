import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from timestamper.cli import add_download_arguments, add_merge_arguments, add_pipeline_arguments


def setup_parsers() -> argparse.ArgumentParser:
    """TimeStamper の統一コマンドラインパーサーを構築します。"""
    parser = argparse.ArgumentParser(
        description="TimeStamper: YouTubeライブ音声のダウンロード、文字起こし、Ollama後処理を行う統合ツール"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="実行するコマンド")

    download_parser = subparsers.add_parser(
        "download",
        help="YouTubeの指定チャンネルの配信アーカイブから音声をダウンロードします。"
    )
    add_download_arguments(download_parser)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="ダウンロード -> GPU文字起こし -> Ollama enrich を一括実行します。"
    )
    add_pipeline_arguments(pipeline_parser)

    merge_parser = subparsers.add_parser(
        "merge",
        help="文字起こしテキストを月次・年次・全件で結合します。"
    )
    add_merge_arguments(merge_parser)

    return parser


def main() -> None:
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

    if args.command == "pipeline":
        from timestamper.pipeline import run_pipeline

        os._exit(run_pipeline(args))

    if args.command == "merge":
        from timestamper.merge import run_merge

        os._exit(run_merge(args))

    raise RuntimeError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
