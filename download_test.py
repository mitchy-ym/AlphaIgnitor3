import argparse
from pathlib import Path

import yt_dlp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video file for live-end testing."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.youtube.com/watch?v=w0-3N5kjmT0",
        help="YouTube video URL",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="downloads",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser name for authenticated downloads (e.g. chrome, edge, firefox)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Limit downloaded video resolution, e.g. 360 or 480 for smaller files",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download audio only and extract it to a separate audio file",
    )
    parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=("mp3", "m4a", "wav", "opus", "flac"),
        help="Audio format to extract when --audio-only is used",
    )
    return parser


def build_format_selector(max_height: int | None, audio_only: bool) -> str:
    if audio_only:
        return "bestaudio/best"

    if max_height is None:
        return "bv*+ba/b"

    return (
        f"bestvideo[height<={max_height}]+bestaudio/"
        f"best[height<={max_height}]/best"
    )


def download_video(
    url: str,
    output_dir: Path,
    cookies_from_browser: str | None,
    max_height: int | None,
    audio_only: bool,
    audio_format: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": build_format_selector(max_height, audio_only),
        "outtmpl": str(output_dir / "%(upload_date)s_%(title).120s_%(id)s.%(ext)s"),
        "noplaylist": True,
        "retries": 20,
        "fragment_retries": 20,
        "skip_unavailable_fragments": True,
        "concurrent_fragment_downloads": 4,
        "wait_for_video": (0, 180),
        "continuedl": True,
        "no_part": False,
    }

    if audio_only:
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ]
    else:
        ydl_opts["merge_output_format"] = "mp4"

    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return download_video(
        args.url,
        Path(args.output_dir),
        args.cookies_from_browser,
        args.max_height,
        args.audio_only,
        args.audio_format,
    )


if __name__ == "__main__":
    raise SystemExit(main())
