import argparse
import sys
from pathlib import Path

# ローカルモジュールへのインポートを可能にする
sys.path.append(str(Path(__file__).parent))

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
    download_parser.add_argument(
        "channel_handle",
        help="@から始まるチャンネル識別子（例: @Google）"
    )
    download_parser.add_argument(
        "-f", "--format",
        default="best",
        choices=["best", "mp3", "m4a", "wav", "opus", "flac"],
        help="出力する音声ファイルのフォーマット (デフォルト: best)"
    )
    download_parser.add_argument(
        "-b", "--bitrate",
        default="128k",
        help="音声のビットレート (デフォルト: 128k)"
    )
    download_parser.add_argument(
        "-o", "--output",
        default=None,
        help="音声ファイルを保存するフォルダパス (デフォルト: ./downloads/[チャンネル名]/)"
    )
    download_parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="ブラウザからCookieを読み込んで認証動画を処理します (例: chrome, edge, firefox)"
    )
    download_parser.add_argument(
        "--cookies",
        default=None,
        help="Netscape形式のCookieファイルパス (例: cookies/cookies.txt)。指定しない場合、cookies/cookies.txtが存在すれば自動的に読み込みます。"
    )
    download_parser.add_argument(
        "--ffmpeg-location",
        default=None,
        help="ffmpegバイナリのパス (システム環境変数に通っていない場合に使用)"
    )
    download_parser.add_argument(
        "--verbose-progress",
        action="store_true",
        dest="verbose_progress",
        default=True,
        help="従来の標準出力テキストによる進捗ログ（[PROGRESS]等）を表示します（デフォルト）。"
    )
    download_parser.add_argument(
        "--no-verbose-progress", "--quiet", "-q",
        action="store_false",
        dest="verbose_progress",
        help="進捗ログ（[INFO]や[PROGRESS]等）の標準出力を無効にし、プログレスバー表示にします。"
    )
    download_parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=2,
        help="最大同時ダウンロード数 (デフォルト: 2)"
    )
    download_parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="ダウンロードする最大動画本数 (デフォルト: すべて)"
    )
    download_parser.add_argument(
        "--debug",
        action="store_true",
        help="yt-dlpの詳細なログ出力を有効にする"
    )

    # ---- 2. transcribe サブコマンド ----
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="ローカルの音声/動画ファイルを読み込み、Whisperで高速に文字起こし処理を行います。"
    )
    transcribe_parser.add_argument(
        "input_path",
        nargs="?",
        default="downloads",
        help="音声/動画ファイルのパス、または対象ファイルが格納されたディレクトリ (デフォルト: downloads)"
    )
    transcribe_parser.add_argument(
        "-o", "--output-dir",
        default="transcripts",
        help="テキストとJSONの文字起こし結果を保存するディレクトリ (デフォルト: transcripts)"
    )
    transcribe_parser.add_argument(
        "--model",
        default="turbo",
        help="使用するWhisperモデルのサイズ、またはHugging Faceのリポジトリ名 (デフォルト: turbo)"
    )
    transcribe_parser.add_argument(
        "--language",
        default="ja",
        help="対象言語コード (デフォルト: ja)"
    )
    transcribe_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="推論デバイス (デフォルト: auto)"
    )
    transcribe_parser.add_argument(
        "--compute-type",
        choices=["auto", "int8", "float16", "int8_float16", "float32"],
        default="float16",
        help="推論の計算精度 (デフォルト: float16)"
    )
    transcribe_parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisperのタスクの種類 (デフォルト: transcribe)"
    )
    transcribe_parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="ビームサーチのサイズ (デフォルト: 5)"
    )
    transcribe_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="並行処理するチャンクのバッチサイズ (デフォルト: 16)"
    )
    transcribe_parser.add_argument(
        "--delete-audio",
        action="store_true",
        help="文字起こしが正常に完了した後に、入力した音声ファイルを削除します。"
    )
    transcribe_parser.add_argument(
        "--initial-prompt",
        default="こんにちは。今日はいい天気ですね。本日はよろしくお願いいたします。",
        help="文字起こしの開始時に与える初期プロンプト"
    )
    transcribe_parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="音声検出（VAD）のしきい値。0.0〜1.0 (デフォルト: 0.5)"
    )
    transcribe_parser.add_argument(
        "--chunk-duration",
        type=float,
        default=600.0,
        help="非同期処理での音声分割のチャンク秒数 (デフォルト: 600.0)"
    )

    # ---- 3. pipeline サブコマンド ----
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="ダウンロード -> デコード -> GPU文字起こしを、並行かつインメモリの3ステージ非同期パイプラインで一括実行します。"
    )
    # ダウンロードオプション
    pipeline_parser.add_argument(
        "channel_handle",
        help="@から始まるチャンネル識別子（例: @Google）"
    )
    pipeline_parser.add_argument(
        "-f", "--format",
        default="best",
        choices=["best", "mp3", "m4a", "wav", "opus", "flac"],
        help="出力する音声ファイルのフォーマット (デフォルト: best)"
    )
    pipeline_parser.add_argument(
        "-b", "--bitrate",
        default="128k",
        help="音声のビットレート (デフォルト: 128k)"
    )
    pipeline_parser.add_argument(
        "-o", "--output",
        default=None,
        help="音声ファイルを保存するフォルダパス (デフォルト: ./downloads/[チャンネル名]/)"
    )
    pipeline_parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="年齢制限などを回避するためにブラウザからCookieを読み込む (例: chrome, edge, firefox)"
    )
    pipeline_parser.add_argument(
        "--cookies",
        default=None,
        help="Netscape形式のCookieファイルパス (例: cookies/cookies.txt)。指定しない場合、cookies/cookies.txtが存在すれば自動的に読み込みます。"
    )
    pipeline_parser.add_argument(
        "--ffmpeg-location",
        default=None,
        help="ffmpegバイナリのパス (システム環境変数に通っていない場合に使用)"
    )
    pipeline_parser.add_argument(
        "--verbose-progress",
        action="store_true",
        dest="verbose_progress",
        default=True,
        help="従来の標準出力テキストによる進捗ログ（[PROGRESS]等）を表示します（デフォルト）。"
    )
    pipeline_parser.add_argument(
        "--no-verbose-progress", "--quiet", "-q",
        action="store_false",
        dest="verbose_progress",
        help="進捗ログ（[INFO]や[PROGRESS]等）の標準出力を無効にし、プログレスバー表示にします。"
    )
    pipeline_parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="YouTubeの動画一覧と、ローカルのキャッシュ・音声ファイル・文字起こし結果（transcriptsフォルダ）の整合性を確認します。"
    )
    pipeline_parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=2,
        help="最大同時ダウンロード数 (デフォルト: 2)"
    )
    pipeline_parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="ダウンロードする最大動画本数 (デフォルト: すべて)"
    )
    pipeline_parser.add_argument(
        "--debug",
        action="store_true",
        help="yt-dlpの詳細なログ出力を有効にする"
    )
    # 文字起こしオプション
    pipeline_parser.add_argument(
        "--transcribe-model",
        default="turbo",
        help="文字起こしに使用するWhisperモデル (デフォルト: turbo)"
    )
    pipeline_parser.add_argument(
        "--transcribe-language",
        default="ja",
        help="文字起こしの対象言語 (デフォルト: ja)"
    )
    pipeline_parser.add_argument(
        "--transcribe-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="文字起こしを実行するデバイス (デフォルト: auto)"
    )
    pipeline_parser.add_argument(
        "--transcribe-compute-type",
        choices=["auto", "int8", "float16", "int8_float16", "float32"],
        default="float16",
        help="モデルの計算精度 (デフォルト: float16)"
    )
    pipeline_parser.add_argument(
        "--transcribe-output-dir",
        default="transcripts",
        help="文字起こし結果を保存するフォルダ (デフォルト: transcripts)"
    )
    pipeline_parser.add_argument(
        "--transcribe-keep-audio",
        action="store_true",
        help="文字起こしが正常に完了した後に、入力した音声ファイルを残します（デフォルトは自動削除）。"
    )
    pipeline_parser.add_argument(
        "--transcribe-vad-threshold",
        type=float,
        default=0.5,
        help="音声検出（VAD）のしきい値。0.0〜1.0 (デフォルト: 0.5)"
    )
    pipeline_parser.add_argument(
        "--transcribe-initial-prompt",
        default="こんにちは。今日はいい天気ですね。本日はよろしくお願いいたします。",
        help="文字起こしの開始時に与える初期プロンプト"
    )
    pipeline_parser.add_argument(
        "--transcribe-chunk-duration",
        type=float,
        default=600.0,
        help="非同期処理での音声分割のチャンク秒数 (デフォルト: 600.0)"
    )
    pipeline_parser.add_argument(
        "--transcribe-batch-size",
        type=int,
        default=16,
        help="並行処理するバッチサイズ (デフォルト: 16)"
    )
    pipeline_parser.add_argument(
        "--transcribe-beam-size",
        type=int,
        default=5,
        help="ビームサーチのサイズ (デフォルト: 5)"
    )
    pipeline_parser.add_argument(
        "--transcribe-task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="タスクの種類 (デフォルト: transcribe)"
    )

    return parser

def main():
    # Windows環境等のエンコーディング対策
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = setup_parsers()
    args = parser.parse_args()

    if args.command == "download":
        from youtube_live_audio_downloader import run_downloader
        sys.exit(run_downloader(args))
    elif args.command == "transcribe":
        from transcribe_local import run_transcribe_worker
        sys.exit(run_transcribe_worker(args))
    elif args.command == "pipeline":
        args.transcribe = True  # パイプライン処理のフラグを有効化
        from pipeline import run_pipeline
        sys.exit(run_pipeline(args))

if __name__ == "__main__":
    main()
