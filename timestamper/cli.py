import argparse


DEFAULT_OLLAMA_ENDPOINT = "http://172.19.208.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3.6:latest"


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    """ダウンロードコマンド用の引数を追加します。"""
    parser.add_argument("channel_handle", help="@から始まるチャンネル識別子（例: @Google）")
    parser.add_argument(
        "-f",
        "--format",
        default="best",
        choices=["best", "mp3", "m4a", "wav", "opus", "flac"],
        help="出力する音声ファイルのフォーマット (デフォルト: best)",
    )
    parser.add_argument("-b", "--bitrate", default="128k", help="音声のビットレート (デフォルト: 128k)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="音声ファイルを保存するフォルダパス (デフォルト: ./downloads/[チャンネル名]/)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="ブラウザからCookieを読み込んで認証動画を処理します (例: chrome, edge, firefox)",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Netscape形式のCookieファイルパス (例: cookies/cookies.txt)",
    )
    parser.add_argument(
        "--ffmpeg-location",
        default=None,
        help="ffmpegバイナリのパス (システム環境変数に通っていない場合に使用)",
    )
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        dest="verbose_progress",
        default=False,
        help="標準出力に進捗ログを表示します。",
    )
    parser.add_argument(
        "--no-verbose-progress",
        "--quiet",
        "-q",
        action="store_false",
        dest="verbose_progress",
        help="進捗ログを無効化し、プログレスバー中心で表示します（デフォルト）。",
    )
    parser.add_argument("-w", "--max-workers", type=int, default=2, help="最大同時ダウンロード数 (デフォルト: 2)")
    parser.add_argument("--max-downloads", type=int, default=None, help="ダウンロードする最大動画本数 (デフォルト: すべて)")
    parser.add_argument("--debug", action="store_true", help="yt-dlpの詳細なログ出力を有効にする")


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """パイプラインコマンド用の引数を追加します。"""
    add_download_arguments(parser)

    parser.add_argument("--transcribe-model", default="turbo", help="使用するWhisperモデル (デフォルト: turbo)")
    parser.add_argument("--transcribe-output-dir", default="transcripts", help="文字起こし結果の保存先 (デフォルト: transcripts)")
    parser.add_argument("--transcribe-json", action="store_true", help="文字起こし結果をJSON形式でも出力します")
    parser.add_argument("--transcribe-language", default="ja", help="対象言語コード。自動検出にする場合は 'auto' を指定")
    parser.add_argument(
        "--transcribe-device",
        choices=["auto", "cuda"],
        default="auto",
        help="推論デバイス。GPU 前提のため auto または cuda のみ指定できます",
    )
    parser.add_argument(
        "--transcribe-compute-type",
        choices=["auto", "int8", "float16", "int8_float16", "float32"],
        default="float16",
        help="推論の計算精度 (デフォルト: float16)",
    )
    parser.add_argument("--transcribe-task", choices=["transcribe", "translate"], default="transcribe", help="Whisperのタスク種別")
    parser.add_argument("--transcribe-beam-size", type=int, default=5, help="ビームサーチのサイズ (デフォルト: 5)")
    parser.add_argument("--transcribe-batch-size", type=int, default=16, help="並行処理するチャンク数 (デフォルト: 16)")
    parser.add_argument("--transcribe-keep-audio", action="store_true", help="文字起こし後も音声ファイルを保持します")
    parser.add_argument(
        "--transcribe-initial-prompt",
        default="こんにちは。今日はいい天気ですね。本日はよろしくお願いいたします。",
        help="文字起こし開始時の初期プロンプト",
    )
    parser.add_argument("--transcribe-vad-threshold", type=float, default=0.5, help="音声検出（VAD）のしきい値")
    parser.add_argument("--transcribe-chunk-duration", type=float, default=600.0, help="音声分割のチャンク秒数")
    parser.add_argument(
        "--transcribe-min-rms",
        type=float,
        default=0.003,
        help="無音区間除外用の最小RMSしきい値 (デフォルト: 0.003)",
    )

    parser.add_argument("--enrich", action="store_true", help="文字起こし完了後、Ollamaで整形・要約・章立てを生成します")
    parser.add_argument("--enrich-output-dir", default="enriched", help="enrich成果物の保存先 (デフォルト: enriched)")
    parser.add_argument(
        "--llm-endpoint",
        default=DEFAULT_OLLAMA_ENDPOINT,
        help=f"OllamaのOpenAI互換APIエンドポイント (デフォルト: {DEFAULT_OLLAMA_ENDPOINT})",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollamaで使用するモデル名 (デフォルト: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument("--llm-timeout", type=float, default=600.0, help="LLM APIリクエストのタイムアウト秒数")
    parser.add_argument("--llm-max-tokens", type=int, default=3072, help="LLM APIレスポンスの最大トークン数")
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help="ローカルLLM APIキーを読む環境変数名。未指定ならAuthorizationヘッダーを付けません。",
    )
    parser.add_argument("--enrich-max-chars", dest="max_chars", type=int, default=1500, help="LLMに渡す1チャンクの最大文字数")
    parser.add_argument("--enrich-force", dest="force", action="store_true", help="既存のenrich成果物がある場合も再生成します")


def add_merge_arguments(parser: argparse.ArgumentParser) -> None:
    """merge コマンド用の引数を追加します。"""
    parser.add_argument(
        "--mode",
        choices=["monthly", "yearly", "all"],
        default="monthly",
        help=(
            "集約モードを選択します。"
            " monthly: 月ごとに結合（デフォルト）。"
            " yearly: 年ごとに結合。"
            " all: チャンネルごとに1ファイルに結合。"
        ),
    )
    parser.add_argument(
        "--input-dir",
        default="transcripts",
        help="結合対象の入力ディレクトリ (デフォルト: transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "結合ファイルの出力先ルート。未指定時は input-dir が transcripts の場合 transcripts、"
            "それ以外は input-dir の親ディレクトリ。"
        ),
    )
    parser.add_argument(
        "--strip-timestamps",
        action="store_true",
        help="各行先頭の [HH:MM:SS] タイムスタンプを除去して結合します（デフォルトは保持）。",
    )
