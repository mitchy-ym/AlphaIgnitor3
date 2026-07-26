import argparse

def add_download_arguments(parser: argparse.ArgumentParser):
    """共通のダウンロードに関する引数をパーサーに追加します。"""
    parser.add_argument(
        "channel_handle",
        help="@から始まるチャンネル識別子（例: @Google）"
    )
    parser.add_argument(
        "-f", "--format",
        default="best",
        choices=["best", "mp3", "m4a", "wav", "opus", "flac"],
        help="出力する音声ファイルのフォーマット (デフォルト: best)"
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
        help="ブラウザからCookieを読み込んで認証動画を処理します (例: chrome, edge, firefox)"
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Netscape形式のCookieファイルパス (例: cookies/cookies.txt)。"
    )
    parser.add_argument(
        "--ffmpeg-location",
        default=None,
        help="ffmpegバイナリのパス (システム環境変数に通っていない場合に使用)"
    )
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        dest="verbose_progress",
        default=False,
        help="従来の標準出力テキストによる進捗ログ（[PROGRESS]等）を表示します。"
    )
    parser.add_argument(
        "--no-verbose-progress", "--quiet", "-q",
        action="store_false",
        dest="verbose_progress",
        help="進捗ログ（[INFO]や[PROGRESS]等）の標準出力を無効にし、プログレスバー表示にします（デフォルト）。"
    )
    parser.add_argument(
        "-w", "--max-workers",
        type=int,
        default=2,
        help="最大同時ダウンロード数 (デフォルト: 2)"
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="ダウンロードする最大動画本数 (デフォルト: すべて)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="yt-dlpの詳細なログ出力を有効にする"
    )
    parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="YouTubeの動画一覧と、ローカルのキャッシュ・音声ファイル・文字起こし結果（transcriptsフォルダ）の整合性を確認します。"
    )
    parser.add_argument(
        "--sync-cache",
        action="store_true",
        help="実ファイル（transcriptsフォルダ内の文字起こしテキスト）の状態に基づき、ローカルのダウンロードキャッシュを自動的に同期・修復します。"
    )

def add_transcribe_arguments(parser: argparse.ArgumentParser, prefix: str = ""):
    """共通の文字起こしに関する引数をパーサーに追加します。Prefix を指定することで変数名の重複を防ぎます。"""
    p = prefix
    
    if not p:
        parser.add_argument(
            "input_path",
            nargs="?",
            default="downloads",
            help="音声/動画ファイルのパス、または対象ファイルが格納されたディレクトリ (デフォルト: downloads)"
        )
        parser.add_argument(
            "-o", "--output-dir",
            default="transcripts",
            help="文字起こし結果を保存するディレクトリ (デフォルト: transcripts)"
        )
        parser.add_argument(
            "--model",
            default="turbo",
            help="使用するWhisperモデルのサイズ、またはHugging Faceのリポジトリ名 (デフォルト: turbo)"
        )
    else:
        parser.add_argument(
            f"--{p}model",
            default="turbo",
            help="使用するWhisperモデルのサイズ、またはHugging Faceのリポジトリ名 (デフォルト: turbo)"
        )
        parser.add_argument(
            f"--{p}output-dir",
            default="transcripts",
            help="文字起こし結果を保存するフォルダ (デフォルト: transcripts)"
        )

    parser.add_argument(
        f"--{p}json",
        action="store_true",
        help="文字起こし結果をJSON形式でも出力します（デフォルト: 無効）"
    )

    parser.add_argument(
        f"--{p}language",
        default="ja",
        help="対象言語コード。自動検出にする場合は 'auto' を指定 (デフォルト: ja)"
    )
    parser.add_argument(
        f"--{p}device",
        choices=["auto", "cuda"],
        default="auto",
        help="推論デバイス。GPU 前提のため auto または cuda のみ指定できます (デフォルト: auto)"
    )
    parser.add_argument(
        f"--{p}compute-type",
        choices=["auto", "int8", "float16", "int8_float16", "float32"],
        default="float16",
        help="推論の計算精度 (デフォルト: float16)"
    )
    parser.add_argument(
        f"--{p}task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisperのタスクの種類 (デフォルト: transcribe)"
    )
    parser.add_argument(
        f"--{p}beam-size",
        type=int,
        default=5,
        help="ビームサーチのサイズ (デフォルト: 5)"
    )
    parser.add_argument(
        f"--{p}batch-size",
        type=int,
        default=16,
        help="並行処理するチャンクのバッチサイズ (デフォルト: 16)"
    )
    
    if not p:
        parser.add_argument(
            "--delete-audio",
            action="store_true",
            help="文字起こしが正常に完了した後に、入力した音声ファイルを削除します。"
        )
    else:
        parser.add_argument(
            f"--{p}keep-audio",
            action="store_true",
            help="文字起こしが正常に完了した後に、入力した音声ファイルを残します（デフォルトは自動削除）。"
        )

    parser.add_argument(
        f"--{p}initial-prompt",
        default="こんにちは。今日はいい天気ですね。本日はよろしくお願いいたします。",
        help="文字起こしの開始時に与える初期プロンプト"
    )
    parser.add_argument(
        f"--{p}vad-threshold",
        type=float,
        default=0.5,
        help="音声検出（VAD）のしきい値。0.0〜1.0 (デフォルト: 0.5)"
    )
    parser.add_argument(
        f"--{p}chunk-duration",
        type=float,
        default=600.0,
        help="非同期処理での音声分割のチャンク秒数 (デフォルト: 600.0)"
    )
    parser.add_argument(
        f"--{p}min-rms",
        type=float,
        default=0.003,
        help="文字起こしセグメントの最小RMS（音量エネルギー）しきい値。これ未満の区間はハルシネーションと見なして除外します。0.0で無効化 (デフォルト: 0.003)"
    )
