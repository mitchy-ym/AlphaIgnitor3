import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

# AMD ROCm RDNA3.5 iGPU/APU 互換性（Radeon 890M など）のための環境変数設定
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.0")
# Windows上での Hugging Face Hub シンボリックリンク警告を抑止
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# OpenMP ランタイムの競合回避（Windows環境）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def _init_openmp_runtime() -> None:
    """ROCm / ctranslate2 向けの OpenMP (libomp.so) ランタイムを事前ロードします。"""
    if sys.platform.startswith("linux"):
        import ctypes
        for candidate in [
            "/opt/rocm/lib/llvm/lib/libomp.so",
            "/usr/lib/llvm-18/lib/libomp.so",
            "/usr/lib/x86_64-linux-gnu/libomp.so",
            "/usr/lib/x86_64-linux-gnu/libomp.so.5",
        ]:
            if os.path.exists(candidate):
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                    break
                except Exception:
                    pass

_init_openmp_runtime()

# グローバルな環境へのインストールなしで ffmpeg を利用可能にする
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

# 進捗ログ表示フラグ (デフォルトで有効、コマンドライン引数 --quiet 等で変更されます)
SHOW_PROGRESS_TEXT = True

# 対象とする音声・動画ファイルの拡張子
MEDIA_EXTENSIONS = {".mp3", ".m4a", ".wav", ".opus", ".flac", ".mp4", ".mkv", ".webm"}

TIMESTAMP_REGEX = re.compile(r"^\[(\d{1,2}:\d{2}:\d{2})\]\s*(.*)$")
TIMESTAMP_PREFIX_REGEX = re.compile(r"^\[\d{1,2}:\d{2}:\d{2}\]\s*")


def get_timestamp() -> str:
    """現在の時刻をフォーマットした文字列を返します。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(message: str) -> None:
    """情報ログを出力します。"""
    if SHOW_PROGRESS_TEXT:
        print(f"[INFO] {get_timestamp()} - {message}", flush=True)


def log_progress(message_or_current: str | int, total: int | None = None, message: str | None = None) -> None:
    """進捗ログを出力します。log_progress(current, total, message) または log_progress(message) を受け付けます。"""
    if SHOW_PROGRESS_TEXT:
        if isinstance(message_or_current, int) and total is not None and message is not None:
            print(f"[PROGRESS] {get_timestamp()} - [{message_or_current}/{total}] {message}", flush=True)
        else:
            print(f"[PROGRESS] {get_timestamp()} - {message_or_current}", flush=True)


def log_success(message_or_current: str | int, total: int | None = None, message: str | None = None) -> None:
    """成功ログを出力します。log_success(current, total, message) または log_success(message) を受け付けます。"""
    if SHOW_PROGRESS_TEXT:
        if isinstance(message_or_current, int) and total is not None and message is not None:
            print(f"[SUCCESS] {get_timestamp()} - [{message_or_current}/{total}] {message}", flush=True)
        else:
            print(f"[SUCCESS] {get_timestamp()} - {message_or_current}", flush=True)


def log_error(message_or_current: str | int, total: int | None = None, message: str | None = None) -> None:
    """エラーログを出力します。log_error(current, total, message) または log_error(message) を受け付けます。"""
    if isinstance(message_or_current, int) and total is not None and message is not None:
        prefix = f"[{message_or_current}/{total}] "
        text = f"{prefix}{message}"
    else:
        text = str(message_or_current)

    formatted = f"[ERROR] {get_timestamp()} - {text}"
    if SHOW_PROGRESS_TEXT:
        print(formatted, flush=True)
    else:
        print(formatted, file=sys.stderr, flush=True)


def log_warn(message: str) -> None:
    """警告ログを出力します。無効時も標準エラー出力に出力されます。"""
    formatted = f"[WARN] {get_timestamp()} - {message}"
    if SHOW_PROGRESS_TEXT:
        print(formatted, flush=True)
    else:
        print(formatted, file=sys.stderr, flush=True)


def sanitize_channel_title(channel_title: str, fallback: str | None = None) -> str:
    """チャンネル名からファイルシステムで安全に使用できる文字列を生成します。"""
    safe_title = "".join(c for c in channel_title if c.isalnum() or c in (" ", "_", "-")).strip()
    if safe_title:
        return safe_title
    if fallback:
        return fallback.replace("@", "")
    return "channel"


def extract_timestamp_and_text(line: str) -> tuple[str, str] | None:
    """行頭のタイムスタンプ [HH:MM:SS] とテキストを抽出します。一致しない場合は None を返します。"""
    match = TIMESTAMP_REGEX.match(line.strip())
    if match:
        return match.group(1), match.group(2).strip()
    return None


def strip_timestamp(line: str) -> str:
    """行頭のタイムスタンプ [HH:MM:SS] または [H:MM:SS] を除去します。"""
    return TIMESTAMP_PREFIX_REGEX.sub("", line)


class ProgressPositionManager:
    """tqdm などのマルチスレッド進捗バーの表示位置を管理するクラスです。"""
    def __init__(self, max_workers: int):
        import threading
        self.lock = threading.Lock()
        self.available_positions = list(range(1, max_workers + 1))
        
    def acquire(self) -> int:
        with self.lock:
            if self.available_positions:
                return self.available_positions.pop(0)
            return 1
            
    def release(self, pos: int):
        with self.lock:
            self.available_positions.append(pos)
            self.available_positions.sort()

def format_seconds(seconds: float) -> str:
    """秒数を HH:MM:SS 形式の文字列に変換します。"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_media_duration(media_path: Path) -> float | None:
    """ffprobe または PyAV を使用してメディアファイルの再生時間（秒）を取得します。"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
    except Exception as e:
        log_warn(f"Failed to get duration with ffprobe: {e}")

    try:
        import av
        with av.open(str(media_path)) as container:
            return float(container.duration) / av.time_base
    except Exception as e:
        log_warn(f"Failed to get duration with PyAV: {e}")
    
    return None


def sanitize_cookie_file(cookie_file_path: str | Path | None) -> None:
    """Netscape形式のCookieファイルに含まれる破損（ヌルバイト）や、
    Python 3.11のhttp.cookiejarによるアサーションエラー（AssertionError）を回避するために、
    Cookieファイルを読み込んで自動的にクリーンアップします。
    """
    if not cookie_file_path:
        return
    path = Path(cookie_file_path)
    if not path.exists() or not path.is_file():
        return

    try:
        # まずバイナリとして読み込む
        with open(path, "rb") as f:
            content = f.read()

        # ヌルバイトを取り除く
        content_clean = content.replace(b"\x00", b"")

        try:
            text = content_clean.decode("utf-8")
        except UnicodeDecodeError:
            text = content_clean.decode("latin-1", errors="ignore")

        lines = text.splitlines()
        fixed_lines = []
        modified = False
        
        # ヌルバイトの除去があった場合、またはファイルに変化がある場合は modified とする
        if len(content_clean) != len(content):
            modified = True

        for line in lines:
            if not line.strip():
                fixed_lines.append(line)
                continue
            if line.startswith("#"):
                fixed_lines.append(line)
                continue

            parts = line.split("\t")
            if len(parts) >= 7:
                domain = parts[0]
                domain_specified = parts[1].upper()  # TRUE / FALSE

                if domain_specified not in ("TRUE", "FALSE"):
                    modified = True
                    continue

                initial_dot = domain.startswith(".")

                # http.cookiejar.py のアサーションバグ対策:
                # 2列目が TRUE の場合はドメインの先頭がドットで始まっていなければならない
                # 2列目が FALSE の場合はドメインの先頭がドットで始まってはならない
                if domain_specified == "TRUE" and not initial_dot:
                    parts[0] = "." + domain
                    modified = True
                elif domain_specified == "FALSE" and initial_dot:
                    parts[0] = domain[1:]
                    modified = True

                fixed_lines.append("\t".join(parts))
            else:
                # 不正なフィールド数の行を除去
                modified = True

        if modified:
            log_info(f"Cookieファイルを自動クリーンアップしました: {path.name}")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(fixed_lines) + "\n")

    except Exception as e:
        log_warn(f"Cookieファイルのクリーンアップ処理中にエラーが発生しました: {e}")

