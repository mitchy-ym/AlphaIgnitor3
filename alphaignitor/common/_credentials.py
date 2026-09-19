"""認証情報ファイルの読み込みユーティリティ。

massive_rest.py と massive_splits.py が共通して必要とする
env ファイル読み込みと認証情報ファイルパス解決をここに集約する。
"""
from __future__ import annotations

import os
from pathlib import Path


def load_simple_env_file(path: Path) -> dict[str, str]:
    """シンプルな KEY=VALUE 形式の env ファイルを読み込む。

    - 空行・コメント行（#）はスキップ
    - 値のクォート（シングル・ダブル）を除去
    """
    out: dict[str, str] = {}
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for raw_line in raw:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def resolve_credentials_file() -> Path | None:
    """secrets/credentials.env が存在すれば返す。存在しなければ None。"""
    cand = Path(__file__).resolve().parents[2] / "secrets" / "credentials.env"
    if cand.exists() and cand.is_file():
        return cand
    return None


def load_credentials_into_environ() -> None:
    """secrets/credentials.env の全キーを os.environ に反映する。

    既に環境変数にセット済みの値は上書きしない。
    """
    path = resolve_credentials_file()
    if path is None:
        return
    for k, v in load_simple_env_file(path).items():
        if k not in os.environ:
            os.environ[k] = v


_PLACEHOLDER_TOKENS = frozenset(["YOUR_", "PLACEHOLDER", "CHANGEME", "TODO"])


def _is_placeholder(value: str | None, *, min_length: int = 1) -> bool:
    """値がプレースホルダーまたは空である場合 True を返す。

    Args:
        value:     検査する文字列。
        min_length: これ未満の長さの場合 True（デフォルト 1）。
    """
    if not value:
        return True
    v = str(value).strip()
    if not v:
        return True
    if len(v) < min_length:
        return True
    upper = v.upper()
    if any(tok in upper for tok in _PLACEHOLDER_TOKENS) or v.endswith("_HERE"):
        return True
    return False


def ensure_api_key_loaded() -> None:
    """secrets/credentials.env が存在すれば env vars に読み込む。

    MASSIVE_API_KEY または API_KEY を探索する。
    """
    creds = resolve_credentials_file()
    if creds is None:
        return
    vals = load_simple_env_file(creds)

    for k, v in vals.items():
        # プレースホルダー値が環境変数に存在する場合はファイル値で上書きする。
        if k in {"MASSIVE_API_KEY", "API_KEY"}:
            if _is_placeholder(os.environ.get(k)):
                os.environ[k] = v
            continue
        os.environ.setdefault(k, v)


def get_api_key() -> str:
    """Massive / Polygon REST API キーを取得する。"""
    ensure_api_key_loaded()

    # API キーは 20文字以上かつプレースホルダーでないもののみ有効とみなす。
    _DUMMY_VALUES = frozenset(["localtest", "test", "dummy"])
    candidates = [
        os.environ.get("MASSIVE_API_KEY"),
        os.environ.get("API_KEY"),
        os.environ.get("SECRET_KEY"),
    ]
    api_key = next(
        (
            c for c in candidates
            if c and c not in _DUMMY_VALUES and not _is_placeholder(c, min_length=20)
        ),
        None,
    )

    if not api_key:
        creds = resolve_credentials_file()
        raise RuntimeError(
            "Massive REST API Key が未設定です。\n"
            f"ファイル: {creds if creds is not None else '(not found)'} に MASSIVE_API_KEY=... を追加するか、環境変数 MASSIVE_API_KEY を設定してください。\n"
        )
    return str(api_key)

