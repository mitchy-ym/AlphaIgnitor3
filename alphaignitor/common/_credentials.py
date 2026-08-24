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
