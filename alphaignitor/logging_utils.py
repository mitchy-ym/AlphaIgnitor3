from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
import sys
import io


import os

_TZ_OFFSET = float(os.environ.get("LOG_TIMEZONE_OFFSET_HOURS", "8"))
LOG_TIMEZONE = dt.timezone(dt.timedelta(hours=_TZ_OFFSET))
_SGT = LOG_TIMEZONE


def sgt_now_str() -> str:
    # Human-friendly, single-line timestamp (no timezone suffix)
    return dt.datetime.now(LOG_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def make_run_id(prefix: str = "daily") -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{prefix}"


class EventLogger:
    def __init__(self, run_id: str, log_dir: Path) -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # 日次ログファイルはSGTの日付境界を基準にする。
        log_day = dt.datetime.now(_SGT).date().isoformat()
        self.log_file = self.log_dir / f"{log_day}.log"
        # ファイルハンドルを持続的に保持し、emit()ごとのopenを回避する。
        self._log_fh: io.TextIOWrapper = self.log_file.open("a", encoding="utf-8")
        # Windows CP1252 などの環境で日本語ログが UnicodeEncodeError になるのを防ぐ。
        for _s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
            if _s is not None and hasattr(_s, "reconfigure"):
                try:
                    _s.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    def close(self) -> None:
        """ログファイルハンドルをクローズする。"""
        try:
            self._log_fh.close()
        except Exception:
            pass

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def emit(self, *, level: str, stage: str, event: str, msg: str, kv: dict[str, Any] | None = None) -> None:
        ts = sgt_now_str()
        lvl = str(level).upper()
        safe_msg = _escape_text(str(msg))

        kv_pairs: list[str] = []
        if kv:
            for k in sorted(kv.keys()):
                key = _safe_key(k)
                kv_pairs.append(f"{key}={_format_value(kv[k])}")

        suffix = (" | " + " ".join(kv_pairs)) if kv_pairs else ""
        # ログ行は人が読みやすい形式にする。run_id プレフィックスは行ごとに付けない。
        line = f"{ts} {lvl} {stage} {event} - {safe_msg}{suffix}"

        # 重要: runtime.run_module_main が sys.stdout/stderr をリダイレクトするため、
        # ロガー出力は sys.__stdout__ を直接使用して無限ループを回避する。
        try:
            print(line, file=sys.__stdout__, flush=True)
        except Exception:
            # 最後の手段としてアクティブな stdout に出力する。
            print(line, flush=True)
        self._log_fh.write(line + "\n")
        self._log_fh.flush()


def _safe_key(k: object) -> str:
    s = str(k)
    # logfmt-ish: avoid spaces and '=' in keys
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    key = "".join(out).strip("_")
    return key or "key"


def _escape_text(s: str) -> str:
    # Keep logs strictly single-line and neutralize terminal control chars.
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


def _format_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)

    if isinstance(v, (list, tuple)):
        s = "[" + ",".join(_escape_text(str(x)) for x in v) + "]"
    elif isinstance(v, dict):
        # 各値は_format_valueを再帰呼び出しして一貫したシリアライズを適用する。
        items = []
        for k in sorted(v.keys()):
            items.append(f"{_safe_key(k)}:{_format_value(v[k])}")
        s = "{" + ",".join(items) + "}"
    else:
        s = _escape_text(str(v))

    needs_quote = any(ch.isspace() for ch in s) or any(ch in s for ch in ['"', "=", "|", "\\"]) or s == ""
    if needs_quote:
        s2 = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s2}"'
    return s
