from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import traceback
from pathlib import Path
from typing import Sequence

from alphaignitor.logging_utils import EventLogger


class _LineCapture(io.TextIOBase):
    """stdout をラインごとに捕捉して EventLogger に転送するラッパー。

    stderr は tqdm バーがターミナルに直接描画できるようリダイレクトしない。
    """

    def __init__(self, on_line) -> None:
        self._on_line = on_line
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._on_line(line)
        return len(s)

    def flush(self) -> None:
        if self._buffer:
            self._on_line(self._buffer)
            self._buffer = ""


def run_module_main(
    *,
    stage: str,
    module: str,
    args: Sequence[str],
    cwd: Path,
    logger: EventLogger,
) -> None:
    logger.emit(level="INFO", stage=stage, event="command.start", msg="モジュール実行", kv={"module": module, "args": list(args)})

    def on_line(line: str) -> None:
        # \r はターミナル上書き用なのでログには不要。
        msg = line.replace("\r", "")
        if msg.strip():
            logger.emit(level="INFO", stage=stage, event="command.output", msg=msg)

    capture = _LineCapture(on_line)
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    code = 0

    try:
        os.chdir(cwd)
        sys.argv = [module, *list(args)]
        target = importlib.import_module(module)
        if not hasattr(target, "main"):
            raise RuntimeError(f"モジュールに main() が定義されていない: {module}")

        # stderr はリダイレクトしない: tqdm バーが直接ターミナルに描画される。
        with contextlib.redirect_stdout(capture):
            try:
                result = target.main()
                code = int(result) if result is not None else 0
            except SystemExit as e:
                if isinstance(e.code, int):
                    code = e.code
                elif e.code is None:
                    code = 0
                else:
                    code = 1
            except Exception:
                traceback.print_exc()
                code = 1
    finally:
        capture.flush()
        sys.argv = old_argv
        os.chdir(old_cwd)

    if code != 0:
        logger.emit(level="ERROR", stage=stage, event="command.failed", msg="コマンド失敗", kv={"exit_code": code})
        raise RuntimeError(f"ステージ {stage} が終了コード {code} で失敗した")

    logger.emit(level="INFO", stage=stage, event="command.done", msg="コマンド完了", kv={"exit_code": code})
