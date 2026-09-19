"""ビルドステージ。

REST API からの日次 OHLCV ダウンロードを担当する。
（レガシーな特徴量パネル生成は Zero-Shot Ensemble 化に伴い廃止）
"""
from __future__ import annotations

from pathlib import Path

from alphaignitor.config import PipelineConfig
from alphaignitor.logging_utils import EventLogger
from alphaignitor.pipeline.download_market_data import run_download


def run(
    cfg: PipelineConfig,
    *,
    logger: EventLogger,
    root: Path,
    skip_download: bool = False,
    skip_build: bool = True,
) -> Path | None:
    """日足データのダウンロードを実行する。

    Args:
        skip_download: REST ダウンロードをスキップし、既存の day_aggs を使う。
        skip_build:    legacy 互換パラメータ（現在は常にダウンロードのみ実行）。
    """
    if skip_download:
        logger.emit(level="INFO", stage="download", event="skipped", msg="ダウンロードをスキップ")
    else:
        if not cfg.start_date or not cfg.end_date:
            raise ValueError("start_date and end_date are required for download stage")
        logger.emit(
            level="INFO",
            stage="download",
            event="start",
            msg="日足データダウンロード開始",
            kv={"start": cfg.start_date, "end": cfg.end_date},
        )
        try:
            stats = run_download(
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                download_dir=root / "aggs" / "us_stock_day",
            )
            logger.emit(level="INFO", stage="download", event="done", msg="ダウンロード完了", kv=stats)
        except Exception as e:
            logger.emit(level="ERROR", stage="download", event="failed", msg=str(e))
            raise

    return None
