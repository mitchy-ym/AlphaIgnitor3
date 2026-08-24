"""ビルドステージ。

REST API からの日次 OHLCV ダウンロードと、
日次パネル parquet の生成を担当する。
"""
from __future__ import annotations

import os
from pathlib import Path

from alphaignitor.config import PipelineConfig
from alphaignitor.logging_utils import EventLogger
from alphaignitor.pipeline.download_market_data import run_download
from alphaignitor.pipeline.build_daily_panel import run_build_daily_panel


def _resolve_canonical_parquet(root: Path) -> Path:
    """aggs/parquet/ 以下の最新パネル parquet を返す。"""
    candidates = sorted((root / "aggs" / "parquet").glob("us_stock_daily_panel_*.parquet"))
    if not candidates:
        raise FileNotFoundError("aggs/parquet に日次パネル parquet が見当たりません")
    return candidates[-1]


def _expected_output_parquet(
    root: Path,
    *,
    start_date: str | None,
    end_date: str | None,
) -> Path | None:
    """start/end が確定している場合のみ期待出力パスを返す。"""
    if not start_date or not end_date:
        return None
    return root / "aggs" / "parquet" / f"us_stock_daily_panel_{start_date}_{end_date}.parquet"


def run(
    cfg: PipelineConfig,
    *,
    logger: EventLogger,
    root: Path,
    skip_download: bool = False,
    skip_build: bool = False,
) -> Path | None:
    """ダウンロードと、必要に応じて legacy パネルビルドを実行する。

    Args:
        skip_download: REST ダウンロードをスキップし、既存の day_aggs を使う。
        skip_build:    ビルドをスキップする。zero-shot ensemble では通常 True。
    """
    expected = _expected_output_parquet(root, start_date=cfg.start_date, end_date=cfg.end_date)

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

    if skip_build:
        logger.emit(level="INFO", stage="build", event="skipped", msg="ビルドをスキップ")
    else:
        if not cfg.start_date or not cfg.end_date:
            raise ValueError("start_date and end_date are required for build stage")
        tickers_csv = os.environ.get("TICKERS_CSV")
        tickers_csv_path = Path(tickers_csv) if tickers_csv else (root / "us_stock_list.csv")

        logger.emit(level="INFO", stage="build", event="start", msg="日次パネル生成開始")
        try:
            out_p = run_build_daily_panel(
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                download_dir=root / "aggs" / "us_stock_day",
                out_parquet=expected,
                tickers_csv=tickers_csv_path,
                max_tickers=cfg.max_tickers,
                splits_cache_dir=root / "aggs" / "splits_cache",
            )
            logger.emit(level="INFO", stage="build", event="done", msg="パネル生成完了", kv={"out": str(out_p)})
        except Exception as e:
            logger.emit(level="ERROR", stage="build", event="failed", msg=str(e))
            raise

    if expected is not None and expected.exists():
        return expected

    if skip_build:
        return None

    # skip-build 時や日付未指定時は既存の最新 parquet を探す
    try:
        return _resolve_canonical_parquet(root)
    except FileNotFoundError as e:
        hint = "--skip-build を指定した場合、aggs/parquet/us_stock_daily_panel_*.parquet が存在するか確認してください。"
        raise FileNotFoundError(f"{e} ({hint})")
