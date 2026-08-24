from __future__ import annotations

import os
from pathlib import Path

from alphaignitor.config import PipelineConfig
from alphaignitor.logging_utils import EventLogger
from alphaignitor.pipeline.zero_shot_ensemble.report import run_report


def _resolve_forecast_path(root: Path, cfg: PipelineConfig) -> Path:
    # Prefer an explicit run_date forecast if present.
    for d in [cfg.run_date, cfg.end_date]:
        if d:
            candidate = root / "predict" / f"{d}_us_stock_ensemble_forecast.parquet"
            if candidate.exists():
                return candidate

    files = list((root / "predict").glob("*_ensemble_forecast.parquet"))
    if not files:
        raise FileNotFoundError("predict/ に ensemble 予測 parquet が見当たりません")

    # Avoid lexicographic traps (e.g., 'smoke_' sorting after dates). Choose most recent by mtime.
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def run(cfg: PipelineConfig, *, logger: EventLogger, root: Path) -> Path:
    forecast = _resolve_forecast_path(root, cfg)
    logger.emit(
        level="INFO",
        stage="report",
        event="start",
        msg="HTML レポート生成開始",
        kv={"forecast": str(forecast), "outdir": cfg.report_outdir},
    )

    try:
        html_path = run_report(
            forecast_path=forecast,
            outdir=root / cfg.report_outdir,
            asof_date=cfg.run_date,
            ticker_meta_csv=root / "us_stock_list.csv",
        )
        logger.emit(
            level="INFO",
            stage="report",
            event="done",
            msg="HTML レポート生成完了",
            kv={"report_file": str(html_path)},
        )
        return html_path
    except Exception as e:
        logger.emit(level="ERROR", stage="report", event="failed", msg=str(e))
        raise
