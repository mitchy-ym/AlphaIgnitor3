"""Zero-shot ensemble forecast stage."""
from __future__ import annotations

import os
from pathlib import Path

from alphaignitor.config import PipelineConfig
from alphaignitor.logging_utils import EventLogger
from alphaignitor.pipeline.zero_shot_ensemble.forecast import run_forecast


def run(
    cfg: PipelineConfig,
    *,
    logger: EventLogger,
    root: Path,
) -> Path:
    """Chronos-2 / TimesFM / TiREX の zero-shot ensemble 予測を生成する。"""
    logger.emit(
        level="INFO",
        stage="forecast",
        event="start",
        msg="Zero-shot アンサンブル予測開始",
        kv={
            "models": cfg.ensemble_models,
            "prediction_days": cfg.prediction_days,
            "trials": cfg.optuna_n_trials,
        },
    )

    tickers = os.environ.get("TICKERS")
    tickers_csv = os.environ.get("TICKERS_CSV")
    tickers_csv_path = Path(tickers_csv) if tickers_csv else (root / "us_stock_list.csv")

    try:
        out_path = run_forecast(
            day_aggs_dir=root / "aggs" / "us_stock_day",
            outdir=root / "predict",
            optuna_storage_path=root / cfg.optuna_storage_path,
            prediction_cache_path=root / cfg.prediction_cache_path,
            run_date=cfg.run_date,
            tickers=tickers,
            tickers_csv=tickers_csv_path,
            max_tickers=cfg.max_tickers,
            context_days=cfg.context_days,
            prediction_days=cfg.prediction_days,
            optuna_window_days=cfg.optuna_window_days,
            optuna_n_trials=cfg.optuna_n_trials,
            optuna_timeout_minutes=cfg.optuna_timeout_minutes,
            models=cfg.ensemble_models,
            chronos2_device_map=cfg.chronos2_device_map,
            timesfm_device=cfg.timesfm_device,
            tirex_backend=cfg.tirex_backend,
            chronos2_batch_size=cfg.chronos2_batch_size,
            timesfm_batch_size=cfg.timesfm_batch_size,
            tirex_batch_size=cfg.tirex_batch_size,
            min_available_models=cfg.min_available_models,
            optimizer_workers=cfg.optimizer_workers,
            root=root,
        )
        logger.emit(
            level="INFO",
            stage="forecast",
            event="done",
            msg="アンサンブル予測完了",
            kv={"output_file": str(out_path)},
        )
        return out_path
    except Exception as e:
        logger.emit(level="ERROR", stage="forecast", event="failed", msg=str(e))
        raise
