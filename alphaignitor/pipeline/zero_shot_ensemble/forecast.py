from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import load_adapters
from .market_data import (
    available_trade_dates,
    forecast_trade_dates,
    latest_asof_date,
    load_price_panel,
    optimization_asof_dates,
    resolve_tickers,
    ticker_series_map,
)
from .optimizer import (
    build_training_rows,
    ensure_prediction_cache_for_universe,
    load_available_predictions,
    optimize_all_tickers,
)
from .metrics import metric_summary
from .schema import (
    CORE_FORECAST_COLUMNS,
    ENSEMBLE_MODELS,
    METRIC_NAMES,
    METRIC_TARGETS,
    display_direction,
    log_return,
    normalize_weights,
    weights_to_json,
)
from .storage import connect, load_all_prediction_details_for_asof, load_best_weights, load_model_prediction_details


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos-2 / TimesFM / TiREX zero-shot ensemble forecast")
    p.add_argument("--run-date", type=str, default=None, help="As-of date upper bound (YYYY-MM-DD)")
    p.add_argument("--day-aggs-dir", type=Path, default=Path("aggs/us_stock_day"))
    p.add_argument("--tickers", type=str, default=None, help="Comma-separated ticker list")
    p.add_argument("--tickers-csv", type=Path, default=None)
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--context-days", type=int, default=60)
    p.add_argument("--prediction-days", type=int, default=5)
    p.add_argument("--optuna-window-days", type=int, default=252)
    p.add_argument("--optuna-n-trials", type=int, default=100)
    p.add_argument("--optuna-timeout-minutes", type=int, default=10)
    p.add_argument("--optuna-storage-path", type=Path, default=Path("models/ensemble_optuna.sqlite3"))
    p.add_argument("--prediction-cache-path", type=Path, default=Path("cache/zero_shot_predictions.sqlite3"))
    p.add_argument("--models", type=str, default=",".join(ENSEMBLE_MODELS))
    p.add_argument("--chronos2-device-map", type=str, default="cuda")
    p.add_argument("--timesfm-device", type=str, default="auto")
    p.add_argument("--tirex-backend", type=str, default="auto")
    p.add_argument("--chronos2-batch-size", type=int, default=64)
    p.add_argument("--timesfm-batch-size", type=int, default=256)
    p.add_argument("--tirex-batch-size", type=int, default=256)
    p.add_argument("--min-available-models", type=int, default=2)
    p.add_argument("--optimizer-workers", type=int, default=None)
    p.add_argument("--outdir", type=Path, default=Path("predict"))
    return p.parse_args()


def run_forecast(
    *,
    day_aggs_dir: Path,
    outdir: Path,
    optuna_storage_path: Path,
    prediction_cache_path: Path,
    run_date: str | None = None,
    tickers: list[str] | str | None = None,
    tickers_csv: Path | None = None,
    max_tickers: int | None = None,
    context_days: int = 60,
    prediction_days: int = 5,
    optuna_window_days: int = 252,
    optuna_n_trials: int = 100,
    optuna_timeout_minutes: int = 10,
    models: list[str] | str | None = None,
    chronos2_device_map: str = "cuda",
    timesfm_device: str = "auto",
    tirex_backend: str = "auto",
    chronos2_batch_size: int = 64,
    timesfm_batch_size: int = 256,
    tirex_batch_size: int = 256,
    min_available_models: int = 2,
    optimizer_workers: int | None = None,
    root: Path | None = None,
) -> Path:
    base_root = root or Path(".")
    model_list: list[str]
    if isinstance(models, str):
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    elif isinstance(models, list):
        model_list = models
    else:
        model_list = ENSEMBLE_MODELS

    if not model_list:
        raise ValueError("At least one model must be specified")

    workers = int(optimizer_workers or (os.cpu_count() or 1))

    day_aggs_path = Path(day_aggs_dir)
    asof = latest_asof_date(day_aggs_path, run_date=run_date)
    all_available_dates = available_trade_dates(day_aggs_path)
    history_dates = [d for d in all_available_dates if d <= asof]
    print(f"[ensemble] asof={asof.isoformat()} available_dates={len(all_available_dates)}", flush=True)

    train_asofs = optimization_asof_dates(
        history_dates,
        current_asof=asof,
        context_days=int(context_days),
        prediction_days=int(prediction_days),
        window_days=int(optuna_window_days),
    )

    resolved_tickers = resolve_tickers(
        root=base_root,
        tickers_arg=tickers if isinstance(tickers, str) else (",".join(tickers) if tickers else None),
        tickers_csv=tickers_csv,
        max_tickers=max_tickers,
    )
    if not resolved_tickers:
        raise ValueError("No tickers resolved")
    print(
        f"[ensemble] tickers={len(resolved_tickers)} train_asofs={len(train_asofs)} "
        f"context_days={context_days} trials={optuna_n_trials}",
        flush=True,
    )

    print("[ensemble] loading price panel...", flush=True)
    price_panel = load_price_panel(day_aggs_path, dates=all_available_dates, tickers=resolved_tickers)
    series_by_ticker = ticker_series_map(price_panel)
    print(f"[ensemble] loaded price rows={len(price_panel)} ticker_series={len(series_by_ticker)}", flush=True)

    print(f"[ensemble] loading zero-shot models: {','.join(model_list)}", flush=True)
    adapters = load_adapters(
        model_list,
        context_days=int(context_days),
        horizon=int(prediction_days),
        chronos2_device_map=str(chronos2_device_map),
        timesfm_device=str(timesfm_device),
        tirex_backend=str(tirex_backend),
    )
    print("[ensemble] model loading done", flush=True)

    Path(optuna_storage_path).parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(prediction_cache_path)
    conn = connect(cache_path)
    try:
        cache_asofs = sorted(set(train_asofs + [asof]))
        print("[ensemble] building inference cache for universe...", flush=True)
        ensure_prediction_cache_for_universe(
            conn=conn,
            tickers=resolved_tickers,
            series_by_ticker=series_by_ticker,
            adapters=adapters,
            asof_dates=cache_asofs,
            prediction_days=int(prediction_days),
            context_days=int(context_days),
            model_batch_sizes={
                "chronos2": int(chronos2_batch_size),
                "timesfm": int(timesfm_batch_size),
                "tirex": int(tirex_batch_size),
            },
        )
        print("[ensemble] inference cache complete", flush=True)
    finally:
        conn.close()

    print("[ensemble] optimizing ticker weights...", flush=True)
    weights_by_ticker = optimize_all_tickers(
        prediction_cache_path=cache_path,
        optuna_storage_path=Path(optuna_storage_path),
        series_by_ticker=series_by_ticker,
        tickers=resolved_tickers,
        adapters=adapters,
        asof_dates=train_asofs,
        prediction_days=int(prediction_days),
        context_days=int(context_days),
        n_trials=int(optuna_n_trials),
        timeout_seconds=int(optuna_timeout_minutes) * 60,
        ticker_workers=workers,
        min_available_models=int(min_available_models),
    )
    print(f"[ensemble] optimized weights for {len(weights_by_ticker)} tickers", flush=True)

    conn = connect(cache_path)
    try:
        print("[ensemble] calculating validation metrics...", flush=True)
        validation_metrics = build_validation_metrics(
            conn=conn,
            series_by_ticker=series_by_ticker,
            tickers=resolved_tickers,
            models=list(adapters),
            weights_by_ticker=weights_by_ticker,
            asof_dates=train_asofs,
            prediction_days=int(prediction_days),
            min_available_models=int(min_available_models),
        )
        print("[ensemble] building current forecast rows...", flush=True)
        rows = build_current_forecast_rows(
            conn=conn,
            series_by_ticker=series_by_ticker,
            tickers=resolved_tickers,
            adapters=adapters,
            weights_by_ticker=weights_by_ticker,
            asof=asof,
            context_days=int(context_days),
            prediction_days=int(prediction_days),
            min_available_models=int(min_available_models),
            validation_metrics=validation_metrics,
        )
    finally:
        conn.close()

    out = pd.DataFrame(rows)
    for col in CORE_FORECAST_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[CORE_FORECAST_COLUMNS].sort_values(["ticker", "horizon"]).reset_index(drop=True)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    forecast_path = outdir_path / f"{asof.isoformat()}_us_stock_ensemble_forecast.parquet"
    out.to_parquet(forecast_path, index=False)
    print(f"[ensemble] asof={asof.isoformat()} tickers={out['ticker'].nunique()} rows={len(out)}")
    print(f"[ensemble] saved forecast: {forecast_path}")
    return forecast_path


def main() -> int:
    args = parse_args()
    run_forecast(
        day_aggs_dir=args.day_aggs_dir,
        outdir=args.outdir,
        optuna_storage_path=args.optuna_storage_path,
        prediction_cache_path=args.prediction_cache_path,
        run_date=args.run_date,
        tickers=args.tickers,
        tickers_csv=args.tickers_csv,
        max_tickers=args.max_tickers,
        context_days=args.context_days,
        prediction_days=args.prediction_days,
        optuna_window_days=args.optuna_window_days,
        optuna_n_trials=args.optuna_n_trials,
        optuna_timeout_minutes=args.optuna_timeout_minutes,
        models=args.models,
        chronos2_device_map=args.chronos2_device_map,
        timesfm_device=args.timesfm_device,
        tirex_backend=args.tirex_backend,
        chronos2_batch_size=args.chronos2_batch_size,
        timesfm_batch_size=args.timesfm_batch_size,
        tirex_batch_size=args.tirex_batch_size,
        min_available_models=args.min_available_models,
        optimizer_workers=args.optimizer_workers,
    )
    return 0


def build_current_forecast_rows(
    *,
    conn,
    series_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str],
    adapters,
    weights_by_ticker: dict[str, dict[str, float]],
    asof,
    context_days: int,
    prediction_days: int,
    min_available_models: int,
    validation_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    asof_str = asof.isoformat()
    future_dates = forecast_trade_dates(asof, prediction_days)
    details_map = load_all_prediction_details_for_asof(conn, asof_trade_date=asof_str)

    for ticker in tickers:
        series = series_by_ticker.get(ticker)
        if series is None or series.empty:
            continue
        if asof_str not in set(series["trade_date"].astype(str)):
            continue
        by_date = {str(row.trade_date): float(row.close) for row in series.itertuples(index=False)}
        asof_close = by_date.get(asof_str)
        if asof_close is None or not math.isfinite(asof_close):
            continue
        base_weights = weights_by_ticker.get(ticker) or load_best_weights(conn, ticker=ticker, models=list(adapters))
        for horizon, forecast_date in enumerate(future_dates, start=1):
            preds = {}
            for model in adapters:
                cached = details_map.get((ticker, model, horizon))
                if cached is not None and cached[4] is None and cached[0] is not None and math.isfinite(float(cached[0])):
                    preds[model] = float(cached[0])
            if len(preds) < min_available_models:
                continue

            local_weights = normalize_weights(base_weights, available_models=set(preds))
            ensemble_pred = sum(preds[model] * local_weights[model] for model in local_weights)
            ensemble_return = (ensemble_pred / asof_close - 1.0) if asof_close > 0 else float("nan")
            actual_close = by_date.get(forecast_date.isoformat())
            q10_close = _weighted_quantile_close_from_map(
                details_map=details_map,
                ticker=ticker,
                horizon=horizon,
                weights=local_weights,
                field_index=1,
            )
            q50_close = _weighted_quantile_close_from_map(
                details_map=details_map,
                ticker=ticker,
                horizon=horizon,
                weights=local_weights,
                field_index=2,
            )
            q90_close = _weighted_quantile_close_from_map(
                details_map=details_map,
                ticker=ticker,
                horizon=horizon,
                weights=local_weights,
                field_index=3,
            )
            q50_logret = log_return(float(q50_close), float(asof_close)) if q50_close is not None else log_return(float(ensemble_pred), float(asof_close))
            row = {
                "ticker": ticker,
                "asof_trade_date": asof_str,
                "horizon": int(horizon),
                "forecast_trade_date": forecast_date.isoformat(),
                "asof_close": float(asof_close),
                "ensemble_pred": float(ensemble_pred),
                "ensemble_return": float(ensemble_return),
                "ensemble_direction": display_direction(asof_close, ensemble_pred),
                "actual_close": float(actual_close) if actual_close is not None else np.nan,
                "actual_direction": display_direction(asof_close, actual_close),
                "weights_json": weights_to_json(local_weights),
                "q0.1": log_return(float(q10_close), float(asof_close)) if q10_close is not None else np.nan,
                "q0.5": q50_logret,
                "q0.9": log_return(float(q90_close), float(asof_close)) if q90_close is not None else np.nan,
            }
            for model in ENSEMBLE_MODELS:
                row[f"{model}_pred"] = preds.get(model, np.nan)
                row[f"weight_{model}"] = float(local_weights.get(model, 0.0))
            row.update(_flatten_metrics((validation_metrics or {}).get(ticker, {})))
            rows.append(row)
    return rows


def _weighted_quantile_close_from_map(
    *,
    details_map: dict,
    ticker: str,
    horizon: int,
    weights: dict[str, float],
    field_index: int,
) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for model, weight in weights.items():
        cached = details_map.get((ticker, model, horizon))
        if cached is None:
            continue
        value = cached[field_index]
        if value is None or not math.isfinite(float(value)):
            continue
        weighted_sum += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def build_validation_metrics(
    *,
    conn,
    series_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str],
    models: list[str],
    weights_by_ticker: dict[str, dict[str, float]],
    asof_dates: list,
    prediction_days: int,
    min_available_models: int,
) -> dict[str, dict[str, dict[str, float]]]:
    metrics_by_ticker: dict[str, dict[str, dict[str, float]]] = {}
    for ticker in tickers:
        series = series_by_ticker.get(ticker)
        if series is None or series.empty:
            continue
        validation_rows = build_training_rows(
            conn=conn,
            ticker=ticker,
            series=series,
            models=models,
            asof_dates=asof_dates,
            prediction_days=prediction_days,
            min_available_models=min_available_models,
        )
        if not validation_rows:
            continue
        base_weights = weights_by_ticker.get(ticker) or normalize_weights({model: 1.0 for model in models})
        metric_rows: list[dict[str, float]] = []
        for row in validation_rows:
            preds: dict[str, float] = row["preds"]
            local_weights = normalize_weights(base_weights, available_models=set(preds))
            metric_row = {
                "asof_close": float(row["asof_close"]),
                "actual_close": float(row["actual_close"]),
            }
            for model in models:
                metric_row[f"{model}_pred"] = preds.get(model, np.nan)
            metric_row["ensemble_pred"] = sum(preds[model] * local_weights[model] for model in local_weights)
            metric_rows.append(metric_row)
        metric_df = pd.DataFrame(metric_rows)
        metrics_by_ticker[ticker] = {
            model: metric_summary(metric_df, pred_col=f"{model}_pred")
            for model in models
            if f"{model}_pred" in metric_df.columns
        }
        metrics_by_ticker[ticker]["ensemble"] = metric_summary(metric_df, pred_col="ensemble_pred")
    return metrics_by_ticker


def _flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for target in METRIC_TARGETS:
        values = metrics.get(target, {})
        for metric in METRIC_NAMES:
            out[f"{target}_{metric}"] = float(values.get(metric, np.nan))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
