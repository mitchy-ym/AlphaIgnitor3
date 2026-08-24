from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import sqlite3
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from .adapters import ZeroShotAdapter
from .market_data import forecast_trade_dates, parse_iso_date
from .schema import equal_weights, evaluable_direction_pair, normalize_weights
from .storage import connect, load_model_prediction, load_model_prediction_details, save_best_weights, save_model_predictions


def _has_complete_cached_prediction(conn: sqlite3.Connection, *, ticker: str, asof_trade_date: str, model: str, horizon: int) -> bool:
    cached = load_model_prediction_details(
        conn,
        ticker=ticker,
        asof_trade_date=asof_trade_date,
        model=model,
        horizon=horizon,
    )
    if cached is None:
        return False
    pred, q10, q50, q90, error = cached
    values = [pred, q10, q50, q90]
    return error is None and all(v is not None and math.isfinite(float(v)) for v in values)


def _load_complete_prediction_keys(conn: sqlite3.Connection) -> set[tuple[str, str, str, int]]:
    """Return set of (ticker, asof_trade_date, model, horizon) for complete, error-free predictions in bulk."""
    rows = conn.execute(
        """
        SELECT ticker, asof_trade_date, model, horizon, pred_close, q10_close, q50_close, q90_close, error
        FROM model_predictions
        WHERE error IS NULL
        """
    ).fetchall()
    complete: set[tuple[str, str, str, int]] = set()
    for ticker, asof_str, model, horizon, pred, q10, q50, q90, err in rows:
        if err is not None:
            continue
        values = [pred, q10, q50, q90]
        if all(v is not None and math.isfinite(float(v)) for v in values):
            complete.add((str(ticker), str(asof_str), str(model), int(horizon)))
    return complete


def optimize_all_tickers(
    *,
    prediction_cache_path: Path,
    optuna_storage_path: Path,
    series_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str],
    adapters: dict[str, ZeroShotAdapter],
    asof_dates: list,
    prediction_days: int,
    context_days: int,
    n_trials: int,
    timeout_seconds: int,
    ticker_workers: int,
    min_available_models: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    worker_count = max(int(ticker_workers), 1)
    total = len(tickers)
    if worker_count == 1:
        for idx, ticker in enumerate(tickers, start=1):
            result = _optimize_ticker_job(
                idx=idx,
                total=total,
                ticker=ticker,
                series=series_by_ticker.get(ticker),
                prediction_cache_path=prediction_cache_path,
                optuna_storage_path=optuna_storage_path,
                adapters=adapters,
                asof_dates=asof_dates,
                prediction_days=prediction_days,
                context_days=context_days,
                n_trials=n_trials,
                timeout_seconds=timeout_seconds,
                trial_workers=1,
                min_available_models=min_available_models,
            )
            if result is not None:
                ticker_name, weights = result
                out[ticker_name] = weights
        return out

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {}
        for idx, ticker in enumerate(tickers, start=1):
            future = pool.submit(
                _optimize_ticker_job,
                idx=idx,
                total=total,
                ticker=ticker,
                series=series_by_ticker.get(ticker),
                prediction_cache_path=prediction_cache_path,
                optuna_storage_path=optuna_storage_path,
                adapters=adapters,
                asof_dates=asof_dates,
                prediction_days=prediction_days,
                context_days=context_days,
                n_trials=n_trials,
                timeout_seconds=timeout_seconds,
                trial_workers=1,
                min_available_models=min_available_models,
            )
            futures[future] = ticker

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[ensemble][optuna] {ticker}: failed {type(exc).__name__}: {exc}", flush=True)
                continue
            if result is not None:
                ticker_name, weights = result
                out[ticker_name] = weights
    return out


def ensure_prediction_cache_for_universe(
    *,
    conn: sqlite3.Connection,
    tickers: list[str],
    series_by_ticker: dict[str, pd.DataFrame],
    adapters: dict[str, ZeroShotAdapter],
    asof_dates: list,
    prediction_days: int,
    context_days: int,
    model_batch_sizes: dict[str, int] | None = None,
) -> None:
    jobs = _build_inference_jobs(
        tickers=tickers,
        series_by_ticker=series_by_ticker,
        asof_dates=asof_dates,
        context_days=context_days,
    )
    if not jobs:
        return

    completed_keys = _load_complete_prediction_keys(conn)

    for model, adapter in adapters.items():
        pending = [
            job
            for job in jobs
            if not all(
                (str(job["ticker"]), str(job["asof_trade_date"]), model, h) in completed_keys
                for h in range(1, int(prediction_days) + 1)
            )
        ]
        if not pending:
            continue

        print(f"[ensemble][cache] model={model} pending={len(pending)}", flush=True)
        configured = (model_batch_sizes or {}).get(model, 64)
        size = max(int(configured), 1)
        print(f"[ensemble][cache] model={model} batch_size={size}", flush=True)
        for start in range(0, len(pending), size):
            chunk = pending[start : start + size]
            contexts = [job["context"] for job in chunk]
            try:
                predicted = adapter.batch_forecast_result(contexts, horizon=prediction_days)
                if len(predicted) != len(chunk):
                    raise RuntimeError(
                        f"batch size mismatch for {model}: got {len(predicted)} expected {len(chunk)}"
                    )
                for job, result in zip(chunk, predicted):
                    save_model_predictions(
                        conn,
                        ticker=job["ticker"],
                        asof_trade_date=job["asof_trade_date"],
                        model=model,
                        predictions=result.point,
                        q10_predictions=result.q10,
                        q50_predictions=result.q50,
                        q90_predictions=result.q90,
                    )
            except Exception as exc:
                print(
                    f"[ensemble][cache] model={model} chunk={start // size + 1}: batch failed {type(exc).__name__}: {exc}",
                    flush=True,
                )
                for job in chunk:
                    try:
                        result = adapter.forecast_result(job["context"], horizon=prediction_days)
                        save_model_predictions(
                            conn,
                            ticker=job["ticker"],
                            asof_trade_date=job["asof_trade_date"],
                            model=model,
                            predictions=result.point,
                            q10_predictions=result.q10,
                            q50_predictions=result.q50,
                            q90_predictions=result.q90,
                        )
                    except Exception as inner:
                        msg = f"{type(inner).__name__}: {inner}"
                        save_model_predictions(
                            conn,
                            ticker=job["ticker"],
                            asof_trade_date=job["asof_trade_date"],
                            model=model,
                            predictions=[None] * int(prediction_days),
                            error=msg,
                        )
                        print(
                            f"[ensemble][cache] {job['ticker']} {job['asof_trade_date']} {model}: failed {msg}",
                            flush=True,
                        )


def _optimize_ticker_job(
    *,
    idx: int,
    total: int,
    ticker: str,
    series: pd.DataFrame | None,
    prediction_cache_path: Path,
    optuna_storage_path: Path,
    adapters: dict[str, ZeroShotAdapter],
    asof_dates: list,
    prediction_days: int,
    context_days: int,
    n_trials: int,
    timeout_seconds: int,
    trial_workers: int,
    min_available_models: int,
) -> tuple[str, dict[str, float]] | None:
    started = time.monotonic()
    print(f"[ensemble][optuna] {idx}/{total} {ticker}: start", flush=True)
    if series is None or series.empty:
        print(f"[ensemble][optuna] {idx}/{total} {ticker}: skipped (no price series)", flush=True)
        return None

    conn = connect(prediction_cache_path)
    try:
        weights = optimize_ticker(
            conn=conn,
            optuna_storage_path=optuna_storage_path,
            ticker=ticker,
            series=series,
            adapters=adapters,
            asof_dates=asof_dates,
            prediction_days=prediction_days,
            context_days=context_days,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            trial_workers=trial_workers,
            min_available_models=min_available_models,
        )
    finally:
        conn.close()

    elapsed = time.monotonic() - started
    weight_text = ", ".join(f"{k}={v:.3f}" for k, v in sorted(weights.items()))
    print(f"[ensemble][optuna] {idx}/{total} {ticker}: done {elapsed:.1f}s {weight_text}", flush=True)
    return ticker, weights


def optimize_ticker(
    *,
    conn: sqlite3.Connection,
    optuna_storage_path: Path,
    ticker: str,
    series: pd.DataFrame,
    adapters: dict[str, ZeroShotAdapter],
    asof_dates: list,
    prediction_days: int,
    context_days: int,
    n_trials: int,
    timeout_seconds: int,
    trial_workers: int,
    min_available_models: int,
) -> dict[str, float]:
    models = list(adapters.keys())
    rows = build_training_rows(
        conn=conn,
        ticker=ticker,
        series=series,
        models=models,
        asof_dates=asof_dates,
        prediction_days=prediction_days,
        min_available_models=min_available_models,
    )
    if not rows:
        weights = equal_weights(models)
        save_best_weights(conn, ticker=ticker, weights=weights, score=float("nan"), n_trials=0)
        print(f"[ensemble][optuna] {ticker}: no training rows; using equal weights", flush=True)
        return weights

    storage_url = f"sqlite:///{Path(optuna_storage_path).as_posix()}"
    study = optuna.create_study(
        study_name=_study_name(ticker=ticker, asof_dates=asof_dates),
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        raw = {model: trial.suggest_float(f"w_{model}", 0.0, 1.0) for model in models}
        weights = normalize_weights(raw)
        return score_rows(rows, weights=weights, min_available_models=min_available_models)

    completed_trials = _finished_trial_count(study)
    trials_to_run = max(int(n_trials) - completed_trials, 0)
    print(
        f"[ensemble][optuna] {ticker}: rows={len(rows)} trials={trials_to_run}/{n_trials} "
        f"timeout={timeout_seconds}s n_jobs={max(int(trial_workers), 1)}",
        flush=True,
    )
    if trials_to_run > 0:
        study.optimize(
            objective,
            n_trials=trials_to_run,
            timeout=int(timeout_seconds),
            n_jobs=max(int(trial_workers), 1),
            show_progress_bar=False,
        )
    best_trial = _best_complete_trial(study)
    if best_trial is None:
        weights = equal_weights(models)
        save_best_weights(conn, ticker=ticker, weights=weights, score=float("nan"), n_trials=len(study.trials))
        print(f"[ensemble][optuna] {ticker}: no completed trials; using equal weights", flush=True)
        return weights
    best_params = best_trial.params
    weights = normalize_weights({model: float(best_params.get(f"w_{model}", 0.0)) for model in models})
    score = float(best_trial.value) if best_trial.value is not None else float("nan")
    save_best_weights(conn, ticker=ticker, weights=weights, score=score, n_trials=len(study.trials))
    print(f"[ensemble][optuna] {ticker}: best_score={score:.4f} total_trials={len(study.trials)}", flush=True)
    return weights



def _build_inference_jobs(
    *,
    tickers: list[str],
    series_by_ticker: dict[str, pd.DataFrame],
    asof_dates: list,
    context_days: int,
) -> list[dict]:
    jobs: list[dict] = []
    for ticker in tickers:
        series = series_by_ticker.get(ticker)
        if series is None or series.empty:
            continue
        by_date = {str(row.trade_date): i for i, row in enumerate(series.itertuples(index=False))}
        closes = pd.to_numeric(series["close"], errors="coerce").to_numpy(dtype="float32")
        for asof in asof_dates:
            asof_str = asof.isoformat() if hasattr(asof, "isoformat") else str(asof)
            idx = by_date.get(asof_str)
            if idx is None or idx + 1 < int(context_days):
                continue
            context = closes[idx + 1 - int(context_days) : idx + 1]
            if not np.isfinite(context).all():
                continue
            jobs.append(
                {
                    "ticker": ticker,
                    "asof_trade_date": asof_str,
                    "context": context,
                }
            )
    return jobs


def load_available_predictions(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    asof_trade_date: str,
    models: list[str],
    horizon: int,
) -> dict[str, float]:
    preds: dict[str, float] = {}
    for model in models:
        cached = load_model_prediction(
            conn,
            ticker=ticker,
            asof_trade_date=asof_trade_date,
            model=model,
            horizon=horizon,
        )
        if cached is None:
            continue
        pred, error = cached
        if error is None and pred is not None and math.isfinite(float(pred)):
            preds[model] = float(pred)
    return preds


def build_training_rows(
    *,
    conn: sqlite3.Connection,
    ticker: str,
    series: pd.DataFrame,
    models: list[str],
    asof_dates: list,
    prediction_days: int,
    min_available_models: int,
) -> list[dict]:
    by_date = {str(row.trade_date): float(row.close) for row in series.itertuples(index=False)}
    rows: list[dict] = []
    for asof in asof_dates:
        asof_date = parse_iso_date(asof)
        asof_str = asof_date.isoformat()
        asof_close = by_date.get(asof_str)
        if asof_close is None or not math.isfinite(asof_close):
            continue
        future_dates = forecast_trade_dates(asof_date, prediction_days)
        for horizon, forecast_date in enumerate(future_dates, start=1):
            actual_close = by_date.get(forecast_date.isoformat())
            if actual_close is None or not math.isfinite(actual_close):
                continue
            preds = load_available_predictions(
                conn,
                ticker=ticker,
                asof_trade_date=asof_str,
                models=models,
                horizon=horizon,
            )
            if len(preds) >= int(min_available_models):
                rows.append(
                    {
                        "horizon": int(horizon),
                        "asof_close": float(asof_close),
                        "actual_close": float(actual_close),
                        "preds": preds,
                    }
                )
    return rows


def score_rows(rows: list[dict], *, weights: dict[str, float], min_available_models: int) -> float:
    by_horizon: dict[int, list[int]] = {}
    for row in rows:
        preds: dict[str, float] = row["preds"]
        available = set(preds)
        if len(available) < int(min_available_models):
            continue
        local_weights = normalize_weights(weights, available_models=available)
        ensemble = sum(preds[model] * local_weights[model] for model in local_weights)
        dirs = evaluable_direction_pair(float(row["asof_close"]), float(ensemble), float(row["actual_close"]))
        if dirs is None:
            continue
        pred_dir, actual_dir = dirs
        bucket = by_horizon.setdefault(int(row["horizon"]), [0, 0])
        bucket[0] += int(pred_dir == actual_dir)
        bucket[1] += 1
    scores = [hits / valid for hits, valid in by_horizon.values() if valid]
    return float(np.mean(scores)) if scores else 0.0


def _study_name(*, ticker: str, asof_dates: list) -> str:
    if asof_dates:
        start = parse_iso_date(asof_dates[0]).isoformat().replace("-", "")
        end = parse_iso_date(asof_dates[-1]).isoformat().replace("-", "")
        period = f"{start}_{end}"
    else:
        period = "no_validation_rows"
    safe_ticker = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(ticker).upper())
    return f"ensemble_weights_{safe_ticker}_{period}"


def _finished_trial_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state in {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL})


def _best_complete_trial(study: optuna.Study):
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    if not complete:
        return None
    return max(complete, key=lambda trial: float(trial.value))
