from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaignitor.backtest.config import StrategyConfig
from alphaignitor.pipeline.zero_shot_ensemble.schema import (
    ENSEMBLE_MODELS,
    normalize_weights,
)
from alphaignitor.pipeline.zero_shot_ensemble.storage import (
    load_all_prediction_details_for_asof,
    load_best_weights,
)


@dataclass
class TradeSignal:
    ticker: str
    asof_date: str
    target_horizon: int
    expected_return: float
    ensemble_pred: float
    asof_close: float
    q10_close: float | None
    q50_close: float | None
    q90_close: float | None
    consensus: str  # "all", "majority", "mixed"
    score: float
    weights: dict[str, float]
    model_preds: dict[str, float]


def evaluate_signals_for_asof(
    *,
    conn: sqlite3.Connection | None = None,
    asof_date: str,
    tickers: list[str],
    series_by_ticker: dict[str, pd.DataFrame],
    strategy_cfg: StrategyConfig,
    models: list[str] | None = None,
    preloaded_details_map: dict | None = None,
    preloaded_weights_map: dict[str, dict[str, float]] | None = None,
    preloaded_close_map: dict[tuple[str, str], float] | None = None,
) -> list[TradeSignal]:
    """Generate and rank trading signals for a specific as-of date based on strategy rules."""
    model_list = models or ENSEMBLE_MODELS
    if preloaded_details_map is not None:
        details_map = preloaded_details_map
    elif conn is not None:
        details_map = load_all_prediction_details_for_asof(conn, asof_trade_date=asof_date)
    else:
        details_map = {}

    signals: list[TradeSignal] = []
    target_h_spec = str(strategy_cfg.target_horizon).strip().lower()

    for ticker in tickers:
        if preloaded_close_map is not None:
            asof_close = preloaded_close_map.get((ticker, asof_date))
        else:
            series = series_by_ticker.get(ticker)
            if series is None or series.empty:
                continue
            row_match = series[series["trade_date"] == asof_date]
            if row_match.empty:
                continue
            asof_close = float(row_match.iloc[0]["close"])

        if asof_close is None or not math.isfinite(asof_close) or asof_close <= 0:
            continue

        if preloaded_weights_map is not None:
            base_weights = preloaded_weights_map.get(ticker, {m: 1.0 / len(model_list) for m in model_list})
        elif conn is not None:
            base_weights = load_best_weights(conn, ticker=ticker, models=model_list)
        else:
            base_weights = {m: 1.0 / len(model_list) for m in model_list}

        # Evaluate candidate horizons
        horizons_to_check = [1, 2, 3, 4, 5] if target_h_spec == "best" else [int(target_h_spec)]
        best_signal_for_ticker: TradeSignal | None = None

        for horizon in horizons_to_check:
            preds: dict[str, float] = {}
            q10s: dict[str, float] = {}
            q50s: dict[str, float] = {}
            q90s: dict[str, float] = {}

            for m in model_list:
                item = details_map.get((ticker, m, horizon))
                if item is not None and item[4] is None and item[0] is not None:
                    p = float(item[0])
                    if math.isfinite(p) and p > 0:
                        preds[m] = p
                    if item[1] is not None and math.isfinite(float(item[1])):
                        q10s[m] = float(item[1])
                    if item[2] is not None and math.isfinite(float(item[2])):
                        q50s[m] = float(item[2])
                    if item[3] is not None and math.isfinite(float(item[3])):
                        q90s[m] = float(item[3])

            if len(preds) < 2:
                continue

            local_weights = normalize_weights(base_weights, available_models=set(preds))
            ensemble_pred = sum(preds[m] * local_weights[m] for m in local_weights)
            expected_ret = (ensemble_pred / asof_close) - 1.0

            if expected_ret < float(strategy_cfg.min_predicted_return):
                continue

            # Model consensus check
            up_models = sum(1 for m, val in preds.items() if val > asof_close)
            consensus_str = "all" if up_models == len(preds) else ("majority" if up_models >= 2 else "mixed")

            if strategy_cfg.consensus_level == "all" and up_models < len(preds):
                continue
            if strategy_cfg.consensus_level == "majority" and up_models < 2:
                continue

            # Quantile (q10) confidence filter
            q10_weighted: float | None = None
            if q10s and any(m in q10s for m in local_weights):
                w_sum = sum(local_weights[m] for m in local_weights if m in q10s)
                if w_sum > 0:
                    q10_weighted = sum(q10s[m] * local_weights[m] for m in local_weights if m in q10s) / w_sum

            if strategy_cfg.use_q10_filter and q10_weighted is not None:
                if q10_weighted < asof_close * 0.995:
                    continue

            q50_weighted: float | None = None
            if q50s:
                w_sum = sum(local_weights[m] for m in local_weights if m in q50s)
                if w_sum > 0:
                    q50_weighted = sum(q50s[m] * local_weights[m] for m in local_weights if m in q50s) / w_sum

            q90_weighted: float | None = None
            if q90s:
                w_sum = sum(local_weights[m] for m in local_weights if m in q90s)
                if w_sum > 0:
                    q90_weighted = sum(q90s[m] * local_weights[m] for m in local_weights if m in q90s) / w_sum

            # Score calculation
            uncertainty_spread = (q90_weighted - q10_weighted) / asof_close if (q90_weighted and q10_weighted) else 0.05
            score = expected_ret / (1.0 + uncertainty_spread)

            candidate_signal = TradeSignal(
                ticker=ticker,
                asof_date=asof_date,
                target_horizon=int(horizon),
                expected_return=float(expected_ret),
                ensemble_pred=float(ensemble_pred),
                asof_close=float(asof_close),
                q10_close=q10_weighted,
                q50_close=q50_weighted,
                q90_close=q90_weighted,
                consensus=consensus_str,
                score=float(score),
                weights=local_weights,
                model_preds=preds,
            )

            if best_signal_for_ticker is None or candidate_signal.score > best_signal_for_ticker.score:
                best_signal_for_ticker = candidate_signal

        if best_signal_for_ticker is not None:
            signals.append(best_signal_for_ticker)

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals
