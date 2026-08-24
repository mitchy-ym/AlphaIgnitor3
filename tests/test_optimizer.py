from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaignitor.pipeline.zero_shot_ensemble.market_data import (
    available_trade_dates,
    load_price_panel,
    optimization_asof_dates,
    ticker_series_map,
)
from alphaignitor.pipeline.zero_shot_ensemble.optimizer import (
    _build_inference_jobs,
    _load_complete_prediction_keys,
    build_training_rows,
    ensure_prediction_cache_for_universe,
    optimize_all_tickers,
    score_rows,
)
from alphaignitor.pipeline.zero_shot_ensemble.storage import connect


class TestOptimizer:
    def test_build_inference_jobs(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "trade_date": dates,
                "close": np.linspace(150, 160, 10),
            }
        )
        series_by_ticker = {"AAPL": df}

        jobs = _build_inference_jobs(
            tickers=["AAPL"],
            series_by_ticker=series_by_ticker,
            asof_dates=[dates[-1]],
            context_days=5,
        )
        assert len(jobs) == 1
        assert jobs[0]["ticker"] == "AAPL"
        assert len(jobs[0]["context"]) == 5

    def test_score_rows(self):
        rows = [
            {
                "horizon": 1,
                "asof_close": 100.0,
                "actual_close": 105.0,  # UP
                "preds": {"m1": 102.0, "m2": 104.0},  # both UP
            },
            {
                "horizon": 2,
                "asof_close": 100.0,
                "actual_close": 95.0,  # DOWN
                "preds": {"m1": 96.0, "m2": 94.0},  # both DOWN
            },
        ]
        weights = {"m1": 0.5, "m2": 0.5}
        score = score_rows(rows, weights=weights, min_available_models=2)
        assert score == 1.0  # 100% accuracy

    def test_full_mock_cache_and_optimization(
        self, temp_dir: Path, mock_day_aggs: Path, mock_adapters
    ):
        dates = available_trade_dates(mock_day_aggs)
        asof = dates[-1]
        tickers = ["AAPL", "MSFT"]

        price_panel = load_price_panel(mock_day_aggs, dates=dates, tickers=tickers)
        series_by_ticker = ticker_series_map(price_panel)

        train_asofs = optimization_asof_dates(
            dates,
            current_asof=asof,
            context_days=10,
            prediction_days=3,
            window_days=10,
        )

        db_path = temp_dir / "cache.sqlite3"
        optuna_db = temp_dir / "optuna.sqlite3"

        conn = connect(db_path)
        try:
            # Build cache
            cache_asofs = sorted(set(train_asofs + [asof]))
            ensure_prediction_cache_for_universe(
                conn=conn,
                tickers=tickers,
                series_by_ticker=series_by_ticker,
                adapters=mock_adapters,
                asof_dates=cache_asofs,
                prediction_days=3,
                context_days=10,
            )

            # Verify complete keys
            complete_keys = _load_complete_prediction_keys(conn)
            assert len(complete_keys) > 0
        finally:
            conn.close()

        # Run optimization
        weights_by_ticker = optimize_all_tickers(
            prediction_cache_path=db_path,
            optuna_storage_path=optuna_db,
            series_by_ticker=series_by_ticker,
            tickers=tickers,
            adapters=mock_adapters,
            asof_dates=train_asofs,
            prediction_days=3,
            context_days=10,
            n_trials=5,
            timeout_seconds=60,
            ticker_workers=1,
            min_available_models=2,
        )

        assert "AAPL" in weights_by_ticker
        assert "MSFT" in weights_by_ticker
        assert math.isclose(sum(weights_by_ticker["AAPL"].values()), 1.0)
