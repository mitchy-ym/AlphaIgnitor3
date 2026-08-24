from __future__ import annotations

from pathlib import Path

import pytest

from alphaignitor.pipeline.zero_shot_ensemble.storage import (
    connect,
    load_all_prediction_details_for_asof,
    load_best_weights,
    load_model_prediction,
    load_model_prediction_details,
    save_best_weights,
    save_model_predictions,
)


class TestStorage:
    def test_connect_and_schema(self, temp_dir: Path):
        db_path = temp_dir / "test_pred.sqlite3"
        conn = connect(db_path)
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        assert "model_predictions" in tables
        assert "ensemble_best_weights" in tables
        conn.close()

    def test_save_and_load_model_predictions(self, temp_dir: Path):
        db_path = temp_dir / "test_pred.sqlite3"
        conn = connect(db_path)

        save_model_predictions(
            conn,
            ticker="AAPL",
            asof_trade_date="2025-01-10",
            model="chronos2",
            predictions=[150.0, 152.0, 155.0],
            q10_predictions=[148.0, 150.0, 152.0],
            q50_predictions=[150.0, 152.0, 155.0],
            q90_predictions=[153.0, 155.0, 158.0],
        )

        # Load single
        pred, err = load_model_prediction(
            conn, ticker="AAPL", asof_trade_date="2025-01-10", model="chronos2", horizon=2
        )
        assert pred == 152.0
        assert err is None

        # Load details
        pred, q10, q50, q90, err = load_model_prediction_details(
            conn, ticker="AAPL", asof_trade_date="2025-01-10", model="chronos2", horizon=3
        )
        assert pred == 155.0
        assert q10 == 152.0
        assert q50 == 155.0
        assert q90 == 158.0
        assert err is None

        # Bulk load for asof
        details = load_all_prediction_details_for_asof(conn, asof_trade_date="2025-01-10")
        assert ("AAPL", "chronos2", 1) in details
        assert ("AAPL", "chronos2", 2) in details
        assert ("AAPL", "chronos2", 3) in details
        assert details[("AAPL", "chronos2", 1)][0] == 150.0

        conn.close()

    def test_save_and_load_best_weights(self, temp_dir: Path):
        db_path = temp_dir / "test_pred.sqlite3"
        conn = connect(db_path)

        weights = {"chronos2": 0.4, "timesfm": 0.3, "tirex": 0.3}
        save_best_weights(conn, ticker="MSFT", weights=weights, score=0.68, n_trials=50)

        loaded = load_best_weights(conn, ticker="MSFT", models=["chronos2", "timesfm", "tirex"])
        assert loaded == weights

        # Unknown ticker fallback
        fallback = load_best_weights(conn, ticker="UNKNOWN", models=["chronos2", "timesfm"])
        assert fallback == {"chronos2": 0.5, "timesfm": 0.5}

        conn.close()
