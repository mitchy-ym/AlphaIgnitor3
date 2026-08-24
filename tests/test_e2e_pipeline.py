from __future__ import annotations

from pathlib import Path

import pytest

from alphaignitor.cli import run_daily
from alphaignitor.pipeline.zero_shot_ensemble.adapters import register_adapter
from conftest import DummyMockAdapter


class TestE2EPipeline:
    def test_run_daily_end_to_end(self, temp_dir: Path, mock_day_aggs: Path):
        # Register mock models into registry
        register_adapter("mock_chronos2", DummyMockAdapter)
        register_adapter("mock_timesfm", DummyMockAdapter)
        register_adapter("mock_tirex", DummyMockAdapter)

        # Create config file
        config_path = temp_dir / "test_pipeline.yaml"
        config_path.write_text(
            f"""
            start_date: null
            end_date: null
            prediction_days: 3
            context_days: 10
            optuna_window_days: 10
            optuna_n_trials: 3
            optuna_timeout_minutes: 1
            optuna_storage_path: {temp_dir / "optuna.sqlite3"}
            prediction_cache_path: {temp_dir / "predictions.sqlite3"}
            ensemble_models:
              - mock_chronos2
              - mock_timesfm
              - mock_tirex
            min_available_models: 2
            optimizer_workers: 1
            max_tickers: 2
            report_outdir: {temp_dir / "reports"}
            """,
            encoding="utf-8",
        )

        # Create stock list CSV
        stock_list = temp_dir / "us_stock_list.csv"
        stock_list.write_text(
            "Ticker,Name,Sector\nAAPL,Apple Inc.,Technology\nMSFT,Microsoft Corp.,Technology\n",
            encoding="utf-8",
        )

        # Execute run_daily with skip_download=True, using mock day aggs
        exit_code = run_daily(
            config_path=config_path,
            run_date=None,
            skip_download=True,
            skip_build=True,
            skip_forecast=False,
            skip_report=False,
            build_panel=False,
            root=temp_dir,
        )

        assert exit_code == 0

        # Check prediction parquet exists
        predict_files = list((temp_dir / "predict").glob("*_us_stock_ensemble_forecast.parquet"))
        assert len(predict_files) == 1

        # Check report HTML exists
        report_files = list((temp_dir / "reports").glob("*/report.html"))
        assert len(report_files) == 1
        html_content = report_files[0].read_text(encoding="utf-8")
        assert "Apple Inc." in html_content
        assert "Microsoft Corp." in html_content
