from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alphaignitor.pipeline.zero_shot_ensemble.report import (
    _max_horizon,
    _mini_forecast_svg,
    _report_columns,
    build_charts_html,
    build_report_table,
    build_summary_dl,
    render_report_html,
    validate_forecast,
)


class TestReport:
    @pytest.fixture
    def sample_forecast_df(self) -> pd.DataFrame:
        rows = []
        for h in range(1, 6):
            rows.append(
                {
                    "ticker": "AAPL",
                    "asof_trade_date": "2025-01-10",
                    "horizon": h,
                    "forecast_trade_date": f"2025-01-1{h}",
                    "asof_close": 150.0,
                    "ensemble_pred": 150.0 + h * 1.5,
                    "ensemble_return": (h * 1.5) / 150.0,
                    "ensemble_direction": 1,
                    "actual_close": 150.0 + h * 1.2,
                    "actual_direction": 1,
                    "chronos2_pred": 150.0 + h * 1.4,
                    "timesfm_pred": 150.0 + h * 1.5,
                    "tirex_pred": 150.0 + h * 1.6,
                    "weight_chronos2": 0.33,
                    "weight_timesfm": 0.33,
                    "weight_tirex": 0.34,
                    "weights_json": '{"chronos2": 0.33, "timesfm": 0.33, "tirex": 0.34}',
                    "q0.1": 0.005 * h,
                    "q0.5": 0.01 * h,
                    "q0.9": 0.015 * h,
                }
            )
        return pd.DataFrame(rows)

    def test_validate_and_max_horizon(self, sample_forecast_df):
        validate_forecast(sample_forecast_df)
        assert _max_horizon(sample_forecast_df) == 5

    def test_report_table_and_columns(self, sample_forecast_df, temp_dir: Path):
        meta_csv = temp_dir / "meta.csv"
        meta_csv.write_text("Ticker,Name,Sector\nAAPL,Apple Inc.,Technology\n", encoding="utf-8")

        cols = _report_columns(max_horizon=5)
        assert len(cols) == 11  # Name, Sector, Signal, Bull, Bear, Day1..Day5, Avg

        table = build_report_table(sample_forecast_df, ticker_meta_csv=meta_csv, max_horizon=5)
        assert len(table) == 1
        assert table.iloc[0]["ticker"] == "AAPL"
        assert table.iloc[0]["name"] == "Apple Inc."
        assert table.iloc[0]["day1"] != ""
        assert table.iloc[0]["day5"] != ""

    def test_mini_forecast_svg(self):
        svg = _mini_forecast_svg(asof_close=100.0, preds=[101.0, 102.0, 103.0, 104.0, 105.0])
        assert "<svg" in svg
        assert "AsOf" in svg
        assert "D5" in svg  # 5th day label present

    def test_full_html_rendering(self, sample_forecast_df, temp_dir: Path):
        meta_csv = temp_dir / "meta.csv"
        meta_csv.write_text("Ticker,Name,Sector\nAAPL,Apple Inc.,Technology\n", encoding="utf-8")

        table = build_report_table(sample_forecast_df, ticker_meta_csv=meta_csv, max_horizon=5)
        summary_dl = build_summary_dl(
            df=sample_forecast_df,
            table=table,
            asof="2025-01-10",
            forecast_path=Path("predict/2025-01-10_us_stock_ensemble_forecast.parquet"),
            max_horizon=5,
        )
        charts_html = build_charts_html(df=sample_forecast_df, table=table, max_horizon=5)
        cols = _report_columns(max_horizon=5)

        html_out = render_report_html(
            table=table,
            cols=cols,
            summary_dl=summary_dl,
            acc_html="<div>Accuracy</div>",
            charts_html=charts_html,
            max_horizon=5,
        )

        assert "<!doctype html>" in html_out
        assert "Apple Inc." in html_out
        assert "Daily Forecast Report" in html_out
