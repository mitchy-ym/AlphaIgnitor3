from __future__ import annotations

import datetime as dt
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

import alphaignitor.common.trading_calendar as tc
from alphaignitor.common._credentials import load_simple_env_file
from alphaignitor.common.day_store import (
    has_partition,
    list_partition_dates,
    partition_path,
    read_day_partition,
    write_day_partition,
)
from alphaignitor.common.massive_splits import (
    SplitEvent,
    adjustment_factor_for_date,
    load_or_fetch_splits,
)
from alphaignitor.logging_utils import EventLogger, make_run_id


class TestTradingCalendar:
    def test_market_holidays(self):
        holidays_2025 = tc.us_stock_market_holidays(2025)
        # New Year's Day
        assert dt.date(2025, 1, 1) in holidays_2025
        # Juneteenth
        assert dt.date(2025, 6, 19) in holidays_2025
        # Christmas
        assert dt.date(2025, 12, 25) in holidays_2025
        # Thanksgiving
        assert dt.date(2025, 11, 27) in holidays_2025

    def test_is_trading_day(self):
        # Saturday is not trading day
        assert not tc.is_us_stock_trading_day(dt.date(2025, 1, 4))
        # Sunday is not trading day
        assert not tc.is_us_stock_trading_day(dt.date(2025, 1, 5))
        # New Year is not trading day
        assert not tc.is_us_stock_trading_day(dt.date(2025, 1, 1))
        # Normal Wednesday (Jan 8, 2025) is trading day
        assert tc.is_us_stock_trading_day(dt.date(2025, 1, 8))

    def test_previous_and_next_trading_days(self):
        # Next trading days after Friday Jan 3, 2025
        fri = dt.date(2025, 1, 3)
        next_3 = tc.next_trading_days(fri, 3)
        assert next_3 == [dt.date(2025, 1, 6), dt.date(2025, 1, 7), dt.date(2025, 1, 8)]

        # Previous trading day before Monday Jan 6, 2025
        prev = tc.previous_trading_day(dt.date(2025, 1, 6))
        assert prev == fri

    def test_trading_days_back_from(self):
        wed = dt.date(2025, 1, 8)
        back_3 = tc.trading_days_back_from(wed, 3)
        assert back_3 == [dt.date(2025, 1, 6), dt.date(2025, 1, 7), dt.date(2025, 1, 8)]


class TestDayStore:
    def test_write_and_read_partition(self, temp_dir: Path):
        day_root = temp_dir / "day_aggs"
        d = dt.date(2025, 3, 1)

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOG"],
                "open": [150.5, 180.2],
                "high": [155.0, 185.0],
                "low": [150.0, 179.0],
                "close": [153.0, 182.0],
                "volume": [1000, 2000],
            }
        )

        out_path = write_day_partition(df, day_root=day_root, trade_date=d)
        assert out_path.exists()
        assert has_partition(day_root, d)
        assert list_partition_dates(day_root) == ["2025-03-01"]

        # Read back
        read_df = read_day_partition(day_root=day_root, trade_date=d)
        assert len(read_df) == 2
        assert set(read_df["ticker"]) == {"AAPL", "GOOG"}

        # Read with ticker filter
        read_aapl = read_day_partition(day_root=day_root, trade_date=d, tickers={"AAPL"})
        assert len(read_aapl) == 1
        assert read_aapl.iloc[0]["ticker"] == "AAPL"

        # Read with specific columns
        read_cols = read_day_partition(day_root=day_root, trade_date=d, columns=["ticker", "close"])
        assert list(read_cols.columns) == ["ticker", "close"]

    def test_read_corrupted_partition_warns(self, temp_dir: Path):
        day_root = temp_dir / "corrupted_aggs"
        day_root.mkdir(parents=True, exist_ok=True)
        bad_file = partition_path(day_root, "2025-01-01")
        bad_file.write_text("not a parquet file")

        with pytest.warns(UserWarning, match="Failed to read parquet partition"):
            df = read_day_partition(day_root=day_root, trade_date="2025-01-01")
        assert df.empty


class TestMassiveSplits:
    def test_adjustment_factor_for_date(self):
        events = [
            SplitEvent(execution_date=dt.date(2024, 6, 10), historical_adjustment_factor=0.1, adjustment_type="forward_split")
        ]
        # Date before split gets the adjustment factor (0.1)
        assert adjustment_factor_for_date(events_asc=events, d=dt.date(2024, 6, 1)) == 0.1
        # Date after split gets 1.0
        assert adjustment_factor_for_date(events_asc=events, d=dt.date(2024, 6, 15)) == 1.0

    def test_load_splits_cache_empty_list_does_not_refetch(self, temp_dir: Path, monkeypatch):
        cache_dir = temp_dir / "splits_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "NO_SPLIT.json"
        # Write empty split event list
        cache_file.write_text("[]", encoding="utf-8")

        fetch_called = False

        def mock_fetch(*args, **kwargs):
            nonlocal fetch_called
            fetch_called = True
            return []

        monkeypatch.setattr("alphaignitor.common.massive_splits.fetch_splits_for_ticker", mock_fetch)

        result = load_or_fetch_splits(
            ticker="NO_SPLIT",
            cache_dir=cache_dir,
            allowed_types={"forward_split"},
            api_key="mock_key",
        )

        assert result == []
        assert not fetch_called, "Should not re-fetch from API when cache file exists with empty list"


class TestCredentialsAndLogger:
    def test_load_simple_env_file(self, temp_dir: Path):
        env_file = temp_dir / "test.env"
        env_file.write_text(
            """
            # Comment
            KEY1=value1
            KEY2="quoted_value"
            KEY3='single_quoted'
            INVALID_LINE_NO_EQUALS
            """,
            encoding="utf-8",
        )
        loaded = load_simple_env_file(env_file)
        assert loaded == {"KEY1": "value1", "KEY2": "quoted_value", "KEY3": "single_quoted"}

    def test_event_logger(self, temp_dir: Path):
        log_dir = temp_dir / "logs"
        run_id = make_run_id("test")
        with EventLogger(run_id=run_id, log_dir=log_dir) as logger:
            logger.emit(level="INFO", stage="test", event="run", msg="Hello test", kv={"count": 10})

        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text(encoding="utf-8")
        assert "INFO test run - Hello test" in content
        assert "count=10" in content
