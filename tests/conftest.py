from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaignitor.common.day_store import write_day_partition
from alphaignitor.pipeline.zero_shot_ensemble.adapters import ForecastResult, ZeroShotAdapter


class DummyMockAdapter(ZeroShotAdapter):
    name = "mock_model"

    def __init__(self, multiplier: float = 1.01) -> None:
        self.multiplier = multiplier

    def forecast(self, close_context: np.ndarray, *, horizon: int) -> list[float]:
        return self.forecast_result(close_context, horizon=horizon).point

    def forecast_result(self, close_context: np.ndarray, *, horizon: int) -> ForecastResult:
        last = float(close_context[-1]) if len(close_context) > 0 else 100.0
        point = [last * (self.multiplier ** h) for h in range(1, horizon + 1)]
        q10 = [p * 0.98 for p in point]
        q50 = point[:]
        q90 = [p * 1.02 for p in point]
        return ForecastResult(point=point, q10=q10, q50=q50, q90=q90)

    def batch_forecast_result(self, close_contexts: list[np.ndarray], *, horizon: int) -> list[ForecastResult]:
        return [self.forecast_result(ctx, horizon=horizon) for ctx in close_contexts]


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def mock_day_aggs(temp_dir: Path) -> Path:
    """Generate 30 trading days of sample OHLCV parquet partitions for AAPL and MSFT."""
    day_root = temp_dir / "aggs" / "us_stock_day"
    tickers = ["AAPL", "MSFT"]
    base_prices = {"AAPL": 150.0, "MSFT": 300.0}

    start_date = dt.date(2025, 1, 1)
    dates: list[dt.date] = []
    cur = start_date
    while len(dates) < 30:
        if cur.weekday() < 5:  # Mon-Fri
            dates.append(cur)
        cur = cur + dt.timedelta(days=1)

    np.random.seed(42)
    for d_idx, trade_date in enumerate(dates):
        rows = []
        for ticker in tickers:
            base = base_prices[ticker] * (1.0 + 0.002 * d_idx)
            o = base + float(np.random.uniform(-1, 1))
            c = o + float(np.random.uniform(-1, 1))
            h = max(o, c) + float(np.random.uniform(0, 1))
            l = min(o, c) - float(np.random.uniform(0, 1))
            v = int(np.random.randint(100000, 500000))
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date.isoformat(),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "transactions": 500,
                    "vwap": (o + c) / 2.0,
                }
            )
        df = pd.DataFrame(rows)
        write_day_partition(df, day_root=day_root, trade_date=trade_date, overwrite=True)

    return day_root


@pytest.fixture
def mock_adapters() -> dict[str, ZeroShotAdapter]:
    return {
        "chronos2": DummyMockAdapter(multiplier=1.01),
        "timesfm": DummyMockAdapter(multiplier=1.005),
        "tirex": DummyMockAdapter(multiplier=1.015),
    }
