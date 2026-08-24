from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd

from alphaignitor.common.day_store import list_partition_dates, read_day_partition
import alphaignitor.common.trading_calendar as tc


def parse_iso_date(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value).strip())


def read_tickers_from_csv(path: Path) -> list[str]:
    df = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    for col in ["Ticker", "ticker", "symbol", "Symbol"]:
        if col in df.columns:
            values = df[col].astype(str).str.strip()
            return [v for v in values.tolist() if v]
    raise ValueError(f"Ticker column not found in: {path}")


def resolve_tickers(
    *,
    root: Path,
    tickers_arg: str | None = None,
    tickers_csv: Path | None = None,
    max_tickers: int | None = None,
) -> list[str]:
    raw = tickers_arg or os.environ.get("TICKERS")
    if raw:
        tickers = [s.strip().upper() for s in raw.split(",") if s.strip()]
    else:
        csv_path = tickers_csv or Path(os.environ.get("TICKERS_CSV", root / "us_stock_list.csv"))
        tickers = [s.upper() for s in read_tickers_from_csv(Path(csv_path))]

    # Preserve input order while removing duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    if max_tickers is not None:
        out = out[: int(max_tickers)]
    return out


def available_trade_dates(day_root: Path, *, end_date: dt.date | None = None) -> list[dt.date]:
    dates = [parse_iso_date(d) for d in list_partition_dates(day_root)]
    dates = [d for d in dates if end_date is None or d <= end_date]
    dates.sort()
    return dates


def latest_asof_date(day_root: Path, *, run_date: str | None) -> dt.date:
    end = parse_iso_date(run_date) if run_date else None
    dates = available_trade_dates(day_root, end_date=end)
    if not dates:
        raise FileNotFoundError(f"No day aggregate parquet files found in {day_root}")
    return dates[-1]


from concurrent.futures import ThreadPoolExecutor


def _load_single_day_partition(
    day_root: Path,
    trade_date: dt.date,
    ticker_set: set[str],
) -> pd.DataFrame:
    df = read_day_partition(
        day_root=day_root,
        trade_date=trade_date,
        columns=["ticker", "open", "close"],
        tickers=ticker_set,
    )
    if df.empty:
        return pd.DataFrame()
    df = df[["ticker", "open", "close"]].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["trade_date"] = trade_date.isoformat()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def load_price_panel(
    day_root: Path,
    *,
    dates: list[dt.date],
    tickers: list[str],
    max_workers: int = 8,
) -> pd.DataFrame:
    ticker_set = set(tickers)
    if not dates or not tickers:
        return pd.DataFrame(columns=["ticker", "trade_date", "open", "close"])

    workers = min(max_workers, len(dates))
    if workers <= 1:
        frames = [_load_single_day_partition(day_root, d, ticker_set) for d in dates]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(lambda d: _load_single_day_partition(day_root, d, ticker_set), dates))

    valid_frames = [f for f in frames if not f.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["ticker", "trade_date", "open", "close"])
    out = pd.concat(valid_frames, ignore_index=True)
    return out.dropna(subset=["ticker", "trade_date", "close"]).sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def ticker_series_map(price_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    series: dict[str, pd.DataFrame] = {}
    for ticker, group in price_panel.groupby("ticker", sort=False):
        g = group.sort_values("trade_date").reset_index(drop=True)
        series[str(ticker)] = g
    return series


def optimization_asof_dates(
    dates: list[dt.date],
    *,
    current_asof: dt.date,
    context_days: int,
    prediction_days: int,
    window_days: int,
) -> list[dt.date]:
    if current_asof not in dates:
        return []
    current_idx = dates.index(current_asof)
    last_train_idx = current_idx - int(prediction_days)
    first_train_idx = int(context_days) - 1
    if last_train_idx < first_train_idx:
        return []
    out = dates[first_train_idx : last_train_idx + 1]
    return out[-int(window_days) :]


def forecast_trade_dates(asof: dt.date, prediction_days: int) -> list[dt.date]:
    return tc.next_trading_days(asof, int(prediction_days))
