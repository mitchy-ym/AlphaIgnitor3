"""データ読み込みユーティリティ。"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import pandas as pd

from alphaignitor.common.day_store import read_day_partition


def _load_ticker_meta(csv_path) -> dict[str, dict]:
    """ティッカーメタデータをCSVから読み込む。"""
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8").fillna("")
        df.columns = df.columns.str.lower()
        if "ticker" not in df.columns:
            return {}
        df = df.set_index("ticker")
        return df.to_dict(orient="index")
    except Exception:
        return {}


def read_day_aggs_by_date(
    day_root: Path | str,
    trade_date: "dt.date | str",
    *,
    tickers: "set[str] | None" = None,
    need_open: bool = False,
) -> pd.DataFrame:
    try:
        if isinstance(trade_date, dt.date):
            td = trade_date.isoformat()
        else:
            td = dt.date.fromisoformat(str(trade_date).strip()).isoformat()
    except Exception:
        return pd.DataFrame(columns=["ticker", "open", "close"])

    cols = ["ticker", "close", "volume"]
    if need_open:
        cols.append("open")
    df = read_day_partition(day_root=Path(day_root), trade_date=td, columns=cols, tickers=tickers)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "open", "close"])  # empty

    for c in ["ticker"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["open", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    if tickers is not None and "ticker" in df.columns:
        df = df[df["ticker"].isin(tickers)].copy()
    return df
