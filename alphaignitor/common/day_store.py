from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["open", "high", "low", "close", "vwap"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype("int64")
    if "transactions" in out.columns:
        out["transactions"] = pd.to_numeric(out["transactions"], errors="coerce").fillna(0).astype("int32")
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype("string")
    if "trade_date" in out.columns:
        out["trade_date"] = out["trade_date"].astype("string")
    return out


def _to_iso_date(trade_date: dt.date | str) -> str:
    if isinstance(trade_date, dt.date):
        return trade_date.isoformat()
    return dt.date.fromisoformat(str(trade_date).strip()).isoformat()


def partition_path(day_root: Path, trade_date: dt.date | str) -> Path:
    """フラット形式のファイルパスを返す。例: aggs/us_stock_day/2022-08-22.parquet"""
    d = _to_iso_date(trade_date)
    return Path(day_root) / f"{d}.parquet"


def has_partition(day_root: Path, trade_date: dt.date | str) -> bool:
    return partition_path(day_root, trade_date).is_file()


def list_partition_dates(day_root: Path) -> list[str]:
    root = Path(day_root)
    if not root.exists():
        return []
    out: list[str] = []
    for f in root.glob("*.parquet"):
        if not f.is_file():
            continue
        try:
            out.append(dt.date.fromisoformat(f.stem).isoformat())
        except Exception:
            continue
    out.sort()
    return out


def write_day_partition(
    df: pd.DataFrame,
    *,
    day_root: Path,
    trade_date: dt.date | str,
    overwrite: bool = False,
    compression: str = "snappy",
) -> Path:
    day_root = Path(day_root)
    day_root.mkdir(parents=True, exist_ok=True)
    out_path = partition_path(day_root, trade_date)
    if out_path.exists() and not overwrite:
        return out_path

    d = _to_iso_date(trade_date)
    w = _optimize_dtypes(df)
    w["trade_date"] = d

    # アトミックリネームで書き込む (書き込み途中のファイルが見えないようにする)
    tmp = day_root / f"{d}.parquet.part"
    w.to_parquet(tmp, index=False, compression=compression)
    tmp.replace(out_path)
    return out_path


def read_day_partition(
    *,
    day_root: Path,
    trade_date: dt.date | str,
    columns: list[str] | None = None,
    tickers: set[str] | None = None,
) -> pd.DataFrame:
    import warnings

    path = partition_path(Path(day_root), trade_date)
    if not path.is_file():
        return pd.DataFrame()

    read_cols = columns
    if columns is not None and tickers is not None and "ticker" not in columns:
        read_cols = list(columns) + ["ticker"]

    try:
        df = pd.read_parquet(path, columns=read_cols)
    except Exception as e:
        warnings.warn(f"Failed to read parquet partition {path}: {e}", stacklevel=2)
        return pd.DataFrame()

    if tickers is not None and "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
        df = df[df["ticker"].isin(tickers)].copy()
        if columns is not None and "ticker" not in columns:
            df = df[columns].copy()
    return df
