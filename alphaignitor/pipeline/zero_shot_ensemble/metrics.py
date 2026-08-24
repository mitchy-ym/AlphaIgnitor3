from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .schema import evaluable_direction_pair


def directional_accuracy(rows: Iterable[tuple[float, float, float]]) -> float:
    """Compute sign accuracy from `(asof_close, pred_close, actual_close)` rows."""
    hits = 0
    valid = 0
    for asof_close, pred_close, actual_close in rows:
        dirs = evaluable_direction_pair(asof_close, pred_close, actual_close)
        if dirs is None:
            continue
        pred_dir, actual_dir = dirs
        valid += 1
        hits += int(pred_dir == actual_dir)
    return float(hits / valid) if valid else float("nan")


def rmse(pred: pd.Series, actual: pd.Series) -> float:
    values = _finite_pair_values(pred, actual)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values[:, 0] - values[:, 1]))))


def mae(pred: pd.Series, actual: pd.Series) -> float:
    values = _finite_pair_values(pred, actual)
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.abs(values[:, 0] - values[:, 1])))


def _finite_pair_values(pred: pd.Series, actual: pd.Series) -> np.ndarray:
    p = pd.to_numeric(pred, errors="coerce").to_numpy(dtype="float64")
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype="float64")
    mask = np.isfinite(p) & np.isfinite(a) & (p != 0.0) & (a != 0.0)
    return np.column_stack([p[mask], a[mask]])


def metric_summary(df: pd.DataFrame, *, pred_col: str) -> dict[str, float]:
    valid = df.dropna(subset=["asof_close", pred_col, "actual_close"])
    if valid.empty:
        return {"directional_accuracy": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    rows = valid[["asof_close", pred_col, "actual_close"]].itertuples(index=False, name=None)
    return {
        "directional_accuracy": directional_accuracy(rows),
        "rmse": rmse(valid[pred_col], valid["actual_close"]),
        "mae": mae(valid[pred_col], valid["actual_close"]),
    }


def fmt_float(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value) * 100:+.{digits}f}%"
