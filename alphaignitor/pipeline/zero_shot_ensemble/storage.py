from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from .schema import equal_weights


def connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_predictions (
            ticker TEXT NOT NULL,
            asof_trade_date TEXT NOT NULL,
            model TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            pred_close REAL,
            q10_close REAL,
            q50_close REAL,
            q90_close REAL,
            error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, asof_trade_date, model, horizon)
        )
        """
    )
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(model_predictions)").fetchall()
    }
    for name in ["q10_close", "q50_close", "q90_close"]:
        if name not in cols:
            conn.execute(f"ALTER TABLE model_predictions ADD COLUMN {name} REAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ensemble_best_weights (
            ticker TEXT PRIMARY KEY,
            weights_json TEXT NOT NULL,
            score REAL,
            n_trials INTEGER,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def load_model_prediction(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    asof_trade_date: str,
    model: str,
    horizon: int,
) -> tuple[float | None, str | None] | None:
    row = conn.execute(
        """
        SELECT pred_close, error FROM model_predictions
        WHERE ticker = ? AND asof_trade_date = ? AND model = ? AND horizon = ?
        """,
        (ticker, asof_trade_date, model, int(horizon)),
    ).fetchone()
    if row is None:
        return None
    return row[0], row[1]


def load_model_prediction_details(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    asof_trade_date: str,
    model: str,
    horizon: int,
) -> tuple[float | None, float | None, float | None, float | None, str | None] | None:
    row = conn.execute(
        """
        SELECT pred_close, q10_close, q50_close, q90_close, error FROM model_predictions
        WHERE ticker = ? AND asof_trade_date = ? AND model = ? AND horizon = ?
        """,
        (ticker, asof_trade_date, model, int(horizon)),
    ).fetchone()
    if row is None:
        return None
    return row[0], row[1], row[2], row[3], row[4]


def load_all_prediction_details_for_asof(
    conn: sqlite3.Connection,
    *,
    asof_trade_date: str,
) -> dict[tuple[str, str, int], tuple[float | None, float | None, float | None, float | None, str | None]]:
    """Bulk load all model predictions for a specific asof date."""
    rows = conn.execute(
        """
        SELECT ticker, model, horizon, pred_close, q10_close, q50_close, q90_close, error
        FROM model_predictions
        WHERE asof_trade_date = ?
        """,
        (asof_trade_date,),
    ).fetchall()
    out = {}
    for ticker, model, horizon, pred, q10, q50, q90, err in rows:
        out[(str(ticker), str(model), int(horizon))] = (pred, q10, q50, q90, err)
    return out


def save_model_predictions(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    asof_trade_date: str,
    model: str,
    predictions: list[float | None] | None = None,
    q10_predictions: list[float | None] | None = None,
    q50_predictions: list[float | None] | None = None,
    q90_predictions: list[float | None] | None = None,
    error: str | None = None,
) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    preds = predictions if predictions is not None else [None] * 5
    q10_predictions = q10_predictions or [None] * len(preds)
    q50_predictions = q50_predictions or [None] * len(preds)
    q90_predictions = q90_predictions or [None] * len(preds)
    rows = [
        (
            ticker,
            asof_trade_date,
            model,
            idx + 1,
            pred,
            q10_predictions[idx] if idx < len(q10_predictions) else None,
            q50_predictions[idx] if idx < len(q50_predictions) else None,
            q90_predictions[idx] if idx < len(q90_predictions) else None,
            error,
            now,
        )
        for idx, pred in enumerate(preds)
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO model_predictions
        (ticker, asof_trade_date, model, horizon, pred_close, q10_close, q50_close, q90_close, error, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def save_best_weights(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    weights: dict[str, float],
    score: float,
    n_trials: int,
) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT OR REPLACE INTO ensemble_best_weights
        (ticker, weights_json, score, n_trials, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ticker, json.dumps(weights, ensure_ascii=False), float(score), int(n_trials), now),
    )
    conn.commit()


def load_best_weights(conn: sqlite3.Connection, *, ticker: str, models: list[str]) -> dict[str, float]:
    row = conn.execute(
        "SELECT weights_json FROM ensemble_best_weights WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    if row is None:
        return equal_weights(models)
    try:
        raw = json.loads(row[0])
    except json.JSONDecodeError:
        return equal_weights(models)
    return {model: float(raw.get(model, 0.0)) for model in models}
