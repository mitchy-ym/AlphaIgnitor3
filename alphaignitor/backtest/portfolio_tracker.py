from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_ACTIVE_POSITIONS_PATH = Path("cache/active_positions.json")


def load_active_positions(path: Path | None = None) -> list[dict[str, Any]]:
    pos_path = path or DEFAULT_ACTIVE_POSITIONS_PATH
    if not pos_path.exists():
        return []
    try:
        with pos_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_active_positions(positions: list[dict[str, Any]], path: Path | None = None) -> None:
    pos_path = path or DEFAULT_ACTIVE_POSITIONS_PATH
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    with pos_path.open("w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)


def record_position_entry(
    *,
    ticker: str,
    shares: int,
    entry_price: float,
    asof_date: str,
    target_horizon: int = 3,
    sl_price: float | None = None,
    tp_price: float | None = None,
    path: Path | None = None,
) -> None:
    positions = load_active_positions(path)
    # Remove existing if already present
    positions = [p for p in positions if p.get("ticker") != ticker.upper()]
    positions.append({
        "ticker": ticker.upper(),
        "shares": int(shares),
        "entry_price": float(entry_price),
        "entry_asof_date": asof_date,
        "holding_days": 0,
        "target_horizon": target_horizon,
        "sl_price": float(sl_price) if sl_price is not None else None,
        "tp_price": float(tp_price) if tp_price is not None else None,
    })
    save_active_positions(positions, path)


def record_position_exit(
    *,
    ticker: str,
    path: Path | None = None,
) -> dict[str, Any] | None:
    positions = load_active_positions(path)
    matched = next((p for p in positions if p.get("ticker") == ticker.upper()), None)
    if matched:
        positions = [p for p in positions if p.get("ticker") != ticker.upper()]
        save_active_positions(positions, path)
    return matched


def update_positions_holding_and_prices(
    *,
    current_prices: dict[str, float],
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Update current prices and mark holding days for active positions."""
    positions = load_active_positions(path)
    for p in positions:
        ticker = p.get("ticker", "")
        if ticker in current_prices:
            p["current_price"] = current_prices[ticker]
    save_active_positions(positions, path)
    return positions
