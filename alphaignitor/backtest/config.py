from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRADING_CONFIG_PATH = PROJECT_ROOT / "config" / "trading.yaml"


@dataclass
class CapitalConfig:
    initial_cash_usd: float = 35000.0
    currency: str = "USD"


@dataclass
class PortfolioConfig:
    min_slots: int = 3
    target_slots: int = 5
    max_slots: int | None = None  # None = flexible / no hard ceiling
    slot_size_pct: float = 0.25
    max_slot_size_pct: float = 0.3333
    min_cash_buffer_usd: float = 500.0


@dataclass
class StrategyConfig:
    min_predicted_return: float = 0.015
    min_predicted_short_return: float = 0.10  # High conviction hurdle for short selling
    target_horizon: str = "2"  # "1", "2", "3", "5", "best"
    holding_days: int = 3
    stop_loss_pct: float | None = None
    emergency_stop_loss_pct: float | None = 0.10  # Black-swan disaster stop (-10%)
    take_profit_pct: float | None = 0.04
    consensus_level: str = "all"  # "all", "majority", "none"
    use_q10_filter: bool = False
    min_directional_accuracy: float = 0.55
    exit_on_down_signal: bool = False
    max_per_sector: int = 2  # Hard cap: max 2 tickers per sector
    sector_policy: str = "tiered_1_to_2"  # 1 recommended, up to 2 if needed (どうしても枠)
    allow_short: bool = True  # Enable short selling in portfolio

    def get_effective_sl_pct(self) -> float | None:
        """Return explicit stop loss, or fallback to emergency disaster stop."""
        if self.stop_loss_pct is not None and self.stop_loss_pct > 0:
            return float(self.stop_loss_pct)
        if self.emergency_stop_loss_pct is not None and self.emergency_stop_loss_pct > 0:
            return float(self.emergency_stop_loss_pct)
        return None


@dataclass
class ExecutionConfig:
    broker: str = "moomoo"
    order_type: str = "LOO"  # Limit On Open (moomoo 指値 / 時間外OFF)
    user_timezone: str = "Asia/Singapore"
    operating_hours: str = "20:00-24:00 SGT"
    slippage_pct: float = 0.0005
    commission_pct: float = 0.0005
    limit_price_mode: str = "atr_adaptive"  # "atr_adaptive", "fixed_pct", "none"
    limit_atr_multiplier: float = 0.30
    limit_expected_return_multiplier: float = 0.25
    limit_max_gap_pct: float = 0.015  # +1.5% ceiling for opening gap
    limit_min_gap_pct: float = 0.005  # +0.5% floor for normal open tick

    def calc_effective_entry(self, open_price: float) -> float:
        return float(open_price) * (1.0 + self.slippage_pct)

    def calc_effective_exit(self, raw_price: float) -> float:
        return float(raw_price) * (1.0 - self.slippage_pct)

    def calc_effective_short_entry(self, open_price: float) -> float:
        return float(open_price) * (1.0 - self.slippage_pct)

    def calc_effective_short_cover(self, raw_price: float) -> float:
        return float(raw_price) * (1.0 + self.slippage_pct)

    def calc_shares(self, available_cash: float, effective_entry: float) -> int:
        if effective_entry <= 0 or available_cash <= 0:
            return 0
        return max(0, int(available_cash // (effective_entry * (1.0 + self.commission_pct))))

    def calc_cost_basis(self, shares: int, effective_entry: float) -> float:
        return shares * effective_entry * (1.0 + self.commission_pct)

    def calc_net_proceeds(self, shares: int, effective_exit: float) -> float:
        return shares * effective_exit * (1.0 - self.commission_pct)


@dataclass
class TradingConfig:
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def calc_sl_price(self, effective_entry: float) -> float | None:
        sl_pct = self.strategy.get_effective_sl_pct()
        if sl_pct is not None:
            return effective_entry * (1.0 - sl_pct)
        return None

    def calc_tp_price(self, effective_entry: float) -> float | None:
        if self.strategy.take_profit_pct is not None and self.strategy.take_profit_pct > 0:
            return effective_entry * (1.0 + float(self.strategy.take_profit_pct))
        return None

    def calc_short_sl_price(self, effective_entry: float) -> float | None:
        """For short positions, stop loss triggers when price rises above entry."""
        sl_pct = self.strategy.get_effective_sl_pct()
        if sl_pct is not None:
            return effective_entry * (1.0 + sl_pct)
        return None

    def calc_short_tp_price(self, effective_entry: float) -> float | None:
        """For short positions, take profit triggers when price falls below entry."""
        if self.strategy.take_profit_pct is not None and self.strategy.take_profit_pct > 0:
            return effective_entry * (1.0 - float(self.strategy.take_profit_pct))
        return None

    def calc_limit_price(
        self,
        asof_close: float,
        *,
        atr: float | None = None,
        expected_return: float | None = None,
    ) -> float:
        """Calculate optimal recommended limit price (LOO) for BUY."""
        base = float(asof_close)
        cand_offsets: list[float] = []
        if atr is not None and atr > 0:
            cand_offsets.append(self.execution.limit_atr_multiplier * float(atr))
        if expected_return is not None and expected_return > 0:
            cand_offsets.append(base * (self.execution.limit_expected_return_multiplier * float(expected_return)))

        if cand_offsets:
            offset = min(cand_offsets)
        else:
            offset = base * self.execution.limit_min_gap_pct

        max_offset = base * self.execution.limit_max_gap_pct
        min_offset = base * self.execution.limit_min_gap_pct
        offset = max(min_offset, min(max_offset, offset))
        limit_p = base + offset
        return round(limit_p, 4) if limit_p < 1.0 else round(limit_p, 2)

    def calc_short_limit_price(
        self,
        asof_close: float,
        *,
        atr: float | None = None,
        expected_return: float | None = None,
    ) -> float:
        """Calculate recommended limit price for SELL SHORT (floor limit: avoid excessive gap-down shorting)."""
        base = float(asof_close)
        cand_offsets: list[float] = []
        if atr is not None and atr > 0:
            cand_offsets.append(self.execution.limit_atr_multiplier * float(atr))
        if expected_return is not None and expected_return < 0:
            cand_offsets.append(base * (self.execution.limit_expected_return_multiplier * abs(float(expected_return))))

        if cand_offsets:
            offset = min(cand_offsets)
        else:
            offset = base * self.execution.limit_min_gap_pct

        max_offset = base * self.execution.limit_max_gap_pct
        min_offset = base * self.execution.limit_min_gap_pct
        offset = max(min_offset, min(max_offset, offset))
        limit_p = base - offset
        return round(limit_p, 4) if limit_p < 1.0 else round(limit_p, 2)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_atr(df: pd.DataFrame, window: int = 14) -> float | None:
    """Calculate 14-day Average True Range from price history DataFrame."""
    if df is None or df.empty or len(df) < 2:
        return None
    try:
        import numpy as np
        w = df.tail(window + 10).copy()
        high = pd.to_numeric(w.get("high", w["close"]), errors="coerce").astype(float)
        low = pd.to_numeric(w.get("low", w["close"]), errors="coerce").astype(float)
        close = pd.to_numeric(w["close"], errors="coerce").astype(float)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna()
        if tr.empty:
            return None
        val = float(tr.tail(window).mean()) if len(tr) >= window else float(tr.mean())
        return val if math.isfinite(val) and val > 0 else None
    except Exception:
        return None


def load_trading_config(path: Path | None = None) -> TradingConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_TRADING_CONFIG_PATH
    if not cfg_path.exists():
        return TradingConfig()

    if yaml is None:
        raise RuntimeError("PyYAML is required to load trading config. Please install pyyaml.")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    capital_dict = raw.get("capital", {})
    portfolio_dict = raw.get("portfolio", {})
    strategy_dict = raw.get("strategy", {})
    execution_dict = raw.get("execution", {})

    capital = CapitalConfig(**{k: v for k, v in capital_dict.items() if k in CapitalConfig.__annotations__})
    portfolio = PortfolioConfig(**{k: v for k, v in portfolio_dict.items() if k in PortfolioConfig.__annotations__})
    strategy = StrategyConfig(**{k: v for k, v in strategy_dict.items() if k in StrategyConfig.__annotations__})
    execution = ExecutionConfig(**{k: v for k, v in execution_dict.items() if k in ExecutionConfig.__annotations__})

    return TradingConfig(
        capital=capital,
        portfolio=portfolio,
        strategy=strategy,
        execution=execution,
    )


def save_trading_config(config: TradingConfig, path: Path | None = None) -> None:
    cfg_path = Path(path) if path is not None else DEFAULT_TRADING_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("PyYAML is required to save trading config. Please install pyyaml.")

    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False, allow_unicode=True)
