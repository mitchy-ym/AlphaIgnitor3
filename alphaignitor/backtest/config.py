from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    max_slots: int = 3
    slot_size_pct: float = 0.3333
    min_cash_buffer_usd: float = 500.0


@dataclass
class StrategyConfig:
    min_predicted_return: float = 0.015
    target_horizon: str = "best"  # "1", "2", "3", "5", "best"
    holding_days: int = 4
    stop_loss_pct: float | None = None
    emergency_stop_loss_pct: float | None = 0.070  # Black-swan disaster stop (-7%)
    take_profit_pct: float | None = 0.040
    consensus_level: str = "majority"  # "all", "majority", "none"
    use_q10_filter: bool = True
    min_directional_accuracy: float = 0.55
    exit_on_down_signal: bool = True

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
    order_type: str = "MOO"  # Market On Open
    user_timezone: str = "Asia/Singapore"
    operating_hours: str = "20:00-24:00 SGT"
    slippage_pct: float = 0.0005
    commission_pct: float = 0.0005

    def calc_effective_entry(self, open_price: float) -> float:
        return float(open_price) * (1.0 + self.slippage_pct)

    def calc_effective_exit(self, raw_price: float) -> float:
        return float(raw_price) * (1.0 - self.slippage_pct)

    def calc_shares(self, available_cash: float, effective_entry: float) -> int:
        if effective_entry <= 0:
            return 0
        return int(available_cash // (effective_entry * (1.0 + self.commission_pct)))

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "capital": dataclasses.asdict(self.capital),
            "portfolio": dataclasses.asdict(self.portfolio),
            "strategy": dataclasses.asdict(self.strategy),
            "execution": dataclasses.asdict(self.execution),
        }


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
