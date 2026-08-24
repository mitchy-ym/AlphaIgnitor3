from __future__ import annotations

from .config import (
    DEFAULT_TRADING_CONFIG_PATH,
    CapitalConfig,
    ExecutionConfig,
    PortfolioConfig,
    StrategyConfig,
    TradingConfig,
    load_trading_config,
    save_trading_config,
)
from .engine import BacktestEngine
from .metrics import BacktestResult, DailyPortfolioState, TradeRecord, calculate_metrics
from .optimizer import run_walk_forward_optimization
from .portfolio_tracker import (
    DEFAULT_ACTIVE_POSITIONS_PATH,
    load_active_positions,
    record_position_entry,
    record_position_exit,
    save_active_positions,
    update_positions_holding_and_prices,
)
from .report import format_action_sheet_text, generate_action_sheet, generate_backtest_html_report
from .strategy import TradeSignal, evaluate_signals_for_asof

__all__ = [
    "DEFAULT_TRADING_CONFIG_PATH",
    "CapitalConfig",
    "PortfolioConfig",
    "StrategyConfig",
    "ExecutionConfig",
    "TradingConfig",
    "load_trading_config",
    "save_trading_config",
    "BacktestEngine",
    "TradeRecord",
    "DailyPortfolioState",
    "BacktestResult",
    "calculate_metrics",
    "TradeSignal",
    "evaluate_signals_for_asof",
    "run_walk_forward_optimization",
    "generate_action_sheet",
    "format_action_sheet_text",
    "generate_backtest_html_report",
    "DEFAULT_ACTIVE_POSITIONS_PATH",
    "load_active_positions",
    "save_active_positions",
    "record_position_entry",
    "record_position_exit",
    "update_positions_holding_and_prices",
]
