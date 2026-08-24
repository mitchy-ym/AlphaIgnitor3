from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    trade_id: int
    ticker: str
    entry_asof_date: str
    entry_trade_date: str
    entry_price: float
    shares: int
    cost_basis: float  # entry_price * shares + fees
    exit_trade_date: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None  # "take_profit", "stop_loss", "time_expiry", "down_reversal"
    proceeds: float | None = None
    net_pnl: float | None = None
    return_pct: float | None = None
    holding_days: int = 0
    signal_expected_return: float = 0.0
    target_horizon: int = 3
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


@dataclass
class DailyPortfolioState:
    trade_date: str
    cash: float
    positions_value: float
    total_equity: float
    active_positions_count: int
    daily_return: float = 0.0
    drawdown: float = 0.0


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    payoff_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    avg_holding_days: float
    max_consecutive_losses: int
    trades: list[TradeRecord] = field(default_factory=list)
    daily_history: list[DailyPortfolioState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "cagr_pct": round(self.cagr * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 3),
            "payoff_ratio": round(self.payoff_ratio, 3),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_trade_return_pct": round(self.avg_trade_return * 100, 2),
            "avg_holding_days": round(self.avg_holding_days, 1),
            "max_consecutive_losses": self.max_consecutive_losses,
        }


def calculate_metrics(
    *,
    initial_capital: float,
    trades: list[TradeRecord],
    daily_history: list[DailyPortfolioState],
) -> BacktestResult:
    if not daily_history:
        return BacktestResult(
            initial_capital=initial_capital,
            final_equity=initial_capital,
            total_return=0.0,
            cagr=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            payoff_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_trade_return=0.0,
            avg_holding_days=0.0,
            max_consecutive_losses=0,
            trades=trades,
            daily_history=daily_history,
        )

    final_equity = daily_history[-1].total_equity
    total_return = (final_equity / initial_capital) - 1.0

    n_days = max(len(daily_history), 1)
    cagr = ((1.0 + total_return) ** (252.0 / n_days)) - 1.0 if (total_return > -1.0 and n_days > 10) else total_return

    # Daily returns & Sharpe
    equities = np.array([d.total_equity for d in daily_history], dtype=np.float64)
    daily_rets = np.diff(equities) / equities[:-1] if len(equities) > 1 else np.array([0.0])

    # Max Drawdown
    peaks = np.maximum.accumulate(equities)
    drawdowns = (peaks - equities) / peaks
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    mean_daily_ret = float(np.mean(daily_rets)) if len(daily_rets) > 0 else 0.0
    std_daily_ret = float(np.std(daily_rets)) if len(daily_rets) > 0 else 0.0

    sharpe = (mean_daily_ret / std_daily_ret * math.sqrt(252.0)) if std_daily_ret > 1e-8 else 0.0

    # Sortino ratio
    neg_rets = daily_rets[daily_rets < 0]
    std_neg = float(np.std(neg_rets)) if len(neg_rets) > 0 else 0.0
    sortino = (mean_daily_ret / std_neg * math.sqrt(252.0)) if std_neg > 1e-8 else (sharpe if sharpe > 0 else 0.0)

    calmar = (cagr / max_dd) if max_dd > 1e-6 else 0.0

    # Trade level statistics
    completed_trades = [t for t in trades if t.net_pnl is not None]
    total_trades = len(completed_trades)
    winning_trades = sum(1 for t in completed_trades if (t.net_pnl or 0) > 0)
    losing_trades = sum(1 for t in completed_trades if (t.net_pnl or 0) <= 0)
    win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

    gross_profit = sum(t.net_pnl for t in completed_trades if (t.net_pnl or 0) > 0)
    gross_loss = abs(sum(t.net_pnl for t in completed_trades if (t.net_pnl or 0) < 0))

    if gross_loss > 1e-6:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 99.9
    else:
        profit_factor = 0.0

    avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
    avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 1e-6 else 0.0

    avg_ret = float(np.mean([t.return_pct for t in completed_trades if t.return_pct is not None])) if completed_trades else 0.0
    avg_holding = float(np.mean([t.holding_days for t in completed_trades])) if completed_trades else 0.0

    # Max consecutive losses
    max_consec_losses = 0
    curr_losses = 0
    for t in completed_trades:
        if (t.net_pnl or 0) < 0:
            curr_losses += 1
            max_consec_losses = max(max_consec_losses, curr_losses)
        else:
            curr_losses = 0

    # Fill daily history drawdown
    for i, state in enumerate(daily_history):
        state.drawdown = float(drawdowns[i])

    return BacktestResult(
        initial_capital=float(initial_capital),
        final_equity=float(final_equity),
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_dd),
        calmar_ratio=float(calmar),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        payoff_ratio=float(payoff_ratio),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        avg_trade_return=float(avg_ret),
        avg_holding_days=float(avg_holding),
        max_consecutive_losses=max_consec_losses,
        trades=trades,
        daily_history=daily_history,
    )
