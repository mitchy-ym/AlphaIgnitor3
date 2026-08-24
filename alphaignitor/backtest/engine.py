from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaignitor.backtest.config import TradingConfig
from alphaignitor.backtest.metrics import (
    BacktestResult,
    DailyPortfolioState,
    TradeRecord,
    calculate_metrics,
)
from alphaignitor.backtest.strategy import TradeSignal


class BacktestEngine:
    def __init__(
        self,
        trading_config: TradingConfig,
        series_by_ticker: dict[str, pd.DataFrame],
    ) -> None:
        self.config = trading_config
        self.exec_cfg = trading_config.execution
        self.series_by_ticker = series_by_ticker
        self._price_lookup: dict[tuple[str, str], dict[str, float]] = {}
        self._build_price_lookup()

    def _build_price_lookup(self) -> None:
        for ticker, df in self.series_by_ticker.items():
            if df.empty:
                continue
            for row in df.itertuples(index=False):
                d_str = str(row.trade_date)
                o = float(row.open) if hasattr(row, "open") and math.isfinite(float(row.open)) else float(row.close)
                h = float(row.high) if hasattr(row, "high") and math.isfinite(float(row.high)) else max(o, float(row.close))
                l = float(row.low) if hasattr(row, "low") and math.isfinite(float(row.low)) else min(o, float(row.close))
                c = float(row.close)
                self._price_lookup[(ticker, d_str)] = {
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                }

    def run(
        self,
        *,
        trade_dates: list[str],
        signals_by_date: dict[str, list[TradeSignal]],
    ) -> BacktestResult:
        initial_cash = float(self.config.capital.initial_cash_usd)
        cash = initial_cash
        max_slots = int(self.config.portfolio.max_slots)
        slot_size_pct = float(self.config.portfolio.slot_size_pct)
        min_cash_buffer = float(self.config.portfolio.min_cash_buffer_usd)

        holding_days_limit = int(self.config.strategy.holding_days)
        exit_on_down = bool(self.config.strategy.exit_on_down_signal)

        active_trades: dict[str, TradeRecord] = {}  # ticker -> TradeRecord
        completed_trades: list[TradeRecord] = []
        daily_history: list[DailyPortfolioState] = []
        trade_counter = 0

        for day_idx, today in enumerate(trade_dates):
            yesterday = trade_dates[day_idx - 1] if day_idx > 0 else None

            # -------------------------------------------------------------
            # 1. MORNING OPEN (09:30 EST): Exits for scheduled / reversed trades
            # -------------------------------------------------------------
            tickers_to_exit_at_open: list[tuple[str, str]] = []
            for ticker, trade in list(active_trades.items()):
                prices = self._price_lookup.get((ticker, today))
                if prices is None:
                    continue

                # Check time expiry (holding days reached limit)
                if trade.holding_days >= holding_days_limit:
                    tickers_to_exit_at_open.append((ticker, "time_expiry"))
                    continue

                # Check down reversal from yesterday's signal
                if exit_on_down and yesterday is not None:
                    yesterday_signals = signals_by_date.get(yesterday, [])
                    matched_sig = next((s for s in yesterday_signals if s.ticker == ticker), None)
                    if matched_sig is not None and matched_sig.expected_return < 0:
                        tickers_to_exit_at_open.append((ticker, "down_reversal"))

            for ticker, reason in tickers_to_exit_at_open:
                trade = active_trades.pop(ticker)
                prices = self._price_lookup[(ticker, today)]
                open_price = prices["open"]
                exit_price = self.exec_cfg.calc_effective_exit(open_price)
                proceeds = self.exec_cfg.calc_net_proceeds(trade.shares, exit_price)
                cash += proceeds

                trade.exit_trade_date = today
                trade.exit_price = exit_price
                trade.exit_reason = reason
                trade.proceeds = proceeds
                trade.net_pnl = proceeds - trade.cost_basis
                trade.return_pct = (exit_price / trade.entry_price) - 1.0
                completed_trades.append(trade)

            # -------------------------------------------------------------
            # 2. MORNING OPEN (09:30 EST): New Buys generated on yesterday
            # -------------------------------------------------------------
            if yesterday is not None:
                pending_signals = signals_by_date.get(yesterday, [])
                eligible_signals = [s for s in pending_signals if s.ticker not in active_trades]

                for signal in eligible_signals:
                    if len(active_trades) >= max_slots:
                        break

                    prices = self._price_lookup.get((signal.ticker, today))
                    if prices is None or prices["open"] <= 0:
                        continue

                    total_equity_est = cash + sum(
                        t.shares * self._price_lookup.get((t.ticker, today), {}).get("open", t.entry_price)
                        for t in active_trades.values()
                    )
                    slot_target_cash = total_equity_est * slot_size_pct
                    available_cash = min(cash - min_cash_buffer, slot_target_cash)

                    if available_cash < 500.0:
                        continue

                    open_price = prices["open"]
                    effective_entry = self.exec_cfg.calc_effective_entry(open_price)
                    shares = self.exec_cfg.calc_shares(available_cash, effective_entry)

                    if shares <= 0:
                        continue

                    cost_basis = self.exec_cfg.calc_cost_basis(shares, effective_entry)
                    cash -= cost_basis

                    sl_price = self.config.calc_sl_price(effective_entry)
                    tp_price = self.config.calc_tp_price(effective_entry)

                    trade_counter += 1
                    new_trade = TradeRecord(
                        trade_id=trade_counter,
                        ticker=signal.ticker,
                        entry_asof_date=yesterday,
                        entry_trade_date=today,
                        entry_price=effective_entry,
                        shares=shares,
                        cost_basis=cost_basis,
                        holding_days=0,
                        signal_expected_return=signal.expected_return,
                        target_horizon=signal.target_horizon,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                    )
                    active_trades[signal.ticker] = new_trade

            # -------------------------------------------------------------
            # 3. INTRADAY SESSION: OCO (Stop-Loss / Take-Profit) trigger checks
            # -------------------------------------------------------------
            intraday_exits: list[tuple[str, str, float]] = []

            for ticker, trade in list(active_trades.items()):
                prices = self._price_lookup.get((ticker, today))
                if prices is None:
                    continue

                o, h, l, c = prices["open"], prices["high"], prices["low"], prices["close"]
                sl = trade.stop_loss_price
                tp = trade.take_profit_price

                # Check gap on open
                if sl is not None and o <= sl:
                    exit_p = self.exec_cfg.calc_effective_exit(o)
                    intraday_exits.append((ticker, "stop_loss", exit_p))
                    continue
                elif tp is not None and o >= tp:
                    exit_p = self.exec_cfg.calc_effective_exit(o)
                    intraday_exits.append((ticker, "take_profit", exit_p))
                    continue

                # Check intraday High / Low
                sl_hit = sl is not None and l <= sl
                tp_hit = tp is not None and h >= tp

                if sl_hit and tp_hit:
                    # Conservative assumption: Stop Loss hit first
                    exit_p = self.exec_cfg.calc_effective_exit(sl)
                    intraday_exits.append((ticker, "stop_loss", exit_p))
                elif sl_hit:
                    exit_p = self.exec_cfg.calc_effective_exit(sl)
                    intraday_exits.append((ticker, "stop_loss", exit_p))
                elif tp_hit:
                    exit_p = self.exec_cfg.calc_effective_exit(tp)
                    intraday_exits.append((ticker, "take_profit", exit_p))

            for ticker, reason, exit_p in intraday_exits:
                trade = active_trades.pop(ticker)
                proceeds = self.exec_cfg.calc_net_proceeds(trade.shares, exit_p)
                cash += proceeds

                trade.exit_trade_date = today
                trade.exit_price = exit_p
                trade.exit_reason = reason
                trade.proceeds = proceeds
                trade.net_pnl = proceeds - trade.cost_basis
                trade.return_pct = (exit_p / trade.entry_price) - 1.0
                trade.holding_days = max(trade.holding_days, 1)
                completed_trades.append(trade)

            # -------------------------------------------------------------
            # 4. MARKET CLOSE: Mark-to-market and Daily State Recording
            # -------------------------------------------------------------
            positions_value = 0.0
            for ticker, trade in active_trades.items():
                trade.holding_days += 1
                prices = self._price_lookup.get((ticker, today))
                close_price = prices["close"] if prices else trade.entry_price
                positions_value += trade.shares * close_price

            total_equity = cash + positions_value
            daily_state = DailyPortfolioState(
                trade_date=today,
                cash=float(cash),
                positions_value=float(positions_value),
                total_equity=float(total_equity),
                active_positions_count=len(active_trades),
            )
            daily_history.append(daily_state)

        # Close any remaining open positions at the end of simulation
        if trade_dates:
            last_date = trade_dates[-1]
            for ticker, trade in list(active_trades.items()):
                prices = self._price_lookup.get((ticker, last_date))
                raw_close = prices["close"] if prices else trade.entry_price
                close_p = self.exec_cfg.calc_effective_exit(raw_close)
                proceeds = self.exec_cfg.calc_net_proceeds(trade.shares, close_p)
                cash += proceeds
                trade.exit_trade_date = last_date
                trade.exit_price = close_p
                trade.exit_reason = "backtest_end"
                trade.proceeds = proceeds
                trade.net_pnl = proceeds - trade.cost_basis
                trade.return_pct = (close_p / trade.entry_price) - 1.0
                completed_trades.append(trade)

        return calculate_metrics(
            initial_capital=initial_cash,
            trades=completed_trades,
            daily_history=daily_history,
        )
