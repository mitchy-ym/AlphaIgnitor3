from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from alphaignitor.backtest.config import (
    CapitalConfig,
    PortfolioConfig,
    StrategyConfig,
    TradingConfig,
    load_trading_config,
    save_trading_config,
)
from alphaignitor.backtest.engine import BacktestEngine
from alphaignitor.backtest.metrics import (
    DailyPortfolioState,
    TradeRecord,
    calculate_metrics,
)
from alphaignitor.backtest.report import (
    format_action_sheet_text,
    generate_action_sheet,
    generate_backtest_html_report,
)
from alphaignitor.backtest.strategy import (
    TradeSignal,
    evaluate_signals_for_asof,
)
from alphaignitor.pipeline.zero_shot_ensemble.storage import (
    init_schema,
    save_best_weights,
    save_model_predictions,
)


def test_trading_config_defaults_and_serialization(tmp_path: Path):
    cfg = TradingConfig()
    assert cfg.capital.initial_cash_usd == 35000.0
    assert cfg.portfolio.max_slots is None
    assert cfg.portfolio.min_slots == 3
    assert cfg.portfolio.target_slots == 5
    assert cfg.execution.broker == "moomoo"

    save_path = tmp_path / "test_trading.yaml"
    save_trading_config(cfg, save_path)
    assert save_path.exists()

    loaded = load_trading_config(save_path)
    assert loaded.capital.initial_cash_usd == 35000.0
    assert loaded.strategy.holding_days == 3
    assert loaded.strategy.min_predicted_return == 0.015
    assert loaded.strategy.min_predicted_short_return == 0.10


def test_strategy_signal_generation(tmp_path: Path):
    db_path = tmp_path / "cache.sqlite3"
    conn = sqlite3.connect(db_path)
    init_schema(conn)

    # Save mock predictions
    save_model_predictions(
        conn,
        ticker="AAPL",
        asof_trade_date="2026-08-20",
        model="chronos2",
        predictions=[105.0, 106.0, 107.0, 108.0, 109.0],
        q10_predictions=[101.0, 102.0, 103.0, 104.0, 105.0],
    )
    save_model_predictions(
        conn,
        ticker="AAPL",
        asof_trade_date="2026-08-20",
        model="timesfm",
        predictions=[104.0, 105.0, 106.0, 107.0, 108.0],
        q10_predictions=[100.5, 101.5, 102.5, 103.5, 104.5],
    )
    save_model_predictions(
        conn,
        ticker="AAPL",
        asof_trade_date="2026-08-20",
        model="tirex",
        predictions=[106.0, 107.0, 108.0, 109.0, 110.0],
        q10_predictions=[102.0, 103.0, 104.0, 105.0, 106.0],
    )
    save_best_weights(
        conn,
        ticker="AAPL",
        weights={"chronos2": 0.34, "timesfm": 0.33, "tirex": 0.33},
        score=0.7,
        n_trials=20,
    )

    series_df = pd.DataFrame([
        {"trade_date": "2026-08-20", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        {"trade_date": "2026-08-21", "open": 101.0, "high": 103.0, "low": 100.5, "close": 102.5},
    ])
    series_map = {"AAPL": series_df}

    cfg = StrategyConfig(min_predicted_return=0.02, consensus_level="all", use_q10_filter=True)
    signals = evaluate_signals_for_asof(
        conn=conn,
        asof_date="2026-08-20",
        tickers=["AAPL"],
        series_by_ticker=series_map,
        strategy_cfg=cfg,
    )

    assert len(signals) == 1
    assert signals[0].ticker == "AAPL"
    assert signals[0].expected_return > 0.04
    assert signals[0].consensus == "all"
    conn.close()


def test_backtest_engine_execution_and_oco():
    cfg = TradingConfig(
        capital=CapitalConfig(initial_cash_usd=35000.0),
        portfolio=PortfolioConfig(max_slots=2, slot_size_pct=0.5, min_cash_buffer_usd=100.0),
        strategy=StrategyConfig(
            holding_days=3,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            exit_on_down_signal=False,
        ),
    )

    series_df = pd.DataFrame([
        {"trade_date": "2026-08-18", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        # Day 1: Enter at Open = 100.0. SL = 98.0, TP = 105.0. High = 102, Low = 99
        {"trade_date": "2026-08-19", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0},
        # Day 2: Take Profit hit! High = 106.0 >= 105.0
        {"trade_date": "2026-08-20", "open": 102.0, "high": 106.0, "low": 101.5, "close": 105.5},
        {"trade_date": "2026-08-21", "open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0},
    ])
    series_map = {"TEST": series_df}

    signal = TradeSignal(
        ticker="TEST",
        asof_date="2026-08-18",
        target_horizon=3,
        expected_return=0.05,
        ensemble_pred=105.0,
        asof_close=100.0,
        q10_close=100.5,
        q50_close=105.0,
        q90_close=108.0,
        consensus="all",
        score=0.045,
        weights={"chronos2": 1.0},
        model_preds={"chronos2": 105.0},
    )
    signals_by_date = {"2026-08-18": [signal]}

    engine = BacktestEngine(cfg, series_map)
    result = engine.run(
        trade_dates=["2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"],
        signals_by_date=signals_by_date,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.ticker == "TEST"
    assert trade.exit_reason == "take_profit"
    assert trade.net_pnl > 0
    assert result.winning_trades == 1
    assert result.final_equity > 35000.0


def test_action_sheet_generation_and_html(tmp_path: Path):
    cfg = TradingConfig(
        capital=CapitalConfig(initial_cash_usd=35000.0),
        portfolio=PortfolioConfig(max_slots=3, slot_size_pct=0.3333),
    )

    signal = TradeSignal(
        ticker="NVDA",
        asof_date="2026-08-21",
        target_horizon=3,
        expected_return=0.035,
        ensemble_pred=198.0,
        asof_close=192.0,
        q10_close=191.0,
        q50_close=198.0,
        q90_close=205.0,
        consensus="all",
        score=0.03,
        weights={"chronos2": 0.5, "timesfm": 0.5},
        model_preds={"chronos2": 198.0, "timesfm": 198.0},
    )

    sheet = generate_action_sheet(
        asof_date="2026-08-21",
        signals=[signal],
        trading_config=cfg,
    )

    assert len(sheet["buys"]) == 1
    assert sheet["buys"][0]["ticker"] == "NVDA"
    assert sheet["buys"][0]["shares"] > 0
    assert sheet["buys"][0]["allocated_usd"] > 10000.0

    text_output = format_action_sheet_text(sheet)
    assert "NVDA" in text_output
    assert "moomoo" in text_output

    # Test HTML report generation
    dummy_state = DailyPortfolioState(
        trade_date="2026-08-21",
        cash=35000.0,
        positions_value=0.0,
        total_equity=35000.0,
        active_positions_count=0,
    )
    result = calculate_metrics(
        initial_capital=35000.0,
        trades=[],
        daily_history=[dummy_state],
    )
    html_path = generate_backtest_html_report(
        result=result,
        trading_config=cfg,
        outdir=tmp_path,
        action_sheet=sheet,
    )
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "NVDA" in content
    assert "35,000" in content


def test_portfolio_tracker_and_positions(tmp_path: Path):
    from alphaignitor.backtest.portfolio_tracker import (
        load_active_positions,
        record_position_entry,
        record_position_exit,
        save_active_positions,
        update_positions_holding_and_prices,
    )

    pos_file = tmp_path / "active_pos.json"
    assert load_active_positions(pos_file) == []

    # Record entry
    record_position_entry(
        ticker="AAPL",
        shares=50,
        entry_price=200.0,
        asof_date="2026-08-20",
        target_horizon=3,
        sl_price=190.0,
        tp_price=210.0,
        path=pos_file,
    )

    loaded = load_active_positions(pos_file)
    assert len(loaded) == 1
    assert loaded[0]["ticker"] == "AAPL"
    assert loaded[0]["shares"] == 50

    # Update holding and prices
    updated = update_positions_holding_and_prices(
        current_prices={"AAPL": 205.0},
        path=pos_file,
    )
    assert updated[0]["current_price"] == 205.0

    # Record exit
    exited = record_position_exit(ticker="AAPL", path=pos_file)
    assert exited is not None
    assert exited["ticker"] == "AAPL"
    assert load_active_positions(pos_file) == []


def test_emergency_stop_loss_mechanism():
    cfg = TradingConfig(
        capital=CapitalConfig(initial_cash_usd=35000.0),
        portfolio=PortfolioConfig(max_slots=1, slot_size_pct=1.0),
        strategy=StrategyConfig(
            holding_days=5,
            stop_loss_pct=None,  # No standard SL
            emergency_stop_loss_pct=0.07,  # Disaster stop at -7%
            take_profit_pct=None,
        ),
    )

    # Effective SL should be 7%
    effective_sl = cfg.calc_sl_price(100.0)
    assert effective_sl == pytest.approx(93.0)


def test_loo_limit_price_calculation_and_atr():
    from alphaignitor.backtest.config import compute_atr

    df = pd.DataFrame([
        {"high": 102.0, "low": 98.0, "close": 100.0},
        {"high": 105.0, "low": 99.0, "close": 104.0},
        {"high": 106.0, "low": 101.0, "close": 103.0},
    ])
    atr = compute_atr(df, window=2)
    assert atr is not None
    assert atr > 0

    cfg = TradingConfig()
    # 1. Normal price ($100.0) with ATR=2.0 and expected return=0.03
    # ATR offset = 0.30 * 2.0 = 0.60 (0.6%)
    # Clamped between min (0.5% = 0.50) and max (2.0% = 2.00) -> 0.60
    # Limit = 100.0 + 0.60 = 100.60
    limit_p = cfg.calc_limit_price(100.0, atr=2.0, expected_return=0.03)
    assert limit_p == 100.60

    # 2. Huge ATR (10.0) -> offset would be 3.0 (3.0%), clamped to 1.5% ($1.50) -> 101.50
    limit_clamped_high = cfg.calc_limit_price(100.0, atr=10.0, expected_return=0.10)
    assert limit_clamped_high == 101.50

    # 3. Tiny ATR (0.01) -> offset would be 0.003, clamped to 0.5% ($0.50) -> 100.50
    limit_clamped_low = cfg.calc_limit_price(100.0, atr=0.01, expected_return=0.001)
    assert limit_clamped_low == 100.50

    # 4. Sub-dollar penny stock rounding to 4 decimals
    sub_dollar_limit = cfg.calc_limit_price(0.50, atr=0.02, expected_return=0.04)
    assert sub_dollar_limit == round(sub_dollar_limit, 4)
    assert sub_dollar_limit > 0.50


def test_loo_engine_execution_and_gap_skipping():
    series_df = pd.DataFrame([
        {"trade_date": "2026-08-20", "open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0},
        {"trade_date": "2026-08-21", "open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0},  # Gap up above limit 101.0
        {"trade_date": "2026-08-22", "open": 105.0, "high": 107.0, "low": 104.0, "close": 106.0},
    ])
    series_map = {"AAPL": series_df}

    cfg = TradingConfig(
        capital=CapitalConfig(initial_cash_usd=10000.0),
        portfolio=PortfolioConfig(max_slots=1, slot_size_pct=1.0),
        strategy=StrategyConfig(holding_days=2),
    )
    cfg.execution.order_type = "LOO"

    # Signal with limit_price of 101.0 on asof 2026-08-20
    signal = TradeSignal(
        ticker="AAPL",
        asof_date="2026-08-20",
        target_horizon=1,
        expected_return=0.03,
        asof_close=100.0,
        consensus="all",
        limit_price=101.0,
    )
    signals_by_date = {"2026-08-20": [signal]}

    engine = BacktestEngine(cfg, series_map)
    result = engine.run(
        trade_dates=["2026-08-20", "2026-08-21", "2026-08-22"],
        signals_by_date=signals_by_date,
    )

    # AAPL open on 2026-08-21 was 105.0 > limit_price 101.0 -> should be skipped, no trades executed!
    assert len(result.trades) == 0


def test_reserve_tickets_and_break_even_in_action_sheet():
    cfg = TradingConfig(portfolio=PortfolioConfig(max_slots=3))
    signals = [
        TradeSignal(ticker="NVDA", asof_date="2026-08-20", target_horizon=2, expected_return=0.04, asof_close=120.0, consensus="all", sector="Tech", limit_price=125.0),
        TradeSignal(ticker="AMD", asof_date="2026-08-20", target_horizon=2, expected_return=0.035, asof_close=145.0, consensus="all", sector="Tech", limit_price=150.0),
        TradeSignal(ticker="XOM", asof_date="2026-08-20", target_horizon=2, expected_return=0.030, asof_close=105.0, consensus="all", sector="Energy", limit_price=110.0),
        TradeSignal(ticker="JPM", asof_date="2026-08-20", target_horizon=2, expected_return=0.025, asof_close=195.0, consensus="all", sector="Financial", limit_price=200.0),
    ]

    # Active positions with one position up +3.0% (triggering suggest_break_even)
    active_positions = [
        {"ticker": "MSFT", "shares": 30, "entry_price": 400.0, "current_price": 412.0, "asof_date": "2026-08-19", "holding_days": 1, "target_horizon": 2}
    ]

    sheet = generate_action_sheet(
        asof_date="2026-08-20",
        signals=signals,
        active_positions=active_positions,
        trading_config=cfg,
    )

    # 1 position held -> 2 slots available
    assert sheet["available_slots"] == 2
    assert len(sheet["buys"]) == 2
    assert len(sheet["reserves"]) >= 1

    # Check break even suggestion on MSFT (3% gain > 2.5% threshold)
    assert sheet["holds"][0]["suggest_break_even"] is True
    assert sheet["holds"][0]["break_even_sl_price"] == 400.0

    # Format text output check
    text_out = format_action_sheet_text(sheet)
    assert "補欠・リザーブ買い候補" in text_out
    assert "建値ストップ引上げ推奨" in text_out


def test_short_selling_execution_and_tiered_sectors():
    cfg = TradingConfig(
        capital=CapitalConfig(initial_cash_usd=35000.0),
        portfolio=PortfolioConfig(min_slots=3, target_slots=5, max_slots=None, slot_size_pct=0.20),
        strategy=StrategyConfig(
            holding_days=2,
            allow_short=True,
            max_per_sector=2,
            sector_policy="tiered_1_to_2",
            take_profit_pct=0.07,
            emergency_stop_loss_pct=0.10,
        ),
    )

    # Series data: SHORT_TICKER drops on day 2 -> take profit triggered!
    series_df = pd.DataFrame([
        {"trade_date": "2026-08-18", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        # Day 1: Enter SHORT at Open 100.0. TP = 100 * (1 - 0.07) = 93.0. SL = 100 * (1 + 0.10) = 110.0
        {"trade_date": "2026-08-19", "open": 100.0, "high": 101.0, "low": 97.0, "close": 98.0},
        # Day 2: Low reaches 92.0 <= 93.0 -> Take Profit cover executed!
        {"trade_date": "2026-08-20", "open": 96.0, "high": 97.0, "low": 92.0, "close": 93.5},
        {"trade_date": "2026-08-21", "open": 94.0, "high": 95.0, "low": 93.0, "close": 94.0},
    ])
    series_map = {"SHORT_T": series_df}

    short_signal = TradeSignal(
        ticker="SHORT_T",
        asof_date="2026-08-18",
        target_horizon=2,
        expected_return=-0.06,
        asof_close=100.0,
        action="SHORT",
        sector="Tech",
        tier=1,
        score=0.06,
        limit_price=99.5,
    )

    engine = BacktestEngine(cfg, series_map)
    result = engine.run(
        trade_dates=["2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"],
        signals_by_date={"2026-08-18": [short_signal]},
    )

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.ticker == "SHORT_T"
    assert t.side == "SHORT"
    assert t.exit_reason == "take_profit"
    assert t.net_pnl > 0  # Profited from short drop!
    assert t.return_pct > 0
    assert result.final_equity > 35000.0

    # Verify daily events recorded for tooltip
    assert len(result.daily_history) == 4
    d1 = result.daily_history[1]  # 2026-08-19 (entry date)
    assert len(d1.entries_today) == 1
    assert d1.entries_today[0]["ticker"] == "SHORT_T"
    assert d1.entries_today[0]["side"] == "SHORT"

    d2 = result.daily_history[2]  # 2026-08-20 (exit date)
    assert len(d2.exits_today) == 1
    assert d2.exits_today[0]["ticker"] == "SHORT_T"
    assert d2.exits_today[0]["side"] == "SHORT"
    assert d2.exits_today[0]["pnl"] > 0


