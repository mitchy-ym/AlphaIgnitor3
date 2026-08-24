from __future__ import annotations

import copy
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from alphaignitor.backtest.config import (
    DEFAULT_TRADING_CONFIG_PATH,
    TradingConfig,
    load_trading_config,
    save_trading_config,
)
from alphaignitor.backtest.engine import BacktestEngine
from alphaignitor.backtest.metrics import BacktestResult
from alphaignitor.backtest.strategy import TradeSignal, evaluate_signals_for_asof
from alphaignitor.pipeline.zero_shot_ensemble.schema import ENSEMBLE_MODELS
from alphaignitor.pipeline.zero_shot_ensemble.storage import (
    load_all_prediction_details_for_asof,
    load_best_weights,
)


def run_walk_forward_optimization(
    *,
    conn: sqlite3.Connection,
    series_by_ticker: dict[str, pd.DataFrame],
    tickers: list[str],
    all_trade_dates: list[str],
    months: int = 6,
    n_trials: int = 100,
    timeout_seconds: int = 180,
    base_config: TradingConfig | None = None,
    apply_best_to_config: bool = True,
    config_save_path: Path | None = None,
) -> tuple[TradingConfig, list[dict[str, Any]], BacktestResult, BacktestResult]:
    """Run Optuna In-Sample/Out-of-Sample Walk-Forward optimization to find maximum profit/robust rules."""
    config = base_config or load_trading_config()

    # Determine date range for backtest
    # 1 month ≈ 21 trading days
    total_days_needed = int(months * 21)
    if len(all_trade_dates) > total_days_needed:
        eval_dates = all_trade_dates[-total_days_needed:]
    else:
        eval_dates = all_trade_dates[:]

    if len(eval_dates) < 20:
        raise ValueError(f"Insufficient trade dates for backtest: got {len(eval_dates)}, need at least 20")

    # In-Sample (first ~67%) and Out-of-Sample (last ~33%)
    split_idx = int(len(eval_dates) * 0.67)
    in_sample_dates = eval_dates[:split_idx]
    out_of_sample_dates = eval_dates[split_idx:]

    print(
        f"[backtest][optimize] Total dates: {len(eval_dates)} "
        f"(In-Sample: {len(in_sample_dates)} dates [{in_sample_dates[0]} ~ {in_sample_dates[-1]}], "
        f"Out-of-Sample: {len(out_of_sample_dates)} dates [{out_of_sample_dates[0]} ~ {out_of_sample_dates[-1]}])",
        flush=True,
    )

    print("[backtest][optimize] Preloading prediction cache & weights into memory for fast trials...", flush=True)
    weights_map = {t: load_best_weights(conn, ticker=t, models=ENSEMBLE_MODELS) for t in tickers}
    details_map_by_date = {d: load_all_prediction_details_for_asof(conn, asof_trade_date=d) for d in eval_dates}
    close_map = {
        (t, str(row.trade_date)): float(row.close)
        for t, df in series_by_ticker.items()
        if not df.empty
        for row in df.itertuples(index=False)
    }
    print("[backtest][optimize] Preloading complete. Ready for Optuna search.", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        # Sample parameters tailored to moomoo swing trading
        slots = trial.suggest_categorical("max_slots", [3, 5])
        holding_days = trial.suggest_categorical("holding_days", [2, 3, 4, 5])
        min_return = trial.suggest_categorical("min_predicted_return", [0.010, 0.015, 0.020, 0.025, 0.030])
        consensus = trial.suggest_categorical("consensus_level", ["all", "majority"])
        use_q10 = trial.suggest_categorical("use_q10_filter", [True, False])
        stop_loss = trial.suggest_categorical("stop_loss_pct", [0.015, 0.020, 0.025, 0.030, None])
        take_profit = trial.suggest_categorical("take_profit_pct", [0.030, 0.040, 0.050, 0.060, 0.080, None])
        exit_on_down = trial.suggest_categorical("exit_on_down_signal", [True, False])

        trial_cfg = copy.deepcopy(config)
        trial_cfg.portfolio.max_slots = slots
        trial_cfg.portfolio.slot_size_pct = 1.0 / slots
        trial_cfg.strategy.holding_days = holding_days
        trial_cfg.strategy.min_predicted_return = min_return
        trial_cfg.strategy.consensus_level = consensus
        trial_cfg.strategy.use_q10_filter = use_q10
        trial_cfg.strategy.stop_loss_pct = stop_loss
        trial_cfg.strategy.take_profit_pct = take_profit
        trial_cfg.strategy.exit_on_down_signal = exit_on_down

        # Generate signals for all dates using in-memory preloaded maps
        signals_by_date: dict[str, list[TradeSignal]] = {}
        for d in eval_dates:
            signals_by_date[d] = evaluate_signals_for_asof(
                asof_date=d,
                tickers=tickers,
                series_by_ticker=series_by_ticker,
                strategy_cfg=trial_cfg.strategy,
                preloaded_details_map=details_map_by_date[d],
                preloaded_weights_map=weights_map,
                preloaded_close_map=close_map,
            )

        # Run In-Sample
        engine_is = BacktestEngine(trial_cfg, series_by_ticker)
        res_is = engine_is.run(trade_dates=in_sample_dates, signals_by_date=signals_by_date)

        # Run Out-of-Sample
        engine_oos = BacktestEngine(trial_cfg, series_by_ticker)
        res_oos = engine_oos.run(trade_dates=out_of_sample_dates, signals_by_date=signals_by_date)

        # Optimization Score = In-Sample Sharpe * Trade count penalty + Return - MaxDD penalty
        trade_penalty = min(1.0, res_is.total_trades / 12.0)
        dd_penalty = res_is.max_drawdown * 3.0
        score = (res_is.sharpe_ratio * trade_penalty) + (res_is.total_return * 2.0) - dd_penalty

        record = {
            "trial_number": trial.number,
            "score": round(score, 3),
            "params": trial.params,
            "is_return_pct": round(res_is.total_return * 100, 2),
            "is_sharpe": round(res_is.sharpe_ratio, 2),
            "is_max_dd_pct": round(res_is.max_drawdown * 100, 2),
            "is_win_rate_pct": round(res_is.win_rate * 100, 2),
            "is_trades": res_is.total_trades,
            "oos_return_pct": round(res_oos.total_return * 100, 2),
            "oos_sharpe": round(res_oos.sharpe_ratio, 2),
            "oos_max_dd_pct": round(res_oos.max_drawdown * 100, 2),
            "oos_win_rate_pct": round(res_oos.win_rate * 100, 2),
            "oos_trades": res_oos.total_trades,
        }
        trial_records.append(record)
        return float(score)

    print(f"[backtest][optimize] Running {n_trials} Optuna trials (timeout: {timeout_seconds}s)...", flush=True)
    started = time.monotonic()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    elapsed = time.monotonic() - started
    print(f"[backtest][optimize] Finished {len(study.trials)} trials in {elapsed:.1f}s", flush=True)

    valid_records = [r for r in trial_records if math.isfinite(r["score"])]
    valid_records.sort(key=lambda r: (r["score"] + r["oos_return_pct"] * 0.1), reverse=True)

    best_params = study.best_params
    best_config = copy.deepcopy(config)
    best_config.portfolio.max_slots = best_params.get("max_slots", 3)
    best_config.portfolio.slot_size_pct = 1.0 / best_config.portfolio.max_slots
    best_config.strategy.holding_days = best_params.get("holding_days", 3)
    best_config.strategy.min_predicted_return = best_params.get("min_predicted_return", 0.015)
    best_config.strategy.consensus_level = best_params.get("consensus_level", "all")
    best_config.strategy.use_q10_filter = best_params.get("use_q10_filter", True)
    best_config.strategy.stop_loss_pct = best_params.get("stop_loss_pct", 0.02)
    best_config.strategy.take_profit_pct = best_params.get("take_profit_pct", 0.05)
    best_config.strategy.exit_on_down_signal = best_params.get("exit_on_down_signal", True)

    # Run full backtests with best parameters
    best_signals: dict[str, list[TradeSignal]] = {}
    for d in eval_dates:
        best_signals[d] = evaluate_signals_for_asof(
            asof_date=d,
            tickers=tickers,
            series_by_ticker=series_by_ticker,
            strategy_cfg=best_config.strategy,
            preloaded_details_map=details_map_by_date[d],
            preloaded_weights_map=weights_map,
            preloaded_close_map=close_map,
        )

    best_is_res = BacktestEngine(best_config, series_by_ticker).run(
        trade_dates=in_sample_dates,
        signals_by_date=best_signals,
    )
    best_oos_res = BacktestEngine(best_config, series_by_ticker).run(
        trade_dates=out_of_sample_dates,
        signals_by_date=best_signals,
    )

    if apply_best_to_config:
        save_trading_config(best_config, path=config_save_path or DEFAULT_TRADING_CONFIG_PATH)
        print(f"[backtest][optimize] Saved best strategy parameters to {config_save_path or DEFAULT_TRADING_CONFIG_PATH}")

    return best_config, valid_records, best_is_res, best_oos_res
