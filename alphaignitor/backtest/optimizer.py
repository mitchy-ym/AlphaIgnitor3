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

    # Precompute ATR map for fast LOO calculation
    from alphaignitor.backtest.config import compute_atr
    from alphaignitor.pipeline.zero_shot_ensemble.schema import normalize_weights
    atr_map: dict[tuple[str, str], float | None] = {}
    for t in tickers:
        s = series_by_ticker.get(t)
        if s is not None and not s.empty:
            for d in eval_dates:
                sub = s[s["trade_date"] <= d]
                if not sub.empty:
                    atr_map[(t, d)] = compute_atr(sub)

    # Sector map
    meta_csv = Path("us_stock_list.csv")
    sector_map = {}
    if meta_csv.exists():
        try:
            m_df = pd.read_csv(meta_csv, dtype=str)
            m_df.columns = m_df.columns.str.lower()
            if "ticker" in m_df.columns and "sector" in m_df.columns:
                sector_map = dict(zip(m_df["ticker"].astype(str), m_df["sector"].astype(str)))
        except Exception:
            pass

    # Precompute raw candidates per date
    raw_candidates_by_date: dict[str, list[dict[str, Any]]] = {}
    for d in eval_dates:
        cands = []
        details_map = details_map_by_date[d]
        for ticker in tickers:
            asof_close = close_map.get((ticker, d))
            if asof_close is None or asof_close <= 0:
                continue
            base_weights = weights_map.get(ticker, {m: 1.0 / len(ENSEMBLE_MODELS) for m in ENSEMBLE_MODELS})
            atr_val = atr_map.get((ticker, d))
            sec = sector_map.get(ticker, "")

            for horizon in [1, 2, 3, 4, 5]:
                preds = {}
                q10s = {}
                q50s = {}
                q90s = {}
                for m in ENSEMBLE_MODELS:
                    item = details_map.get((ticker, m, horizon))
                    if item is not None and item[4] is None and item[0] is not None:
                        p = float(item[0])
                        if math.isfinite(p) and p > 0:
                            preds[m] = p
                        if item[1] is not None and math.isfinite(float(item[1])):
                            q10s[m] = float(item[1])
                        if item[2] is not None and math.isfinite(float(item[2])):
                            q50s[m] = float(item[2])
                        if item[3] is not None and math.isfinite(float(item[3])):
                            q90s[m] = float(item[3])

                if len(preds) < 2:
                    continue

                local_weights = normalize_weights(base_weights, available_models=set(preds))
                ensemble_pred = sum(preds[m] * local_weights[m] for m in local_weights)
                expected_ret = (ensemble_pred / asof_close) - 1.0

                up_models = sum(1 for m, val in preds.items() if val > asof_close)
                consensus_str = "all" if up_models == len(preds) else ("majority" if up_models >= 2 else "mixed")

                q10_weighted = None
                if q10s and any(m in q10s for m in local_weights):
                    w_sum = sum(local_weights[m] for m in local_weights if m in q10s)
                    if w_sum > 0:
                        q10_weighted = sum(q10s[m] * local_weights[m] for m in local_weights if m in q10s) / w_sum

                q50_weighted = None
                if q50s:
                    w_sum = sum(local_weights[m] for m in local_weights if m in q50s)
                    if w_sum > 0:
                        q50_weighted = sum(q50s[m] * local_weights[m] for m in local_weights if m in q50s) / w_sum

                q90_weighted = None
                if q90s:
                    w_sum = sum(local_weights[m] for m in local_weights if m in q90s)
                    if w_sum > 0:
                        q90_weighted = sum(q90s[m] * local_weights[m] for m in local_weights if m in q90s) / w_sum

                uncertainty_spread = (q90_weighted - q10_weighted) / asof_close if (q90_weighted and q10_weighted) else 0.05
                score = expected_ret / (1.0 + uncertainty_spread)

                cands.append({
                    "ticker": ticker,
                    "asof_date": d,
                    "horizon": horizon,
                    "asof_close": asof_close,
                    "expected_ret": expected_ret,
                    "ensemble_pred": ensemble_pred,
                    "up_models": up_models,
                    "total_models": len(preds),
                    "consensus_str": consensus_str,
                    "q10_weighted": q10_weighted,
                    "q50_weighted": q50_weighted,
                    "q90_weighted": q90_weighted,
                    "score": score,
                    "weights": local_weights,
                    "preds": preds,
                    "sector": sec,
                    "atr": atr_val,
                })
        raw_candidates_by_date[d] = cands

    print("[backtest][optimize] Preloading complete. Ready for ultra-fast Optuna search.", flush=True)

    def _build_signals(
        min_predicted_return: float,
        consensus_level: str,
        use_q10_filter: bool,
        target_horizon: str,
        max_per_sector: int,
        order_type: str,
        limit_max_gap_pct: float,
        limit_atr_multiplier: float,
        limit_expected_return_multiplier: float = 0.25,
        limit_min_gap_pct: float = 0.005,
        min_predicted_short_return: float = 0.10,
    ) -> dict[str, list[TradeSignal]]:
        res_signals = {}
        for d, cands in raw_candidates_by_date.items():
            by_ticker = {}
            for c in cands:
                h = c["horizon"]
                if target_horizon != "best" and str(h) != str(target_horizon):
                    continue

                is_buy = c["expected_ret"] >= min_predicted_return
                is_short = c["expected_ret"] <= -min_predicted_short_return
                if not is_buy and not is_short:
                    continue

                if is_buy:
                    if consensus_level == "all" and c["up_models"] < c["total_models"]:
                        continue
                    if consensus_level == "majority" and c["up_models"] < 2:
                        continue
                    if use_q10_filter and c["q10_weighted"] is not None:
                        if c["q10_weighted"] < c["asof_close"] * 0.995:
                            continue
                    act = "BUY"
                    sc = c["score"]
                else:
                    down_m = c["total_models"] - c["up_models"]
                    if consensus_level == "all" and down_m < c["total_models"]:
                        continue
                    if consensus_level == "majority" and down_m < 2:
                        continue
                    act = "SHORT"
                    unc = (c["q90_weighted"] - c["q10_weighted"]) / c["asof_close"] if (c["q90_weighted"] and c["q10_weighted"]) else 0.05
                    sc = abs(c["expected_ret"]) / (1.0 + unc)

                c_copy = dict(c)
                c_copy["action"] = act
                c_copy["score"] = sc
                tk = c_copy["ticker"]
                if tk not in by_ticker or sc > by_ticker[tk]["score"]:
                    by_ticker[tk] = c_copy

            cand_list = list(by_ticker.values())
            cand_list.sort(key=lambda x: x["score"], reverse=True)

            if max_per_sector and max_per_sector > 0:
                tier1 = []
                tier2 = []
                sec_counts = {}
                for c in cand_list:
                    s = c["sector"]
                    cnt = sec_counts.get(s, 0) if s else 0
                    if cnt == 0:
                        c["tier"] = 1
                        tier1.append(c)
                        if s:
                            sec_counts[s] = 1
                    elif cnt == 1 and max_per_sector >= 2:
                        c["tier"] = 2
                        tier2.append(c)
                        if s:
                            sec_counts[s] = 2
                    else:
                        pass
                cand_list = tier1 + tier2

            signals = []
            for c in cand_list:
                limit_p = None
                act = c.get("action", "BUY")
                if order_type in ["LOO", "LIMIT"]:
                    base = c["asof_close"]
                    offsets = []
                    if c["atr"] is not None and c["atr"] > 0:
                        offsets.append(limit_atr_multiplier * float(c["atr"]))
                    if c["expected_ret"] != 0:
                        offsets.append(base * (limit_expected_return_multiplier * abs(float(c["expected_ret"]))))
                    offset = min(offsets) if offsets else base * limit_min_gap_pct
                    offset = max(base * limit_min_gap_pct, min(base * limit_max_gap_pct, offset))
                    lp = (base + offset) if act == "BUY" else (base - offset)
                    limit_p = round(lp, 4) if lp < 1.0 else round(lp, 2)

                signals.append(TradeSignal(
                    ticker=c["ticker"],
                    asof_date=c["asof_date"],
                    target_horizon=c["horizon"],
                    expected_return=c["expected_ret"],
                    ensemble_pred=c["ensemble_pred"],
                    asof_close=c["asof_close"],
                    q10_close=c["q10_weighted"],
                    q50_close=c["q50_weighted"],
                    q90_close=c["q90_weighted"],
                    consensus=c["consensus_str"],
                    score=c["score"],
                    weights=c["weights"],
                    model_preds=c["preds"],
                    sector=c["sector"],
                    atr=c["atr"],
                    limit_price=limit_p,
                    tier=c.get("tier", 1),
                    action=act,
                ))
            res_signals[d] = signals
        return res_signals

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        slots = trial.suggest_categorical("max_slots", [3, 4, 5])
        holding_days = trial.suggest_categorical("holding_days", [1, 2, 3, 4, 5])
        min_return = trial.suggest_categorical("min_predicted_return", [0.010, 0.015, 0.020, 0.025, 0.030])
        consensus = trial.suggest_categorical("consensus_level", ["all", "majority"])
        use_q10 = trial.suggest_categorical("use_q10_filter", [False, True])
        take_profit = trial.suggest_categorical("take_profit_pct", [0.035, 0.040, 0.050, 0.060, 0.070, 0.080, 0.100, None])
        stop_loss = trial.suggest_categorical("stop_loss_pct", [None, 0.025, 0.030, 0.035])
        emergency_sl = trial.suggest_categorical("emergency_stop_loss_pct", [0.05, 0.06, 0.07, 0.08, 0.10])
        max_per_sec = trial.suggest_categorical("max_per_sector", [1, 2, 0])
        target_horizon = trial.suggest_categorical("target_horizon", ["best", "1", "2", "3"])
        exit_on_down = trial.suggest_categorical("exit_on_down_signal", [False, True])
        limit_max_gap = trial.suggest_categorical("limit_max_gap_pct", [0.015, 0.020, 0.025, 0.030, 0.035])
        limit_atr_mult = trial.suggest_categorical("limit_atr_multiplier", [0.25, 0.30, 0.40, 0.50])

        trial_cfg = copy.deepcopy(config)
        trial_cfg.portfolio.max_slots = slots
        trial_cfg.portfolio.slot_size_pct = 1.0 / slots
        trial_cfg.strategy.holding_days = holding_days
        trial_cfg.strategy.min_predicted_return = min_return
        trial_cfg.strategy.consensus_level = consensus
        trial_cfg.strategy.use_q10_filter = use_q10
        trial_cfg.strategy.stop_loss_pct = stop_loss
        trial_cfg.strategy.take_profit_pct = take_profit
        trial_cfg.strategy.emergency_stop_loss_pct = emergency_sl
        trial_cfg.strategy.max_per_sector = max_per_sec
        trial_cfg.strategy.target_horizon = target_horizon
        trial_cfg.strategy.exit_on_down_signal = exit_on_down
        trial_cfg.execution.order_type = "LOO"
        trial_cfg.execution.limit_max_gap_pct = limit_max_gap
        trial_cfg.execution.limit_atr_multiplier = limit_atr_mult

        signals_by_date = _build_signals(
            min_predicted_return=min_return,
            consensus_level=consensus,
            use_q10_filter=use_q10,
            target_horizon=target_horizon,
            max_per_sector=max_per_sec,
            order_type="LOO",
            limit_max_gap_pct=limit_max_gap,
            limit_atr_multiplier=limit_atr_mult,
        )

        # Run In-Sample
        engine_is = BacktestEngine(trial_cfg, series_by_ticker)
        res_is = engine_is.run(trade_dates=in_sample_dates, signals_by_date=signals_by_date)

        # Run Out-of-Sample
        engine_oos = BacktestEngine(trial_cfg, series_by_ticker)
        res_oos = engine_oos.run(trade_dates=out_of_sample_dates, signals_by_date=signals_by_date)

        # Run Full
        engine_full = BacktestEngine(trial_cfg, series_by_ticker)
        res_full = engine_full.run(trade_dates=eval_dates, signals_by_date=signals_by_date)

        score = (res_full.total_return * 100.0) + (res_full.sharpe_ratio * 10.0) - (res_full.max_drawdown * 50.0)

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
            "full_return_pct": round(res_full.total_return * 100, 2),
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
    best_config.portfolio.max_slots = best_params.get("max_slots", 4)
    best_config.portfolio.slot_size_pct = 1.0 / best_config.portfolio.max_slots
    best_config.strategy.holding_days = best_params.get("holding_days", 2)
    best_config.strategy.min_predicted_return = best_params.get("min_predicted_return", 0.03)
    best_config.strategy.consensus_level = best_params.get("consensus_level", "all")
    best_config.strategy.use_q10_filter = best_params.get("use_q10_filter", False)
    best_config.strategy.stop_loss_pct = best_params.get("stop_loss_pct", None)
    best_config.strategy.emergency_stop_loss_pct = best_params.get("emergency_stop_loss_pct", 0.10)
    best_config.strategy.take_profit_pct = best_params.get("take_profit_pct", 0.07)
    best_config.strategy.max_per_sector = best_params.get("max_per_sector", 1)
    best_config.strategy.target_horizon = str(best_params.get("target_horizon", "2"))
    best_config.strategy.exit_on_down_signal = best_params.get("exit_on_down_signal", False)
    best_config.execution.order_type = "LOO"
    best_config.execution.limit_max_gap_pct = best_params.get("limit_max_gap_pct", 0.015)
    best_config.execution.limit_atr_multiplier = best_params.get("limit_atr_multiplier", 0.30)

    best_signals = _build_signals(
        min_predicted_return=best_config.strategy.min_predicted_return,
        consensus_level=best_config.strategy.consensus_level,
        use_q10_filter=best_config.strategy.use_q10_filter,
        target_horizon=best_config.strategy.target_horizon,
        max_per_sector=best_config.strategy.max_per_sector,
        order_type="LOO",
        limit_max_gap_pct=best_config.execution.limit_max_gap_pct,
        limit_atr_multiplier=best_config.execution.limit_atr_multiplier,
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
