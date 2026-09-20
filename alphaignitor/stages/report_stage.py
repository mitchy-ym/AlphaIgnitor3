from __future__ import annotations

import os
from pathlib import Path

from alphaignitor.config import PipelineConfig
from alphaignitor.logging_utils import EventLogger
from alphaignitor.pipeline.zero_shot_ensemble.report import run_report


def _resolve_forecast_path(root: Path, cfg: PipelineConfig) -> Path:
    # Prefer an explicit run_date forecast if present.
    for d in [cfg.run_date, cfg.end_date]:
        if d:
            candidate = root / "predict" / f"{d}_us_stock_ensemble_forecast.parquet"
            if candidate.exists():
                return candidate

    files = list((root / "predict").glob("*_ensemble_forecast.parquet"))
    if not files:
        raise FileNotFoundError("predict/ に ensemble 予測 parquet が見当たりません")

    # Avoid lexicographic traps (e.g., 'smoke_' sorting after dates). Choose most recent by mtime.
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def run(cfg: PipelineConfig, *, logger: EventLogger, root: Path) -> Path:
    forecast = _resolve_forecast_path(root, cfg)
    logger.emit(
        level="INFO",
        stage="report",
        event="start",
        msg="HTML レポート生成開始",
        kv={"forecast": str(forecast), "outdir": cfg.report_outdir},
    )

    action_sheet = None
    backtest_kpi = None

    # Load backtest KPI summary if available
    try:
        from alphaignitor.backtest import load_backtest_kpi
        backtest_kpi = load_backtest_kpi(report_dir=root / cfg.report_outdir)
        if backtest_kpi:
            logger.emit(
                level="INFO",
                stage="report",
                event="backtest_kpi_loaded",
                msg=f"バックテストKPI取得完了: 勝率={backtest_kpi.get('win_rate_str')}, Sharpe={backtest_kpi.get('sharpe_ratio_str')}",
            )
    except Exception as be:
        logger.emit(level="DEBUG", stage="report", event="kpi_load_skipped", msg=str(be))

    # Evaluate trading signals & action sheet if trading config exists
    try:
        trading_cfg_path = root / "config/trading.yaml"
        if trading_cfg_path.exists():
            import datetime as dt
            import sqlite3
            import pandas as pd
            from alphaignitor.backtest import (
                evaluate_signals_for_asof,
                generate_action_sheet,
                load_trading_config,
            )
            from alphaignitor.pipeline.zero_shot_ensemble.market_data import (
                available_trade_dates,
                load_price_panel,
                ticker_series_map,
            )

            trading_cfg = load_trading_config(trading_cfg_path)
            cache_file = root / getattr(cfg, "prediction_cache_path", "cache/zero_shot_predictions.sqlite3")
            cache_conn = sqlite3.connect(cache_file) if cache_file.exists() else None

            from alphaignitor.pipeline.zero_shot_ensemble.market_data import (
                available_trade_dates,
                latest_asof_date,
                load_price_panel,
                resolve_tickers,
                ticker_series_map,
            )

            resolved_tickers = resolve_tickers(root=root, tickers_csv=root / "us_stock_list.csv")
            day_aggs = root / "aggs/us_stock_day"
            asof = latest_asof_date(day_aggs, run_date=cfg.run_date).isoformat()

            if day_aggs.exists() and resolved_tickers:
                avail_dates = [d.isoformat() for d in available_trade_dates(day_aggs)]
                needed_dates = [dt.date.fromisoformat(d) for d in avail_dates[-60:]]
                price_panel = load_price_panel(day_aggs, dates=needed_dates, tickers=resolved_tickers)
                series_by_ticker = ticker_series_map(price_panel)
            else:
                series_by_ticker = {}
            signals = evaluate_signals_for_asof(
                conn=cache_conn,
                asof_date=asof,
                tickers=resolved_tickers,
                series_by_ticker=series_by_ticker,
                strategy_cfg=trading_cfg.strategy,
            )
            if cache_conn:
                cache_conn.close()

            action_sheet = generate_action_sheet(
                asof_date=asof,
                signals=signals,
                trading_config=trading_cfg,
                series_by_ticker=series_by_ticker,
            )
            logger.emit(
                level="INFO",
                stage="report",
                event="action_sheet_ready",
                msg=f"moomoo アクションシート生成完了: 買い推奨 {len(action_sheet.get('buys', []))}件 / 手仕舞い {len(action_sheet.get('exits', []))}件",
            )
    except Exception as ae:
        logger.emit(
            level="WARN",
            stage="report",
            event="action_sheet_eval_skipped",
            msg=f"アクションシート生成スキップ (通常レポート生成を継続): {ae}",
        )

    try:
        html_path = run_report(
            forecast_path=forecast,
            outdir=root / cfg.report_outdir,
            asof_date=cfg.run_date,
            ticker_meta_csv=root / "us_stock_list.csv",
            action_sheet=action_sheet,
            backtest_kpi=backtest_kpi,
        )
        logger.emit(
            level="INFO",
            stage="report",
            event="done",
            msg="HTML レポート生成完了",
            kv={"report_file": str(html_path)},
        )

        # ポータル index.html の自動更新
        try:
            from alphaignitor.pipeline.portal import generate_portal

            portal_path = generate_portal(report_dir=root / cfg.report_outdir)
            logger.emit(
                level="INFO",
                stage="report",
                event="portal_done",
                msg="ポータル index.html 生成完了",
                kv={"portal_file": str(portal_path)},
            )
        except Exception as pe:
            logger.emit(
                level="WARN",
                stage="report",
                event="portal_failed",
                msg=f"ポータル生成失敗 (スキップ): {pe}",
            )

        return html_path
    except Exception as e:
        logger.emit(level="ERROR", stage="report", event="failed", msg=str(e))
        raise
