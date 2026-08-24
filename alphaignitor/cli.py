"""AlphaIgnitor CLI オーケストレーター。

run-daily コマンド:
  ダウンロード → zero-shot ensemble予測 → レポート

optimize-strategy コマンド:
  過去データを用いた Optuna Walk-Forward 売買戦略最適化 (moomoo手動運用向け)

backtest コマンド:
  指定ルールでのバックテスト実行および詳細HTMLレポート出力

action-sheet コマンド:
  最新予測に基づく今夜の moomoo 発注指示シート表示
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import os
import sqlite3

from alphaignitor.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_pipeline_config
from alphaignitor.logging_utils import EventLogger, make_run_id
from alphaignitor.stages import build_stage, forecast_stage, report_stage
from alphaignitor.common._credentials import load_credentials_into_environ
from alphaignitor.backtest import (
    DEFAULT_TRADING_CONFIG_PATH,
    TradingConfig,
    load_trading_config,
    save_trading_config,
    BacktestEngine,
    run_walk_forward_optimization,
    evaluate_signals_for_asof,
    generate_action_sheet,
    format_action_sheet_text,
    generate_backtest_html_report,
)
from alphaignitor.pipeline.zero_shot_ensemble.market_data import (
    available_trade_dates,
    latest_asof_date,
    load_price_panel,
    resolve_tickers,
    ticker_series_map,
)
from alphaignitor.pipeline.zero_shot_ensemble.schema import ENSEMBLE_MODELS
from alphaignitor.pipeline.zero_shot_ensemble.storage import (
    load_all_prediction_details_for_asof,
    load_best_weights,
)

# secrets/credentials.env の環境変数 (HF_TOKEN 等) をプロセスに反映する
load_credentials_into_environ()
# Windows環境でのHugging Face Hubのシンボリックリンク警告を無効化
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaIgnitor 日次パイプライン & バックテスト オーケストレーター")
    sub = p.add_subparsers(dest="command", required=True)

    # ── run-daily ────────────────────────────────────────────────
    run = sub.add_parser(
        "run-daily",
        help="ダウンロード→zero-shot ensemble予測→レポートを順に実行する",
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML 設定ファイルのパス")
    run.add_argument("--trading-config", type=Path, default=DEFAULT_TRADING_CONFIG_PATH, help="売買設定ファイルのパス")
    run.add_argument("--run-date", type=str, default=None, help="実行日を上書きする (YYYY-MM-DD)")
    run.add_argument(
        "--skip-download", action="store_true",
        help="ダウンロードステージをスキップする (ローカルデータを使用)",
    )
    run.add_argument(
        "--skip-build", action="store_true",
        help="legacy パネルビルドをスキップする",
    )
    run.add_argument(
        "--build-panel",
        action="store_true",
        help="legacy の日次特徴量パネルも生成する。zero-shot ensemble では通常不要。",
    )
    run.add_argument(
        "--skip-forecast", action="store_true",
        help="予測ステージをスキップする (predict/<date>_us_stock_ensemble_forecast.parquet が存在する場合)",
    )
    run.add_argument("--skip-report", action="store_true", help="HTML レポート生成をスキップする")
    run.add_argument("--optuna-trials", type=int, default=None, help="Optuna trial 数を上書きする")
    run.add_argument("--optuna-timeout-minutes", type=int, default=None, help="Optuna 1銘柄あたりの時間上限を上書きする")

    # ── optimize-strategy ─────────────────────────────────────────
    opt = sub.add_parser(
        "optimize-strategy",
        help="過去半年間のデータから moomoo 手動運用に最適な売買ルールを探索する",
    )
    opt.add_argument("--months", type=int, default=6, help="バックテスト探索期間 (月数, default: 6)")
    opt.add_argument("--trials", type=int, default=100, help="Optuna trial 数 (default: 100)")
    opt.add_argument("--timeout-seconds", type=int, default=180, help="最適化タイムアウト秒数 (default: 180)")
    opt.add_argument("--trading-config", type=Path, default=DEFAULT_TRADING_CONFIG_PATH, help="売買設定ファイル")
    opt.add_argument("--day-aggs-dir", type=Path, default=Path("aggs/us_stock_day"))
    opt.add_argument("--prediction-cache-path", type=Path, default=Path("cache/zero_shot_predictions.sqlite3"))
    opt.add_argument("--tickers-csv", type=Path, default=Path("us_stock_list.csv"))
    opt.add_argument("--outdir", type=Path, default=Path("report/backtest"))
    opt.add_argument("--no-apply", action="store_true", help="最適化結果を設定ファイルに保存しない")

    # ── backtest ──────────────────────────────────────────────────
    bt = sub.add_parser(
        "backtest",
        help="設定ファイルに基いて詳細バックテストを実行し、HTMLレポートを出力する",
    )
    bt.add_argument("--months", type=int, default=6, help="バックテスト期間 (月数, default: 6)")
    bt.add_argument("--trading-config", type=Path, default=DEFAULT_TRADING_CONFIG_PATH, help="売買設定ファイル")
    bt.add_argument("--day-aggs-dir", type=Path, default=Path("aggs/us_stock_day"))
    bt.add_argument("--prediction-cache-path", type=Path, default=Path("cache/zero_shot_predictions.sqlite3"))
    bt.add_argument("--tickers-csv", type=Path, default=Path("us_stock_list.csv"))
    bt.add_argument("--outdir", type=Path, default=Path("report/backtest"))

    # ── action-sheet ──────────────────────────────────────────────
    act = sub.add_parser(
        "action-sheet",
        help="最新予測データに基づき今夜の moomoo 発注シートを表示する",
    )
    act.add_argument("--asof-date", type=str, default=None, help="As-of 日付 (YYYY-MM-DD, default: 最新日)")
    act.add_argument("--trading-config", type=Path, default=DEFAULT_TRADING_CONFIG_PATH, help="売買設定ファイル")
    act.add_argument("--day-aggs-dir", type=Path, default=Path("aggs/us_stock_day"))
    act.add_argument("--prediction-cache-path", type=Path, default=Path("cache/zero_shot_predictions.sqlite3"))
    act.add_argument("--tickers-csv", type=Path, default=Path("us_stock_list.csv"))
    act.add_argument("--record-buys", action="store_true", help="提案された新規買い発注を保有中ポジションとして記録する")
    act.add_argument("--record-exit", type=str, default=None, help="指定銘柄の保有ポジションを手仕舞い記録する (例: --record-exit AAPL)")
    act.add_argument("--list-positions", action="store_true", help="現在保有中のポジション一覧を表示する")
    act.add_argument("--clear-positions", action="store_true", help="保有中ポジションをすべてクリアする")

    return p


def run_daily(
    *,
    config_path: Path,
    trading_config_path: Path = DEFAULT_TRADING_CONFIG_PATH,
    run_date: str | None,
    skip_download: bool,
    skip_build: bool,
    skip_forecast: bool = False,
    skip_report: bool = False,
    build_panel: bool = False,
    optuna_trials: int | None = None,
    optuna_timeout_minutes: int | None = None,
    root: Path,
) -> int:
    cfg = load_pipeline_config(config_path)
    trading_cfg = load_trading_config(trading_config_path)

    if run_date:
        cfg.run_date = run_date
    if optuna_trials is not None:
        cfg.optuna_n_trials = int(optuna_trials)
    if optuna_timeout_minutes is not None:
        cfg.optuna_timeout_minutes = int(optuna_timeout_minutes)

    if cfg.start_date is None and cfg.end_date is None:
        end_ymd = cfg.run_date or str(date.today())
        try:
            end_d = date.fromisoformat(end_ymd)
        except Exception:
            end_d = date.today()
            end_ymd = str(end_d)
        start_d = end_d - timedelta(days=1300)
        cfg.start_date = str(start_d)
        cfg.end_date = end_ymd

    if cfg.run_date is None and cfg.end_date is not None:
        cfg.run_date = cfg.end_date

    run_id = make_run_id("daily")
    with EventLogger(run_id=run_id, log_dir=root / "log") as logger:
        logger.emit(
            level="INFO",
            stage="pipeline",
            event="start",
            msg="日次パイプライン開始",
            kv={"config": str(config_path), "log_file": str(logger.log_file)},
        )

        # ── Stage 1: Download (+ optional legacy panel build) ────
        build_stage.run(
            cfg,
            logger=logger,
            root=root,
            skip_download=skip_download,
            skip_build=skip_build or not build_panel,
        )

        # ── Stage 2: Forecast / Optuna ensemble ──────────────────
        if skip_forecast:
            logger.emit(level="INFO", stage="forecast", event="skipped", msg="予測をスキップ")
        else:
            forecast_stage.run(cfg, logger=logger, root=root)

        # ── Stage 3: Report ──────────────────────────────────────
        if skip_report:
            logger.emit(level="INFO", stage="report", event="skipped", msg="レポートをスキップ")
        else:
            report_stage.run(cfg, logger=logger, root=root)

        # ── Stage 4: Generate & Print moomoo Action Sheet ────────
        try:
            cache_conn = sqlite3.connect(root / cfg.prediction_cache_path)
            asof = latest_asof_date(root / "aggs/us_stock_day", run_date=cfg.run_date).isoformat()
            resolved_tickers = resolve_tickers(root=root, tickers_csv=root / "us_stock_list.csv")
            available_dates = [d.isoformat() for d in available_trade_dates(root / "aggs/us_stock_day")]
            price_panel = load_price_panel(root / "aggs/us_stock_day", dates=[date.fromisoformat(d) for d in available_dates[-60:]], tickers=resolved_tickers)
            series_by_ticker = ticker_series_map(price_panel)

            signals = evaluate_signals_for_asof(
                conn=cache_conn,
                asof_date=asof,
                tickers=resolved_tickers,
                series_by_ticker=series_by_ticker,
                strategy_cfg=trading_cfg.strategy,
            )
            cache_conn.close()

            action_sheet = generate_action_sheet(
                asof_date=asof,
                signals=signals,
                trading_config=trading_cfg,
            )
            print("\n" + format_action_sheet_text(action_sheet) + "\n")
        except Exception as e:
            logger.emit(level="WARN", stage="action_sheet", event="error", msg=f"Action sheet generation failed: {e}")

        logger.emit(level="INFO", stage="pipeline", event="done", msg="日次パイプライン完了")
    return 0


def run_optimize_strategy_cli(
    *,
    months: int,
    trials: int,
    timeout_seconds: int,
    trading_config_path: Path,
    day_aggs_dir: Path,
    prediction_cache_path: Path,
    tickers_csv: Path,
    outdir: Path,
    apply_best: bool,
    root: Path,
) -> int:
    trading_cfg = load_trading_config(root / trading_config_path)
    cache_conn = sqlite3.connect(root / prediction_cache_path)
    resolved_tickers = resolve_tickers(root=root, tickers_csv=root / tickers_csv)
    all_dates_dt = available_trade_dates(root / day_aggs_dir)
    all_dates = [d.isoformat() for d in all_dates_dt]

    total_days_needed = int(months * 21)
    needed_dates_dt = all_dates_dt[-(total_days_needed + 10):] if len(all_dates_dt) > (total_days_needed + 10) else all_dates_dt

    print(f"[backtest] Loading price history for {len(resolved_tickers)} tickers ({len(needed_dates_dt)} dates)...")
    price_panel = load_price_panel(root / day_aggs_dir, dates=needed_dates_dt, tickers=resolved_tickers)
    series_by_ticker = ticker_series_map(price_panel)
    print(f"[backtest] Loaded {len(price_panel):,} price rows.")

    best_cfg, top_strategies, res_is, res_oos = run_walk_forward_optimization(
        conn=cache_conn,
        series_by_ticker=series_by_ticker,
        tickers=resolved_tickers,
        all_trade_dates=all_dates,
        months=months,
        n_trials=trials,
        timeout_seconds=timeout_seconds,
        base_config=trading_cfg,
        apply_best_to_config=apply_best,
        config_save_path=root / trading_config_path,
    )

    # Run full period backtest with best config
    engine = BacktestEngine(best_cfg, series_by_ticker)
    total_days_needed = int(months * 21)
    eval_dates = all_dates[-total_days_needed:] if len(all_dates) > total_days_needed else all_dates[:]

    best_signals = {
        d: evaluate_signals_for_asof(
            conn=cache_conn,
            asof_date=d,
            tickers=resolved_tickers,
            series_by_ticker=series_by_ticker,
            strategy_cfg=best_cfg.strategy,
        )
        for d in eval_dates
    }
    full_result = engine.run(trade_dates=eval_dates, signals_by_date=best_signals)

    # Generate Action sheet for latest asof
    latest_asof = eval_dates[-1]
    action_sheet = generate_action_sheet(
        asof_date=latest_asof,
        signals=best_signals.get(latest_asof, []),
        trading_config=best_cfg,
    )
    cache_conn.close()

    # Generate HTML report
    html_path = generate_backtest_html_report(
        result=full_result,
        trading_config=best_cfg,
        outdir=root / outdir,
        top_strategies=top_strategies,
        action_sheet=action_sheet,
    )

    print("\n" + "=" * 80)
    print("🏆 最適化完了サマリー (Best Walk-Forward Strategy)")
    print("=" * 80)
    print(f"・初期資金 (Capital)       : ${full_result.initial_capital:,.2f} USD")
    print(f"・最終資産 (Final Equity)  : ${full_result.final_equity:,.2f} USD (累積リターン: {full_result.total_return * 100:+.2f}%)")
    print(f"・勝率 (Win Rate)          : {full_result.win_rate * 100:.1f}% (PF: {full_result.profit_factor:.2f})")
    print(f"・シャープレシオ (Sharpe)  : {full_result.sharpe_ratio:.2f}")
    print(f"・最大ドローダウン (MaxDD) : -{full_result.max_drawdown * 100:.2f}%")
    print(f"・総トレード数 (Trades)    : {full_result.total_trades} 回 (平均保有: {full_result.avg_holding_days:.1f} 日)")
    print(f"・最適パラメータ           : 保有{best_cfg.strategy.holding_days}日 | SL -{float(best_cfg.strategy.stop_loss_pct or 0)*100:.1f}% | TP +{float(best_cfg.strategy.take_profit_pct or 0)*100:.1f}% | 合意 {best_cfg.strategy.consensus_level}")
    print(f"・詳細HTMLレポート         : {html_path}")
    print("\n" + format_action_sheet_text(action_sheet) + "\n")
    return 0


def run_backtest_cli(
    *,
    months: int,
    trading_config_path: Path,
    day_aggs_dir: Path,
    prediction_cache_path: Path,
    tickers_csv: Path,
    outdir: Path,
    root: Path,
) -> int:
    trading_cfg = load_trading_config(root / trading_config_path)
    cache_conn = sqlite3.connect(root / prediction_cache_path)
    resolved_tickers = resolve_tickers(root=root, tickers_csv=root / tickers_csv)
    all_dates_dt = available_trade_dates(root / day_aggs_dir)
    all_dates = [d.isoformat() for d in all_dates_dt]
    total_days_needed = int(months * 21)
    eval_dates = all_dates[-total_days_needed:] if len(all_dates) > total_days_needed else all_dates[:]
    needed_dates_dt = all_dates_dt[-(total_days_needed + 10):] if len(all_dates_dt) > (total_days_needed + 10) else all_dates_dt

    print(f"[backtest] Running {months}-month backtest on {len(eval_dates)} trading dates [{eval_dates[0]} ~ {eval_dates[-1]}]...")
    price_panel = load_price_panel(root / day_aggs_dir, dates=needed_dates_dt, tickers=resolved_tickers)
    series_by_ticker = ticker_series_map(price_panel)

    weights_map = {t: load_best_weights(cache_conn, ticker=t, models=ENSEMBLE_MODELS) for t in resolved_tickers}
    details_map_by_date = {d: load_all_prediction_details_for_asof(cache_conn, asof_trade_date=d) for d in eval_dates}
    close_map = {
        (t, str(row.trade_date)): float(row.close)
        for t, df in series_by_ticker.items()
        if not df.empty
        for row in df.itertuples(index=False)
    }

    signals_by_date = {
        d: evaluate_signals_for_asof(
            asof_date=d,
            tickers=resolved_tickers,
            series_by_ticker=series_by_ticker,
            strategy_cfg=trading_cfg.strategy,
            preloaded_details_map=details_map_by_date[d],
            preloaded_weights_map=weights_map,
            preloaded_close_map=close_map,
        )
        for d in eval_dates
    }

    engine = BacktestEngine(trading_cfg, series_by_ticker)
    result = engine.run(trade_dates=eval_dates, signals_by_date=signals_by_date)

    latest_asof = eval_dates[-1]
    action_sheet = generate_action_sheet(
        asof_date=latest_asof,
        signals=signals_by_date.get(latest_asof, []),
        trading_config=trading_cfg,
    )
    cache_conn.close()

    html_path = generate_backtest_html_report(
        result=result,
        trading_config=trading_cfg,
        outdir=root / outdir,
        action_sheet=action_sheet,
    )

    print("\n" + "=" * 80)
    print("📊 バックテスト結果サマリー")
    print("=" * 80)
    print(f"・初期資金 (Capital)       : ${result.initial_capital:,.2f} USD")
    print(f"・最終資産 (Final Equity)  : ${result.final_equity:,.2f} USD (累積リターン: {result.total_return * 100:+.2f}%)")
    print(f"・勝率 (Win Rate)          : {result.win_rate * 100:.1f}% (PF: {result.profit_factor:.2f})")
    print(f"・シャープレシオ (Sharpe)  : {result.sharpe_ratio:.2f}")
    print(f"・最大ドローダウン (MaxDD) : -{result.max_drawdown * 100:.2f}%")
    print(f"・総トレード数 (Trades)    : {result.total_trades} 回 (平均保有: {result.avg_holding_days:.1f} 日)")
    print(f"・詳細HTMLレポート         : {html_path}")
    print("\n" + format_action_sheet_text(action_sheet) + "\n")
    return 0


def run_action_sheet_cli(
    *,
    asof_date: str | None,
    trading_config_path: Path,
    day_aggs_dir: Path,
    prediction_cache_path: Path,
    tickers_csv: Path,
    record_buys: bool = False,
    record_exit_ticker: str | None = None,
    list_positions: bool = False,
    clear_positions: bool = False,
    root: Path,
) -> int:
    from alphaignitor.backtest import (
        load_active_positions,
        record_position_entry,
        record_position_exit,
        save_active_positions,
        update_positions_holding_and_prices,
    )

    if clear_positions:
        save_active_positions([], root / "cache/active_positions.json")
        print("✅ 保有中ポジションをすべてクリアしました。")
        return 0

    if record_exit_ticker:
        exited = record_position_exit(ticker=record_exit_ticker, path=root / "cache/active_positions.json")
        if exited:
            print(f"✅ {record_exit_ticker.upper()} の手仕舞いを記録しました。")
        else:
            print(f"⚠️ {record_exit_ticker.upper()} は保有中ポジションに見当たりませんでした。")
        return 0

    if list_positions:
        positions = load_active_positions(root / "cache/active_positions.json")
        if not positions:
            print("📦 現在保有中のポジションはありません。")
        else:
            print(f"📦 現在保有中ポジション ({len(positions)} 件):")
            for p in positions:
                tp_str = f"${p.get('tp_price')}" if p.get("tp_price") else "None"
                sl_str = f"${p.get('sl_price')}" if p.get("sl_price") else "None"
                print(f"  ・{p.get('ticker')}: {p.get('shares')}株 @ ${p.get('entry_price'):.2f} (Entry: {p.get('entry_asof_date')}, 保有: {p.get('holding_days')}日, TP: {tp_str}, SL: {sl_str})")
        return 0

    trading_cfg = load_trading_config(root / trading_config_path)
    cache_conn = sqlite3.connect(root / prediction_cache_path)
    resolved_tickers = resolve_tickers(root=root, tickers_csv=root / tickers_csv)
    target_asof = asof_date or latest_asof_date(root / day_aggs_dir, run_date=None).isoformat()
    all_dates_dt = available_trade_dates(root / day_aggs_dir)

    price_panel = load_price_panel(root / day_aggs_dir, dates=all_dates_dt[-60:], tickers=resolved_tickers)
    series_by_ticker = ticker_series_map(price_panel)

    # Update active positions holding days & current prices
    current_prices = {
        ticker: float(df.iloc[-1]["close"])
        for ticker, df in series_by_ticker.items()
        if not df.empty
    }
    active_positions = update_positions_holding_and_prices(
        current_prices=current_prices,
        path=root / "cache/active_positions.json",
    )

    signals = evaluate_signals_for_asof(
        conn=cache_conn,
        asof_date=target_asof,
        tickers=resolved_tickers,
        series_by_ticker=series_by_ticker,
        strategy_cfg=trading_cfg.strategy,
    )
    cache_conn.close()

    sheet = generate_action_sheet(
        asof_date=target_asof,
        signals=signals,
        trading_config=trading_cfg,
        active_positions=active_positions,
    )
    print("\n" + format_action_sheet_text(sheet) + "\n")

    if record_buys and sheet.get("buys"):
        for b in sheet["buys"]:
            record_position_entry(
                ticker=b["ticker"],
                shares=b["shares"],
                entry_price=b["asof_close"],
                asof_date=target_asof,
                target_horizon=b.get("target_horizon_days", 3),
                sl_price=b.get("stop_loss_price"),
                tp_price=b.get("take_profit_price"),
                path=root / "cache/active_positions.json",
            )
        print(f"✅ {len(sheet['buys'])} 件の新規買い発注を保有中ポジションとして記録しました。")

    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "run-daily":
        return run_daily(
            config_path=args.config,
            trading_config_path=args.trading_config,
            run_date=args.run_date,
            skip_download=bool(args.skip_download),
            skip_build=bool(args.skip_build),
            skip_forecast=bool(args.skip_forecast),
            skip_report=bool(args.skip_report),
            build_panel=bool(args.build_panel),
            optuna_trials=args.optuna_trials,
            optuna_timeout_minutes=args.optuna_timeout_minutes,
            root=PROJECT_ROOT,
        )

    if args.command == "optimize-strategy":
        return run_optimize_strategy_cli(
            months=int(args.months),
            trials=int(args.trials),
            timeout_seconds=int(args.timeout_seconds),
            trading_config_path=args.trading_config,
            day_aggs_dir=args.day_aggs_dir,
            prediction_cache_path=args.prediction_cache_path,
            tickers_csv=args.tickers_csv,
            outdir=args.outdir,
            apply_best=not bool(args.no_apply),
            root=PROJECT_ROOT,
        )

    if args.command == "backtest":
        return run_backtest_cli(
            months=int(args.months),
            trading_config_path=args.trading_config,
            day_aggs_dir=args.day_aggs_dir,
            prediction_cache_path=args.prediction_cache_path,
            tickers_csv=args.tickers_csv,
            outdir=args.outdir,
            root=PROJECT_ROOT,
        )

    if args.command == "action-sheet":
        return run_action_sheet_cli(
            asof_date=args.asof_date,
            trading_config_path=args.trading_config,
            day_aggs_dir=args.day_aggs_dir,
            prediction_cache_path=args.prediction_cache_path,
            tickers_csv=args.tickers_csv,
            record_buys=bool(getattr(args, "record_buys", False)),
            record_exit_ticker=getattr(args, "record_exit", None),
            list_positions=bool(getattr(args, "list_positions", False)),
            clear_positions=bool(getattr(args, "clear_positions", False)),
            root=PROJECT_ROOT,
        )

    raise RuntimeError(f"未対応コマンド: {args.command}")
