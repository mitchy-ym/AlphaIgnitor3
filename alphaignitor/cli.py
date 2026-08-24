"""AlphaIgnitor CLI オーケストレーター。

run-daily コマンドで以下のステージを順に実行する:
  download → zero-shot ensemble forecast → report

各ステージは --skip-* フラグで個別にスキップできる。
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import os

from alphaignitor.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_pipeline_config
from alphaignitor.logging_utils import EventLogger, make_run_id
from alphaignitor.stages import build_stage, forecast_stage, report_stage
from alphaignitor.common._credentials import load_credentials_into_environ

# secrets/credentials.env の環境変数 (HF_TOKEN 等) をプロセスに反映する
load_credentials_into_environ()
# Windows環境でのHugging Face Hubのシンボリックリンク警告を無効化
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaIgnitor 日次パイプライン オーケストレーター")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run-daily",
        help="ダウンロード→zero-shot ensemble予測→レポートを順に実行する",
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML 設定ファイルのパス")
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

    return p


def run_daily(
    *,
    config_path: Path,
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
    if run_date:
        cfg.run_date = run_date
    if optuna_trials is not None:
        cfg.optuna_n_trials = int(optuna_trials)
    if optuna_timeout_minutes is not None:
        cfg.optuna_timeout_minutes = int(optuna_timeout_minutes)

    # zero-shot モデルの context と Optuna の walk-forward 窓を確保するため、
    # 未指定時は run_date (または today) から十分な履歴を遡る。
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

    # run_date が未設定の場合は end_date とそろえる。
    # これにより出力パス (models/forecast/report) が決定論的になる。
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

        logger.emit(level="INFO", stage="pipeline", event="done", msg="日次パイプライン完了")
    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "run-daily":
        return run_daily(
            config_path=args.config,
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

    raise RuntimeError(f"未対応コマンド: {args.command}")
