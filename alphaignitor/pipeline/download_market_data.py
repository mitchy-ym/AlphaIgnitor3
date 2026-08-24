from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from alphaignitor.common.massive_rest import download_daily_parquet


def _parse_iso_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(s).strip())
    except Exception as e:
        raise ValueError(f"invalid ISO date: {s}") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Massive REST grouped endpoint から日次OHLCVを取得します。\n"
            "- minute_aggs はサポートしません（本プロジェクトは日足専用）\n"
            "- ファイルレイアウト: aggs/us_stock_day/YYYY-MM-DD.parquet\n"
            "- マルチスレッド並列ダウンロード (MASSIVE_REST_WORKERS または --workers で指定が可能)"
        )
    )
    p.add_argument("--start-date", type=str, required=True, help="開始日 (YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, required=True, help="終了日 (YYYY-MM-DD)")
    p.add_argument(
        "--download-dir",
        type=Path,
        default=Path("aggs/us_stock_day"),
        help="ダウンロード先（デフォルト: aggs/us_stock_day）",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="既存ファイルがあっても再DLします。",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="並列ダウンロードのスレッド数 (デフォルト: 環境変数 MASSIVE_REST_WORKERS または 8)。",
    )
    return p.parse_args()


def run_download(
    *,
    start_date: dt.date | str,
    end_date: dt.date | str,
    download_dir: Path,
    overwrite: bool = False,
    workers: int | None = None,
) -> dict[str, int]:
    s_date = _parse_iso_date(str(start_date)) if not isinstance(start_date, dt.date) else start_date
    e_date = _parse_iso_date(str(end_date)) if not isinstance(end_date, dt.date) else end_date
    if e_date < s_date:
        raise ValueError("end-date must be >= start-date")

    return download_daily_parquet(
        start_date=s_date,
        end_date=e_date,
        download_dir=Path(download_dir),
        overwrite=overwrite,
        max_workers=workers,
    )


def main() -> int:
    args = parse_args()
    stats = run_download(
        start_date=args.start_date,
        end_date=args.end_date,
        download_dir=args.download_dir,
        overwrite=bool(args.overwrite),
        workers=args.workers,
    )
    print("[DAY_AGGS] download done:", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
