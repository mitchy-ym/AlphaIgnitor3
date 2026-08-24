from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from tqdm import tqdm

import alphaignitor.common.trading_calendar as tc
from alphaignitor.common.day_store import has_partition, write_day_partition
from alphaignitor.common.massive_splits import get_api_key


_DEFAULT_GROUPED_URL = "https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/{date}"
_MISSING_RATE_THRESHOLD = 0.05


class DayDataQualityError(RuntimeError):
    pass


def _generate_date_range(start_date: date, end_date: date) -> Iterator[date]:
    cur = start_date
    while cur <= end_date:
        yield cur
        cur = cur + timedelta(days=1)


def _extract_ticker(rec: dict) -> str | None:
    for key in ("T", "ticker", "symbol", "t"):
        v = rec.get(key)
        if v:
            return str(v)
    return None


def _extract_num(rec: dict, *keys: str) -> float | None:
    for k in keys:
        if k in rec:
            try:
                return float(rec.get(k))
            except Exception:
                return None
    return None


def fetch_grouped_daily_aggs_rest(
    *,
    trade_date: date,
    api_key: str,
    adjusted: bool = True,
    timeout_sec: float = 60.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    endpoint_tpl = os.environ.get("MASSIVE_GROUPED_URL", _DEFAULT_GROUPED_URL)
    url = endpoint_tpl.format(date=trade_date.isoformat())

    params = {
        "adjusted": "true" if adjusted else "false",
        "apiKey": api_key,
    }

    session = requests.Session()
    backoff = 1.0
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_sec)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
                continue
            raise

        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results") or []
            rows: list[dict] = []
            total_records = len(results) if isinstance(results, list) else 0
            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    ticker = _extract_ticker(r)
                    if not ticker:
                        continue
                    o = _extract_num(r, "o", "open")
                    h = _extract_num(r, "h", "high")
                    l = _extract_num(r, "l", "low")
                    c = _extract_num(r, "c", "close")
                    v = _extract_num(r, "v", "volume")
                    if None in (o, h, l, c, v):
                        continue
                    n = _extract_num(r, "n", "transactions")
                    vw = _extract_num(r, "vw", "vwap")
                    rows.append(
                        {
                            "trade_date": trade_date.isoformat(),
                            "ticker": ticker,
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": int(v),
                            "transactions": int(n) if n is not None else 0,
                            "vwap": vw,
                        }
                    )

            if total_records > 0:
                missing_rate = 1.0 - (len(rows) / float(total_records))
                if missing_rate > _MISSING_RATE_THRESHOLD:
                    raise DayDataQualityError(
                        f"grouped day data quality check failed: trade_date={trade_date.isoformat()} "
                        f"valid={len(rows)} total={total_records} missing_rate={missing_rate:.4f} "
                        f"threshold={_MISSING_RATE_THRESHOLD:.4f}"
                    )

            return pd.DataFrame(rows)

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt < max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        sleep_sec = float(retry_after)
                    except Exception:
                        sleep_sec = backoff
                else:
                    sleep_sec = backoff
                time.sleep(max(0.1, sleep_sec))
                backoff = min(backoff * 2.0, 10.0)
                continue

        resp.raise_for_status()

    if last_err:
        raise last_err
    raise RuntimeError("unexpected REST fetch failure")


# ── 1日分フェッチ + 保存 (スレッドワーカー) ─────────────────────

_R_SKIPPED = "skipped"
_R_SUCCESS = "success"
_R_NOT_FOUND = "not_found"


def _fetch_and_save(
    d: date,
    *,
    api_key: str,
    adjusted: bool,
    max_retries: int,
    download_dir: Path,
    overwrite: bool,
) -> str:
    """1日分のデータを取得・保存するスレッドワーカー。

    Returns:
        "skipped"   — 既存ファイルがありスキップ
        "success"   — 取得・保存成功
        "not_found" — その日のデータが空

    Raises:
        DayDataQualityError: 品質チェック失敗 (即座に全タスクを中断)
        Exception: その他のネットワークエラー等
    """
    if (not overwrite) and has_partition(download_dir, d):
        return _R_SKIPPED

    df = fetch_grouped_daily_aggs_rest(
        trade_date=d,
        api_key=api_key,
        adjusted=adjusted,
        max_retries=max_retries,
    )
    if df.empty:
        return _R_NOT_FOUND

    write_day_partition(df, day_root=download_dir, trade_date=d, overwrite=True, compression="snappy")
    return _R_SUCCESS


def download_daily_parquet(
    *,
    start_date: date,
    end_date: date,
    download_dir: Path,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> dict[str, int]:
    """REST grouped endpoint から日次OHLCVをマルチスレッドで取得し parquet に保存する。

    ファイルレイアウト: aggs/us_stock_day/YYYY-MM-DD.parquet

    Args:
        max_workers: 並列スレッド数。未指定時は環境変数 MASSIVE_REST_WORKERS
                     (デフォルト: 8)。
    """
    api_key = get_api_key()
    adjusted = True
    max_retries = int(os.environ.get("MASSIVE_REST_MAX_RETRIES", "3"))
    workers = max_workers or int(os.environ.get("MASSIVE_REST_WORKERS", "8"))

    date_list = list(_generate_date_range(start_date, end_date))
    trading_dates, weekend_dates, holiday_dates = tc.classify_us_stock_days(date_list)

    print(
        (
            "day_aggs_download_start: "
            f"start_date={start_date.isoformat()} end_date={end_date.isoformat()} "
            f"trading_days={len(trading_dates)} weekends={len(weekend_dates)} holidays={len(holiday_dates)} "
            f"overwrite={overwrite} workers={workers} download_dir={str(download_dir)} source=rest_grouped"
        ),
        flush=True,
    )

    success_count = 0
    skip_count = 0
    error_count = 0
    not_found_count = 0

    submit_kwargs = dict(
        api_key=api_key,
        adjusted=adjusted,
        max_retries=max_retries,
        download_dir=download_dir,
        overwrite=overwrite,
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_date: dict[Future[str], date] = {
            executor.submit(_fetch_and_save, d, **submit_kwargs): d
            for d in trading_dates
        }

        pbar = tqdm(
            as_completed(future_to_date),
            total=len(future_to_date),
            desc=f"download day_aggs(rest, workers={workers})",
            unit="day",
            dynamic_ncols=True,
            ascii=True,
        )
        for fut in pbar:
            try:
                result = fut.result()
            except DayDataQualityError:
                # 品質エラーは即座に外部に伝播し全タスクをキャンセルする
                for f in future_to_date:
                    f.cancel()
                raise
            except Exception as e:
                error_count += 1
                d = future_to_date[fut]
                print(f"\n[ERROR] {d.isoformat()}: {e}", flush=True)
            else:
                if result == _R_SKIPPED:
                    skip_count += 1
                elif result == _R_SUCCESS:
                    success_count += 1
                elif result == _R_NOT_FOUND:
                    not_found_count += 1

            pbar.set_postfix(
                ok=success_count,
                skip=skip_count,
                err=error_count,
                nf=not_found_count,
                refresh=False,
            )

    return {
        "success": int(success_count),
        "skipped": int(skip_count),
        "errors": int(error_count),
        "not_found": int(not_found_count),
        "trading_days": int(len(trading_dates)),
        "weekends": int(len(weekend_dates)),
        "holidays": int(len(holiday_dates)),
    }
