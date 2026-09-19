from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qsl, urlunparse
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from alphaignitor.common._credentials import (
    load_simple_env_file,
    resolve_credentials_file,
    _PLACEHOLDER_TOKENS,
    _is_placeholder,
    ensure_api_key_loaded,
    get_api_key,
)


@dataclass(frozen=True)
class SplitEvent:
    execution_date: date
    historical_adjustment_factor: float
    adjustment_type: str



def _add_api_key(url: str, api_key: str) -> str:
    """Ensure apiKey query param exists."""

    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q.setdefault("apiKey", api_key)
    new_query = urlencode(q)
    return urlunparse(parsed._replace(query=new_query))


def _http_get_json(url: str, *, api_key: str | None = None) -> dict:
    headers = {"User-Agent": "AlphaIgnitor3"}
    # Default to query-param auth (`apiKey=...`) which is known to work.
    # If you need bearer auth, set MASSIVE_SPLITS_AUTH=bearer or both.
    if api_key:
        auth_mode = (os.environ.get("MASSIVE_SPLITS_AUTH") or "query").strip().lower()
        if auth_mode in {"bearer", "both"}:
            headers["Authorization"] = f"Bearer {api_key}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        # Provide actionable guidance for the most common failure: auth.
        if int(getattr(e, "code", 0) or 0) == 401:
            raise RuntimeError(
                "Massive Splits REST API が 401 Unauthorized を返しました。\n"
                "MASSIVE_API_KEY (または API_KEY) が未設定、もしくは無効です。\n"
                "- 環境変数 MASSIVE_API_KEY=... を設定するか\n"
                "- secrets/credentials.env に MASSIVE_API_KEY=... を追記してください。\n"
            ) from e
        raise


def fetch_splits_for_ticker(*, ticker: str, api_key: str) -> list[SplitEvent]:
    base = os.environ.get("MASSIVE_SPLITS_URL", "https://api.massive.com/stocks/v1/splits")
    params = {
        "ticker": ticker,
        "limit": 5000,
        "sort": "execution_date.desc",
        "apiKey": api_key,
    }
    url = base + "?" + urlencode(params)

    out: list[SplitEvent] = []
    while True:
        data = _http_get_json(url, api_key=api_key)
        results = data.get("results") or []
        if isinstance(results, list):
            for r in results:
                try:
                    d = date.fromisoformat(str(r.get("execution_date")))
                    f = float(r.get("historical_adjustment_factor"))
                    at = str(r.get("adjustment_type") or "")
                except Exception:
                    continue
                out.append(SplitEvent(execution_date=d, historical_adjustment_factor=f, adjustment_type=at))

        next_url = data.get("next_url")
        if not next_url:
            break
        url = _add_api_key(str(next_url), api_key)

    # Sort ascending by execution_date for downstream lookup.
    out.sort(key=lambda x: x.execution_date)
    return out


def load_or_fetch_splits(
    *,
    ticker: str,
    cache_dir: Path,
    allowed_types: set[str],
    api_key: str,
) -> list[SplitEvent]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}.json"

    events: list[SplitEvent] = []
    cache_loaded = False

    if cache_path.exists() and cache_path.is_file() and cache_path.stat().st_size > 0:
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for r in raw:
                    try:
                        d = date.fromisoformat(str(r["execution_date"]))
                        f = float(r["historical_adjustment_factor"])
                        at = str(r.get("adjustment_type") or "")
                    except Exception:
                        continue
                    events.append(SplitEvent(execution_date=d, historical_adjustment_factor=f, adjustment_type=at))
                cache_loaded = True
        except Exception:
            events = []
            cache_loaded = False

    if not cache_loaded:
        fetched = fetch_splits_for_ticker(ticker=ticker, api_key=api_key)
        # write raw cache as list[dict]
        payload = [
            {
                "execution_date": e.execution_date.isoformat(),
                "historical_adjustment_factor": float(e.historical_adjustment_factor),
                "adjustment_type": str(e.adjustment_type),
            }
            for e in fetched
        ]
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        events = fetched

    # Filter allowed types
    events = [e for e in events if str(e.adjustment_type) in allowed_types]
    events.sort(key=lambda x: x.execution_date)
    return events


def adjustment_factor_for_date(*, events_asc: list[SplitEvent], d: date) -> float:
    """Return factor for historical date d.

    Uses Massive rule: find first split whose execution_date is after date d.
    If none, factor=1.
    """

    for e in events_asc:
        if e.execution_date > d:
            f = float(e.historical_adjustment_factor)
            if f > 0 and not math.isnan(f):
                return f
            break
    return 1.0
