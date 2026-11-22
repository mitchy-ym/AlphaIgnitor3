from __future__ import annotations

import datetime as dt
import email.utils
import html
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from .schema import NewsArticle

logger = logging.getLogger(__name__)


def _mask_api_key(text: str) -> str:
    return re.sub(r"([?&](?:apiKey|key)=)[^&\s'\"]+", r"\1***", str(text))

# Public financial RSS feeds (Zero-cost, authoritative, reliable)
DEFAULT_RSS_FEEDS = [
    {
        "name": "FRB Press Releases",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "category": "macro_policy",
    },
    {
        "name": "CNBC Economy",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
        "category": "macro_economy",
    },
    {
        "name": "CNBC Finance",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "category": "markets",
    },
    {
        "name": "MarketWatch Top Stories",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "category": "markets",
    },
]

_CLEANR = re.compile(r"<.*?>")


def _clean_html(raw_html: str) -> str:
    cleantext = re.sub(_CLEANR, " ", raw_html or "")
    cleantext = html.unescape(cleantext)
    return " ".join(cleantext.split())


def _parse_rfc822_date(date_str: str) -> dt.datetime | None:
    if not date_str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
    except Exception:
        pass
    try:
        # ISO fallback
        return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except Exception:
        return None


def fetch_rss_feed(
    url: str,
    *,
    source_name: str,
    cutoff_utc: dt.datetime,
    timeout_sec: float = 8.0,
) -> list[NewsArticle]:
    articles: list[NewsArticle] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
        if resp.status_code != 200 or not resp.content:
            logger.warning("RSS feed returned non-200 or empty body (%s): status=%s", source_name, resp.status_code)
            return []
        root = ET.fromstring(resp.content)
    except Exception as e:
        logger.warning("RSS feed fetch failed (%s): %s", source_name, _mask_api_key(str(e)))
        return []

    # Parse channel -> item
    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        desc_el = item.find("description")

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        if not title or not link:
            continue

        pub_text = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
        pub_dt = _parse_rfc822_date(pub_text)
        if pub_dt is not None and pub_dt < cutoff_utc:
            continue

        desc = _clean_html(desc_el.text) if desc_el is not None and desc_el.text else ""
        pub_iso = pub_dt.isoformat() if pub_dt else dt.datetime.now(dt.timezone.utc).isoformat()

        articles.append(
            NewsArticle(
                title=title,
                url=link,
                published_at=pub_iso,
                source=source_name,
                summary=desc[:300],
                tickers=[],
            )
        )
    return articles


def fetch_polygon_market_news(
    *,
    api_key: str,
    cutoff_utc: dt.datetime,
    limit: int = 40,
    timeout_sec: float = 10.0,
    base_url: str = "https://api.polygon.io",
) -> list[NewsArticle]:
    if not api_key:
        return []

    # Massive/Polygon reference news endpoint
    url = f"{base_url.rstrip('/')}/v2/reference/news"
    params: dict[str, Any] = {
        "limit": int(limit),
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout_sec)
        if resp.status_code != 200:
            # If standard polygon base fails, try massive endpoint if different
            if "polygon.io" in base_url:
                alt_url = "https://api.massive.com/v2/reference/news"
                resp = requests.get(alt_url, params=params, timeout=timeout_sec)
                if resp.status_code != 200:
                    return []
            else:
                return []
        data = resp.json()
        results = data.get("results") or []
    except Exception as e:
        logger.warning("Polygon market news fetch failed: %s", _mask_api_key(str(e)))
        return []

    articles: list[NewsArticle] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        article_url = str(item.get("article_url", "")).strip()
        pub_str = str(item.get("published_utc", "")).strip()
        if not title or not article_url:
            continue

        pub_dt = _parse_rfc822_date(pub_str)
        if pub_dt is not None and pub_dt < cutoff_utc:
            continue

        pub_iso = pub_dt.isoformat() if pub_dt else pub_str
        tickers = [str(t).upper() for t in item.get("tickers", []) if t]
        desc = _clean_html(str(item.get("description", "")))
        source = str((item.get("publisher") or {}).get("name", "Polygon News"))

        articles.append(
            NewsArticle(
                title=title,
                url=article_url,
                published_at=pub_iso,
                source=source,
                summary=desc[:300],
                tickers=tickers,
            )
        )
    return articles


def fetch_polygon_ticker_news(
    *,
    ticker: str,
    api_key: str,
    cutoff_utc: dt.datetime,
    limit: int = 5,
    timeout_sec: float = 8.0,
    base_url: str = "https://api.polygon.io",
) -> list[NewsArticle]:
    if not api_key or not ticker:
        return []
    url = f"{base_url.rstrip('/')}/v2/reference/news"
    params = {
        "ticker": ticker.upper(),
        "limit": int(limit),
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout_sec)
        if resp.status_code != 200 and "polygon.io" in base_url:
            resp = requests.get("https://api.massive.com/v2/reference/news", params=params, timeout=timeout_sec)
            if resp.status_code != 200:
                return []
        data = resp.json()
        results = data.get("results") or []
    except Exception as e:
        logger.warning("Polygon market news fetch failed: %s", _mask_api_key(str(e)))
        return []

    articles: list[NewsArticle] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        article_url = str(item.get("article_url", "")).strip()
        pub_str = str(item.get("published_utc", "")).strip()
        if not title or not article_url:
            continue

        pub_dt = _parse_rfc822_date(pub_str)
        if pub_dt is not None and pub_dt < cutoff_utc:
            continue

        desc = _clean_html(str(item.get("description", "")))
        articles.append(
            NewsArticle(
                title=title,
                url=article_url,
                published_at=pub_dt.isoformat() if pub_dt else pub_str,
                source=str((item.get("publisher") or {}).get("name", "Polygon News")),
                summary=desc[:300],
                tickers=[ticker.upper()],
            )
        )
    return articles


def collect_all_news(
    *,
    api_key: str | None = None,
    target_tickers: list[str] | None = None,
    lookback_hours: int = 36,
    max_workers: int = 6,
    cache_dir: Any = None,
) -> list[NewsArticle]:
    """Collect macro, sector, and target-ticker news concurrently."""
    if not api_key:
        try:
            from alphaignitor.common._credentials import get_api_key
            api_key = get_api_key()
        except Exception:
            api_key = None
    cutoff_utc = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=int(lookback_hours))
    all_articles: list[NewsArticle] = []
    seen_urls: set[str] = set()

    def _add_article(a: NewsArticle) -> None:
        norm_url = urllib.parse.urldefrag(a.url)[0].strip().lower()
        if norm_url and norm_url not in seen_urls:
            seen_urls.add(norm_url)
            all_articles.append(a)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # 1. Fetch RSS feeds
        for feed in DEFAULT_RSS_FEEDS:
            futures.append(
                executor.submit(
                    fetch_rss_feed,
                    feed["url"],
                    source_name=feed["name"],
                    cutoff_utc=cutoff_utc,
                )
            )

        # 2. Fetch general market news from Polygon/Massive
        if api_key:
            futures.append(
                executor.submit(
                    fetch_polygon_market_news,
                    api_key=api_key,
                    cutoff_utc=cutoff_utc,
                    limit=40,
                )
            )

            # 3. Fetch specific ticker news for target watchlist (limit to top 15 tickers)
            tickers_to_query = (target_tickers or [])[:15]
            for t in tickers_to_query:
                futures.append(
                    executor.submit(
                        fetch_polygon_ticker_news,
                        ticker=t,
                        api_key=api_key,
                        cutoff_utc=cutoff_utc,
                        limit=3,
                    )
                )

        for f in as_completed(futures):
            try:
                res = f.result()
                if isinstance(res, list):
                    for a in res:
                        _add_article(a)
            except Exception as e:
                logger.warning("News collection worker failed: %s", _mask_api_key(str(e)))

    all_articles.sort(key=lambda a: a.published_at, reverse=True)
    return all_articles

