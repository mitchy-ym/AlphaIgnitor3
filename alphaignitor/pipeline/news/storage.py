from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from .schema import MacroNewsReport, NewsArticle

DEFAULT_NEWS_DB_PATH = Path("cache/news_sentiment.sqlite3")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_NEWS_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS macro_news_reports (
                asof_date TEXT PRIMARY KEY,
                macro_score REAL NOT NULL,
                macro_regime TEXT NOT NULL,
                event_risk_level TEXT NOT NULL,
                event_risk_note TEXT,
                report_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cached_raw_articles (
                url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                published_at TEXT NOT NULL,
                source TEXT NOT NULL,
                summary TEXT,
                tickers_json TEXT,
                fetched_at TEXT NOT NULL
            );
            """
        )


def save_macro_report(conn: sqlite3.Connection, report: MacroNewsReport) -> None:
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    report_dict = report.to_dict()
    if not report.generated_at:
        report_dict["generated_at"] = now_iso
    with conn:
        conn.execute(
            """
            INSERT INTO macro_news_reports (
                asof_date, macro_score, macro_regime, event_risk_level, event_risk_note, report_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asof_date) DO UPDATE SET
                macro_score = excluded.macro_score,
                macro_regime = excluded.macro_regime,
                event_risk_level = excluded.event_risk_level,
                event_risk_note = excluded.event_risk_note,
                report_json = excluded.report_json,
                updated_at = excluded.updated_at;
            """,
            (
                report.asof_date,
                float(report.macro_score),
                report.macro_regime,
                report.event_risk_level,
                report.event_risk_note,
                json.dumps(report_dict, ensure_ascii=False),
                now_iso,
            ),
        )


def load_macro_report(conn: sqlite3.Connection, asof_date: str) -> MacroNewsReport | None:
    row = conn.execute(
        """
        SELECT report_json FROM macro_news_reports WHERE asof_date = ?;
        """,
        (str(asof_date),),
    ).fetchone()
    if row and row[0]:
        try:
            return MacroNewsReport.from_dict(json.loads(row[0]))
        except Exception:
            return None
    return None


def save_raw_articles(conn: sqlite3.Connection, articles: list[NewsArticle]) -> int:
    if not articles:
        return 0
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = [
        (
            a.url,
            a.title,
            a.published_at,
            a.source,
            a.summary,
            json.dumps(a.tickers, ensure_ascii=False),
            now_iso,
        )
        for a in articles
        if a.url and a.title
    ]
    with conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO cached_raw_articles (
                url, title, published_at, source, summary, tickers_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
    return len(rows)


def load_raw_articles_since(conn: sqlite3.Connection, since_iso: str) -> list[NewsArticle]:
    rows = conn.execute(
        """
        SELECT title, url, published_at, source, summary, tickers_json
        FROM cached_raw_articles
        WHERE published_at >= ?
        ORDER BY published_at DESC;
        """,
        (str(since_iso),),
    ).fetchall()
    out: list[NewsArticle] = []
    for t, u, p, s, sm, tk_json in rows:
        try:
            tickers = json.loads(tk_json) if tk_json else []
        except Exception:
            tickers = []
        out.append(NewsArticle(title=t, url=u, published_at=p, source=s, summary=sm or "", tickers=tickers))
    return out


class NewsSentimentStorage:
    """Convenience wrapper for persisting news sentiment and articles."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_NEWS_DB_PATH

    def get_connection(self) -> sqlite3.Connection:
        return connect(self.db_path)

    def save_macro_report(self, report: MacroNewsReport) -> None:
        conn = self.get_connection()
        try:
            save_macro_report(conn, report)
        finally:
            conn.close()

    def load_macro_report(self, asof_date: str) -> MacroNewsReport | None:
        conn = self.get_connection()
        try:
            return load_macro_report(conn, asof_date)
        finally:
            conn.close()

    def save_raw_articles(self, articles: list[NewsArticle]) -> int:
        conn = self.get_connection()
        try:
            return save_raw_articles(conn, articles)
        finally:
            conn.close()

    def load_raw_articles(self, lookback_hours: int = 36) -> list[NewsArticle]:
        since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lookback_hours)).isoformat()
        conn = self.get_connection()
        try:
            return load_raw_articles_since(conn, since)
        finally:
            conn.close()


