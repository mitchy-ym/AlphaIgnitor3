from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alphaignitor.pipeline.news.collector import (
    collect_all_news,
    fetch_polygon_market_news,
    fetch_rss_feed,
)
from alphaignitor.pipeline.news.analyzer import (
    _parse_llm_json_to_report,
    analyze_news_report,
    analyze_news_with_gemini,
    generate_fallback_macro_report,
)
from alphaignitor.pipeline.news.integration import evaluate_news_hybrid_verdict
from alphaignitor.pipeline.news.schema import (
    MacroNewsReport,
    NewsArticle,
    SectorSentiment,
    TickerNewsImpact,
)
from alphaignitor.pipeline.news.storage import NewsSentimentStorage


class TestNewsSchemaAndStorage:
    def test_schema_serialization_and_scores(self):
        report = MacroNewsReport(
            asof_date="2025-01-10",
            macro_score=0.40,
            macro_regime="Bullish",
            event_risk_level="MEDIUM",
            event_risk_note="CPI Announcement tomorrow",
            key_drivers_ja=["インフレ指標の鈍化", "ハイテク決算の好調"],
            sector_scores={
                "Information Technology": SectorSentiment(score=0.60, reason_ja="半導体需要の回復"),
                "Energy": SectorSentiment(score=-0.30, reason_ja="原油安"),
            },
            ticker_impacts={
                "NVDA": TickerNewsImpact(score=0.80, summary_ja="新AIチップ発表", is_adverse_alert=False),
                "TSLA": TickerNewsImpact(score=-0.70, summary_ja="リコール報道", is_adverse_alert=True),
            },
            articles_analyzed_count=15,
            provider="gemini",
            model="gemini-2.5-flash",
        )

        # Composite score tests
        # NVDA: Technology -> 0.35 * 0.40 + 0.35 * 0.60 + 0.30 * 0.80 = 0.14 + 0.21 + 0.24 = 0.59
        nvda_score = report.get_composite_ticker_score("NVDA", "Information Technology")
        assert round(nvda_score, 2) == 0.59

        # AAPL (no ticker impact): Technology -> (0.35 * 0.40 + 0.35 * 0.60) / 0.70 = 0.50
        aapl_score = report.get_composite_ticker_score("AAPL", "Information Technology")
        assert round(aapl_score, 2) == 0.50

        # Serialization roundtrip
        d = report.to_dict()
        assert d["macro_score"] == 0.40
        assert d["event_risk_level"] == "MEDIUM"

        restored = MacroNewsReport.from_dict(d)
        assert restored.asof_date == "2025-01-10"
        assert restored.macro_score == 0.40
        assert restored.sector_scores["Information Technology"].score == 0.60
        assert restored.ticker_impacts["TSLA"].is_adverse_alert is True

    def test_sqlite_storage(self, tmp_path: Path):
        db_path = tmp_path / "news_sentiment.sqlite3"
        storage = NewsSentimentStorage(db_path)

        report = MacroNewsReport(
            asof_date="2025-01-15",
            macro_score=0.25,
            macro_regime="Moderately Bullish",
            key_drivers_ja=["雇用統計の堅調"],
        )
        storage.save_macro_report(report)

        loaded = storage.load_macro_report("2025-01-15")
        assert loaded is not None
        assert loaded.asof_date == "2025-01-15"
        assert loaded.macro_score == 0.25
        assert loaded.key_drivers_ja == ["雇用統計の堅調"]

        # Cache misses return None
        assert storage.load_macro_report("2025-01-16") is None

        # Raw articles storage
        import datetime as dt
        now_str = dt.datetime.now(dt.timezone.utc).isoformat()
        articles = [
            NewsArticle(title="Fed holds rates", url="https://example.com/1", published_at=now_str, source="FRB"),
            NewsArticle(title="Tech leads rally", url="https://example.com/2", published_at=now_str, source="CNBC"),
        ]
        storage.save_raw_articles(articles)
        cached_articles = storage.load_raw_articles(lookback_hours=48)
        assert len(cached_articles) >= 2


class TestNewsCollector:
    @patch("requests.get")
    def test_fetch_rss_feed(self, mock_get):
        rss_content = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Fed News</title>
            <item>
              <title>FOMC releases statement</title>
              <link>https://www.federalreserve.gov/newsevents/pressreleases/monetary20250115a.htm</link>
              <pubDate>Wed, 15 Jan 2025 14:00:00 GMT</pubDate>
              <description>The Federal Open Market Committee decided today to keep rates unchanged.</description>
            </item>
          </channel>
        </rss>"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = rss_content
        mock_get.return_value = mock_resp

        import datetime as dt
        cutoff = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        articles = fetch_rss_feed("https://example.com/rss", source_name="Federal Reserve", cutoff_utc=cutoff)
        assert len(articles) == 1
        assert articles[0].title == "FOMC releases statement"
        assert articles[0].source == "Federal Reserve"
        assert "keep rates unchanged" in articles[0].summary

    @patch("requests.get")
    def test_fetch_polygon_market_news(self, mock_get):
        poly_json = {
            "results": [
                {
                    "title": "Nvidia expands AI ecosystem",
                    "article_url": "https://polygon.io/news/123",
                    "published_utc": "2025-01-15T16:00:00Z",
                    "description": "Nvidia announces new architecture.",
                    "tickers": ["NVDA"],
                    "publisher": {"name": "Polygon News"},
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = poly_json
        mock_get.return_value = mock_resp

        import datetime as dt
        cutoff = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
        articles = fetch_polygon_market_news(api_key="mock_key", cutoff_utc=cutoff)
        assert len(articles) == 1
        assert articles[0].title == "Nvidia expands AI ecosystem"
        assert articles[0].tickers == ["NVDA"]

    @patch("alphaignitor.pipeline.news.collector.fetch_polygon_ticker_news")
    @patch("alphaignitor.pipeline.news.collector.fetch_polygon_market_news")
    @patch("alphaignitor.pipeline.news.collector.fetch_rss_feed")
    def test_collect_all_news(self, mock_rss, mock_poly_market, mock_poly_ticker):
        mock_rss.return_value = [
            NewsArticle(
                title="Mock Headline",
                url="https://example.com/mock",
                published_at="2025-01-15T12:00:00Z",
                source="MockSource",
            )
        ]
        mock_poly_market.return_value = []
        mock_poly_ticker.return_value = []
        articles = collect_all_news(lookback_hours=24)
        assert len(articles) >= 1
        assert articles[0].title == "Mock Headline"


class TestNewsAnalyzer:
    def test_parse_llm_json_to_report(self):
        data = {
            "macro_score": 0.35,
            "macro_regime": "Moderately Bullish",
            "event_risk_level": "LOW",
            "event_risk_note": None,
            "key_drivers_ja": ["利下げ期待の継続", "企業業績の安定"],
            "sector_scores": {
                "Information Technology": {"score": 0.5, "reason_ja": "AI関連投資"},
            },
            "ticker_impacts": {
                "NVDA": {"score": 0.7, "catalyst_type": "Earnings", "summary_ja": "大幅増益", "is_adverse_alert": False},
            },
        }
        report = _parse_llm_json_to_report(data, asof_date="2025-01-15", provider="gemini", model="gemini-2.5-flash", articles_count=10)
        assert report.macro_score == 0.35
        assert report.macro_regime == "Moderately Bullish"
        assert report.sector_scores["Information Technology"].score == 0.5
        assert report.ticker_impacts["NVDA"].score == 0.7

    def test_generate_fallback_macro_report(self):
        report = generate_fallback_macro_report(asof_date="2025-01-15")
        assert report.macro_score == 0.0
        assert "Quant Fallback" in report.macro_regime
        assert report.event_risk_level == "LOW"
        assert len(report.key_drivers_ja) == 1

    @patch("requests.post")
    def test_analyze_news_with_gemini(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "macro_score": 0.40,
                                    "macro_regime": "Bullish",
                                    "event_risk_level": "LOW",
                                    "key_drivers_ja": ["好景気期待"],
                                    "sector_scores": {},
                                    "ticker_impacts": {},
                                })
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        articles = [
            NewsArticle(title="Market Rallies", url="https://example.com/rally", published_at="2025-01-15T12:00:00Z", source="CNBC")
        ]
        report = analyze_news_with_gemini(articles=articles, asof_date="2025-01-15", api_key="test_key")
        assert report.macro_score == 0.40
        assert report.macro_regime == "Bullish"

    def test_analyze_news_report_fallback(self):
        # Without API keys, analyze_news_report gracefully falls back to quant fallback
        articles = [
            NewsArticle(title="Market Rallies", url="https://example.com/rally", published_at="2025-01-15T12:00:00Z", source="CNBC")
        ]
        with patch.dict("os.environ", {}, clear=True):
            report = analyze_news_report(articles=articles, asof_date="2025-01-15")
            assert report.macro_score == 0.0
            assert "Quant Fallback" in report.macro_regime


class TestNewsIntegration:
    def test_hybrid_verdict_neutral_when_no_news(self):
        # 6 bull, 0 bear -> quant bullish
        v, t, items = evaluate_news_hybrid_verdict(quant_bull=6, quant_bear=0, ticker="AAPL", sector="Information Technology")
        assert v == "bullish"
        assert t == "🟢 Bull"
        assert len(items) == 0

    def test_hybrid_verdict_adverse_safety_gate(self):
        report = MacroNewsReport(
            asof_date="2025-01-15",
            macro_score=0.20,
            ticker_impacts={
                "AAPL": TickerNewsImpact(score=-0.80, is_adverse_alert=True, summary_ja="司法省が反トラスト法提訴"),
            },
        )
        # Quant says bullish, but acute adverse alert must demote to Neutral
        v, t, items = evaluate_news_hybrid_verdict(
            quant_bull=6, quant_bear=0, ticker="AAPL", sector="Information Technology", news_report=report
        )
        assert v == "neutral"
        assert t == "🟡 Neutral"
        assert any("Safety Filter" in item["label"] for item in items)

    def test_hybrid_verdict_short_squeeze_protection(self):
        report = MacroNewsReport(
            asof_date="2025-01-15",
            macro_score=0.60,
            ticker_impacts={
                "NVDA": TickerNewsImpact(score=0.85, summary_ja="空前の好決算"),
            },
        )
        # Quant says bearish, but strong positive news must demote to Neutral
        v, t, items = evaluate_news_hybrid_verdict(
            quant_bull=0, quant_bear=6, ticker="NVDA", sector="Information Technology", news_report=report
        )
        assert v == "neutral"
        assert t == "🟡 Neutral"
        assert any("Safety Filter: Bear downgraded" in item["label"] for item in items)

    def test_hybrid_verdict_catalyst_boost(self):
        report = MacroNewsReport(
            asof_date="2025-01-15",
            macro_score=0.40,
            sector_scores={"Information Technology": SectorSentiment(score=0.60, reason_ja="強気")},
            ticker_impacts={"MSFT": TickerNewsImpact(score=0.70, summary_ja="大型提携発表")},
        )
        # Quant is slightly positive (3 vs 2: neutral), composite score >= 0.50 triggers boost
        v, t, items = evaluate_news_hybrid_verdict(
            quant_bull=3, quant_bear=2, ticker="MSFT", sector="Information Technology", news_report=report
        )
        assert v == "bullish"
        assert t == "🟢 Bull"
        assert any("Catalyst Boost" in item["label"] for item in items)


class TestNewsStrategyAndReportIntegration:
    def test_evaluate_signals_with_news_filter(self):
        from alphaignitor.backtest.config import StrategyConfig, TradingConfig
        from alphaignitor.backtest.strategy import evaluate_signals_for_asof
        from alphaignitor.pipeline.zero_shot_ensemble.storage import (
            init_schema,
            save_best_weights,
            save_model_predictions,
        )

        strategy_cfg = StrategyConfig(
            min_predicted_return=0.01,
            consensus_level="all",
            allow_short=True,
            min_predicted_short_return=0.01,
        )

        import sqlite3
        conn = sqlite3.connect(":memory:")
        init_schema(conn)

        for m in ["chronos2", "timesfm", "tirex"]:
            save_model_predictions(
                conn,
                ticker="AAPL",
                asof_trade_date="2025-01-15",
                model=m,
                predictions=[105.0, 106.0, 107.0],
            )
            save_model_predictions(
                conn,
                ticker="TSLA",
                asof_trade_date="2025-01-15",
                model=m,
                predictions=[90.0, 89.0, 88.0],
            )
        save_best_weights(conn, ticker="AAPL", weights={"chronos2": 0.33, "timesfm": 0.33, "tirex": 0.34}, score=0.8, n_trials=10)
        save_best_weights(conn, ticker="TSLA", weights={"chronos2": 0.33, "timesfm": 0.33, "tirex": 0.34}, score=0.8, n_trials=10)

        series_df = pd.DataFrame([
            {"trade_date": "2025-01-14", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"trade_date": "2025-01-15", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        ])
        series_map = {"AAPL": series_df, "TSLA": series_df}

        # Case 1: Without news, AAPL is BUY, TSLA is SHORT
        sigs = evaluate_signals_for_asof(
            conn=conn,
            asof_date="2025-01-15",
            tickers=["AAPL", "TSLA"],
            series_by_ticker=series_map,
            strategy_cfg=strategy_cfg,
        )
        actions = {s.ticker: s.action for s in sigs}
        assert actions.get("AAPL") == "BUY"
        assert actions.get("TSLA") == "SHORT"

        # Case 2: AAPL has acute adverse alert -> BUY blocked
        report_adverse_aapl = MacroNewsReport(
            asof_date="2025-01-15",
            ticker_impacts={
                "AAPL": TickerNewsImpact(score=-0.80, is_adverse_alert=True, summary_ja="米司法省提訴"),
            },
        )
        sigs_adv = evaluate_signals_for_asof(
            conn=conn,
            asof_date="2025-01-15",
            tickers=["AAPL", "TSLA"],
            series_by_ticker=series_map,
            strategy_cfg=strategy_cfg,
            news_report=report_adverse_aapl,
        )
        actions_adv = {s.ticker: s.action for s in sigs_adv}
        assert "AAPL" not in actions_adv  # Excluded!
        assert actions_adv.get("TSLA") == "SHORT"

        # Case 3: TSLA has strong positive news (score >= 0.45) -> SHORT blocked to prevent short squeeze
        report_bull_tsla = MacroNewsReport(
            asof_date="2025-01-15",
            ticker_impacts={
                "TSLA": TickerNewsImpact(score=0.85, summary_ja="納車台数が市場予想を大幅超過"),
            },
        )
        sigs_tsla = evaluate_signals_for_asof(
            conn=conn,
            asof_date="2025-01-15",
            tickers=["AAPL", "TSLA"],
            series_by_ticker=series_map,
            strategy_cfg=strategy_cfg,
            news_report=report_bull_tsla,
        )
        actions_tsla = {s.ticker: s.action for s in sigs_tsla}
        assert actions_tsla.get("AAPL") == "BUY"
        assert "TSLA" not in actions_tsla  # Short blocked!

        conn.close()

    def test_action_sheet_adverse_news_emergency_exit(self):
        from alphaignitor.backtest.config import TradingConfig
        from alphaignitor.backtest.report import format_action_sheet_text, generate_action_sheet
        from alphaignitor.backtest.strategy import TradeSignal

        trading_cfg = TradingConfig()

        active_positions = [
            {"ticker": "NVDA", "shares": 50, "entry_price": 120.0, "current_price": 125.0, "holding_days": 1},
            {"ticker": "MSFT", "shares": 30, "entry_price": 400.0, "current_price": 402.0, "holding_days": 1},
        ]

        news_report = MacroNewsReport(
            asof_date="2025-01-15",
            macro_score=0.30,
            event_risk_level="HIGH",
            event_risk_note="米中半導体輸出規制の緊急発効",
            key_drivers_ja=["半導体規制の強化", "金利見通しの小幅修正"],
            ticker_impacts={
                "NVDA": TickerNewsImpact(score=-0.85, is_adverse_alert=True, summary_ja="対中AIチップ即時禁輸令"),
            },
        )

        sheet = generate_action_sheet(
            asof_date="2025-01-15",
            signals=[],
            trading_config=trading_cfg,
            active_positions=active_positions,
            news_report=news_report,
        )

        # NVDA must be promoted to emergency EXIT due to adverse news alert even though holding_days == 1
        exit_tickers = [e["ticker"] for e in sheet["exits"]]
        assert "NVDA" in exit_tickers
        nvda_exit = [e for e in sheet["exits"] if e["ticker"] == "NVDA"][0]
        assert "悪材料ニュース検知" in nvda_exit["reason"]

        # MSFT remains in holds
        hold_tickers = [h["ticker"] for h in sheet["holds"]]
        assert "MSFT" in hold_tickers

        # Sheet text formatting
        text = format_action_sheet_text(sheet)
        assert "マクロ警戒アラート" in text
        assert "半導体規制の強化" in text

    def test_html_report_rendering_with_news_pulse(self, tmp_path: Path):
        from alphaignitor.pipeline.zero_shot_ensemble.report import (
            _report_columns,
            build_report_table,
            build_summary_dl,
            render_report_html,
        )

        meta_csv = tmp_path / "meta.csv"
        meta_csv.write_text("Ticker,Name,Sector\nAAPL,Apple Inc.,Information Technology\n", encoding="utf-8")

        df = pd.DataFrame([
            {
                "ticker": "AAPL",
                "asof_trade_date": "2025-01-10",
                "horizon": 1,
                "forecast_trade_date": "2025-01-11",
                "asof_close": 150.0,
                "ensemble_pred": 153.0,
                "ensemble_return": 0.02,
                "ensemble_direction": 1,
                "chronos2_pred": 153.0,
                "timesfm_pred": 153.0,
                "tirex_pred": 153.0,
            }
        ])

        news_report = MacroNewsReport(
            asof_date="2025-01-10",
            macro_score=0.45,
            macro_regime="Bullish",
            event_risk_level="LOW",
            key_drivers_ja=["利下げ期待と好調な雇用統計"],
            sector_scores={"Information Technology": SectorSentiment(score=0.55, reason_ja="生成AI需要加速")},
            ticker_impacts={"AAPL": TickerNewsImpact(score=0.40, summary_ja="新製品発表好調")},
            articles_analyzed_count=12,
        )

        table = build_report_table(df, ticker_meta_csv=meta_csv, max_horizon=1, news_report=news_report)
        assert len(table) == 1
        sig_json = json.loads(table.iloc[0]["signals_json"])
        # Should include both Model Vote and Macro/Sector News in items
        categories = {item["category"] for item in sig_json["items"]}
        assert "Model Vote" in categories
        assert "Macro & Sector News" in categories

        cols = _report_columns(max_horizon=1)
        summary_dl = build_summary_dl(df=df, table=table, asof="2025-01-10", forecast_path=Path("mock.parquet"), max_horizon=1)

        html = render_report_html(
            table=table,
            cols=cols,
            summary_dl=summary_dl,
            acc_html="<div>acc</div>",
            charts_html="<div>charts</div>",
            max_horizon=1,
            news_report=news_report,
        )

        assert "Political & Economic News Pulse" in html or "Macro Political & Economic Pulse" in html
        assert "利下げ期待と好調な雇用統計" in html
        assert "Information Technology" in html


    def test_portal_rendering_with_macro_pulse(self, tmp_path: Path):
        from alphaignitor.pipeline.portal import render_portal_html, scan_reports
        from alphaignitor.pipeline.news.storage import NewsSentimentStorage

        report_dir = tmp_path / "report"
        day_dir = report_dir / "2025-01-10"
        day_dir.mkdir(parents=True)
        (day_dir / "report.html").write_text("<html>Report</html>", encoding="utf-8")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        db_path = cache_dir / "news_sentiment.sqlite3"
        storage = NewsSentimentStorage(db_path)
        storage.save_macro_report(
            MacroNewsReport(
                asof_date="2025-01-10",
                macro_score=0.35,
                macro_regime="Moderately Bullish",
                event_risk_level="MEDIUM",
                event_risk_note="米CPI発表直前",
            )
        )

        data = scan_reports(report_dir)
        assert data["latest_macro"] is not None
        assert data["latest_macro"]["macro_score"] == 0.35

        portal_html = render_portal_html(data)
        assert "政治・経済マクロ地合い" in portal_html
        assert "Moderately Bullish (+0.35)" in portal_html
        assert "米CPI発表直前" in portal_html
