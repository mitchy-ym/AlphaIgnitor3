from __future__ import annotations

import math
from typing import Any

from .schema import MacroNewsReport


def evaluate_news_hybrid_verdict(
    *,
    quant_bull: int,
    quant_bear: int,
    ticker: str,
    sector: str = "",
    news_report: MacroNewsReport | None = None,
    adverse_threshold: float = -0.45,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Combine quantitative model votes with qualitative macro/sector news sentiment.

    Returns:
        (verdict_str, display_text, overlay_items)
        where verdict_str is one of: "bullish", "bearish", "neutral"
        display_text is one of: "🟢 Bull", "🔴 Bear", "🟡 Neutral"
    """
    # 1. Base quantitative rule (Model Vote: Chronos-2, TimesFM, TiRex-2)
    if quant_bull >= quant_bear + 2:
        quant_verdict = "bullish"
        quant_text = "🟢 Bull"
    elif quant_bear >= quant_bull + 2:
        quant_verdict = "bearish"
        quant_text = "🔴 Bear"
    else:
        quant_verdict = "neutral"
        quant_text = "🟡 Neutral"

    if news_report is None:
        return quant_verdict, quant_text, []

    # 2. Extract composite news score and items
    comp_score = news_report.get_composite_ticker_score(ticker=ticker, sector=sector)
    m_score = news_report.macro_score
    s_sentiment = news_report.sector_scores.get(sector)
    s_score = s_sentiment.score if s_sentiment else 0.0
    t_impact = news_report.ticker_impacts.get(ticker)

    overlay_items: list[dict[str, Any]] = []

    # Macro item
    m_state = True if m_score > 0.15 else (False if m_score < -0.15 else None)
    m_sign = "+" if m_score >= 0 else ""
    overlay_items.append({
        "label": f"Macro: {news_report.macro_regime} ({m_sign}{m_score:.2f})",
        "category": "Macro & Sector News",
        "bullish": m_state,
    })

    # Sector item
    if sector:
        s_state = True if s_score > 0.15 else (False if s_score < -0.15 else None)
        s_sign = "+" if s_score >= 0 else ""
        s_reason = f" - {s_sentiment.reason_ja}" if s_sentiment and s_sentiment.reason_ja else ""
        overlay_items.append({
            "label": f"Sector ({sector}): {s_sign}{s_score:.2f}{s_reason}",
            "category": "Macro & Sector News",
            "bullish": s_state,
        })

    # Ticker specific item if present
    if t_impact is not None:
        t_state = True if t_impact.score > 0.15 else (False if t_impact.score < -0.15 else None)
        t_sign = "+" if t_impact.score >= 0 else ""
        adv_str = " [⚠️ Acute Adverse Alert]" if t_impact.is_adverse_alert else ""
        t_summary = f": {t_impact.summary_ja}" if t_impact.summary_ja else ""
        overlay_items.append({
            "label": f"Ticker Catalysts ({t_sign}{t_impact.score:.2f}){adv_str}{t_summary}",
            "category": "Ticker News",
            "bullish": t_state,
        })

    # 3. Hybrid safety rules (Asymmetric Gateway)
    final_verdict = quant_verdict
    final_text = quant_text

    is_acute_adverse = t_impact.is_adverse_alert if t_impact else False

    if quant_verdict == "bullish":
        # If severe bad news or acute adverse alert, demote to Neutral (safety gate)
        if is_acute_adverse or comp_score <= adverse_threshold:
            final_verdict = "neutral"
            final_text = "🟡 Neutral"
            overlay_items.append({
                "label": "Safety Filter: Bull downgraded to Neutral due to severe adverse news/macro headwinds",
                "category": "Hybrid Filter",
                "bullish": False,
            })
    elif quant_verdict == "bearish":
        # If strong positive catalyst/macro rally, demote Bear to Neutral to prevent short squeeze
        if comp_score >= 0.45:
            final_verdict = "neutral"
            final_text = "🟡 Neutral"
            overlay_items.append({
                "label": "Safety Filter: Bear downgraded to Neutral due to strong macro/ticker tailwinds",
                "category": "Hybrid Filter",
                "bullish": True,
            })
    elif quant_verdict == "neutral":
        # High-conviction macro/sector boost for close margin
        if quant_bull > quant_bear and comp_score >= 0.50:
            final_verdict = "bullish"
            final_text = "🟢 Bull"
            overlay_items.append({
                "label": "Catalyst Boost: Neutral upgraded to Bull supported by strong macro & sector tailwinds",
                "category": "Hybrid Filter",
                "bullish": True,
            })
        elif quant_bear > quant_bull and comp_score <= -0.50:
            final_verdict = "bearish"
            final_text = "🔴 Bear"
            overlay_items.append({
                "label": "Catalyst Boost: Neutral upgraded to Bear supported by strong macro & sector headwinds",
                "category": "Hybrid Filter",
                "bullish": False,
            })

    return final_verdict, final_text, overlay_items

