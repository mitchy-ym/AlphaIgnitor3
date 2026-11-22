from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from typing import Any

import requests

from .schema import MacroNewsReport, NewsArticle, SectorSentiment, TickerNewsImpact

logger = logging.getLogger(__name__)

GICS_SECTORS = [
    "Information Technology",
    "Financials",
    "Health Care",
    "Consumer Discretionary",
    "Communication",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

DEFAULT_PROMPT_TEMPLATE = """You are an institutional macro quantitative analyst and equity strategist for AlphaIgnitor3.
Analyze the following latest US and global financial, political, and economic news headlines and snippets for date {asof_date}.

Input News Articles ({num_articles} items):
{news_text}

Additional Market Context:
- Target watchlist tickers evaluated: {target_tickers}
- Current quantitative baseline model vote: Bull {quant_bull_pct}%, Bear {quant_bear_pct}%, Neutral {quant_neutral_pct}%

Your Task:
Extract macroeconomic & political market sentiment, evaluate sector impacts for the 11 GICS sectors, identify scheduled high-impact events (FOMC, CPI, NFP, etc.), and detect any acute adverse company-specific catalysts for the target tickers.

You MUST respond strictly with a valid JSON object matching this exact schema:
{{
  "macro_score": <float between -1.0 (extreme panic/bearish) and +1.0 (extreme bullish), 0.0 is neutral>,
  "macro_regime": "<one of: Strong Bullish, Moderately Bullish, Neutral, Moderately Bearish, Strong Bearish>",
  "event_risk_level": "<one of: LOW, MEDIUM, HIGH>",
  "event_risk_note": "<string in Japanese summarizing any upcoming high-impact scheduled events like FOMC, CPI, or null if none>",
  "key_drivers_ja": [
    "<3 concise bullet points in Japanese explaining the primary macroeconomic and political drivers>",
    "<bullet 2 in Japanese>",
    "<bullet 3 in Japanese>"
  ],
  "sector_scores": {{
    "Information Technology": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Financials": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Health Care": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Consumer Discretionary": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Communication": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Industrials": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Consumer Staples": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Energy": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Utilities": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Real Estate": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}},
    "Materials": {{"score": <float -1.0 to 1.0>, "reason_ja": "<brief Japanese reason>"}}
  }},
  "ticker_impacts": {{
    "<TICKER>": {{
      "score": <float -1.0 to 1.0>,
      "catalyst_type": "<e.g. Earnings, Guidance, Regulatory, Policy, Demand, General>",
      "summary_ja": "<1 sentence in Japanese describing specific catalyst>",
      "is_adverse_alert": <true if this company suffered an acute severe negative event (e.g. guidance cut, fraud, investigation, disastrous earnings) that warrants avoiding or exiting>
    }}
  }}
}}
"""


def _build_news_text(articles: list[NewsArticle], max_items: int = 50) -> str:
    lines: list[str] = []
    for idx, a in enumerate(articles[:max_items], 1):
        tk_str = f" [Tickers: {', '.join(a.tickers)}]" if a.tickers else ""
        desc_str = f" - {a.summary}" if a.summary else ""
        lines.append(f"[{idx}] {a.source} ({a.published_at[:16]}): {a.title}{desc_str}{tk_str}")
    return "\n".join(lines)


def analyze_news_with_gemini(
    *,
    articles: list[NewsArticle],
    asof_date: str,
    api_key: str,
    target_tickers: list[str] | None = None,
    quant_bull_pct: float = 33.3,
    quant_bear_pct: float = 33.3,
    quant_neutral_pct: float = 33.4,
    model: str = "gemini-3.8-flash",
    thinking_level: str = "HIGH",
    timeout_sec: float = 35.0,
) -> MacroNewsReport:
    """Call Google Gemini API via lightweight HTTP REST to extract macro & sector sentiment."""
    news_text = _build_news_text(articles)
    prompt = DEFAULT_PROMPT_TEMPLATE.format(
        asof_date=asof_date,
        num_articles=len(articles[:50]),
        news_text=news_text if news_text else "(No recent news collected)",
        target_tickers=", ".join(target_tickers or []) or "(None)",
        quant_bull_pct=round(quant_bull_pct, 1),
        quant_bear_pct=round(quant_bear_pct, 1),
        quant_neutral_pct=round(quant_neutral_pct, 1),
    )

    models_to_try = [model]
    for alt in ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-flash-latest"]:
        if alt not in models_to_try:
            models_to_try.append(alt)

    generation_config: dict[str, Any] = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
    }
    if thinking_level:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for cur_model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code == 200:
                resp_json = resp.json()
                candidates = resp_json.get("candidates") or []
                if candidates:
                    part = candidates[0].get("content", {}).get("parts", [{}])[0]
                    raw_text = part.get("text", "")
                    data = json.loads(raw_text)
                    return _parse_llm_json_to_report(data, asof_date=asof_date, provider="gemini", model=cur_model, articles_count=len(articles))
            last_error = RuntimeError(f"Gemini API ({cur_model}) status {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            last_error = e

    if last_error:
        raise last_error
    raise RuntimeError("All Gemini models failed")


def analyze_news_with_openai(
    *,
    articles: list[NewsArticle],
    asof_date: str,
    api_key: str,
    target_tickers: list[str] | None = None,
    quant_bull_pct: float = 33.3,
    quant_bear_pct: float = 33.3,
    quant_neutral_pct: float = 33.4,
    model: str = "gpt-4o-mini",
    timeout_sec: float = 30.0,
) -> MacroNewsReport:
    """Call OpenAI Chat Completions API via HTTP REST."""
    news_text = _build_news_text(articles)
    prompt = DEFAULT_PROMPT_TEMPLATE.format(
        asof_date=asof_date,
        num_articles=len(articles[:50]),
        news_text=news_text if news_text else "(No recent news collected)",
        target_tickers=", ".join(target_tickers or []) or "(None)",
        quant_bull_pct=round(quant_bull_pct, 1),
        quant_bear_pct=round(quant_bear_pct, 1),
        quant_neutral_pct=round(quant_neutral_pct, 1),
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a quantitative macro and equity analyst. Respond in strict JSON."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API returned status {resp.status_code}: {resp.text[:300]}")

    resp_json = resp.json()
    choices = resp_json.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI API returned empty choices")

    raw_text = choices[0].get("message", {}).get("content", "")
    data = json.loads(raw_text)

    return _parse_llm_json_to_report(data, asof_date=asof_date, provider="openai", model=model, articles_count=len(articles))


def _parse_llm_json_to_report(
    data: dict[str, Any],
    *,
    asof_date: str,
    provider: str,
    model: str,
    articles_count: int,
) -> MacroNewsReport:
    m_score = float(data.get("macro_score", 0.0))
    m_regime = str(data.get("macro_regime", "Neutral"))
    event_risk = str(data.get("event_risk_level", "LOW")).upper()
    if event_risk not in {"LOW", "MEDIUM", "HIGH"}:
        event_risk = "LOW"
    event_note = data.get("event_risk_note")

    drivers = [str(d) for d in data.get("key_drivers_ja", []) if d]
    if not drivers:
        drivers = ["特段の突発的マクロニュースは観測されず、通常取引環境が継続"]

    sec_map: dict[str, SectorSentiment] = {}
    for s_name, s_data in (data.get("sector_scores") or {}).items():
        if isinstance(s_data, dict):
            sec_map[s_name] = SectorSentiment(
                score=float(s_data.get("score", 0.0)),
                reason_ja=str(s_data.get("reason_ja", "")),
            )
        elif isinstance(s_data, (int, float)):
            sec_map[s_name] = SectorSentiment(score=float(s_data))

    # Fill default GICS sectors if omitted
    for gs in GICS_SECTORS:
        if gs not in sec_map:
            sec_map[gs] = SectorSentiment(score=0.0, reason_ja="中立")

    tk_map: dict[str, TickerNewsImpact] = {}
    for t_name, t_data in (data.get("ticker_impacts") or {}).items():
        if isinstance(t_data, dict):
            tk_map[t_name.upper()] = TickerNewsImpact(
                score=float(t_data.get("score", 0.0)),
                catalyst_type=str(t_data.get("catalyst_type", "General")),
                summary_ja=str(t_data.get("summary_ja", "")),
                is_adverse_alert=bool(t_data.get("is_adverse_alert", False)),
                headlines=[],
            )

    return MacroNewsReport(
        asof_date=asof_date,
        macro_score=max(-1.0, min(1.0, m_score)),
        macro_regime=m_regime,
        event_risk_level=event_risk,
        event_risk_note=event_note,
        key_drivers_ja=drivers[:3],
        sector_scores=sec_map,
        ticker_impacts=tk_map,
        articles_analyzed_count=articles_count,
        provider=provider,
        model=model,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def generate_fallback_macro_report(asof_date: str) -> MacroNewsReport:
    """Return a neutral fallback report when LLM keys or network are unavailable."""
    sec_map = {s: SectorSentiment(score=0.0, reason_ja="中立 (時系列モデル準拠)") for s in GICS_SECTORS}
    return MacroNewsReport(
        asof_date=asof_date,
        macro_score=0.0,
        macro_regime="Neutral (Quant Fallback)",
        event_risk_level="LOW",
        event_risk_note=None,
        key_drivers_ja=[
            "ニュース分析エンジン未設定または通信スキップのため、純粋な時系列数値モデル（Chronos-2 / TimesFM / TiRex-2）のみで判定しています。"
        ],
        sector_scores=sec_map,
        ticker_impacts={},
        articles_analyzed_count=0,
        provider="fallback",
        model="none",
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def analyze_news_report(
    *,
    articles: list[NewsArticle],
    asof_date: str,
    provider: str = "gemini",
    model: str = "gemini-3.8-flash",
    thinking_level: str = "HIGH",
    target_tickers: list[str] | None = None,
    quant_bull_pct: float = 33.3,
    quant_bear_pct: float = 33.3,
    quant_neutral_pct: float = 33.4,
) -> MacroNewsReport:
    """Orchestrate news analysis across providers with safe fallback."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        try:
            from alphaignitor.common._credentials import get_gemini_api_key
            gemini_key = get_gemini_api_key()
        except Exception:
            gemini_key = None
    openai_key = os.environ.get("OPENAI_API_KEY")

    prov = provider.lower().strip()
    try:
        if prov == "gemini" and gemini_key:
            return analyze_news_with_gemini(
                articles=articles,
                asof_date=asof_date,
                api_key=gemini_key,
                target_tickers=target_tickers,
                quant_bull_pct=quant_bull_pct,
                quant_bear_pct=quant_bear_pct,
                quant_neutral_pct=quant_neutral_pct,
                model=model,
                thinking_level=thinking_level,
            )
        elif prov == "openai" and openai_key:
            return analyze_news_with_openai(
                articles=articles,
                asof_date=asof_date,
                api_key=openai_key,
                target_tickers=target_tickers,
                quant_bull_pct=quant_bull_pct,
                quant_bear_pct=quant_bear_pct,
                quant_neutral_pct=quant_neutral_pct,
                model=model if "gpt" in model else "gpt-4o-mini",
            )
        elif gemini_key:
            return analyze_news_with_gemini(
                articles=articles,
                asof_date=asof_date,
                api_key=gemini_key,
                target_tickers=target_tickers,
                quant_bull_pct=quant_bull_pct,
                quant_bear_pct=quant_bear_pct,
                quant_neutral_pct=quant_neutral_pct,
                model="gemini-3.8-flash",
                thinking_level=thinking_level,
            )
        elif openai_key:
            return analyze_news_with_openai(
                articles=articles,
                asof_date=asof_date,
                api_key=openai_key,
                target_tickers=target_tickers,
                quant_bull_pct=quant_bull_pct,
                quant_bear_pct=quant_bear_pct,
                quant_neutral_pct=quant_neutral_pct,
                model="gpt-4o-mini",
            )
    except Exception as e:
        logger.warning("LLM news analysis failed; falling back to keyword analysis: %s", e)
        print(f"[news_analyzer] Warning: LLM analysis failed ({e}). Using quant fallback.")

    return generate_fallback_macro_report(asof_date=asof_date)

