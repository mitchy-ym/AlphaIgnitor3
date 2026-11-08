from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NewsArticle:
    title: str
    url: str
    published_at: str
    source: str
    summary: str = ""
    tickers: list[str] = field(default_factory=list)


@dataclass
class SectorSentiment:
    score: float  # -1.0 to +1.0
    reason_ja: str = ""


@dataclass
class TickerNewsImpact:
    score: float  # -1.0 to +1.0
    catalyst_type: str = "General"  # Earnings, Macro, Policy, Guidance, Regulatory
    summary_ja: str = ""
    is_adverse_alert: bool = False  # True for sudden severe bad news
    headlines: list[str] = field(default_factory=list)


@dataclass
class MacroNewsReport:
    asof_date: str
    macro_score: float = 0.0  # -1.0 (Extreme Bear) to +1.0 (Extreme Bull)
    macro_regime: str = "Neutral"  # Bullish, Moderately Bullish, Neutral, Moderately Bearish, Bearish
    event_risk_level: str = "LOW"  # LOW, MEDIUM, HIGH
    event_risk_note: str | None = None
    key_drivers_ja: list[str] = field(default_factory=list)
    sector_scores: dict[str, SectorSentiment] = field(default_factory=dict)
    ticker_impacts: dict[str, TickerNewsImpact] = field(default_factory=dict)
    articles_analyzed_count: int = 0
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    generated_at: str = ""

    def get_composite_ticker_score(
        self,
        ticker: str,
        sector: str | None = None,
        *,
        macro_weight: float = 0.35,
        sector_weight: float = 0.35,
        ticker_weight: float = 0.30,
    ) -> float:
        """Compute composite news score for a specific ticker (-1.0 to +1.0)."""
        m_score = float(self.macro_score)
        sec_sentiment = self.sector_scores.get(sector or "")
        s_score = float(sec_sentiment.score) if sec_sentiment else 0.0

        t_impact = self.ticker_impacts.get(ticker)
        if t_impact is not None:
            t_score = float(t_impact.score)
            composite = (macro_weight * m_score) + (sector_weight * s_score) + (ticker_weight * t_score)
        else:
            denom = macro_weight + sector_weight
            composite = ((macro_weight * m_score) + (sector_weight * s_score)) / denom if denom > 0 else m_score

        return max(-1.0, min(1.0, composite))

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof_date": self.asof_date,
            "macro_score": round(self.macro_score, 4),
            "macro_regime": self.macro_regime,
            "event_risk_level": self.event_risk_level,
            "event_risk_note": self.event_risk_note,
            "key_drivers_ja": self.key_drivers_ja,
            "sector_scores": {k: asdict(v) for k, v in self.sector_scores.items()},
            "ticker_impacts": {k: asdict(v) for k, v in self.ticker_impacts.items()},
            "articles_analyzed_count": self.articles_analyzed_count,
            "provider": self.provider,
            "model": self.model,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MacroNewsReport:
        sec_scores = {
            k: SectorSentiment(**v) if isinstance(v, dict) else SectorSentiment(score=float(v))
            for k, v in d.get("sector_scores", {}).items()
        }
        tk_impacts = {
            k: TickerNewsImpact(**v) if isinstance(v, dict) else TickerNewsImpact(score=float(v))
            for k, v in d.get("ticker_impacts", {}).items()
        }
        return cls(
            asof_date=str(d.get("asof_date", "")),
            macro_score=float(d.get("macro_score", 0.0)),
            macro_regime=str(d.get("macro_regime", "Neutral")),
            event_risk_level=str(d.get("event_risk_level", "LOW")),
            event_risk_note=d.get("event_risk_note"),
            key_drivers_ja=list(d.get("key_drivers_ja", [])),
            sector_scores=sec_scores,
            ticker_impacts=tk_impacts,
            articles_analyzed_count=int(d.get("articles_analyzed_count", 0)),
            provider=str(d.get("provider", "gemini")),
            model=str(d.get("model", "gemini-2.5-flash")),
            generated_at=str(d.get("generated_at", "")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> MacroNewsReport:
        return cls.from_dict(json.loads(s))

