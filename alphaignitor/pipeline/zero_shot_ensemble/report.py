from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alphaignitor.pipeline.report_daily_forecast._chart import _svg_line_chart
from alphaignitor.pipeline.report_daily_forecast._data_loader import _load_ticker_meta, read_day_aggs_by_date
from alphaignitor.pipeline.report_daily_forecast._html import render_html

from .metrics import fmt_float, fmt_pct, metric_summary
from .market_data import available_trade_dates, load_price_panel, parse_iso_date, ticker_series_map
from .schema import METRIC_NAMES, METRIC_TARGETS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate zero-shot ensemble forecast HTML report")
    p.add_argument("--forecast", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("report"))
    p.add_argument("--asof-date", type=str, default=None)
    p.add_argument("--ticker-meta-csv", type=Path, default=Path("us_stock_list.csv"))
    return p.parse_args()


def run_report(
    *,
    forecast_path: Path | None = None,
    outdir: Path = Path("report"),
    asof_date: str | None = None,
    ticker_meta_csv: Path | None = None,
) -> Path:
    f_path = Path(forecast_path) if forecast_path else _latest_forecast_path(Path("predict"))
    df = pd.read_parquet(f_path)
    validate_forecast(df)
    max_horizon = _max_horizon(df)

    asof = asof_date or str(df["asof_trade_date"].iloc[0])
    meta_csv = Path(ticker_meta_csv) if ticker_meta_csv else Path("us_stock_list.csv")
    table = build_report_table(df, ticker_meta_csv=meta_csv, max_horizon=max_horizon)
    summary_dl = build_summary_dl(df=df, table=table, asof=asof, forecast_path=f_path, max_horizon=max_horizon)
    acc_html = build_accuracy_html(predict_dir=f_path.parent, max_horizon=max_horizon)
    charts_html = build_charts_html(df=df, table=table, max_horizon=max_horizon)
    cols = _report_columns(max_horizon)
    outdir_path = Path(outdir) / asof
    outdir_path.mkdir(parents=True, exist_ok=True)
    html_path = outdir_path / "report.html"
    html_path.write_text(
        render_report_html(
            table=table,
            cols=cols,
            summary_dl=summary_dl,
            acc_html=acc_html,
            charts_html=charts_html,
            max_horizon=max_horizon,
        ),
        encoding="utf-8",
    )
    print(f"[report] saved: {html_path}")
    return html_path


def main() -> int:
    args = parse_args()
    run_report(
        forecast_path=args.forecast,
        outdir=args.outdir,
        asof_date=args.asof_date,
        ticker_meta_csv=args.ticker_meta_csv,
    )
    return 0


def validate_forecast(df: pd.DataFrame) -> None:
    required = {
        "ticker",
        "asof_trade_date",
        "horizon",
        "forecast_trade_date",
        "asof_close",
        "ensemble_pred",
        "ensemble_return",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"forecast parquet is missing required columns: {missing}")


def _max_horizon(df: pd.DataFrame) -> int:
    work = pd.to_numeric(df.get("horizon"), errors="coerce").dropna()
    if work.empty:
        return 1
    return max(1, int(work.max()))


def _report_columns(max_horizon: int) -> list[tuple[str, str]]:
    cols = [
        ("Name", "name"),
        ("Sector", "sector"),
        ("Signal", "signal"),
        ("🟢", "bull"),
        ("🔴", "bear"),
    ]
    for horizon in range(1, max_horizon + 1):
        cols.append((f"Day {horizon}", f"day{horizon}"))
    cols.append(("Avg", "avg"))
    return cols


def build_report_table(df: pd.DataFrame, *, ticker_meta_csv: Path, max_horizon: int) -> pd.DataFrame:
    work = df.copy()
    work["ticker"] = work["ticker"].astype(str)
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce").astype("Int64")
    work["ensemble_return"] = pd.to_numeric(work["ensemble_return"], errors="coerce")
    work["asof_close"] = pd.to_numeric(work["asof_close"], errors="coerce")

    meta = _load_ticker_meta(ticker_meta_csv)
    rows: list[dict] = []
    for ticker, g in work.groupby("ticker", sort=True):
        g2 = g.sort_values("horizon")
        row_by_h: dict[int, pd.Series] = {}
        for row in g2.itertuples(index=False):
            h = int(getattr(row, "horizon")) if pd.notna(getattr(row, "horizon")) else None
            if h is not None:
                row_by_h[h] = pd.Series(row._asdict())
        if not row_by_h:
            continue

        meta_row = meta.get(str(ticker), {})
        horizons = list(range(1, max_horizon + 1))
        bull, bear, sig_text, sig_payload = _signal_for_ticker(row_by_h, horizons=horizons)
        row_out = {
            "ticker": str(ticker),
            "name": meta_row.get("name") or meta_row.get("company_name") or str(ticker),
            "sector": meta_row.get("sector", ""),
            "signal": sig_text,
            "bull": bull,
            "bear": bear,
            "avg": _format_avg_return([row_by_h.get(h) for h in horizons]),
            "sort_avg": _avg_return_value([row_by_h.get(h) for h in horizons]),
            "signals_json": sig_payload,
        }
        for horizon in horizons:
            row_out[f"day{horizon}"] = _format_day_return(row_by_h.get(horizon))
        rows.append(row_out)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["bull", "sort_avg", "ticker"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def render_report_html(
    *,
    table: pd.DataFrame,
    cols: list[tuple[str, str]],
    summary_dl: str,
    acc_html: str,
    charts_html: str,
    max_horizon: int,
) -> str:
    return render_html(
        summary_dl=summary_dl,
        acc_html=acc_html,
        all_tbl=table,
        cols=cols,
        charts_html_block=charts_html,
        max_horizon=max_horizon,
    )


def build_summary_dl(*, df: pd.DataFrame, table: pd.DataFrame, asof: str, forecast_path: Path, max_horizon: int) -> str:
    work = df.copy()
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce")
    work["forecast_trade_date"] = pd.to_datetime(work["forecast_trade_date"], errors="coerce")
    max_h = int(work["horizon"].dropna().max()) if not work["horizon"].dropna().empty else 3
    tgt = (
        work.loc[work["horizon"] == max_h, "forecast_trade_date"].dropna().max()
        if "forecast_trade_date" in work.columns
        else pd.NaT
    )
    target_date = str(tgt.date()) if pd.notna(tgt) else ""
    return (
        f"<dt>As-Of</dt><dd>{html.escape(asof)}</dd>"
        f"<dt>Target Date</dt><dd>{html.escape(target_date)}</dd>"
        f"<dt>Source File</dt><dd>{html.escape(str(forecast_path))}</dd>"
        f"<dt>Tickers</dt><dd>{len(table)}</dd>"
        "<dt>Forecast Mode</dt><dd>Zero-shot ensemble (Chronos2/TimesFM/TiREX)</dd>"
        f"<dt>Ranking Basis</dt><dd>Avg of Day 1-{max_horizon} ensemble returns</dd>"
    )


def build_accuracy_html(*, predict_dir: Path, max_horizon: int) -> str:
    horizons = list(range(1, max_horizon + 1))
    rows = _collect_recent_accuracy_rows(predict_dir=predict_dir, max_days=5, horizons=horizons)
    if not rows:
        return "<div class=\"acc-panel\"><h2>Directional Accuracy (Last 5 Trading Days)</h2><p class=\"note\">No historical forecast files with actuals were found.</p></div>"

    body = []
    totals = {h: {"hit": 0, "valid": 0} for h in horizons}
    total_valid_count = 0
    total_count = 0
    for r in rows:
        total_valid_count += r["valid_total"]
        total_count += r["total_total"]
        cells = []
        for horizon in horizons:
            hit = r["by_h"][horizon]["hit"]
            valid = r["by_h"][horizon]["valid"]
            cells.append(f"<td>{_fmt_acc_cell(hit, valid)}</td>")
            totals[horizon]["hit"] += hit
            totals[horizon]["valid"] += valid
        body.append(
            "<tr>"
            f"<td>{html.escape(r['asof'])}</td>"
            f"<td>{r['valid_total']}/{r['total_total']}</td>"
            f"{''.join(cells)}"
            "</tr>"
        )

    head_cols = ''.join(f"<th>Hit Day {h}</th>" for h in horizons)
    foot_cols = ''.join(
        f"<td>{_fmt_acc_cell(totals[h]["hit"], totals[h]["valid"] )}</td>" for h in horizons
    )
    return (
        "<div class=\"acc-panel\">"
        "<h2>Directional Accuracy (Last 5 Trading Days)</h2>"
        "<table class=\"acc-table\">"
        f"<thead><tr><th>As-Of Date</th><th>Valid / Total</th>{head_cols}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "<tfoot><tr>"
        "<td>Total (Last 5 Days)</td>"
        f"<td>{total_valid_count}/{total_count}</td>"
        f"{foot_cols}"
        "</tr></tfoot>"
        "</table>"
        "</div>"
    )


def build_charts_html(*, df: pd.DataFrame, table: pd.DataFrame, max_horizon: int) -> str:
    if table.empty:
        return '<p id="no-charts-msg" class="note">No chart data.</p>'

    work = df.copy()
    work["ticker"] = work["ticker"].astype(str)
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce").astype("Int64")
    work["q0.1"] = pd.to_numeric(work.get("q0.1", np.nan), errors="coerce")
    work["q0.5"] = pd.to_numeric(work.get("q0.5", np.nan), errors="coerce")
    work["q0.9"] = pd.to_numeric(work.get("q0.9", np.nan), errors="coerce")
    work["ensemble_return"] = pd.to_numeric(work.get("ensemble_return", np.nan), errors="coerce")
    work["ensemble_pred"] = pd.to_numeric(work["ensemble_pred"], errors="coerce")
    work["asof_close"] = pd.to_numeric(work["asof_close"], errors="coerce")
    work["forecast_trade_date"] = pd.to_datetime(work["forecast_trade_date"], errors="coerce")
    work["asof_trade_date"] = pd.to_datetime(work["asof_trade_date"], errors="coerce")

    info = {}
    for row in table.itertuples(index=False):
        info[str(row.ticker)] = {
            "name": str(row.name),
            "sector": str(row.sector),
            "signals_json": str(row.signals_json),
        }

    asof_date = _infer_asof_date(work)
    tickers = [str(t) for t in table["ticker"].astype(str).tolist()]
    day_root = Path("aggs/us_stock_day")
    history_map: dict[str, pd.DataFrame] = {}
    open_map_by_date: dict[str, dict[str, float]] = {}

    if day_root.exists() and asof_date is not None and tickers:
        try:
            all_dates = available_trade_dates(day_root, end_date=asof_date)
            needed = all_dates[-300:] if len(all_dates) > 300 else all_dates
            panel = load_price_panel(day_root, dates=needed, tickers=tickers)
            history_map = ticker_series_map(panel)
        except Exception:
            history_map = {}

        try:
            f_dates = (
                work["forecast_trade_date"].dropna().dt.date.drop_duplicates().sort_values().tolist()
            )
            ticker_set = set(tickers)
            for d in f_dates:
                d_iso = d.isoformat()
                day_df = read_day_aggs_by_date(day_root=day_root, trade_date=d, tickers=ticker_set, need_open=True)
                if day_df.empty or "open" not in day_df.columns:
                    open_map_by_date[d_iso] = {}
                    continue
                open_map_by_date[d_iso] = {
                    str(r.ticker): float(r.open)
                    for r in day_df.itertuples(index=False)
                    if pd.notna(r.open)
                }
        except Exception:
            open_map_by_date = {}

    blocks = ['<p id="no-charts-msg" class="note">Select ticker rows to show charts.</p>']
    for ticker, g in work.groupby("ticker", sort=True):
        g2 = g.sort_values("horizon")
        if g2.empty:
            continue
        rec = info.get(str(ticker), {"name": str(ticker), "sector": "", "signals_json": "{}"})
        try:
            asof_close = float(g2["asof_close"].dropna().iloc[0])
        except Exception:
            asof_close = float("nan")
        preds = []
        q10s = []
        q50s = []
        q90s = []
        fc_dates: list[dt.date | None] = []
        fc_opens: list[float | None] = []
        for h in range(1, max_horizon + 1):
            row_h = g2[g2["horizon"] == h]
            if row_h.empty:
                preds.append(None)
                q10s.append(None)
                q50s.append(None)
                q90s.append(None)
                fc_dates.append(None)
                fc_opens.append(None)
                continue
            row = row_h.iloc[0]
            v = pd.to_numeric(row_h["ensemble_pred"], errors="coerce").dropna()
            pred = float(v.iloc[0]) if not v.empty else None
            preds.append(pred)

            q10 = _safe_qlogret(row, "q0.1")
            q50 = _safe_qlogret(row, "q0.5")
            q90 = _safe_qlogret(row, "q0.9")
            if q10 is None or q50 is None or q90 is None:
                qv = _safe_logret_from_prices(asof_close, pred)
                q10 = q10 if q10 is not None else qv
                q50 = q50 if q50 is not None else qv
                q90 = q90 if q90 is not None else qv
            q10s.append(q10)
            q50s.append(q50)
            q90s.append(q90)

            fdt = row.get("forecast_trade_date")
            if pd.notna(fdt):
                fdate = pd.Timestamp(fdt).date()
                fc_dates.append(fdate)
                fc_opens.append(open_map_by_date.get(fdate.isoformat(), {}).get(str(ticker)))
            else:
                fc_dates.append(None)
                fc_opens.append(None)

        history = history_map.get(str(ticker))

        svg = _build_technical_svg(
            ticker=ticker,
            asof_close=asof_close,
            preds=preds,
            history=history,
            asof_date=asof_date,
            forecast_dates=fc_dates,
            q10s=q10s,
            q50s=q50s,
            q90s=q90s,
            forecast_opens=fc_opens,
        )
        title = f"{rec['name']}, {rec['sector']}, {ticker}" if rec["sector"] else f"{rec['name']}, {ticker}"
        signals_attr = html.escape(rec["signals_json"], quote=True)
        blocks.append(
            f'<div class="chart-block" data-ticker="{html.escape(str(ticker))}" data-signals="{signals_attr}" style="display:none">'
            f"<h3>{html.escape(title)}</h3>"
            f"{svg}"
            "</div>"
        )
    return "".join(blocks)


def _infer_asof_date(df: pd.DataFrame) -> dt.date | None:
    if "asof_trade_date" not in df.columns:
        return None
    s = pd.to_datetime(df["asof_trade_date"], errors="coerce").dropna()
    if s.empty:
        return None
    return pd.Timestamp(s.max()).date()


def _safe_qlogret(row: pd.Series, col: str) -> float | None:
    try:
        v = float(row.get(col, np.nan))
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_logret_from_prices(asof_close: float, pred_close: float | None) -> float | None:
    try:
        if pred_close is None:
            return None
        a = float(asof_close)
        p = float(pred_close)
    except Exception:
        return None
    if not (math.isfinite(a) and math.isfinite(p)):
        return None
    if a <= 0 or p <= 0:
        return None
    return math.log(p / a)


def _safe_id_prefix(ticker: str) -> str:
    chars = []
    for ch in str(ticker):
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars) + "-"


def _build_technical_svg(
    *,
    ticker: str,
    asof_close: float,
    preds: list[float | None],
    history: pd.DataFrame | None,
    asof_date: dt.date | None,
    forecast_dates: list[dt.date | None],
    q10s: list[float | None],
    q50s: list[float | None],
    q90s: list[float | None],
    forecast_opens: list[float | None],
) -> str:
    if history is None or history.empty or asof_date is None:
        return _mini_forecast_svg(asof_close=asof_close, preds=preds)

    try:
        h = history.copy()
        h["trade_date"] = pd.to_datetime(h["trade_date"], errors="coerce")
        h["close"] = pd.to_numeric(h["close"], errors="coerce")
        h["volume"] = pd.to_numeric(h.get("volume", np.nan), errors="coerce")
        h = h.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
        h = h[h["trade_date"].dt.date <= asof_date]
        if len(h) < 3:
            return _mini_forecast_svg(asof_close=asof_close, preds=preds)

        total_keep = 260
        display_keep = 140
        h_tail = h.tail(total_keep)
        closes_all = h_tail["close"].astype(float).tolist()
        dates_all = [pd.Timestamp(d).date() for d in h_tail["trade_date"].tolist()]
        vols_all = h_tail["volume"].astype(float).tolist() if "volume" in h_tail.columns else [float("nan")] * len(h_tail)

        # Match sample report display range (~70 trading days)
        display_keep = 70
        if len(closes_all) <= display_keep:
            dates = dates_all
            closes = closes_all
            vols = vols_all
            warmup = None
        else:
            split = len(closes_all) - display_keep
            dates = dates_all[split:]
            closes = closes_all[split:]
            vols = vols_all[split:]
            warmup = closes_all[:split]

        if len(dates) < 3:
            return _mini_forecast_svg(asof_close=asof_close, preds=preds)

        return _svg_line_chart(
            dates=dates,
            closes=closes,
            volumes=vols,
            asof=asof_date,
            forecast_dates=forecast_dates,
            q10s=q10s,
            q50s=q50s,
            q90s=q90s,
            forecast_opens=forecast_opens,
            past_forecast_dates=None,
            past_forecast_q50_closes=None,
            past_forecast_q10_closes=None,
            past_forecast_q90_closes=None,
            warmup_closes=warmup,
            id_prefix=_safe_id_prefix(ticker),
        )
    except Exception:
        return _mini_forecast_svg(asof_close=asof_close, preds=preds)


def _collect_recent_accuracy_rows(*, predict_dir: Path, max_days: int, horizons: list[int]) -> list[dict]:
    files = sorted(Path(predict_dir).glob("*_ensemble_forecast.parquet"))
    if not files:
        return []

    records = []
    for path in files[-20:]:
        try:
            part = pd.read_parquet(
                path,
                columns=["asof_trade_date", "horizon", "ensemble_direction", "actual_direction"],
            )
        except Exception:
            continue
        if part.empty:
            continue

        part["asof_trade_date"] = pd.to_datetime(part["asof_trade_date"], errors="coerce")
        part["horizon"] = pd.to_numeric(part["horizon"], errors="coerce").astype("Int64")
        part["ensemble_direction"] = pd.to_numeric(part["ensemble_direction"], errors="coerce")
        part["actual_direction"] = pd.to_numeric(part["actual_direction"], errors="coerce")
        part = part.dropna(subset=["asof_trade_date", "horizon"])
        if part.empty:
            continue

        asof = str(part["asof_trade_date"].max().date())
        by_h = {}
        valid_total = 0
        total_total = 0
        for h in horizons:
            s = part[part["horizon"] == h]
            valid = s[
                s["ensemble_direction"].isin([-1, 1])
                & s["actual_direction"].isin([-1, 1])
            ]
            hit = int((valid["ensemble_direction"] == valid["actual_direction"]).sum())
            valid_count = int(len(valid))
            total_count = int(len(s))
            by_h[h] = {"hit": hit, "valid": valid_count, "total": total_count}
            valid_total += valid_count
            total_total += total_count
        records.append(
            {
                "asof": asof,
                "by_h": by_h,
                "valid_total": int(valid_total),
                "total_total": int(total_total),
            }
        )

    records.sort(key=lambda x: x["asof"])
    return records[-max_days:]


def _fmt_acc_cell(hit: int, valid: int) -> str:
    if valid <= 0:
        return "N/A"
    rate = hit / valid
    cls = "acc-rate-hi" if rate >= 0.5 else "acc-rate-lo"
    return f'<span class="{cls}">{rate * 100:.1f}%</span>'


def _signal_for_ticker(row_by_h: dict[int, pd.Series], *, horizons: list[int]) -> tuple[int, int, str, str]:
    bull = 0
    bear = 0
    items = []
    model_labels = [("chronos2_pred", "Chronos2"), ("timesfm_pred", "TimesFM"), ("tirex_pred", "TiREX")]
    for col, name in model_labels:
        b = 0
        r = 0
        votes_detail = []  # Track individual horizon votes for detailed label
        for h in horizons:
            s = row_by_h.get(h)
            if s is None:
                continue
            try:
                pred = float(s.get(col, np.nan))
                base = float(s.get("asof_close", np.nan))
            except Exception:
                continue
            if not (math.isfinite(pred) and math.isfinite(base)):
                continue
            if pred > base:
                b += 1
                votes_detail.append(f"D{h}↑")
            elif pred < base:
                r += 1
                votes_detail.append(f"D{h}↓")
        bull += b
        bear += r
        if b + r > 0:
            state = True if b > r else (False if r > b else None)
            detail_str = " ".join(votes_detail) if votes_detail else ""
            label = f"{name}: Bull {b} / Bear {r} ({detail_str})"
            items.append({"label": label, "category": "Model Vote", "bullish": state})

    if bull >= bear + 2:
        verdict = "bullish"
        text = "🟢 Bull"
    elif bear >= bull + 2:
        verdict = "bearish"
        text = "🔴 Bear"
    else:
        verdict = "neutral"
        text = "🟡 Neutral"
    payload = json.dumps({"items": items, "bull": bull, "bear": bear, "verdict": verdict}, ensure_ascii=True)
    return bull, bear, text, payload


def _format_day_return(row: pd.Series | None) -> str:
    if row is None:
        return ""
    try:
        r = float(row.get("ensemble_return", np.nan))
    except Exception:
        return ""
    if not math.isfinite(r):
        return ""
    return f"{r * 100:+.2f}%"


def _avg_return_value(rows: list[pd.Series | None]) -> float:
    vals = []
    for row in rows:
        if row is None:
            continue
        try:
            v = float(row.get("ensemble_return", np.nan))
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _format_avg_return(rows: list[pd.Series | None]) -> str:
    v = _avg_return_value(rows)
    if not math.isfinite(v):
        return ""
    return f"{v * 100:+.2f}%"


def _mini_forecast_svg(*, asof_close: float, preds: list[float | None]) -> str:
    width = 560
    height = 220
    ml, mr, mt, mb = 42, 20, 16, 26
    iw = width - ml - mr
    ih = height - mt - mb

    values = []
    if math.isfinite(asof_close):
        values.append(float(asof_close))
    for p in preds:
        if p is not None and math.isfinite(float(p)):
            values.append(float(p))
    if len(values) < 2:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="560" height="220" viewBox="0 0 560 220"><text x="16" y="24" font-size="12">No chart data</text></svg>'

    ymin = min(values)
    ymax = max(values)
    if ymax <= ymin:
        ymin -= 1.0
        ymax += 1.0
    pad = (ymax - ymin) * 0.15
    ymin -= pad
    ymax += pad

    pts = [asof_close] + [float(p) if p is not None and math.isfinite(float(p)) else float("nan") for p in preds]
    n = len(pts)

    def x(i: int) -> float:
        if n <= 1:
            return float(ml)
        return float(ml + iw * (i / (n - 1)))

    def y(v: float) -> float:
        return float(mt + ih * (1.0 - (v - ymin) / (ymax - ymin)))

    segs = []
    circles = []
    for i, v in enumerate(pts):
        if not math.isfinite(v):
            continue
        xi = x(i)
        yi = y(v)
        circles.append(f'<circle cx="{xi:.2f}" cy="{yi:.2f}" r="3" fill="#0b5"/>')
        if i > 0 and math.isfinite(pts[i - 1]):
            x0 = x(i - 1)
            y0 = y(pts[i - 1])
            segs.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{xi:.2f}" y2="{yi:.2f}" stroke="#1565c0" stroke-width="2"/>')

    labels = ["AsOf"] + [f"D{i}" for i in range(1, n)]
    xlabels = []
    for i, t in enumerate(labels[:n]):
        xlabels.append(f'<text x="{x(i):.2f}" y="{height - 8}" text-anchor="middle" font-size="10" fill="#555">{t}</text>')

    ytop = fmt_float(ymax, 2)
    ybot = fmt_float(ymin, 2)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="560" height="220" viewBox="0 0 560 220">'
        f'<rect x="{ml}" y="{mt}" width="{iw}" height="{ih}" fill="#fff" stroke="#ddd"/>'
        f"{''.join(segs)}"
        f"{''.join(circles)}"
        f"{''.join(xlabels)}"
        f'<text x="8" y="{mt + 10}" font-size="10" fill="#666">{html.escape(str(ytop))}</text>'
        f'<text x="8" y="{mt + ih}" font-size="10" fill="#666">{html.escape(str(ybot))}</text>'
        "</svg>"
    )


def _metrics_by_ticker(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ticker, group in df.groupby("ticker"):
        out[str(ticker)] = {}
        for model, col in [
            ("chronos2", "chronos2_pred"),
            ("timesfm", "timesfm_pred"),
            ("tirex", "tirex_pred"),
            ("ensemble", "ensemble_pred"),
        ]:
            stored = _stored_metric_summary(group, target=model)
            if stored:
                out[str(ticker)][model] = stored
            elif col in group.columns:
                out[str(ticker)][model] = metric_summary(group, pred_col=col)
    return out


def _stored_metric_summary(group: pd.DataFrame, *, target: str) -> dict[str, float]:
    values: dict[str, float] = {}
    if target not in METRIC_TARGETS:
        return values
    for metric in METRIC_NAMES:
        col = f"{target}_{metric}"
        if col not in group.columns:
            return {}
        series = pd.to_numeric(group[col], errors="coerce").dropna()
        if series.empty:
            return {}
        values[metric] = float(series.iloc[0])
    return values


def _fmt_optional(value: object, digits: int = 2) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if not math.isfinite(v):
        return ""
    return fmt_float(v, digits)


def _fmt_metric(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if not math.isfinite(v):
        return ""
    return f"{v * 100:.1f}%"


def _fmt_direction(value: object) -> str:
    try:
        v = int(value)
    except Exception:
        return ""
    if v > 0:
        return "UP"
    if v < 0:
        return "DOWN"
    return ""


def _fmt_weights(row) -> str:
    parts = []
    for label, attr in [("C2", "weight_chronos2"), ("TF", "weight_timesfm"), ("TR", "weight_tirex")]:
        try:
            v = float(getattr(row, attr))
        except Exception:
            continue
        if math.isfinite(v):
            parts.append(f"{label}:{v:.2f}")
    return " ".join(parts)


def _latest_forecast_path(predict_dir: Path) -> Path:
    candidates = sorted(Path(predict_dir).glob("*_ensemble_forecast.parquet"))
    if not candidates:
        raise FileNotFoundError("predict に *_ensemble_forecast.parquet が見当たりません")
    return candidates[-1]


if __name__ == "__main__":
    raise SystemExit(main())
