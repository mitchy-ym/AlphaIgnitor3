"""HTMLレンダリングモジュール（エグゼクティブ・コックピット、テーブル、チャート、ページ全体）。"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _load_asset(filename: str) -> str:
    return (_ASSETS_DIR / filename).read_text(encoding="utf-8")

import numpy as np
import pandas as pd


def _render_table(
    df: pd.DataFrame,
    cols: list[tuple[str, str]],
    *,
    table_id: str | None = None,
    sortable: bool = False,
    checkbox_col: bool = False,
) -> str:
    """DataFrameをHTMLテーブルに変換する。"""
    if df.empty:
        return '<p class="note">(no rows)</p>'

    ths: list[str] = []
    if checkbox_col:
        ths.append('<th><input type="checkbox" id="check-all-master" title="Select / Deselect All"></th>')
    for label, _key in cols:
        is_day_col = label.startswith("Day ")
        if sortable:
            if label in {"Sector", "Signal"}:
                filterable_key = "sector" if label == "Sector" else "signal"
                ths.append(
                    f'<th class="filterable" data-filterable="{filterable_key}">{html.escape(label)}<span class="filter-ind"> ▾</span></th>'
                )
            else:
                sort_type = "string"
                if is_day_col or label == "Avg":
                    sort_type = "forecast"
                elif label in {"🟢", "🔴"}:
                    sort_type = "number"
                extra_class = " day-horizon-col" if is_day_col else ""
                ths.append(
                    f'<th class="sortable{extra_class}" data-sort-type="{html.escape(sort_type)}">{html.escape(label)}<span class="sort-ind"></span></th>'
                )
        else:
            ths.append(f"<th>{html.escape(label)}</th>")
    head = "".join(ths)

    rows_html = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        ticker_val = html.escape(str(row_dict.get("ticker", "")))
        is_rec = bool(row_dict.get("is_recommended", False))
        rec_attr = ' data-recommended="1"' if is_rec else ''
        row_class = ' class="recommended-row"' if is_rec else ''

        tds = []
        if checkbox_col:
            tds.append(f'<td><input type="checkbox" class="row-check" data-ticker="{ticker_val}"{rec_attr}></td>')
        for label, key in cols:
            v = row_dict.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                s = ""
            else:
                s = str(v)
            if key == "ticker":
                badge = ' <span class="badge-buy-sm">🔥 BUY</span>' if is_rec else ''
                tds.append(f'<td class="col-ticker"><span class="ticker-pill">{html.escape(s)}</span>{badge}</td>')
            elif key == "name":
                tds.append(f'<td class="col-name">{html.escape(s)}</td>')
            elif key == "signal":
                sig_tooltip = ""
                sig_json_str = row_dict.get("signals_json")
                if sig_json_str:
                    try:
                        sig_data = json.loads(sig_json_str)
                        items = sig_data.get("items", [])
                        lines = []
                        for item in items:
                            cat = item.get("category", "")
                            lbl = item.get("label", "")
                            if cat and lbl:
                                lines.append(f"[{cat}] {lbl}")
                            elif lbl:
                                lines.append(lbl)
                        sig_tooltip = "\n".join(lines)
                    except Exception:
                        pass
                title_attr = f' title="{html.escape(sig_tooltip)}"' if sig_tooltip else ''
                cursor_style = ' style="cursor:help"' if sig_tooltip else ''
                tds.append(f'<td class="col-signal"{title_attr}{cursor_style}>{html.escape(s)}</td>')
            elif label.startswith("Day "):
                tds.append(f'<td class="day-horizon-col num-cell">{html.escape(s)}</td>')
            elif label == "Avg":
                tds.append(f'<td class="avg-col num-cell">{html.escape(s)}</td>')
            elif label in {"🟢", "🔴"}:
                tds.append(f'<td class="num-cell" style="text-align:center">{html.escape(s)}</td>')
            else:
                tds.append(f"<td>{html.escape(s)}</td>")
        rows_html.append(f"<tr{row_class}{rec_attr}>" + "".join(tds) + "</tr>")

    id_attr = f' id="{html.escape(table_id)}"' if table_id else ""
    return f"<table{id_attr}><thead><tr>{head}</tr></thead>\n<tbody>\n" + "\n".join(rows_html) + "\n</tbody></table>"


def _render_macro_pulse(nr_dict: dict[str, Any] | None) -> str:
    """政治・経済マクロ概況およびセクターセンチメントのパルスウィジェットを生成する。"""
    if not nr_dict:
        return ""
    macro_score = float(nr_dict.get("macro_score", 0.0))
    macro_regime = str(nr_dict.get("macro_regime", "Neutral"))
    event_risk = str(nr_dict.get("event_risk_level", "LOW")).upper()
    event_note = nr_dict.get("event_risk_note")
    drivers = nr_dict.get("key_drivers_ja", [])
    sector_scores = nr_dict.get("sector_scores", {})
    count = int(nr_dict.get("articles_analyzed_count", 0))

    if macro_score >= 0.20:
        regime_badge = f'<span class="macro-regime-badge badge-bull">🟢 {html.escape(macro_regime)} ({macro_score:+.2f})</span>'
    elif macro_score <= -0.20:
        regime_badge = f'<span class="macro-regime-badge badge-bear">🔴 {html.escape(macro_regime)} ({macro_score:+.2f})</span>'
    else:
        regime_badge = f'<span class="macro-regime-badge badge-neutral">🟡 {html.escape(macro_regime)} ({macro_score:+.2f})</span>'

    risk_badge = ""
    if event_risk == "HIGH":
        risk_badge = f'<span class="macro-risk-badge risk-high">⚠️ イベント警戒: {html.escape(str(event_note or "高リスク"))}</span>'
    elif event_risk == "MEDIUM":
        risk_badge = f'<span class="macro-risk-badge risk-med">⚡ イベント注視{": " + html.escape(str(event_note)) if event_note else ""}</span>'

    drivers_html = ""
    if drivers:
        items = "".join(f"<li>{html.escape(str(d))}</li>" for d in drivers[:3])
        drivers_html = f'<ul class="macro-drivers-list">{items}</ul>'

    sectors_html = ""
    if sector_scores:
        sec_chips = []
        for sec, val in sorted(
            sector_scores.items(),
            key=lambda x: -float(x[1].get("score", 0.0) if isinstance(x[1], dict) else x[1]),
        ):
            s_score = float(val.get("score", 0.0) if isinstance(val, dict) else val)
            s_reason = str(val.get("reason_ja", "") if isinstance(val, dict) else "")
            if s_score >= 0.15:
                chip_cls = "sec-chip-bull"
            elif s_score <= -0.15:
                chip_cls = "sec-chip-bear"
            else:
                chip_cls = "sec-chip-neutral"
            title_attr = html.escape(f"{sec}: {s_score:+.2f} ({s_reason})" if s_reason else f"{sec}: {s_score:+.2f}")
            sec_chips.append(f'<span class="sec-chip {chip_cls}" title="{title_attr}">{html.escape(sec)} <b>{s_score:+.2f}</b></span>')
        sectors_html = f'<div class="macro-sectors-row">{"".join(sec_chips)}</div>'

    return f"""
    <div class="macro-pulse-container">
      <div class="macro-pulse-header">
        <div class="macro-title-box">
          <span class="macro-pulse-icon">🌐</span>
          <span class="macro-pulse-heading">Macro Political & Economic Pulse</span>
          {regime_badge}
          {risk_badge}
        </div>
        <div class="macro-meta-box">
          <span>解析ニュース: <b>{count}件</b> (FRB/CNBC/Bloomberg/Polygon)</span>
        </div>
      </div>
      <div class="macro-pulse-content">
        {drivers_html}
        {sectors_html}
      </div>
    </div>
    """


def _render_executive_cockpit(
    action_sheet: dict[str, Any] | None,
    backtest_kpi: dict[str, Any] | None,
    asof: str,
    news_report: Any | None = None,
) -> str:
    """最上部のエグゼクティブ・コックピット（Action Sheet & Strategy Evidence）を生成する。"""
    nr_dict = None
    if news_report is not None:
        if hasattr(news_report, "to_dict"):
            nr_dict = news_report.to_dict()
        elif isinstance(news_report, dict):
            nr_dict = news_report
    elif action_sheet and isinstance(action_sheet.get("news_report"), dict):
        nr_dict = action_sheet["news_report"]

    macro_pulse_html = _render_macro_pulse(nr_dict)

    # ── 1. Action Sheet (左カード) ──────────────────────────────────────
    if action_sheet:
        available_slots = int(action_sheet.get("available_slots", 5))
        max_slots = action_sheet.get("max_slots")
        initial_capital = float(action_sheet.get("initial_capital_usd", 35000.0))
        op_time = action_sheet.get("operating_time_sgt", "平日 20:00〜21:30 SGT")
        buys = action_sheet.get("buys", [])
        reserves = action_sheet.get("reserves", [])
        exits = action_sheet.get("exits", [])
        holds = action_sheet.get("holds", [])
        max_per_sec = int(action_sheet.get("max_per_sector", 2))
    else:
        available_slots = 5
        max_slots = None
        initial_capital = 35000.0
        op_time = "平日 20:00〜21:30 SGT"
        buys = []
        reserves = []
        exits = []
        holds = []
        max_per_sec = 2

    # Slot indicator dots & slot strategy description
    current_held = len(holds)
    slots_desc = f"現在保有: <b>{current_held} 銘柄</b> (最低 3 / 目標 5 / 上限なし)"
    dots_html = "".join('<span class="slot-dot filled" title="保有中銘柄"></span>' for _ in range(min(current_held, 8)))
    if current_held < 5:
        dots_html += "".join('<span class="slot-dot empty" title="目標枠までの空き"></span>' for _ in range(5 - current_held))

    # Buys & Shorts list
    buys_html = ""
    if buys:
        for b in buys:
            tk = html.escape(str(b.get("ticker", "")))
            shares = int(b.get("shares", 0))
            alloc_usd = float(b.get("allocated_usd", 0.0))
            is_short = b.get("action") == "SHORT"
            limit_price = b.get("limit_price")
            limit_gap_pct = b.get("limit_gap_pct")
            tp_price = b.get("take_profit_price")
            tp_pct = b.get("take_profit_pct")
            sl_price = b.get("stop_loss_price")
            sl_pct = b.get("stop_loss_pct")
            exp_ret = b.get("expected_return_pct", 0)
            consensus = html.escape(str(b.get("consensus", "all")))
            sector = html.escape(str(b.get("sector", "")))
            tier = b.get("tier", 1)

            if is_short:
                ticket_cls = "ticket-short"
                badge_cls = "ticket-badge-short"
                badge_lbl = "SHORT LIMIT"
                exp_col = "text-red"
                limit_lbl = "推奨指値 (LOO売建)"
                tp_lbl = "利確買戻 (TP)"
                sl_lbl = "損切買戻 (SL)"
                tp_str = f"${tp_price} (-{tp_pct}%)" if tp_price else "なし (満期決済)"
                sl_str = f"${sl_price} (+{sl_pct}%)" if sl_price else "なし (満期/反転決済)"
                copy_action = "指値 売建"
            else:
                ticket_cls = "ticket-buy"
                badge_cls = "ticket-badge-buy"
                badge_lbl = "BUY LIMIT"
                exp_col = "text-green"
                limit_lbl = "推奨指値 (LOO買付)"
                tp_lbl = "利確指値 (TP)"
                sl_lbl = "損切逆指値 (SL)"
                tp_str = f"${tp_price} (+{tp_pct}%)" if tp_price else "なし (満期決済)"
                sl_str = f"${sl_price} (-{sl_pct}%)" if sl_price else "なし (満期/反転決済)"
                copy_action = "指値 買付"

            gap_sign = "+" if (limit_gap_pct or 0) >= 0 else ""
            limit_str = f"${limit_price}" if limit_price else "成行"
            limit_info = f"${limit_price} ({gap_sign}{limit_gap_pct}%)" if limit_price and limit_gap_pct is not None else (f"${limit_price}" if limit_price else "寄付成行")

            copy_payload = html.escape(
                f"{tk} {copy_action} (${limit_price}) | {shares}株 (約${alloc_usd:,.0f} USD) | TP: {tp_str} | SL: {sl_str} ※時間外OFF",
                quote=True,
            )

            sector_badge = f'<span class="ticket-sector">{sector}</span>' if sector else ""
            tier_badge = '<span class="ticket-tier">拡張枠(2銘柄目)</span>' if tier == 2 else ""

            n_summary = html.escape(str(b.get("news_summary") or ""))
            n_score = b.get("news_score")
            news_badge = ""
            if n_score is not None:
                n_sign = "+" if n_score >= 0 else ""
                n_col = "text-green" if n_score > 0.15 else ("text-red" if n_score < -0.15 else "")
                news_badge = f'<span class="ticket-news-badge {n_col}">📰 {n_sign}{n_score:.2f}</span>'
            news_note = f'<div class="ticket-news-note">📰 <b>ニュース・材料</b>: {n_summary}</div>' if n_summary else ""

            ret_sign = "+" if exp_ret >= 0 else ""
            buys_html += f"""
            <div class="action-ticket {ticket_cls}">
              <div class="ticket-top">
                <div class="ticket-title-row">
                  <span class="{badge_cls}">{badge_lbl}</span>
                  <span class="ticket-ticker">{tk}</span>
                  {sector_badge}
                  {tier_badge}
                  {news_badge}
                  <span class="ticket-exp">期待値: <b class="{exp_col}">{ret_sign}{exp_ret}%</b> ({consensus})</span>
                </div>
                <button class="copy-order-btn" data-copy="{copy_payload}" onclick="copyOrderText(this)" title="発注パラメータをコピー">
                  📋 コピー
                </button>
              </div>
              <div class="ticket-grid">
                <div class="t-cell">
                  <span class="t-lbl">{limit_lbl}</span>
                  <span class="t-val text-blue"><b>{limit_info}</b></span>
                </div>
                <div class="t-cell">
                  <span class="t-lbl">発注数量</span>
                  <span class="t-val"><b class="shares-highlight">{shares:,} 株</b> <span class="alloc-sub">(約 ${alloc_usd:,.2f})</span></span>
                </div>
                <div class="t-cell">
                  <span class="t-lbl">{tp_lbl}</span>
                  <span class="t-val text-green"><b>{tp_str}</b></span>
                </div>
                <div class="t-cell">
                  <span class="t-lbl">{sl_lbl}</span>
                  <span class="t-val text-red"><b>{sl_str}</b></span>
                </div>
              </div>
              {news_note}
              <div class="ticket-footer-note">
                💡 moomoo設定: <b>指値 ({limit_str})</b> / 有効期間 <b>Day</b> / 時間外取引 (Outside RTH) <b>OFF</b>
              </div>
            </div>
            """
    else:
        buys_html = """
        <div class="action-empty">
          <div class="empty-icon">☕</div>
          <div class="empty-msg">本日の新規発注（買付/売建）はありません</div>
          <div class="empty-sub">シグナル閾値（期待リターン・全Horizon合意）を満たす銘柄がないか、セクター制限（推奨1・最大2）による調整です</div>
        </div>
        """

    # Reserves list (補欠・リザーブ銘柄)
    reserves_html = ""
    if reserves:
        reserve_items = []
        for r in reserves:
            rtk = html.escape(str(r.get("ticker", "")))
            rshares = int(r.get("shares", 0))
            ralloc = float(r.get("allocated_usd", 0.0))
            r_is_short = r.get("action") == "SHORT"
            rlimit = r.get("limit_price")
            rgap = r.get("limit_gap_pct")
            rexp = r.get("expected_return_pct", 0)
            rsec = html.escape(str(r.get("sector", "")))
            rtp = r.get("take_profit_price")
            rsl = r.get("stop_loss_price")
            rtp_pct = r.get("take_profit_pct")
            rsl_pct = r.get("stop_loss_pct")
            tp_sign = "-" if r_is_short else "+"
            sl_sign = "+" if r_is_short else "-"
            rtp_str = f"${rtp} ({tp_sign}{rtp_pct}%)" if rtp else "なし"
            rsl_str = f"${rsl} ({sl_sign}{rsl_pct}%)" if rsl else "なし"
            rcopy_act = "指値 売建" if r_is_short else "指値 買付"
            rcopy = html.escape(
                f"{rtk} [補欠] {rcopy_act} (${rlimit}) | {rshares}株 (約${ralloc:,.0f} USD) | 利確: {rtp_str} | 損切: {rsl_str} ※時間外OFF",
                quote=True,
            )
            rsec_str = f"[{rsec}] " if rsec else ""
            rbadge = '<span class="reserve-badge" style="background:#fee2e2; color:#b91c1c;">空売補欠</span>' if r_is_short else '<span class="reserve-badge">補欠</span>'
            rgap_sign = "+" if (rgap or 0) >= 0 else ""
            rexp_sign = "+" if rexp >= 0 else ""
            rexp_col = "text-red" if r_is_short else "text-green"
            reserve_items.append(
                f"""
                <div class="reserve-row">
                  <div class="reserve-left">
                    {rbadge}
                    <span class="reserve-ticker">{rtk}</span>
                    <span class="reserve-sec">{rsec_str}予測: <b class="{rexp_col}">{rexp_sign}{rexp}%</b></span>
                  </div>
                  <div class="reserve-center">
                    <span>推奨指値: <b class="text-blue">${rlimit}</b> ({rgap_sign}{rgap}%)</span>
                    <span>数量: <b>{rshares}株</b> (約${ralloc:,.0f})</span>
                    <span>TP: <b>{rtp_str}</b> / SL: <b>{rsl_str}</b></span>
                  </div>
                  <button class="copy-order-btn-sm" data-copy="{rcopy}" onclick="copyOrderText(this)" title="補欠発注をコピー">📋</button>
                </div>
                """
            )
        reserves_html = f"""
        <div class="reserves-box">
          <div class="reserves-header">
            <span>🛡️ 補欠・リザーブ候補（※本命が乖離等で未約定の場合のみ検討）</span>
          </div>
          <div class="reserves-body">
            {''.join(reserve_items)}
          </div>
        </div>
        """

    # Exits list
    exits_html = ""
    if exits:
        for e in exits:
            tk = html.escape(str(e.get("ticker", "")))
            shares = int(e.get("shares", 0))
            action = html.escape(str(e.get("action", "寄付成行 (MOO) 全株売却")))
            reason = html.escape(str(e.get("reason", "満期到達")))
            exits_html += f"""
            <div class="action-ticket ticket-exit">
              <div class="ticket-top">
                <span class="ticket-badge-exit">⚠️ SELL MOO</span>
                <span class="ticket-ticker">{tk}</span>
                <span class="ticket-shares">{shares:,} 株</span>
                <span class="ticket-reason">➔ {action} ({reason})</span>
              </div>
            </div>
            """

    # Holds list
    holds_html = ""
    if holds:
        hold_items = []
        for h in holds:
            tk = html.escape(str(h.get("ticker", "")))
            shares = int(h.get("shares", 0))
            hdays = int(h.get("holding_days", 0))
            pnl = float(h.get("pnl_pct", 0.0))
            side_str = "売" if h.get("side") == "SHORT" else "買"
            pnl_cls = "text-green" if pnl >= 0 else "text-red"
            be_badge = ""
            if h.get("suggest_break_even"):
                be_price = h.get("break_even_sl_price")
                be_badge = f' <span class="be-pill" title="含み益+2.5%超: 建値への逆指値引き上げ推奨">🛡️建値SL (${be_price})</span>'
            hold_items.append(
                f'<span class="hold-pill"><b>[{side_str}] {tk}</b> {shares}株 (保有 {hdays}日目 / <span class="{pnl_cls}">{pnl:+.2f}%</span>){be_badge}</span>'
            )
        holds_html = f"""
        <div class="holds-bar">
          <span class="holds-title">🔒 継続保有 ({current_held}銘柄):</span>
          <div class="holds-list">{''.join(hold_items)}</div>
        </div>
        """

    # ── 2. Strategy Config parameters ──────────────────────────────────
    if backtest_kpi:
        strat = backtest_kpi.get("strategy", {})
        hold_days_limit = strat.get("holding_days", 3)
        tp_pct_cfg = f"+{float(strat.get('take_profit_pct', 0.04)) * 100:.1f}%" if strat.get("take_profit_pct") else "なし"
        sl_pct_cfg = f"-{float(strat.get('emergency_stop_loss_pct', 0.10)) * 100:.1f}%" if strat.get("emergency_stop_loss_pct") else "なし"
    else:
        hold_days_limit = 3
        tp_pct_cfg = "+4.0%"
        sl_pct_cfg = "-10.0%"

    return f"""
    <div class="cockpit-wrapper">
      <div class="cockpit-header-bar">
        <div class="cockpit-title-group">
          <span class="cockpit-badge">🚀 moomoo Trading Action</span>
          <h2 class="cockpit-main-title">Executive Strategy Cockpit & moomoo Action Plan</h2>
        </div>
        <div class="cockpit-meta-group">
          <span class="meta-pill time-pill">⏱️ 推奨操作: {op_time}</span>
        </div>
      </div>

      {macro_pulse_html}

      <div class="cockpit-grid">
        <!-- Action Sheet -->
        <div class="cockpit-card action-card">
          <div class="card-top-header">
            <div>
              <h3 class="card-heading">📋 今夜の moomoo 発注アクション</h3>
              <p class="card-caption">米国市場オープン前執行（寄付指値 LOO ＋ OCO指値設定 ※時間外OFF）</p>
            </div>
            <div class="slot-status-box">
              <div class="slot-text">{slots_desc}</div>
              <div class="slot-dots">{dots_html}</div>
            </div>
          </div>

          <div class="action-tickets-container">
            {exits_html}
            {buys_html}
            {reserves_html}
            {holds_html}
          </div>

          <div class="card-bottom-bar">
            <span>運用元本: <b>${initial_capital:,.2f} USD</b></span>
            <span>枠方針: <b>最低3銘柄 / 目標5銘柄 (上限なし) | 空売り対応 | 各セクター1銘柄推奨 (どうしても時最大2銘柄)</b></span>
            <span>OCO設定: <b>利確 {tp_pct_cfg} / 緊急損切 {sl_pct_cfg} (最長{hold_days_limit}日保有)</b></span>
          </div>
        </div>
      </div>
    </div>
    """


def render_html(
    *,
    summary_dl: str,
    acc_html: str,
    all_tbl: pd.DataFrame,
    cols: list[tuple[str, str]],
    charts_html_block: str,
    max_horizon: int,
    action_sheet: dict[str, Any] | None = None,
    backtest_kpi: dict[str, Any] | None = None,
    asof: str | None = None,
    news_report: Any | None = None,
) -> str:
    """完全なHTMLレポートドキュメントを生成する。"""
    # 銘柄センチメント集計
    bull_count = 0
    bear_count = 0
    neutral_count = 0
    total_tickers = len(all_tbl)
    if "signal" in all_tbl.columns:
        for s in all_tbl["signal"].astype(str):
            if "Bull" in s:
                bull_count += 1
            elif "Bear" in s:
                bear_count += 1
            else:
                neutral_count += 1
    bull_pct = round((bull_count / total_tickers) * 100, 1) if total_tickers > 0 else 0
    bear_pct = round((bear_count / total_tickers) * 100, 1) if total_tickers > 0 else 0
    neutral_pct = round((neutral_count / total_tickers) * 100, 1) if total_tickers > 0 else 0

    asof_label = html.escape(str(asof or (all_tbl.iloc[0]["asof_trade_date"] if "asof_trade_date" in all_tbl.columns and not all_tbl.empty else "Today")))

    cockpit_html = _render_executive_cockpit(
        action_sheet,
        backtest_kpi,
        asof=asof_label,
        news_report=news_report,
    )

    return f"""<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Daily Forecast Report & Strategy Cockpit</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
{_load_asset("report.css")}
    </style>
  </head>
<body>
  <!-- Toast message element -->
  <div id="copy-toast">✓ クリップボードにコピーしました</div>

  <!-- Header -->
  <header class="top-header">
    <div>
      <h1 class="brand-title">AlphaIgnitor3 Daily Forecast & Strategy Cockpit</h1>
      <p class="brand-sub">As-of: <b>{asof_label}</b> &middot; Multi-Step Zero-Shot Ensemble (Chronos2 / TimesFM / TiREX) &middot; Day {max_horizon} 予測</p>
    </div>
    <a href="../index.html" target="_top" class="portal-nav-link">
      🏠 ポータル一覧へ戻る
    </a>
  </header>

  <!-- ZONE 1: Executive Cockpit -->
  {cockpit_html}

  <!-- ZONE 2: Market Sentiment & Accuracy -->
  <div class="zone-2-row">
    <!-- Left: Market Sentiment & Tech Meta -->
    <div class="panel-box">
      <h2 class="panel-title">📊 マーケット地合い & 予測モデルサマリー</h2>
      <div class="sentiment-bar-container">
        <div class="sentiment-labels">
          <span class="text-green">🟢 強気 (Bull): {bull_count} ({bull_pct}%)</span>
          <span style="color:#d97706">🟡 中立 (Neutral): {neutral_count} ({neutral_pct}%)</span>
          <span class="text-red">🔴 弱気 (Bear): {bear_count} ({bear_pct}%)</span>
        </div>
        <div class="sentiment-bar">
          <div class="bar-bull" style="width:{bull_pct}%" title="強気: {bull_pct}%"></div>
          <div class="bar-neutral" style="width:{neutral_pct}%" title="中立: {neutral_pct}%"></div>
          <div class="bar-bear" style="width:{bear_pct}%" title="弱気: {bear_pct}%"></div>
        </div>
      </div>
      <details class="meta-details">
        <summary>⚙️ 予測モデル・生成メタデータ詳細を表示</summary>
        <dl>{summary_dl}</dl>
      </details>
    </div>

    <!-- Right: Directional Accuracy -->
    <div class="panel-box">
      <h2 class="panel-title">🎯 予測方向の的中率 (Directional Accuracy)</h2>
      {acc_html}
      <p class="note" style="margin:8px 0 0 0; font-size:11px; color:var(--slate-500)">
        直近5営業日のマルチステップ予測（Day 1〜Day {max_horizon}）について、実績終値との方向一致率（的中率）を集計。
      </p>
    </div>
  </div>

  <!-- ZONE 3: Table & Charts Panel -->
  <div class="split-layout">
    <!-- Left: Table -->
    <div class="left-panel">
      <div class="panel-box">
        <div class="table-toolbar">
          <div class="toolbar-left">
            <div class="search-box">
              <input type="text" id="table-search" placeholder="🔍 Ticker または 企業名で検索..." class="search-input">
            </div>
            <button id="btn-sel-rec" class="pill-btn btn-rec" title="推奨銘柄のみ選択・チャート表示">🔥 推奨銘柄</button>
            <button id="btn-sel-top5" class="pill-btn" title="上位5銘柄を選択">上位 5 件</button>
            <button id="btn-sel-all" class="pill-btn" title="表示中の全銘柄を選択">全選択</button>
            <button id="btn-sel-clear" class="pill-btn" title="選択をすべて解除">全解除</button>
          </div>
          <div class="toolbar-right">
            <button id="toggle-day-cols" class="toggle-btn" title="Day 1〜5の日別詳細予測列を表示切替">📅 Horizons</button>
            <span id="row-count-badge" class="count-indicator">{total_tickers} 銘柄</span>
          </div>
        </div>

        <div class="table-scroll">
          {_render_table(all_tbl, cols, table_id="all-tickers", sortable=True, checkbox_col=True)}
        </div>
      </div>
    </div>

    <!-- Right: Charts -->
    <div class="right-panel">
      <div class="panel-box">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <h2 class="panel-title" style="margin:0;">📈 チャート分析</h2>
          <span class="count-indicator" id="chart-count-label"></span>
        </div>
        <div class="charts-container" id="charts-container">
          {charts_html_block}
        </div>
      </div>
    </div>
  </div>

  <script>
{_load_asset("report.js")}
  </script>
</body>
</html>
"""
