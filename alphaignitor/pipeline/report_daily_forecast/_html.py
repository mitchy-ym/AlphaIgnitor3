"""HTMLレンダリングモジュール（エグゼクティブ・コックピット、テーブル、チャート、ページ全体）。"""
from __future__ import annotations

import html
import math
from typing import Any

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


def _render_executive_cockpit(
    action_sheet: dict[str, Any] | None,
    backtest_kpi: dict[str, Any] | None,
    asof: str,
) -> str:
    """最上部のエグゼクティブ・コックピット（Action Sheet & Strategy Evidence）を生成する。"""
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

            ret_sign = "+" if exp_ret >= 0 else ""
            buys_html += f"""
            <div class="action-ticket {ticket_cls}">
              <div class="ticket-top">
                <div class="ticket-title-row">
                  <span class="{badge_cls}">{badge_lbl}</span>
                  <span class="ticket-ticker">{tk}</span>
                  {sector_badge}
                  {tier_badge}
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

    cockpit_html = _render_executive_cockpit(action_sheet, backtest_kpi, asof=asof_label)

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
    :root {{
      --primary: #4f46e5;
      --primary-hover: #4338ca;
      --primary-light: #e0e7ff;
      --success: #16a34a;
      --success-light: #dcfce7;
      --danger: #dc2626;
      --danger-light: #fee2e2;
      --warning: #d97706;
      --warning-light: #fef3c7;
      --slate-900: #0f172a;
      --slate-800: #1e293b;
      --slate-700: #334155;
      --slate-600: #475569;
      --slate-500: #64748b;
      --slate-400: #94a3b8;
      --slate-200: #e2e8f0;
      --slate-100: #f1f5f9;
      --slate-50: #f8fafc;
      --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      --card-shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 24px;
      color: var(--slate-800);
      background-color: var(--slate-50);
      line-height: 1.5;
    }}

    /* ── Header ──────────────────────────────────────────────────────────── */
    .top-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--slate-200);
    }}
    .brand-title {{
      margin: 0 0 4px 0;
      font-family: 'Outfit', sans-serif;
      font-size: 26px;
      font-weight: 800;
      background: linear-gradient(135deg, #4f46e5, #06b6d4);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.02em;
    }}
    .brand-sub {{
      margin: 0;
      color: var(--slate-500);
      font-size: 13px;
      font-weight: 500;
    }}
    .portal-nav-link {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 16px;
      background: var(--primary-light);
      color: var(--primary-hover);
      text-decoration: none;
      border-radius: 8px;
      font-weight: 600;
      font-size: 13px;
      transition: all 0.2s ease;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    .portal-nav-link:hover {{
      background: #c7d2fe;
      transform: translateY(-1px);
    }}

    /* ── Executive Cockpit ───────────────────────────────────────────────── */
    .cockpit-wrapper {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }}
    .cockpit-header-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 16px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--slate-200);
    }}
    .cockpit-title-group {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .cockpit-badge {{
      background: #e0e7ff;
      color: #4338ca;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .cockpit-main-title {{
      margin: 0;
      font-family: 'Outfit', sans-serif;
      font-size: 19px;
      font-weight: 700;
      color: var(--slate-900);
    }}
    .cockpit-meta-group {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .meta-pill {{
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
    }}
    .time-pill {{
      background: #fef3c7;
      color: #92400e;
      border: 1px solid #fde68a;
    }}
    .valid-pill {{
      background: #dcfce7;
      color: #166534;
      border: 1px solid #bbf7d0;
    }}

    .cockpit-grid {{
      display: block;
    }}

    .cockpit-card {{
      background: var(--slate-50);
      border: 1px solid var(--slate-200);
      border-radius: 10px;
      padding: 16px;
      display: flex;
      flex-direction: column;
    }}
    .card-top-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .card-heading {{
      margin: 0 0 2px 0;
      font-family: 'Outfit', sans-serif;
      font-size: 15px;
      font-weight: 700;
      color: var(--slate-900);
    }}
    .card-caption {{
      margin: 0;
      font-size: 11px;
      color: var(--slate-500);
    }}

    /* Slot status */
    .slot-status-box {{
      text-align: right;
    }}
    .slot-text {{
      font-size: 12px;
      color: var(--slate-600);
    }}
    .slot-dots {{
      display: inline-flex;
      gap: 4px;
      margin-top: 4px;
    }}
    .slot-dot {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}
    .slot-dot.filled {{
      background: #4f46e5;
    }}
    .slot-dot.empty {{
      background: #cbd5e1;
    }}

    /* Action tickets */
    .action-tickets-container {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .action-ticket {{
      background: #ffffff;
      border-radius: 8px;
      padding: 12px 14px;
      border: 1px solid var(--slate-200);
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }}
    .ticket-buy {{
      border-left: 5px solid var(--primary);
    }}
    .ticket-short {{
      border-left: 5px solid #dc2626;
      background: #fffafa;
    }}
    .ticket-exit {{
      border-left: 5px solid var(--danger);
      background: #fff5f5;
    }}
    .ticket-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}
    .ticket-title-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .ticket-badge-buy {{
      background: #2563eb;
      color: #ffffff;
      font-size: 10px;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }}
    .ticket-badge-exit {{
      background: var(--danger);
      color: #ffffff;
      font-size: 10px;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .ticket-badge-short {{
      background: #dc2626;
      color: #ffffff;
      font-size: 10px;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }}
    .ticket-ticker {{
      font-family: 'Outfit', sans-serif;
      font-size: 16px;
      font-weight: 800;
      color: var(--slate-900);
    }}
    .ticket-sector {{
      background: var(--slate-100);
      color: var(--slate-600);
      font-size: 11px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 4px;
      border: 1px solid var(--slate-200);
    }}
    .ticket-tier {{
      background: #fef3c7;
      color: #b45309;
      font-size: 10px;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 4px;
      border: 1px solid #fde68a;
    }}
    .ticket-exp {{
      font-size: 12px;
      color: var(--slate-600);
    }}
    .copy-order-btn {{
      font-family: 'Inter', sans-serif;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 8px;
      background: var(--slate-100);
      color: var(--slate-700);
      border: 1px solid var(--slate-200);
      border-radius: 5px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .copy-order-btn:hover {{
      background: #e2e8f0;
      color: var(--slate-900);
    }}

    .ticket-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      background: var(--slate-50);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
    }}
    @media (max-width: 640px) {{
      .ticket-grid {{
        grid-template-columns: repeat(2, 1fr);
      }}
    }}
    .t-cell {{
      display: flex;
      flex-direction: column;
    }}
    .t-lbl {{
      font-size: 10px;
      color: var(--slate-500);
      text-transform: uppercase;
    }}
    .t-val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: var(--slate-900);
      margin-top: 2px;
    }}
    .shares-highlight {{
      color: var(--primary);
      font-weight: 700;
    }}
    .alloc-sub {{
      font-size: 10px;
      color: var(--slate-500);
    }}
    .ticket-footer-note {{
      margin-top: 8px;
      padding-top: 6px;
      border-top: 1px dashed var(--slate-200);
      font-size: 11px;
      color: var(--slate-600);
    }}
    .ticket-footer-note b {{
      color: var(--slate-800);
    }}

    /* Reserves styles */
    .reserves-box {{
      background: #f8fafc;
      border: 1px dashed #cbd5e1;
      border-radius: 8px;
      padding: 8px 12px;
    }}
    .reserves-header {{
      font-size: 11px;
      font-weight: 700;
      color: var(--slate-600);
      margin-bottom: 6px;
    }}
    .reserves-body {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .reserve-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #ffffff;
      border: 1px solid var(--slate-200);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 11px;
    }}
    .reserve-left {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .reserve-badge {{
      background: #e0e7ff;
      color: #4338ca;
      font-size: 9px;
      font-weight: 800;
      padding: 1px 5px;
      border-radius: 3px;
    }}
    .reserve-ticker {{
      font-family: 'Outfit', sans-serif;
      font-weight: 800;
      font-size: 13px;
      color: var(--slate-900);
    }}
    .reserve-sec {{
      color: var(--slate-500);
      font-size: 10px;
    }}
    .reserve-center {{
      display: flex;
      gap: 12px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--slate-700);
    }}
    .copy-order-btn-sm {{
      font-size: 11px;
      padding: 2px 6px;
      background: var(--slate-100);
      border: 1px solid var(--slate-200);
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .copy-order-btn-sm:hover {{
      background: #e2e8f0;
    }}

    .action-empty {{
      text-align: center;
      padding: 24px 16px;
      color: var(--slate-500);
    }}
    .empty-icon {{ font-size: 28px; margin-bottom: 4px; }}
    .empty-msg {{ font-weight: 600; font-size: 13px; color: var(--slate-700); }}
    .empty-sub {{ font-size: 11px; margin-top: 2px; }}

    .holds-bar {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      background: #ffffff;
      border-radius: 6px;
      border: 1px solid var(--slate-200);
      font-size: 11px;
    }}
    .holds-title {{ font-weight: 700; color: var(--slate-600); }}
    .holds-list {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .hold-pill {{
      background: var(--slate-100);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
    }}
    .be-pill {{
      background: #fef3c7;
      color: #92400e;
      border: 1px solid #fde68a;
      font-size: 10px;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 4px;
      margin-left: 4px;
    }}

    .card-bottom-bar {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--slate-500);
      padding-top: 8px;
      border-top: 1px solid var(--slate-200);
    }}

    /* Backtest Evidence Card */
    .btn-backtest-link {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 11px;
      font-weight: 700;
      color: var(--primary);
      text-decoration: none;
      background: #ffffff;
      border: 1px solid #c7d2fe;
      padding: 4px 8px;
      border-radius: 6px;
      transition: all 0.15s ease;
    }}
    .btn-backtest-link:hover {{
      background: var(--primary-light);
    }}

    .kpi-quad-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }}
    .kpi-mini-card {{
      background: #ffffff;
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
    }}
    .kpi-m-label {{
      font-size: 10px;
      font-weight: 600;
      color: var(--slate-500);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .kpi-m-val {{
      font-family: 'Outfit', sans-serif;
      font-size: 20px;
      font-weight: 700;
      margin: 2px 0 1px 0;
    }}
    .kpi-m-sub {{
      font-size: 10px;
      color: var(--slate-400);
    }}

    .rules-summary-box {{
      background: #ffffff;
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 11px;
    }}
    .rules-title {{
      font-weight: 700;
      color: var(--slate-800);
      margin-bottom: 6px;
    }}
    .rules-list {{
      margin: 0;
      padding-left: 16px;
      color: var(--slate-600);
      line-height: 1.6;
    }}

    /* ── Zone 2: Market Sentiment & Accuracy ─────────────────────────────── */
    .zone-2-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 24px;
    }}
    @media (max-width: 900px) {{
      .zone-2-row {{
        grid-template-columns: 1fr;
      }}
    }}
    .panel-box {{
      background: #ffffff;
      border: 1px solid var(--slate-200);
      border-radius: 12px;
      padding: 16px;
      box-shadow: var(--card-shadow);
    }}
    .panel-title {{
      margin: 0 0 10px 0;
      font-family: 'Outfit', sans-serif;
      font-size: 15px;
      font-weight: 700;
      color: var(--slate-900);
    }}

    /* Sentiment bar */
    .sentiment-bar-container {{
      margin-bottom: 14px;
    }}
    .sentiment-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .sentiment-bar {{
      height: 10px;
      border-radius: 5px;
      display: flex;
      overflow: hidden;
      background: var(--slate-200);
    }}
    .bar-bull {{ background: #16a34a; }}
    .bar-neutral {{ background: #eab308; }}
    .bar-bear {{ background: #dc2626; }}

    .meta-details {{
      font-size: 12px;
      margin-top: 8px;
    }}
    .meta-details summary {{
      cursor: pointer;
      color: var(--primary);
      font-weight: 600;
      user-select: none;
    }}
    .meta-details dl {{
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 10px;
      margin: 10px 0 0 0;
      background: var(--slate-50);
      padding: 10px;
      border-radius: 6px;
    }}
    .meta-details dt {{ color: var(--slate-500); font-weight: 600; }}
    .meta-details dd {{ margin: 0; color: var(--slate-800); font-family: 'JetBrains Mono', monospace; word-break: break-all; }}

    /* Accuracy table */
    .acc-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      border-radius: 6px;
      overflow: hidden;
    }}
    .acc-table th, .acc-table td {{
      padding: 6px 10px;
      border-bottom: 1px solid var(--slate-200);
      text-align: right;
    }}
    .acc-table th:first-child, .acc-table td:first-child {{
      text-align: left;
    }}
    .acc-table th:nth-child(2), .acc-table td:nth-child(2) {{
      text-align: center;
    }}
    .acc-table thead th {{
      background: var(--slate-50);
      color: var(--slate-600);
      font-weight: 600;
    }}
    .acc-table tfoot td {{
      background: var(--slate-100);
      font-weight: 700;
    }}
    .acc-rate-hi {{ color: #16a34a; font-weight: 700; }}
    .acc-rate-lo {{ color: #dc2626; font-weight: 700; }}

    /* ── Zone 3: Table & Charts ──────────────────────────────────────────── */
    .split-layout {{
      display: flex;
      gap: 20px;
      align-items: flex-start;
    }}
    .left-panel {{
      flex: 0 0 60%;
      min-width: 0;
    }}
    .right-panel {{
      flex: 0 0 calc(40% - 20px);
      min-width: 0;
      position: sticky;
      top: 24px;
      max-height: calc(100vh - 48px);
      overflow-y: auto;
    }}
    @media (max-width: 1024px) {{
      .split-layout {{
        flex-direction: column;
      }}
      .left-panel, .right-panel {{
        flex: 1 1 100%;
        width: 100%;
        position: static;
        max-height: none;
        overflow-y: visible;
      }}
    }}

    /* Table Toolbar */
    .table-toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--slate-200);
    }}
    .toolbar-left {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .search-box {{
      position: relative;
      display: inline-flex;
      align-items: center;
    }}
    .search-input {{
      font-family: 'Inter', sans-serif;
      font-size: 12px;
      padding: 6px 12px;
      border: 1px solid var(--slate-300);
      border-radius: 6px;
      width: 220px;
      background: #ffffff;
      outline: none;
      transition: all 0.2s ease;
    }}
    .search-input:focus {{
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15);
      width: 260px;
    }}
    .pill-btn {{
      font-family: 'Inter', sans-serif;
      font-size: 11px;
      font-weight: 600;
      padding: 5px 10px;
      border: 1px solid var(--slate-200);
      border-radius: 6px;
      background: #ffffff;
      color: var(--slate-700);
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .pill-btn:hover {{
      background: var(--slate-100);
      border-color: var(--slate-400);
    }}
    .pill-btn.btn-rec {{
      background: #f0fdf4;
      border-color: #bbf7d0;
      color: #166534;
      font-weight: 700;
    }}
    .pill-btn.btn-rec:hover {{
      background: #dcfce7;
    }}
    .toggle-btn {{
      font-family: 'Inter', sans-serif;
      font-size: 11px;
      font-weight: 600;
      padding: 5px 10px;
      border: 1px solid var(--slate-300);
      border-radius: 6px;
      background: #ffffff;
      color: var(--slate-600);
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .toggle-btn:hover {{
      background: var(--slate-50);
      border-color: var(--slate-400);
    }}
    .toggle-btn.active {{
      background: var(--primary-light);
      border-color: #c7d2fe;
      color: var(--primary);
      font-weight: 700;
    }}
    .count-indicator {{
      font-size: 11px;
      color: var(--slate-500);
      font-weight: 500;
    }}

    /* Table styling */
    .table-scroll {{
      overflow-x: auto;
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      background: #ffffff;
    }}
    table {{
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 12px;
      border-bottom: 1px solid var(--slate-100);
      white-space: nowrap;
    }}
    th {{
      font-weight: 600;
      background: var(--slate-50);
      color: var(--slate-600);
      border-bottom: 2px solid var(--slate-200);
      text-align: left;
    }}
    th.sortable {{
      cursor: pointer;
      user-select: none;
    }}
    th.sortable:hover {{
      background: var(--slate-100);
      color: var(--slate-900);
    }}
    th.sortable .sort-ind {{
      margin-left: 4px;
      color: var(--primary);
      font-weight: 700;
    }}

    td.num-cell {{
      font-family: 'JetBrains Mono', monospace;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    .day-horizon-col {{
      display: none;
    }}

    /* Ticker & Name cells */
    .ticker-pill {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: var(--slate-900);
    }}
    .badge-buy-sm {{
      background: #16a34a;
      color: #ffffff;
      font-size: 10px;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 4px;
      margin-left: 4px;
      vertical-align: middle;
    }}
    .col-name {{
      color: var(--slate-600);
      font-size: 12px;
      max-width: 180px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    /* Row highlight for recommended */
    tr.recommended-row td {{
      background-color: #f0fdf4 !important;
    }}
    tr.recommended-row:hover td {{
      background-color: #dcfce7 !important;
    }}
    tr.checked-row td {{
      background-color: #e0e7ff !important;
    }}

    /* Dropdowns */
    th.filterable {{
      cursor: pointer;
      user-select: none;
      position: relative;
    }}
    th.filterable .filter-ind {{ color: var(--slate-400); }}
    th.filterable.filter-active {{ color: var(--primary); }}
    th.filterable.filter-active .filter-ind {{ color: var(--primary); font-weight: 900; }}
    .sector-dropdown {{
      display: none;
      position: absolute;
      top: 100%;
      left: 0;
      z-index: 200;
      background: #ffffff;
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
      min-width: 200px;
      max-height: 320px;
      overflow-y: auto;
      padding: 8px 0;
      font-size: 12px;
      font-weight: normal;
      white-space: nowrap;
    }}
    .sector-dropdown.open {{ display: block; }}
    .sector-dropdown label {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      cursor: pointer;
      color: var(--slate-700);
    }}
    .sector-dropdown label:hover {{ background: var(--slate-100); color: var(--slate-900); }}
    .sector-dropdown .dd-actions {{
      display: flex;
      gap: 8px;
      padding: 4px 14px 6px;
    }}
    .sector-dropdown .dd-actions button {{
      font-size: 11px;
      font-weight: 600;
      padding: 4px 8px;
      border: 1px solid var(--slate-200);
      border-radius: 4px;
      background: #ffffff;
      color: var(--slate-600);
      cursor: pointer;
    }}

    /* ── Charts Panel ────────────────────────────────────────────────────── */
    .charts-container {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .chart-block {{
      position: relative;
      background: #ffffff;
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
      cursor: pointer;
    }}
    .chart-block h3 {{
      margin: 0 0 8px 0;
      font-family: 'Outfit', sans-serif;
      font-size: 13px;
      font-weight: 700;
      color: var(--slate-900);
    }}
    .chart-block img, .chart-block svg {{
      width: 100%;
      height: auto;
      border: 1px solid var(--slate-100);
      border-radius: 6px;
    }}
    .signal-overlay {{
      position: absolute;
      top: 36px;
      right: 12px;
      z-index: 100;
      background: rgba(255, 255, 255, 0.98);
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 11px;
      line-height: 1.6;
      max-width: 230px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.15);
      display: none;
      backdrop-filter: blur(4px);
    }}
    .signal-overlay .ov-badge {{
      font-family: 'Outfit', sans-serif;
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 6px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--slate-100);
    }}
    .signal-overlay .ov-cat {{
      font-weight: 700;
      color: var(--slate-400);
      margin: 6px 0 3px;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .signal-overlay .ov-item {{
      padding: 2px 0;
      font-weight: 500;
    }}

    /* Utility text colors */
    .text-green {{ color: #16a34a !important; font-weight: 700; }}
    .text-red {{ color: #dc2626 !important; font-weight: 700; }}
    .text-blue {{ color: #2563eb !important; font-weight: 700; }}
    .sig-bull {{ color: #16a34a; }}
    .sig-bear {{ color: #dc2626; }}
    .sig-neutral {{ color: var(--slate-600); }}

    /* Toast notification */
    #copy-toast {{
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #0f172a;
      color: #ffffff;
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      box-shadow: 0 10px 25px rgba(0,0,0,0.2);
      z-index: 9999;
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.25s ease;
      pointer-events: none;
    }}
    #copy-toast.show {{
      opacity: 1;
      transform: translateY(0);
    }}
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
  // Toast notification helper & resilient clipboard copy (works on HTTP LAN and mobile)
  function showCopyToast() {{
    const toast = document.getElementById('copy-toast');
    if (toast) {{
      toast.textContent = '✓ 発注指示をコピーしました';
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2500);
    }}
  }}

  function fallbackCopy(text) {{
    try {{
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const success = document.execCommand('copy');
      document.body.removeChild(ta);
      if (success) {{
        showCopyToast();
        return;
      }}
    }} catch (e) {{}}
    prompt('発注指示をコピーしてください:', text);
  }}

  function copyOrderText(btn) {{
    const text = btn.getAttribute('data-copy');
    if (!text) return;
    if (navigator.clipboard && window.isSecureContext) {{
      navigator.clipboard.writeText(text).then(showCopyToast).catch(() => {{
        fallbackCopy(text);
      }});
    }} else {{
      fallbackCopy(text);
    }}
  }}

  (function () {{
    const table = document.getElementById('all-tickers');
    if (!table) return;
    const tbody = table.tBodies && table.tBodies[0];
    if (!tbody) return;

    const ths = Array.from(table.querySelectorAll('thead th'));
    const rows = Array.from(tbody.rows);

    // Initial state: default sort indicator on Avg or Bull
    let state = {{ idx: -1, asc: false }};
    // Locate the 'Avg' or '🟢' column index
    const bullColIdx = ths.findIndex(th => th.textContent.includes('🟢'));
    if (bullColIdx >= 0) state = {{ idx: bullColIdx, asc: false }};

    // ── Search & Filter Logic ─────────────────────────────────────────────
    let searchTerm = '';
    let activeSectors = null; // null = no filter (show all)
    let activeSignals = null;

    const sectorThIdx = ths.findIndex(th => th.getAttribute('data-filterable') === 'sector');
    const signalThIdx = ths.findIndex(th => th.getAttribute('data-filterable') === 'signal');
    const tickerThIdx = ths.findIndex(th => th.textContent.trim().startsWith('Ticker'));
    const nameThIdx   = ths.findIndex(th => th.textContent.trim().startsWith('Name'));

    function applyFilters() {{
      let visibleCount = 0;
      rows.forEach(r => {{
        let show = true;

        // Search text filter
        if (searchTerm) {{
          const tText = (tickerThIdx >= 0 && r.children[tickerThIdx]) ? r.children[tickerThIdx].textContent.toLowerCase() : '';
          const nText = (nameThIdx >= 0 && r.children[nameThIdx]) ? r.children[nameThIdx].textContent.toLowerCase() : '';
          if (!tText.includes(searchTerm) && !nText.includes(searchTerm)) {{
            show = false;
          }}
        }}

        // Sector filter
        if (show && activeSectors !== null && sectorThIdx >= 0) {{
          const sec = r.children[sectorThIdx] ? r.children[sectorThIdx].textContent.trim() : '';
          if (!activeSectors.has(sec)) show = false;
        }}

        // Signal filter
        if (show && activeSignals !== null && signalThIdx >= 0) {{
          const sig = r.children[signalThIdx] ? r.children[signalThIdx].textContent.trim() : '';
          if (!activeSignals.has(sig)) show = false;
        }}

        r.style.display = show ? '' : 'none';
        if (show) visibleCount++;
      }});

      const badge = document.getElementById('row-count-badge');
      if (badge) badge.textContent = `${{visibleCount}} / ${{rows.length}} 銘柄`;
    }}

    // Search input event
    const searchInput = document.getElementById('table-search');
    if (searchInput) {{
      searchInput.addEventListener('input', () => {{
        searchTerm = searchInput.value.trim().toLowerCase();
        applyFilters();
      }});
    }}

    // ── Sector dropdown filter ────────────────────────────────────────────
    function allColValues(idx) {{
      const s = new Set();
      rows.forEach(r => {{
        const td = r.children[idx];
        if (td) s.add(td.textContent.trim());
      }});
      return Array.from(s).sort();
    }}

    if (sectorThIdx >= 0) {{
      const sectorTh = ths[sectorThIdx];
      const dd = document.createElement('div');
      dd.className = 'sector-dropdown';

      function rebuildSectorDd() {{
        dd.innerHTML = `
          <div class="dd-actions">
            <button type="button" class="btn-dd-all">Select All</button>
            <button type="button" class="btn-dd-none">Deselect All</button>
          </div>
        `;
        dd.querySelector('.btn-dd-all').onclick = (e) => {{
          e.stopPropagation();
          dd.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = true);
        }};
        dd.querySelector('.btn-dd-none').onclick = (e) => {{
          e.stopPropagation();
          dd.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = false);
        }};

        allColValues(sectorThIdx).forEach(sec => {{
          const lbl = document.createElement('label');
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.value = sec;
          cb.checked = activeSectors === null || activeSectors.has(sec);
          cb.onclick = (e) => e.stopPropagation();
          lbl.appendChild(cb);
          lbl.appendChild(document.createTextNode(sec || '(blank)'));
          dd.appendChild(lbl);
        }});
      }}

      sectorTh.style.position = 'relative';
      sectorTh.appendChild(dd);

      sectorTh.addEventListener('click', (e) => {{
        e.stopPropagation();
        const isOpen = dd.classList.contains('open');
        document.querySelectorAll('.sector-dropdown.open').forEach(d => d.classList.remove('open'));
        if (!isOpen) {{
          rebuildSectorDd();
          dd.classList.add('open');
        }}
      }});

      document.addEventListener('click', () => {{
        if (!dd.classList.contains('open')) return;
        dd.classList.remove('open');
        const checked = Array.from(dd.querySelectorAll('input[type=checkbox]:checked')).map(c => c.value);
        const allVals = allColValues(sectorThIdx);
        activeSectors = (checked.length === allVals.length) ? null : new Set(checked);
        sectorTh.classList.toggle('filter-active', activeSectors !== null);
        applyFilters();
      }});
      dd.addEventListener('click', (e) => e.stopPropagation());
    }}

    // ── Signal dropdown filter ────────────────────────────────────────────
    if (signalThIdx >= 0) {{
      const signalTh = ths[signalThIdx];
      const dd = document.createElement('div');
      dd.className = 'sector-dropdown';

      function rebuildSignalDd() {{
        dd.innerHTML = `
          <div class="dd-actions">
            <button type="button" class="btn-sig-all">Select All</button>
            <button type="button" class="btn-sig-none">Deselect All</button>
          </div>
        `;
        dd.querySelector('.btn-sig-all').onclick = (e) => {{
          e.stopPropagation();
          dd.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = true);
        }};
        dd.querySelector('.btn-sig-none').onclick = (e) => {{
          e.stopPropagation();
          dd.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = false);
        }};

        allColValues(signalThIdx).forEach(sig => {{
          const lbl = document.createElement('label');
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.value = sig;
          cb.checked = activeSignals === null || activeSignals.has(sig);
          cb.onclick = (e) => e.stopPropagation();
          lbl.appendChild(cb);
          lbl.appendChild(document.createTextNode(sig || '(blank)'));
          dd.appendChild(lbl);
        }});
      }}

      signalTh.style.position = 'relative';
      signalTh.appendChild(dd);

      signalTh.addEventListener('click', (e) => {{
        e.stopPropagation();
        const isOpen = dd.classList.contains('open');
        document.querySelectorAll('.sector-dropdown.open').forEach(d => d.classList.remove('open'));
        if (!isOpen) {{
          rebuildSignalDd();
          dd.classList.add('open');
        }}
      }});

      document.addEventListener('click', () => {{
        if (!dd.classList.contains('open')) return;
        dd.classList.remove('open');
        const checked = Array.from(dd.querySelectorAll('input[type=checkbox]:checked')).map(c => c.value);
        const allVals = allColValues(signalThIdx);
        activeSignals = (checked.length === allVals.length) ? null : new Set(checked);
        signalTh.classList.toggle('filter-active', activeSignals !== null);
        applyFilters();
      }});
      dd.addEventListener('click', (e) => e.stopPropagation());
    }}

    // ── Chart Visibility Sync ─────────────────────────────────────────────
    function updateCharts() {{
      const checkedTickers = new Set(
        Array.from(table.querySelectorAll('.row-check:checked')).map(cb => cb.dataset.ticker)
      );
      let visible = 0;
      document.querySelectorAll('#charts-container .chart-block').forEach(div => {{
        const show = checkedTickers.has(div.dataset.ticker);
        div.style.display = show ? '' : 'none';
        if (show) visible++;
      }});
      const noMsg = document.getElementById('no-charts-msg');
      if (noMsg) noMsg.style.display = visible === 0 ? '' : 'none';
      const label = document.getElementById('chart-count-label');
      if (label) label.textContent = visible > 0 ? `(${{visible}} 銘柄表示中)` : '';

      rows.forEach(r => {{
        const cb = r.querySelector('.row-check');
        r.classList.toggle('checked-row', !!(cb && cb.checked));
      }});
    }}

    table.addEventListener('change', e => {{
      if (e.target.classList.contains('row-check')) {{
        updateCharts();
        syncMasterCheckbox();
      }}
    }});

    // ── Master Checkbox ───────────────────────────────────────────────────
    const masterCb = document.getElementById('check-all-master');
    function syncMasterCheckbox() {{
      if (!masterCb) return;
      const visibleCbs = Array.from(table.querySelectorAll('tbody tr')).filter(r => r.style.display !== 'none').map(r => r.querySelector('.row-check')).filter(Boolean);
      const checkedCount = visibleCbs.filter(cb => cb.checked).length;
      if (checkedCount === 0) {{
        masterCb.checked = false;
        masterCb.indeterminate = false;
      }} else if (checkedCount === visibleCbs.length) {{
        masterCb.checked = true;
        masterCb.indeterminate = false;
      }} else {{
        masterCb.checked = false;
        masterCb.indeterminate = true;
      }}
    }}
    if (masterCb) {{
      masterCb.addEventListener('change', () => {{
        const visibleCbs = Array.from(table.querySelectorAll('tbody tr')).filter(r => r.style.display !== 'none').map(r => r.querySelector('.row-check')).filter(Boolean);
        visibleCbs.forEach(cb => cb.checked = masterCb.checked);
        updateCharts();
      }});
    }}

    // ── Quick Select Toolbar Buttons ──────────────────────────────────────
    const btnSelRec = document.getElementById('btn-sel-rec');
    if (btnSelRec) {{
      btnSelRec.addEventListener('click', () => {{
        let hasRec = false;
        rows.forEach(r => {{
          const cb = r.querySelector('.row-check');
          const isRec = r.getAttribute('data-recommended') === '1';
          if (cb) {{
            cb.checked = isRec;
            if (isRec) hasRec = true;
          }}
        }});
        // If no recommended, select top 3
        if (!hasRec) {{
          rows.slice(0, 3).forEach(r => {{
            const cb = r.querySelector('.row-check');
            if (cb) cb.checked = true;
          }});
        }}
        updateCharts();
        syncMasterCheckbox();
      }});
    }}

    const btnSelTop5 = document.getElementById('btn-sel-top5');
    if (btnSelTop5) {{
      btnSelTop5.addEventListener('click', () => {{
        let checked = 0;
        rows.forEach(r => {{
          const cb = r.querySelector('.row-check');
          if (cb) {{
            if (r.style.display !== 'none' && checked < 5) {{
              cb.checked = true;
              checked++;
            }} else {{
              cb.checked = false;
            }}
          }}
        }});
        updateCharts();
        syncMasterCheckbox();
      }});
    }}

    const btnSelAll = document.getElementById('btn-sel-all');
    if (btnSelAll) {{
      btnSelAll.addEventListener('click', () => {{
        rows.forEach(r => {{
          const cb = r.querySelector('.row-check');
          if (cb && r.style.display !== 'none') cb.checked = true;
        }});
        updateCharts();
        syncMasterCheckbox();
      }});
    }}

    const btnSelClear = document.getElementById('btn-sel-clear');
    if (btnSelClear) {{
      btnSelClear.addEventListener('click', () => {{
        rows.forEach(r => {{
          const cb = r.querySelector('.row-check');
          if (cb) cb.checked = false;
        }});
        updateCharts();
        syncMasterCheckbox();
      }});
    }}

    // ── Column Sorting ────────────────────────────────────────────────────
    function cellText(tr, idx) {{
      const td = tr.children[idx];
      return (td && td.textContent ? td.textContent : '').trim();
    }}

    function parseSortValue(txt, sortType) {{
      const s = (txt || '').trim();
      if (!s) return null;
      if (sortType === 'forecast') {{
        const m = s.match(/^([+-]?[\\d.]+)%/);
        const v = m ? parseFloat(m[1]) : NaN;
        return Number.isFinite(v) ? v : null;
      }}
      if (sortType === 'number') {{
        const v = parseFloat(s);
        return Number.isFinite(v) ? v : null;
      }}
      const v = parseFloat(s.replace(/,/g, ''));
      if (Number.isFinite(v) && /^[+-]?\\d/.test(s)) return v;
      return s.toLowerCase();
    }}

    function compareValues(a, b) {{
      if (a === null && b === null) return 0;
      if (a === null) return 1;
      if (b === null) return -1;
      if (typeof a === 'number' && typeof b === 'number') return a - b;
      return String(a).localeCompare(String(b), 'en', {{ numeric: true, sensitivity: 'base' }});
    }}

    ths.forEach((th, idx) => {{
      if (idx === 0) return; // Checkbox column
      if (th.getAttribute('data-filterable')) return;
      th.addEventListener('click', () => {{
        const sortType = th.getAttribute('data-sort-type') || 'string';
        let asc = (state.idx === idx) ? !state.asc : (sortType === 'string');
        state = {{ idx, asc }};

        rows.sort((ra, rb) => {{
          const va = parseSortValue(cellText(ra, idx), sortType);
          const vb = parseSortValue(cellText(rb, idx), sortType);
          const c = compareValues(va, vb);
          return asc ? c : -c;
        }});

        rows.forEach(r => tbody.appendChild(r));

        ths.forEach(t => {{
          const ind = t.querySelector('.sort-ind');
          if (ind) ind.textContent = '';
        }});
        const ind = th.querySelector('.sort-ind');
        if (ind) ind.textContent = asc ? '↑' : '↓';
      }});
    }});

    // ── Horizon Columns Toggle ────────────────────────────────────────────
    const toggleDayBtn = document.getElementById('toggle-day-cols');
    if (toggleDayBtn) {{
      toggleDayBtn.addEventListener('click', () => {{
        const isActive = toggleDayBtn.classList.toggle('active');
        const displayVal = isActive ? 'table-cell' : 'none';
        table.querySelectorAll('.day-horizon-col').forEach(el => {{ el.style.display = displayVal; }});
      }});
    }}

    // ── Initial Selection on Page Load ────────────────────────────────────
    // Prioritize recommended buy rows!
    const recCheckboxes = Array.from(table.querySelectorAll('.row-check[data-recommended="1"]'));
    if (recCheckboxes.length > 0) {{
      recCheckboxes.forEach(cb => cb.checked = true);
    }} else {{
      // Fallback: select top 3
      rows.slice(0, 3).forEach(r => {{
        const cb = r.querySelector('.row-check');
        if (cb) cb.checked = true;
      }});
    }}

    updateCharts();
    syncMasterCheckbox();

    // ── Chart Signal Overlay (Hover + Tap for Mobile) ─────────────────────
    document.querySelectorAll('.chart-block[data-signals]').forEach(function(block) {{
      const overlay = document.createElement('div');
      overlay.className = 'signal-overlay';
      block.appendChild(overlay);

      function renderOverlayContent() {{
        try {{
          const data = JSON.parse(block.dataset.signals);
          const v = data.verdict;
          const verdictText = v === 'bullish' ? '🟢 Bullish' : v === 'bearish' ? '🔴 Bearish' : '🟡 Neutral';
          const verdictCls  = v === 'bullish' ? 'sig-bull' : v === 'bearish' ? 'sig-bear' : 'sig-neutral';
          let h = '<div class="ov-badge ' + verdictCls + '">' + verdictText
                + ' <span style="font-weight:normal">🟢' + data.bull + ' 🔴' + data.bear + '</span></div>';
          const cats = {{}};
          (data.items || []).forEach(item => {{
            if (!cats[item.category]) cats[item.category] = [];
            cats[item.category].push(item);
          }});
          for (const cat in cats) {{
            h += '<div class="ov-cat">' + cat + '</div>';
            cats[cat].forEach(item => {{
              const icon = item.bullish === true ? '🟢' : item.bullish === false ? '🔴' : '🟡';
              const cls  = item.bullish === true ? 'sig-bull' : item.bullish === false ? 'sig-bear' : 'sig-neutral';
              h += '<div class="ov-item ' + cls + '">' + icon + ' ' + item.label + '</div>';
            }});
          }}
          overlay.innerHTML = h;
        }} catch(e) {{}}
      }}

      // Desktop hover
      block.addEventListener('mouseenter', () => {{
        renderOverlayContent();
        overlay.style.display = 'block';
      }});
      block.addEventListener('mouseleave', () => {{
        overlay.style.display = 'none';
      }});

      // Mobile touch toggle
      block.addEventListener('click', (e) => {{
        if (overlay.style.display === 'block') {{
          overlay.style.display = 'none';
        }} else {{
          renderOverlayContent();
          overlay.style.display = 'block';
        }}
      }});
    }});

  }})();
  </script>
</body>
</html>
"""
