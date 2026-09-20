from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from alphaignitor.backtest.config import TradingConfig
from alphaignitor.backtest.metrics import BacktestResult
from alphaignitor.backtest.strategy import TradeSignal


def _round_price(price: float | None) -> float | None:
    if price is None:
        return None
    p = float(price)
    return round(p, 4) if p < 1.0 else round(p, 2)


def generate_action_sheet(
    *,
    asof_date: str,
    signals: list[TradeSignal],
    trading_config: TradingConfig,
    active_positions: list[dict[str, Any]] | None = None,
    series_by_ticker: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Generate precise, human-executable moomoo trading instructions for tonight's SGT 20:00-21:30 session."""
    from alphaignitor.backtest.portfolio_tracker import load_active_positions

    active = active_positions if active_positions is not None else load_active_positions()
    initial_cash = float(trading_config.capital.initial_cash_usd)
    target_slots = int(getattr(trading_config.portfolio, "target_slots", 5))
    max_slots = int(trading_config.portfolio.max_slots) if trading_config.portfolio.max_slots is not None else target_slots
    slot_size_pct = float(trading_config.portfolio.slot_size_pct)
    min_cash_buffer = float(trading_config.portfolio.min_cash_buffer_usd)
    exec_cfg = trading_config.execution

    holding_days_limit = int(trading_config.strategy.holding_days)
    active_tickers = {p["ticker"] for p in active}

    # Determine EXITS
    exits: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    for pos in active:
        ticker = pos["ticker"]
        holding_days = int(pos.get("holding_days", 0))
        entry_price = float(pos.get("entry_price", 0.0))
        current_price = float(pos.get("current_price", entry_price))
        shares = int(pos.get("shares", 0))
        pnl_pct = ((current_price / entry_price) - 1.0) * 100 if entry_price > 0 else 0.0

        if holding_days >= holding_days_limit:
            exits.append({
                "ticker": ticker,
                "shares": shares,
                "reason": f"満期到達 ({holding_days}日保有)",
                "action": "寄付成行 (MOO) 全株売却",
                "estimated_price": _round_price(current_price),
            })
        else:
            suggest_be = pnl_pct >= 2.5
            holds.append({
                "ticker": ticker,
                "shares": shares,
                "holding_days": holding_days,
                "current_price": _round_price(current_price),
                "pnl_pct": round(pnl_pct, 2),
                "sl_price": _round_price(pos.get("sl_price")),
                "tp_price": _round_price(pos.get("tp_price")),
                "suggest_break_even": suggest_be,
                "break_even_sl_price": _round_price(entry_price) if suggest_be else None,
            })

    # Available slots for new entries
    remaining_active_count = len(active) - len(exits)
    available_slots = max(0, max_slots - remaining_active_count)

    # Helper to build an order record with limit price
    target_cash_per_slot = (initial_cash * slot_size_pct)

    def _build_order_record(sig: TradeSignal, *, is_reserve: bool = False) -> dict[str, Any] | None:
        atr_val = getattr(sig, "atr", None)
        if atr_val is None and series_by_ticker is not None:
            from alphaignitor.backtest.config import compute_atr
            s_df = series_by_ticker.get(sig.ticker)
            if s_df is not None and not s_df.empty:
                atr_val = compute_atr(s_df[s_df["trade_date"] <= asof_date])

        is_short = getattr(sig, "action", "BUY") == "SHORT"
        limit_p = getattr(sig, "limit_price", None)
        if limit_p is None:
            if is_short:
                limit_p = trading_config.calc_short_limit_price(
                    sig.asof_close, atr=atr_val, expected_return=sig.expected_return
                )
            else:
                limit_p = trading_config.calc_limit_price(
                    sig.asof_close, atr=atr_val, expected_return=sig.expected_return
                )
        limit_gap_pct = round(((limit_p / sig.asof_close) - 1.0) * 100, 1) if sig.asof_close > 0 else 0.0

        if is_short:
            est_price = exec_cfg.calc_effective_short_entry(limit_p)
            shares = exec_cfg.calc_shares(target_cash_per_slot - 50.0, est_price)
            if shares <= 0:
                return None
            alloc_usd = shares * est_price
            sl_price = trading_config.calc_short_sl_price(est_price)
            tp_price = trading_config.calc_short_tp_price(est_price)
            sl_pct = round(((sl_price - est_price) / est_price) * 100, 1) if sl_price else None
            tp_pct = round(((est_price - tp_price) / est_price) * 100, 1) if tp_price else None
            order_type_label = "空売指値 (Sell Short Limit) ※時間外OFF" if exec_cfg.order_type in ["LOO", "LIMIT"] else "空売成行 (MOO Short)"
            action_label = "SHORT"
        else:
            est_price = exec_cfg.calc_effective_entry(limit_p)
            shares = exec_cfg.calc_shares(target_cash_per_slot - 50.0, est_price)
            if shares <= 0:
                return None
            alloc_usd = shares * est_price
            sl_price = trading_config.calc_sl_price(est_price)
            tp_price = trading_config.calc_tp_price(est_price)
            sl_pct = round(((est_price - sl_price) / est_price) * 100, 1) if sl_price else None
            tp_pct = round(((tp_price - est_price) / est_price) * 100, 1) if tp_price else None
            order_type_label = "買付指値 (Buy Limit) ※時間外OFF" if exec_cfg.order_type in ["LOO", "LIMIT"] else "買付成行 (MOO Buy)"
            action_label = "BUY"

        return {
            "ticker": sig.ticker,
            "action": action_label,
            "order_type": order_type_label,
            "shares": shares,
            "allocated_usd": round(alloc_usd, 2),
            "asof_close": _round_price(sig.asof_close),
            "limit_price": _round_price(limit_p),
            "limit_gap_pct": limit_gap_pct,
            "expected_return_pct": round(sig.expected_return * 100, 2),
            "target_horizon_days": sig.target_horizon,
            "consensus": sig.consensus,
            "sector": getattr(sig, "sector", ""),
            "tier": getattr(sig, "tier", 1),
            "tier_label": "第1優先 (セクター分散推奨)" if getattr(sig, "tier", 1) == 1 else "拡張枠 (どうしても枠・セクター2銘柄目)",
            "q10_close": _round_price(sig.q10_close),
            "take_profit_price": _round_price(tp_price),
            "take_profit_pct": tp_pct,
            "stop_loss_price": _round_price(sl_price),
            "stop_loss_pct": sl_pct,
            "is_reserve": is_reserve,
        }

    # Candidate BUYS (Primary & Reserve)
    eligible_signals = [s for s in signals if s.ticker not in active_tickers]
    buys: list[dict[str, Any]] = []
    for sig in eligible_signals[:available_slots]:
        rec = _build_order_record(sig, is_reserve=False)
        if rec:
            buys.append(rec)

    reserves: list[dict[str, Any]] = []
    for sig in eligible_signals[available_slots:available_slots + 3]:
        rec = _build_order_record(sig, is_reserve=True)
        if rec:
            reserves.append(rec)

    sheet = {
        "asof_date": asof_date,
        "operating_time_sgt": "平日 20:00〜21:30 SGT",
        "initial_capital_usd": initial_cash,
        "max_slots": max_slots,
        "available_slots": available_slots,
        "max_per_sector": getattr(trading_config.strategy, "max_per_sector", 2),
        "sector_policy": getattr(trading_config.strategy, "sector_policy", "tiered_1_to_2"),
        "buys": buys,
        "reserves": reserves,
        "exits": exits,
        "holds": holds,
    }
    return sheet


def format_action_sheet_text(sheet: dict[str, Any]) -> str:
    lines = [
        "================================================================================",
        f"📋 moomoo 発注アクションシート (As-of: {sheet['asof_date']} / 操作推奨: {sheet['operating_time_sgt']})",
        f"   運用資金: ${sheet['initial_capital_usd']:,.2f} USD | 最大保有枠: {sheet['max_slots']} 銘柄 | 空き枠: {sheet['available_slots']} 銘柄",
        "================================================================================",
    ]

    if sheet["buys"]:
        lines.append("\n【1. 新規買い発注 (本命)】（米国市場オープン前: 指値 + OCO設定 ※時間外OFF）")
        for i, b in enumerate(sheet["buys"], 1):
            tp_str = f"${b['take_profit_price']} (+{b['take_profit_pct']}%)" if b.get('take_profit_price') else "なし (満期決済)"
            sl_str = f"${b['stop_loss_price']} (-{b['stop_loss_pct']}%)" if b.get('stop_loss_price') else "なし (満期/反転決済)"
            lines.append(
                f"  [{i}] {b['ticker']} (予測リターン: +{b['expected_return_pct']}%, 合意: {b['consensus']})\n"
                f"      ・注文種別 : {b['order_type']}\n"
                f"      ・推奨指値 : ${b['limit_price']} (前日終値 ${b['asof_close']} 比 +{b['limit_gap_pct']}% 上限)\n"
                f"      ・発注数量 : {b['shares']} 株 (想定約定額: 約 ${b['allocated_usd']:,.2f} USD)\n"
                f"      ・OCO設定 : 利確(TP) {tp_str} / 損切(SL) {sl_str}"
            )
    else:
        lines.append("\n【1. 新規買い発注 (本命)】\n  ・新規発注なし (空きスロットなし、またはシグナル閾値未満)")

    if sheet.get("reserves"):
        lines.append("\n【1-B. 補欠・リザーブ買い候補】（※本命が寄付き高寄りで未約定の場合、推奨指値以下で代替発注）")
        for i, r in enumerate(sheet["reserves"], 1):
            tp_str = f"${r['take_profit_price']} (+{r['take_profit_pct']}%)" if r.get('take_profit_price') else "なし"
            sl_str = f"${r['stop_loss_price']} (-{r['stop_loss_pct']}%)" if r.get('stop_loss_price') else "なし"
            lines.append(
                f"  [補欠{i}] {r['ticker']} (予測: +{r['expected_return_pct']}%, セクター: {r.get('sector', '-')})\n"
                f"          ・推奨指値: ${r['limit_price']} | 数量: {r['shares']}株 (約${r['allocated_usd']:,.2f} USD) | TP: {tp_str} / SL: {sl_str}"
            )

    if sheet["exits"]:
        lines.append("\n【2. 手仕舞い発注】（米国市場オープン前: 寄付成行 MOO 売）")
        for e in sheet["exits"]:
            lines.append(f"  ・{e['ticker']} : {e['shares']} 株 ➔ {e['action']} ({e['reason']})")
    else:
        lines.append("\n【2. 手仕舞い発注】\n  ・手仕舞い対象なし")

    if sheet["holds"]:
        lines.append("\n【3. 継続保有 (放置 / 建値ストップ)】")
        for h in sheet["holds"]:
            pnl_sign = "+" if h['pnl_pct'] >= 0 else ""
            be_note = f" 🛡️【建値ストップ引上げ推奨】SL ➔ ${h['break_even_sl_price']}" if h.get('suggest_break_even') else ""
            lines.append(
                f"  ・{h['ticker']} : {h['shares']} 株 (保有 {h['holding_days']} 日目 / 損益: {pnl_sign}{h['pnl_pct']}%){be_note}"
            )

    lines.append("================================================================================")
    return "\n".join(lines)


def generate_backtest_html_report(
    *,
    result: BacktestResult,
    trading_config: TradingConfig,
    outdir: Path,
    report_title: str = "AlphaIgnitor3 売買戦略バックテスト & moomoo運用レポート",
    top_strategies: list[dict[str, Any]] | None = None,
    action_sheet: dict[str, Any] | None = None,
) -> Path:
    """Generate an interactive, executive-grade HTML report."""
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    html_file = outdir_path / "backtest_report.html"

    # Format equity curve chart data
    dates = [d.trade_date for d in result.daily_history]
    equities = [round(d.total_equity, 2) for d in result.daily_history]
    drawdowns = [round(d.drawdown * 100, 2) for d in result.daily_history]

    daily_events = []
    for d in result.daily_history:
        daily_events.append({
            "date": d.trade_date,
            "equity": round(d.total_equity, 2),
            "cash": round(d.cash, 2),
            "positions_count": d.active_positions_count,
            "entries": getattr(d, "entries_today", []),
            "exits": getattr(d, "exits_today", []),
            "holdings": getattr(d, "holdings_today", []),
        })
    daily_events_json = json.dumps(daily_events, ensure_ascii=False)

    # Convert trades to rows
    trade_rows = []
    for t in result.trades:
        trade_rows.append({
            "ticker": t.ticker,
            "side": getattr(t, "side", "LONG"),
            "entry_date": t.entry_trade_date,
            "entry_price": round(t.entry_price, 2),
            "shares": t.shares,
            "exit_date": t.exit_trade_date or "Open",
            "exit_price": round(t.exit_price, 2) if t.exit_price else None,
            "pnl": round(t.net_pnl, 2) if t.net_pnl is not None else None,
            "return_pct": round(t.return_pct * 100, 2) if t.return_pct is not None else None,
            "holding_days": t.holding_days,
            "reason": t.exit_reason or "Holding",
        })

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{report_title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {{
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-heading: #f0f6fc;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --gold: #d29922;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
  }}
  .container {{ max-width: 1280px; margin: 0 auto; }}
  h1, h2, h3 {{ color: var(--text-heading); }}
  .header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .badge {{
    background: #1f6feb;
    color: #fff;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: bold;
  }}
  .action-box {{
    background: #1c2128;
    border: 1px solid #388bfd;
    border-left: 6px solid #388bfd;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 28px;
  }}
  .grid-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }}
  .card .label {{ font-size: 12px; color: #8b949e; text-transform: uppercase; }}
  .card .val {{ font-size: 24px; font-weight: bold; margin-top: 6px; color: var(--text-heading); }}
  .val.green {{ color: var(--green); }}
  .val.red {{ color: var(--red); }}
  .chart-container {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 28px;
    height: 420px;
    position: relative;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
    font-size: 14px;
  }}
  th, td {{
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{ background: #21262d; color: var(--text-heading); font-weight: 600; }}
  tr:hover {{ background: #1f242c; }}
  .pnl-plus {{ color: var(--green); font-weight: bold; }}
  .pnl-minus {{ color: var(--red); font-weight: bold; }}
  .custom-chart-tooltip {{
    position: absolute;
    background: rgba(15, 23, 42, 0.96);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 10px;
    color: #f8fafc;
    padding: 12px 14px;
    pointer-events: none;
    transition: opacity 0.15s ease-out, transform 0.15s ease-out;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.8);
    z-index: 1000;
    min-width: 290px;
    max-width: 380px;
    font-size: 12px;
    line-height: 1.4;
  }}
  .tt-header {{
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    padding-bottom: 6px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .tt-date {{ font-weight: 700; font-size: 13px; color: #38bdf8; }}
  .tt-equity {{ font-size: 12px; color: #f1f5f9; }}
  .tt-section {{ margin-top: 8px; }}
  .tt-sec-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
  .tt-title-buy {{ color: #10b981; }}
  .tt-title-exit {{ color: #f87171; }}
  .tt-title-hold {{ color: #94a3b8; }}
  .tt-item {{ display: flex; justify-content: space-between; align-items: center; padding: 3px 0; font-size: 11.5px; border-bottom: 1px dashed rgba(148, 163, 184, 0.12); }}
  .tt-badge-buy {{ background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 600; margin-right: 4px; }}
  .tt-badge-short {{ background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 600; margin-right: 4px; }}
  .tt-badge-exit {{ background: rgba(148, 163, 184, 0.2); color: #94a3b8; padding: 1px 5px; border-radius: 3px; font-size: 10px; margin-right: 4px; }}
  .tt-empty {{ color: #64748b; font-style: italic; font-size: 11px; padding: 2px 0; }}
  .text-green {{ color: var(--green); }}
  .text-red {{ color: var(--red); }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>📈 {report_title}</h1>
      <p style="margin:4px 0 0 0; color:#8b949e;">moomoo 手動執行モデル (SGT 20:00-24:00) | 初期資金: ${result.initial_capital:,.2f} USD</p>
    </div>
    <div style="display:flex; gap:12px; align-items:center;">
      <a href="../index.html" target="_top" style="display:inline-flex; align-items:center; gap:6px; padding:6px 14px; background:#21262d; border:1px solid #30363d; color:#58a6ff; text-decoration:none; border-radius:6px; font-weight:600; font-size:13px; transition:all 0.2s;">
        🏠 ポータル一覧へ
      </a>
      <div class="badge">Walk-Forward 検証済</div>
    </div>
  </div>

  {_render_action_sheet_html(action_sheet) if action_sheet else ""}

  <h2>📊 パフォーマンス KPI サマリー</h2>
  <div class="grid-cards">
    <div class="card">
      <div class="label">累積リターン</div>
      <div class="val {'green' if result.total_return >= 0 else 'red'}">{result.total_return * 100:+.2f}%</div>
    </div>
    <div class="card">
      <div class="label">最終資産</div>
      <div class="val">${result.final_equity:,.2f}</div>
    </div>
    <div class="card">
      <div class="label">CAGR (年率リターン)</div>
      <div class="val {'green' if result.cagr >= 0 else 'red'}">{result.cagr * 100:+.2f}%</div>
    </div>
    <div class="card">
      <div class="label">シャープレシオ</div>
      <div class="val {'green' if result.sharpe_ratio >= 1.0 else ''}">{result.sharpe_ratio:.2f}</div>
    </div>
    <div class="card">
      <div class="label">最大ドローダウン</div>
      <div class="val red">-{result.max_drawdown * 100:.2f}%</div>
    </div>
    <div class="card">
      <div class="label">勝率 (Win Rate)</div>
      <div class="val">{result.win_rate * 100:.1f}%</div>
    </div>
    <div class="card">
      <div class="label">プロフィットファクター</div>
      <div class="val">{result.profit_factor:.2f}</div>
    </div>
    <div class="card">
      <div class="label">総トレード数</div>
      <div class="val">{result.total_trades} 回</div>
    </div>
    <div class="card">
      <div class="label">平均保有日数</div>
      <div class="val">{result.avg_holding_days:.1f} 日</div>
    </div>
  </div>

  <h2>📈 資産推移エクイティカーブ (Equity Curve)</h2>
  <p style="color:#8b949e; font-size:13px; margin-top:-6px; margin-bottom:12px;">※グラフの各ポイントにマウスを乗せると、その日の<b>【新規約定 (買/売)】【決済損益】【当日保有銘柄】</b>の詳細ポップアップが表示されます。</p>
  <div class="chart-container">
    <canvas id="equityChart"></canvas>
  </div>

  {_render_top_strategies_html(top_strategies) if top_strategies else ""}

  <h2>📜 全トレード履歴明細 ({len(trade_rows)} 件)</h2>
  <div class="card" style="padding:0; overflow-x:auto;">
    <table>
      <thead>
        <tr>
          <th>銘柄</th>
          <th>売買種別</th>
          <th>Entry日</th>
          <th>約定値</th>
          <th>株数</th>
          <th>Exit日</th>
          <th>決済値</th>
          <th>純損益 ($)</th>
          <th>リターン (%)</th>
          <th>保有日数</th>
          <th>エグジット理由</th>
        </tr>
      </thead>
      <tbody>
        {"".join(_render_trade_row(t) for t in trade_rows)}
      </tbody>
    </table>
  </div>
</div>

<script>
const dailyEvents = {daily_events_json};

function getOrCreateTooltip(chart) {{
  let tooltipEl = chart.canvas.parentNode.querySelector('.custom-chart-tooltip');
  if (!tooltipEl) {{
    tooltipEl = document.createElement('div');
    tooltipEl.className = 'custom-chart-tooltip';
    chart.canvas.parentNode.appendChild(tooltipEl);
  }}
  return tooltipEl;
}}

function externalTooltipHandler(context) {{
  const {{chart, tooltip}} = context;
  const tooltipEl = getOrCreateTooltip(chart);

  if (tooltip.opacity === 0) {{
    tooltipEl.style.opacity = '0';
    tooltipEl.style.transform = 'translateY(6px)';
    return;
  }}

  if (tooltip.dataPoints && tooltip.dataPoints.length > 0) {{
    const idx = tooltip.dataPoints[0].dataIndex;
    const evt = dailyEvents[idx];
    if (!evt) return;

    let entriesHtml = '';
    if (evt.entries && evt.entries.length > 0) {{
      entriesHtml = evt.entries.map(e => {{
        const isShort = (e.side || e.action || '').toUpperCase().includes('SHORT');
        const badge = isShort
          ? '<span class="tt-badge-short">売建</span>'
          : '<span class="tt-badge-buy">買付</span>';
        const sec = e.sector ? ` <span style="color:#8b949e; font-size:10px;">[${{e.sector}}]</span>` : '';
        return `<div class="tt-item">
          <div>${{badge}}<b>${{e.ticker}}</b>${{sec}}</div>
          <div>${{e.shares}}株 @ $${{Number(e.price).toFixed(2)}}</div>
        </div>`;
      }}).join('');
    }} else {{
      entriesHtml = '<div class="tt-empty">新規約定なし</div>';
    }}

    let exitsHtml = '';
    if (evt.exits && evt.exits.length > 0) {{
      exitsHtml = evt.exits.map(x => {{
        const isShort = (x.side || '').toUpperCase().includes('SHORT');
        const pnl = Number(x.pnl || x.net_pnl || 0);
        const ret = Number(x.return_pct || 0);
        const pnlColor = pnl >= 0 ? 'text-green' : 'text-red';
        const pnlSign = pnl >= 0 ? '+' : '';
        const badge = isShort
          ? '<span class="tt-badge-short">買戻</span>'
          : '<span class="tt-badge-exit">売却</span>';
        return `<div class="tt-item">
          <div>${{badge}}<b>${{x.ticker}}</b> <span style="color:#8b949e; font-size:10px;">(${{x.reason || 'Exit'}})</span></div>
          <div class="${{pnlColor}}">${{pnlSign}}$${{pnl.toFixed(1)}} (${{pnlSign}}${{ret.toFixed(1)}}%)</div>
        </div>`;
      }}).join('');
    }} else {{
      exitsHtml = '<div class="tt-empty">当日手仕舞いなし</div>';
    }}

    let holdingsHtml = '';
    if (evt.holdings && evt.holdings.length > 0) {{
      holdingsHtml = evt.holdings.map(h => {{
        const isShort = (h.side || '').toUpperCase().includes('SHORT');
        const pnl = Number(h.unrealized_pnl || 0);
        const ret = Number(h.return_pct || 0);
        const pnlColor = pnl >= 0 ? 'text-green' : 'text-red';
        const pnlSign = pnl >= 0 ? '+' : '';
        const badge = isShort
          ? '<span class="tt-badge-short">売</span>'
          : '<span class="tt-badge-buy">買</span>';
        const sec = h.sector ? ` <span style="color:#8b949e; font-size:10px;">[${{h.sector}}]</span>` : '';
        return `<div class="tt-item">
          <div>${{badge}}<b>${{h.ticker}}</b>${{sec}} <span style="color:#8b949e; font-size:10px;">${{h.shares}}株 (${{h.holding_days}}日目)</span></div>
          <div class="${{pnlColor}}">${{pnlSign}}$${{pnl.toFixed(1)}} (${{pnlSign}}${{ret.toFixed(1)}}%)</div>
        </div>`;
      }}).join('');
    }} else {{
      holdingsHtml = '<div class="tt-empty">ノーポジ (Cash 100%)</div>';
    }}

    tooltipEl.innerHTML = `
      <div class="tt-header">
        <div class="tt-date">📅 ${{evt.date}}</div>
        <div class="tt-equity">総資産: <b>$${{Number(evt.equity).toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}})}}</b></div>
      </div>
      <div class="tt-section">
        <div class="tt-sec-title tt-title-buy">🟢 当日新規約定</div>
        ${{entriesHtml}}
      </div>
      <div class="tt-section">
        <div class="tt-sec-title tt-title-exit">🔴 当日決済損益</div>
        ${{exitsHtml}}
      </div>
      <div class="tt-section">
        <div class="tt-sec-title tt-title-hold">💼 当日保有銘柄 (${{evt.positions_count || 0}}銘柄)</div>
        ${{holdingsHtml}}
      </div>
    `;
  }}

  const {{offsetLeft: positionX, offsetTop: positionY}} = chart.canvas;
  tooltipEl.style.opacity = '1';
  tooltipEl.style.transform = 'translateY(0)';

  let leftPos = positionX + tooltip.caretX + 16;
  if (leftPos + 340 > chart.width) {{
    leftPos = positionX + tooltip.caretX - 350;
  }}
  let topPos = positionY + tooltip.caretY - 40;
  if (topPos < 10) topPos = 10;

  tooltipEl.style.left = leftPos + 'px';
  tooltipEl.style.top = topPos + 'px';
}}

const ctx = document.getElementById('equityChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(dates)},
    datasets: [{{
      label: 'Portfolio Equity ($)',
      data: {json.dumps(equities)},
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88, 166, 255, 0.1)',
      borderWidth: 2,
      pointRadius: 2,
      pointHoverRadius: 6,
      pointHoverBackgroundColor: '#58a6ff',
      pointHoverBorderColor: '#ffffff',
      pointHoverBorderWidth: 2,
      fill: true,
      tension: 0.1,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{
      mode: 'index',
      intersect: false,
    }},
    plugins: {{
      legend: {{ labels: {{ color: '#c9d1d9' }} }},
      tooltip: {{
        enabled: false,
        external: externalTooltipHandler
      }}
    }},
    scales: {{
      x: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }},
      y: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""
    with html_file.open("w", encoding="utf-8") as f:
        f.write(html_content)

    summary_file = outdir_path / "backtest_summary.json"
    summary_data = {
        "report_title": report_title,
        "initial_capital": float(result.initial_capital),
        "final_equity": float(result.final_equity),
        "total_return": float(result.total_return),
        "total_return_str": f"{result.total_return * 100:+.2f}%",
        "final_equity_str": f"${result.final_equity:,.2f}",
        "cagr": float(result.cagr),
        "sharpe_ratio": float(result.sharpe_ratio),
        "sharpe_ratio_str": f"{result.sharpe_ratio:.2f}",
        "sortino_ratio": float(result.sortino_ratio),
        "max_drawdown": float(result.max_drawdown),
        "max_drawdown_str": f"-{result.max_drawdown * 100:.2f}%",
        "win_rate": float(result.win_rate),
        "win_rate_str": f"{result.win_rate * 100:.1f}%",
        "profit_factor": float(result.profit_factor),
        "profit_factor_str": f"{result.profit_factor:.2f}",
        "total_trades": int(result.total_trades),
        "total_trades_str": f"{result.total_trades} 回",
        "avg_holding_days": float(result.avg_holding_days),
        "avg_holding_days_str": f"{result.avg_holding_days:.1f} 日",
        "strategy": {
            "holding_days": int(trading_config.strategy.holding_days),
            "take_profit_pct": float(trading_config.strategy.take_profit_pct) if trading_config.strategy.take_profit_pct else None,
            "emergency_stop_loss_pct": float(trading_config.strategy.emergency_stop_loss_pct) if trading_config.strategy.emergency_stop_loss_pct else None,
            "consensus_level": str(trading_config.strategy.consensus_level),
            "min_predicted_return": float(trading_config.strategy.min_predicted_return),
            "max_slots": int(trading_config.portfolio.max_slots) if trading_config.portfolio.max_slots is not None else None,
            "min_slots": int(getattr(trading_config.portfolio, "min_slots", 3)),
            "target_slots": int(getattr(trading_config.portfolio, "target_slots", 5)),
            "allow_short": bool(getattr(trading_config.strategy, "allow_short", True)),
            "max_per_sector": int(getattr(trading_config.strategy, "max_per_sector", 2)),
            "sector_policy": "各セクター1銘柄推奨 (どうしてもという場合最大2銘柄まで許容)",
        },
    }
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return html_file


def load_backtest_kpi(report_dir: Path | None = None) -> dict[str, Any] | None:
    """Load cached backtest summary KPIs from report/backtest/, with regex HTML fallback."""
    import re
    bt_dir = (report_dir or Path("report")) / "backtest"
    json_path = bt_dir / "backtest_summary.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    html_path = bt_dir / "backtest_report.html"
    if html_path.exists():
        try:
            content = html_path.read_text(encoding="utf-8")

            def _extract_metric(pattern: str) -> str | None:
                m = re.search(pattern, content)
                return m.group(1).strip() if m else None

            total_return_str = _extract_metric(r"累積リターン</div>\s*<div[^>]*>([^<]+)</div>")
            final_equity_str = _extract_metric(r"最終資産</div>\s*<div[^>]*>([^<]+)</div>")
            win_rate_str = _extract_metric(r"勝率[^<]*</div>\s*<div[^>]*>([^<]+)</div>")
            pf_str = _extract_metric(r"プロフィットファクター[^<]*</div>\s*<div[^>]*>([^<]+)</div>")
            sharpe_str = _extract_metric(r"シャープレシオ</div>\s*<div[^>]*>([^<]+)</div>")
            max_dd_str = _extract_metric(r"最大ドローダウン</div>\s*<div[^>]*>([^<]+)</div>")
            total_trades_str = _extract_metric(r"総トレード数</div>\s*<div[^>]*>([^<]+)</div>")
            avg_holding_str = _extract_metric(r"平均保有日数</div>\s*<div[^>]*>([^<]+)</div>")

            if total_return_str or win_rate_str:
                res = {
                    "total_return_str": total_return_str or "+0.0%",
                    "final_equity_str": final_equity_str or "$0",
                    "win_rate_str": win_rate_str or "0.0%",
                    "profit_factor_str": pf_str or "0.0",
                    "sharpe_ratio_str": sharpe_str or "0.0",
                    "max_drawdown_str": max_dd_str or "0.0%",
                    "total_trades_str": total_trades_str or "0",
                    "avg_holding_days_str": avg_holding_str or "0",
                    "strategy": {
                        "holding_days": 2,
                        "take_profit_pct": 0.05,
                        "emergency_stop_loss_pct": 0.07,
                        "consensus_level": "all",
                        "min_predicted_return": 0.02,
                        "max_slots": 3,
                    },
                }
                try:
                    json_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                return res
        except Exception:
            pass
    return None



def _render_action_sheet_html(sheet: dict[str, Any]) -> str:
    buys_html = ""
    if sheet.get("buys"):
        for b in sheet["buys"]:
            is_short = b.get("action") == "SHORT"
            tp_txt = f"利確指値 <b>${b['take_profit_price']}</b> (+{b['take_profit_pct']}%)" if b.get('take_profit_price') else "利確指値: なし (満期決済)"
            sl_txt = f"損切逆指値 <b>${b['stop_loss_price']}</b> (-{b['stop_loss_pct']}%)" if b.get('stop_loss_price') else "損切逆指値: なし (満期/反転決済)"
            gap_sign = "+" if (b.get('limit_gap_pct') or 0) >= 0 else ""
            limit_txt = f"推奨指値 (LOO): <b>${b.get('limit_price')}</b> ({gap_sign}{b.get('limit_gap_pct')}%)" if b.get('limit_price') else f"注文: {b['order_type']}"
            sec_txt = f"[{b.get('sector')}] " if b.get('sector') else ""
            tier_badge = f"<span style='font-size:11px; color:#d29922; background:rgba(210,153,34,0.15); padding:1px 6px; border-radius:4px; margin-left:6px;'>{b.get('tier_label')}</span>" if b.get("tier", 1) == 2 else ""
            badge = '<span style="background:rgba(239,68,68,0.2); color:#f85149; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:6px;">🔻 空売り (SELL SHORT)</span>' if is_short else '<span style="background:rgba(63,185,80,0.2); color:#3fb950; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:12px; margin-right:6px;">🚀 新規買い (BUY)</span>'
            title_color = "#f85149" if is_short else "#58a6ff"
            exp_ret = b.get('expected_return_pct', 0)
            ret_sign = "+" if exp_ret >= 0 else ""
            buys_html += f"""
            <div style="background:#21262d; border-radius:6px; padding:12px; margin-top:8px;">
              <div style="font-size:16px; font-weight:bold; color:{title_color};">{badge}{sec_txt}{b['ticker']} (予測リターン: {ret_sign}{exp_ret}%, {b['consensus']}){tier_badge}</div>
              <div style="margin-top:6px; font-size:14px; line-height:1.6;">
                ・発注: <b>{limit_txt}</b> | 数量: <b>{b['shares']} 株</b> (約 ${b['allocated_usd']:,.2f} USD)<br>
                ・moomoo OCO設定: {tp_txt} / {sl_txt}<br>
                <span style="font-size:12px; color:#8b949e;">※moomoo設定: 指値注文 / 時間外取引 (Outside RTH) OFF / Time-in-Force: Day</span>
              </div>
            </div>
            """
    else:
        buys_html = "<div style='color:#8b949e; margin-top:8px;'>新規発注なし</div>"

    reserves_html = ""
    if sheet.get("reserves"):
        reserves_items = ""
        for r in sheet["reserves"]:
            is_short = r.get("action") == "SHORT"
            gap_sign = "+" if (r.get('limit_gap_pct') or 0) >= 0 else ""
            r_type = "空売り" if is_short else "買い"
            r_col = "#f85149" if is_short else "#3fb950"
            reserves_items += f"""
            <li style="margin-top:4px;">
              <span style="color:{r_col}; font-weight:bold;">[{r_type}]</span> <b>{r['ticker']}</b> ({r.get('sector', '-')}) | 指値: <b>${r.get('limit_price')}</b> ({gap_sign}{r.get('limit_gap_pct')}%) | {r['shares']}株 (約${r['allocated_usd']:,.0f}) | 予測: {r['expected_return_pct']:+}%
            </li>
            """
        reserves_html = f"""
        <div style="margin-top:12px; padding:10px; background:#161b22; border-radius:6px; border:1px dashed #30363d;">
          <div style="font-size:13px; font-weight:bold; color:#d2a8ff;">🛡️ 補欠・リザーブ候補（本命が不利寄りで指値未約定の場合）</div>
          <ul style="margin:6px 0 0 16px; padding:0; font-size:13px; color:#c9d1d9;">
            {reserves_items}
          </ul>
        </div>
        """

    exits_html = ""
    if sheet.get("exits"):
        for e in sheet["exits"]:
            exits_html += f"<div>・<b>{e['ticker']}</b> ({e['shares']}株) ➔ {e['action']} ({e['reason']})</div>"
    else:
        exits_html = "<div style='color:#8b949e;'>手仕舞い対象なし</div>"

    holds_html = ""
    if sheet.get("holds"):
        hold_items = ""
        for h in sheet["holds"]:
            pnl_sign = "+" if h['pnl_pct'] >= 0 else ""
            pnl_col = "#3fb950" if h['pnl_pct'] >= 0 else "#f85149"
            side_str = "空売" if h.get("side") == "SHORT" else "買付"
            be_note = f" <b style='color:#e3b341;'>🛡️建値ストップ引上げ推奨 (SL➔${h['break_even_sl_price']})</b>" if h.get('suggest_break_even') else ""
            hold_items += f"<div>・[{side_str}] <b>{h['ticker']}</b> {h['shares']}株 (保有 {h['holding_days']}日目 / 損益: <span style='color:{pnl_col};'>{pnl_sign}{h['pnl_pct']}%</span>){be_note}</div>"
        holds_html = f"""
        <div style="margin-top:12px;">
          <h3 style="margin-bottom:4px; color:#f0f6fc;">【継続保有ポジション】</h3>
          {hold_items}
        </div>
        """

    return f"""
    <div class="action-box">
      <h2 style="margin-top:0; color:#58a6ff;">📋 今夜の moomoo 発注アクション ({sheet.get('operating_time_sgt')})</h2>
      <p style="color:#8b949e; margin-bottom:12px;">保有枠方針: 最低3銘柄 / 目標5銘柄 (上限なし) | 空売り対応 | 各セクター1銘柄推奨 (どうしても時最大2銘柄)<br>現在保有: <b>{sheet.get('holds_count', len(sheet.get('holds', [])))}</b> 銘柄 | 運用資金: ${sheet.get('initial_capital_usd', 35000):,.2f} USD</p>
      <div style="margin-bottom:16px;">
        <h3 style="margin-bottom:4px; color:#f0f6fc;">【新規発注 (本命)】</h3>
        {buys_html}
        {reserves_html}
      </div>
      <div style="margin-bottom:16px;">
        <h3 style="margin-bottom:4px; color:#f0f6fc;">【手仕舞い発注】</h3>
        {exits_html}
      </div>
      {holds_html}
    </div>
    """


def _render_top_strategies_html(strategies: list[dict[str, Any]]) -> str:
    rows = ""
    for s in strategies[:5]:
        p = s.get("params", {})
        rows += f"""
        <tr>
          <td>#{s.get('trial_number')}</td>
          <td><b>{s.get('score')}</b></td>
          <td class="{'pnl-plus' if s.get('is_return_pct', 0) >= 0 else 'pnl-minus'}">{s.get('is_return_pct'):+.1f}% (Sharpe: {s.get('is_sharpe')})</td>
          <td class="{'pnl-plus' if s.get('oos_return_pct', 0) >= 0 else 'pnl-minus'}">{s.get('oos_return_pct'):+.1f}% (Sharpe: {s.get('oos_sharpe')})</td>
          <td>{s.get('oos_win_rate_pct')}%</td>
          <td>{p.get('holding_days')}日 / SL: {f"{p.get('stop_loss_pct')*100:.1f}%" if p.get('stop_loss_pct') else 'None'} / TP: {f"{p.get('take_profit_pct')*100:.1f}%" if p.get('take_profit_pct') else 'None'}</td>
        </tr>
        """
    return f"""
    <h2>🏆 最適戦略候補ランキング (In-Sample vs Out-of-Sample)</h2>
    <div class="card" style="padding:0; overflow-x:auto; margin-bottom:28px;">
      <table>
        <thead>
          <tr>
            <th>Trial</th>
            <th>総合Score</th>
            <th>In-Sample 成績</th>
            <th>Out-of-Sample (検証) 成績</th>
            <th>検証勝率</th>
            <th>ルール設定 (保有/SL/TP)</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
    """


def _render_trade_row(t: dict[str, Any]) -> str:
    pnl = t.get("pnl")
    ret = t.get("return_pct")
    pnl_class = "pnl-plus" if (pnl or 0) >= 0 else "pnl-minus"
    pnl_str = f"{pnl:+,.2f}" if pnl is not None else "-"
    ret_str = f"{ret:+.2f}%" if ret is not None else "-"
    exit_p = f"${t.get('exit_price'):,.2f}" if t.get('exit_price') else "-"

    side = t.get("side", "LONG")
    if side == "SHORT":
        side_badge = '<span style="background:rgba(239,68,68,0.2); color:#f85149; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:11px;">空売り (SHORT)</span>'
    else:
        side_badge = '<span style="background:rgba(63,185,80,0.2); color:#3fb950; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:11px;">現物買い (LONG)</span>'

    return f"""
    <tr>
      <td><b>{t.get('ticker')}</b></td>
      <td>{side_badge}</td>
      <td>{t.get('entry_date')}</td>
      <td>${t.get('entry_price'):,.2f}</td>
      <td>{t.get('shares')}</td>
      <td>{t.get('exit_date')}</td>
      <td>{exit_p}</td>
      <td class="{pnl_class}">{pnl_str}</td>
      <td class="{pnl_class}">{ret_str}</td>
      <td>{t.get('holding_days')} 日</td>
      <td><span style="font-size:12px; color:#8b949e;">{t.get('reason')}</span></td>
    </tr>
    """

