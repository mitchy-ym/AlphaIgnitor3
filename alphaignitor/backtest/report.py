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
    max_slots = int(trading_config.portfolio.max_slots)
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
                "estimated_price": current_price,
            })
        else:
            holds.append({
                "ticker": ticker,
                "shares": shares,
                "holding_days": holding_days,
                "current_price": current_price,
                "pnl_pct": round(pnl_pct, 2),
                "sl_price": pos.get("sl_price"),
                "tp_price": pos.get("tp_price"),
            })

    # Available slots for new entries
    remaining_active_count = len(active) - len(exits)
    available_slots = max(0, max_slots - remaining_active_count)

    # Candidate BUYS
    buys: list[dict[str, Any]] = []
    eligible_signals = [s for s in signals if s.ticker not in active_tickers]
    target_cash_per_slot = (initial_cash * slot_size_pct)

    for sig in eligible_signals[:available_slots]:
        est_price = exec_cfg.calc_effective_entry(sig.asof_close)
        shares = exec_cfg.calc_shares(target_cash_per_slot - 50.0, est_price)
        if shares <= 0:
            continue

        alloc_usd = shares * est_price
        sl_price = trading_config.calc_sl_price(est_price)
        tp_price = trading_config.calc_tp_price(est_price)
        sl_pct = round(((est_price - sl_price) / est_price) * 100, 1) if sl_price else None
        tp_pct = round(((tp_price - est_price) / est_price) * 100, 1) if tp_price else None

        buys.append({
            "ticker": sig.ticker,
            "order_type": "MOO (寄付成行 買)",
            "shares": shares,
            "allocated_usd": round(alloc_usd, 2),
            "asof_close": round(sig.asof_close, 2),
            "expected_return_pct": round(sig.expected_return * 100, 2),
            "target_horizon_days": sig.target_horizon,
            "consensus": sig.consensus,
            "q10_close": round(sig.q10_close, 2) if sig.q10_close else None,
            "take_profit_price": round(tp_price, 2) if tp_price else None,
            "take_profit_pct": tp_pct,
            "stop_loss_price": round(sl_price, 2) if sl_price else None,
            "stop_loss_pct": sl_pct,
        })

    sheet = {
        "asof_date": asof_date,
        "operating_time_sgt": "平日 20:00〜21:30 SGT",
        "initial_capital_usd": initial_cash,
        "max_slots": max_slots,
        "available_slots": available_slots,
        "buys": buys,
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
        lines.append("\n【1. 新規買い発注】（米国市場オープン前: MOO 寄付成行 + OCO設定）")
        for i, b in enumerate(sheet["buys"], 1):
            tp_str = f"${b['take_profit_price']} (+{b['take_profit_pct']}%)" if b.get('take_profit_price') else "なし (満期決済)"
            sl_str = f"${b['stop_loss_price']} (-{b['stop_loss_pct']}%)" if b.get('stop_loss_price') else "なし (満期/反転決済)"
            lines.append(
                f"  [{i}] {b['ticker']} (予測リターン: +{b['expected_return_pct']}%, 合意: {b['consensus']})\n"
                f"      ・注文種別 : {b['order_type']}\n"
                f"      ・発注数量 : {b['shares']} 株 (約 ${b['allocated_usd']:,.2f} USD)\n"
                f"      ・OCO設定 : 利確(TP) {tp_str} / 損切(SL) {sl_str}"
            )
    else:
        lines.append("\n【1. 新規買い発注】\n  ・新規発注なし (空きスロットなし、またはシグナル閾値未満)")

    if sheet["exits"]:
        lines.append("\n【2. 手仕舞い発注】（米国市場オープン前: 寄付成行 MOO 売）")
        for e in sheet["exits"]:
            lines.append(f"  ・{e['ticker']} : {e['shares']} 株 ➔ {e['action']} ({e['reason']})")
    else:
        lines.append("\n【2. 手仕舞い発注】\n  ・手仕舞い対象なし")

    if sheet["holds"]:
        lines.append("\n【3. 継続保有 (放置)】")
        for h in sheet["holds"]:
            pnl_sign = "+" if h['pnl_pct'] >= 0 else ""
            lines.append(
                f"  ・{h['ticker']} : {h['shares']} 株 (保有 {h['holding_days']} 日目 / 損益: {pnl_sign}{h['pnl_pct']}%) ➔ OCO維持のままホールド"
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

    # Convert trades to rows
    trade_rows = []
    for t in result.trades:
        trade_rows.append({
            "ticker": t.ticker,
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
    height: 380px;
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
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>📈 {report_title}</h1>
      <p style="margin:4px 0 0 0; color:#8b949e;">moomoo 手動執行モデル (SGT 20:00-24:00) | 初期資金: ${result.initial_capital:,.2f} USD</p>
    </div>
    <div class="badge">Walk-Forward 検証済</div>
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
      <div class="label">勝率 (Win Rate)</div>
      <div class="val">{result.win_rate * 100:.1f}%</div>
    </div>
    <div class="card">
      <div class="label">プロフィットファクター (PF)</div>
      <div class="val">{result.profit_factor:.2f}</div>
    </div>
    <div class="card">
      <div class="label">シャープレシオ</div>
      <div class="val">{result.sharpe_ratio:.2f}</div>
    </div>
    <div class="card">
      <div class="label">最大ドローダウン</div>
      <div class="val red">-{result.max_drawdown * 100:.2f}%</div>
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
          <th>Entry日</th>
          <th>買付値</th>
          <th>株数</th>
          <th>Exit日</th>
          <th>売却値</th>
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
      fill: true,
      tension: 0.1,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
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
    return html_file


def _render_action_sheet_html(sheet: dict[str, Any]) -> str:
    buys_html = ""
    if sheet.get("buys"):
        for b in sheet["buys"]:
            tp_txt = f"利確指値 <b>${b['take_profit_price']}</b> (+{b['take_profit_pct']}%)" if b.get('take_profit_price') else "利確指値: なし (満期決済)"
            sl_txt = f"損切逆指値 <b>${b['stop_loss_price']}</b> (-{b['stop_loss_pct']}%)" if b.get('stop_loss_price') else "損切逆指値: なし (満期/反転決済)"
            buys_html += f"""
            <div style="background:#21262d; border-radius:6px; padding:12px; margin-top:8px;">
              <div style="font-size:16px; font-weight:bold; color:#58a6ff;">🚀 {b['ticker']} (予測リターン: +{b['expected_return_pct']}%, {b['consensus']})</div>
              <div style="margin-top:6px; font-size:14px;">
                ・発注: <b>{b['order_type']}</b> | 数量: <b>{b['shares']} 株</b> (約 ${b['allocated_usd']:,.2f} USD)<br>
                ・moomoo OCO設定: {tp_txt} / {sl_txt}
              </div>
            </div>
            """
    else:
        buys_html = "<div style='color:#8b949e; margin-top:8px;'>新規発注なし</div>"

    exits_html = ""
    if sheet.get("exits"):
        for e in sheet["exits"]:
            exits_html += f"<div>・<b>{e['ticker']}</b> ({e['shares']}株) ➔ {e['action']} ({e['reason']})</div>"
    else:
        exits_html = "<div style='color:#8b949e;'>手仕舞い対象なし</div>"

    return f"""
    <div class="action-box">
      <h2 style="margin-top:0; color:#58a6ff;">📋 今夜の moomoo 発注アクション ({sheet.get('operating_time_sgt')})</h2>
      <p style="color:#8b949e; margin-bottom:12px;">空きスロット: <b>{sheet.get('available_slots')}</b> / {sheet.get('max_slots')} 枠 | 資金: ${sheet.get('initial_capital_usd', 35000):,.2f} USD</p>
      <div style="margin-bottom:16px;">
        <h3 style="margin-bottom:4px; color:#f0f6fc;">【新規買い発注】</h3>
        {buys_html}
      </div>
      <div>
        <h3 style="margin-bottom:4px; color:#f0f6fc;">【手仕舞い発注】</h3>
        {exits_html}
      </div>
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

    return f"""
    <tr>
      <td><b>{t.get('ticker')}</b></td>
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
