"""
AlphaIgnitor3 - Portal Generator
Scans the report directory and generates an executive-grade index.html portal
for serving from ASUSTOR Drivestor 2 NAS Web root or local viewing.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path


def format_size(num_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def get_weekday_jp(date_str: str) -> str:
    """Return Japanese day of week for a YYYY-MM-DD string."""
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        return weekdays[dt.weekday()]
    except Exception:
        return ""


def scan_reports(report_dir: Path) -> dict:
    """Scan report directory for daily forecast reports and backtest reports."""
    daily_reports = []
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    if report_dir.exists():
        for entry in sorted(report_dir.iterdir(), reverse=True):
            if entry.is_dir() and date_pattern.match(entry.name):
                report_file = entry / "report.html"
                if report_file.exists():
                    stat = report_file.stat()
                    daily_reports.append(
                        {
                            "date": entry.name,
                            "weekday": get_weekday_jp(entry.name),
                            "path": f"./{entry.name}/report.html",
                            "size_bytes": stat.st_size,
                            "size_str": format_size(stat.st_size),
                            "modified": datetime.datetime.fromtimestamp(
                                stat.st_mtime
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

    # Check backtest report
    backtest_info = None
    bt_file = report_dir / "backtest" / "backtest_report.html"
    summary_file = report_dir / "backtest" / "backtest_summary.json"
    if bt_file.exists():
        stat = bt_file.stat()
        summary_data = {}
        if summary_file.exists():
            try:
                summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        backtest_info = {
            "title": "AlphaIgnitor3 売買戦略バックテスト & moomoo運用レポート",
            "path": "./backtest/backtest_report.html",
            "size_str": format_size(stat.st_size),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "summary": summary_data,
        }

    return {
        "daily_reports": daily_reports,
        "backtest_info": backtest_info,
        "latest_daily": daily_reports[0] if daily_reports else None,
        "total_days": len(daily_reports),
        "generated_at": datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "tz_name": datetime.datetime.now().astimezone().strftime("%Z") or "SGT",
    }


def render_portal_html(data: dict) -> str:
    """Render the executive-grade dark dashboard portal HTML."""
    daily_reports = data["daily_reports"]
    backtest_info = data["backtest_info"]
    latest_daily = data["latest_daily"]
    total_days = data["total_days"]
    generated_at = data["generated_at"]

    reports_json = json.dumps(daily_reports, ensure_ascii=False)

    bt_kpi_val = "利用可能" if backtest_info else "未生成"
    bt_kpi_color = "#38bdf8" if backtest_info else "#94a3b8"
    bt_kpi_meta = "Trading Strategy & Backtest"
    bt_desc = "累積リターン推移、シャープレシオ、最大ドローダウン、勝率、および moomoo 実口座連携シミュレーションの詳細レポートです。"

    if backtest_info and backtest_info.get("summary"):
        s = backtest_info["summary"]
        if s.get("total_return_str"):
            bt_kpi_val = s["total_return_str"]
            bt_kpi_color = "#10b981" if not s["total_return_str"].startswith("-") else "#ef4444"
            bt_kpi_meta = f"Sharpe {s.get('sharpe_ratio_str', '-')} | DD {s.get('max_drawdown_str', '-')}"
            st = s.get("strategy", {})
            policy_txt = "空売り対応 / 最低3銘柄・目標5銘柄(上限無) / セクター1推奨(最大2)"
            bt_desc = (
                f"直近6ヶ月検証: 累積リターン <strong>{s.get('total_return_str')}</strong> "
                f"(Sharpe <strong>{s.get('sharpe_ratio_str')}</strong>, MaxDD <strong>{s.get('max_drawdown_str')}</strong>, "
                f"勝率 {s.get('win_rate_str')})。LOO推奨指値 + {policy_txt}。"
            )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AlphaIgnitor3 Analytics Portal | ASUSTOR NAS</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-base: #090d16;
      --bg-surface: #0f172a;
      --bg-card: rgba(30, 41, 59, 0.7);
      --bg-card-hover: rgba(51, 65, 85, 0.85);
      --border: rgba(148, 163, 184, 0.15);
      --border-highlight: rgba(99, 102, 241, 0.4);
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent-primary: #6366f1;
      --accent-secondary: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --gradient-brand: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
      --gradient-card: linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
      --gradient-glow: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.18), transparent 70%);
      --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.3);
      --shadow-lg: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
      --radius-lg: 16px;
      --radius-md: 12px;
      --radius-sm: 8px;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-base);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.6;
      background-image: var(--gradient-glow);
      background-attachment: fixed;
      padding-bottom: 60px;
    }}

    .container {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 0 24px;
    }}

    /* Top Navigation Bar */
    header.navbar {{
      border-bottom: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.8);
      backdrop-filter: blur(16px);
      position: sticky;
      top: 0;
      z-index: 50;
    }}
    .nav-content {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      height: 70px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
    }}
    .brand-icon {{
      width: 40px;
      height: 40px;
      background: var(--gradient-brand);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }}
    .brand-text {{
      font-family: 'Outfit', sans-serif;
      font-size: 22px;
      font-weight: 800;
      background: var(--gradient-brand);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.02em;
    }}
    .nas-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      color: #34d399;
    }}
    .pulse-dot {{
      width: 8px;
      height: 8px;
      background-color: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0% {{ transform: scale(0.95); opacity: 0.8; }}
      50% {{ transform: scale(1.3); opacity: 1; }}
      100% {{ transform: scale(0.95); opacity: 0.8; }}
    }}

    /* Hero Section */
    .hero {{
      padding: 48px 0 32px;
    }}
    .hero-title {{
      font-family: 'Outfit', sans-serif;
      font-size: clamp(28px, 4vw, 42px);
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.2;
      margin-bottom: 12px;
    }}
    .hero-subtitle {{
      font-size: 16px;
      color: var(--text-secondary);
      max-width: 720px;
      margin-bottom: 32px;
    }}

    /* Quick KPI Stats Bar */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 20px;
      transition: all 0.25s ease;
    }}
    .kpi-card:hover {{
      border-color: var(--border-highlight);
      transform: translateY(-2px);
    }}
    .kpi-label {{
      font-size: 13px;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .kpi-value {{
      font-family: 'Outfit', sans-serif;
      font-size: 26px;
      font-weight: 700;
      color: var(--text-primary);
    }}
    .kpi-meta {{
      font-size: 12px;
      color: var(--text-secondary);
      margin-top: 4px;
      font-family: 'JetBrains Mono', monospace;
    }}

    /* Featured Action Cards */
    .featured-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 24px;
      margin-bottom: 48px;
    }}
    .feature-card {{
      background: var(--gradient-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 32px;
      position: relative;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      box-shadow: var(--shadow-sm);
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .feature-card::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: var(--gradient-brand);
      opacity: 0.8;
    }}
    .feature-card:hover {{
      transform: translateY(-4px);
      border-color: var(--border-highlight);
      box-shadow: var(--shadow-lg), 0 0 30px rgba(99, 102, 241, 0.15);
    }}
    .feature-tag {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 16px;
      width: fit-content;
    }}
    .tag-latest {{
      background: rgba(99, 102, 241, 0.2);
      color: #818cf8;
      border: 1px solid rgba(99, 102, 241, 0.4);
    }}
    .tag-backtest {{
      background: rgba(6, 182, 212, 0.2);
      color: #38bdf8;
      border: 1px solid rgba(6, 182, 212, 0.4);
    }}
    .feature-title {{
      font-family: 'Outfit', sans-serif;
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 8px;
      color: var(--text-primary);
    }}
    .feature-desc {{
      font-size: 14px;
      color: var(--text-secondary);
      margin-bottom: 24px;
    }}
    .feature-action {{
      display: flex;
      gap: 12px;
      margin-top: auto;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 20px;
      border-radius: var(--radius-sm);
      font-size: 14px;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
      transition: all 0.2s ease;
      border: none;
    }}
    .btn-primary {{
      background: var(--gradient-brand);
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }}
    .btn-primary:hover {{
      opacity: 0.95;
      transform: scale(1.02);
      box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }}
    .btn-secondary {{
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-primary);
      border: 1px solid var(--border);
    }}
    .btn-secondary:hover {{
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.25);
    }}

    /* Archive Section */
    .section-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 24px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }}
    .section-title {{
      font-family: 'Outfit', sans-serif;
      font-size: 24px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .search-box {{
      position: relative;
      min-width: 260px;
    }}
    .search-input {{
      width: 100%;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 10px 16px 10px 38px;
      color: var(--text-primary);
      font-size: 14px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .search-input:focus {{
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }}
    .search-icon {{
      position: absolute;
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-muted);
      font-size: 14px;
    }}

    /* Daily Reports Grid */
    .reports-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }}
    .report-card {{
      background: var(--bg-card);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 20px;
      display: flex;
      flex-direction: column;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
    }}
    .report-card:hover {{
      background: var(--bg-card-hover);
      border-color: var(--border-highlight);
      transform: translateY(-3px);
      box-shadow: var(--shadow-sm);
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    .card-date {{
      font-family: 'Outfit', sans-serif;
      font-size: 20px;
      font-weight: 700;
      color: var(--text-primary);
    }}
    .card-weekday {{
      font-size: 13px;
      color: var(--accent-secondary);
      font-weight: 600;
      margin-left: 6px;
    }}
    .badge-pill {{
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 12px;
      background: rgba(99, 102, 241, 0.25);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.4);
    }}
    .card-meta {{
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 18px;
      font-family: 'JetBrains Mono', monospace;
    }}
    .card-actions {{
      display: flex;
      gap: 8px;
      margin-top: auto;
    }}
    .card-actions .btn {{
      padding: 8px 14px;
      font-size: 13px;
      flex: 1;
    }}

    /* Preview Modal (Drawer) */
    .modal-overlay {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      z-index: 100;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }}
    .modal-overlay.active {{
      display: flex;
    }}
    .modal-container {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      width: 95vw;
      max-width: 1400px;
      height: 90vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
      overflow: hidden;
      animation: modalIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    @keyframes modalIn {{
      from {{ transform: scale(0.96); opacity: 0; }}
      to {{ transform: scale(1); opacity: 1; }}
    }}
    .modal-header {{
      padding: 16px 24px;
      background: rgba(30, 41, 59, 0.9);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .modal-title {{
      font-family: 'Outfit', sans-serif;
      font-size: 18px;
      font-weight: 700;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .modal-actions {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .close-btn {{
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-size: 24px;
      cursor: pointer;
      line-height: 1;
      padding: 4px;
      transition: color 0.2s;
    }}
    .close-btn:hover {{
      color: #ef4444;
    }}
    .modal-iframe {{
      flex: 1;
      width: 100%;
      border: none;
      background: #ffffff;
    }}

    /* Footer */
    footer {{
      margin-top: 64px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      text-align: center;
      font-size: 13px;
      color: var(--text-muted);
    }}
    footer a {{
      color: var(--accent-secondary);
      text-decoration: none;
    }}

    /* Responsive adjustments */
    @media (max-width: 640px) {{
      .hero {{ padding: 24px 0 20px; }}
      .featured-grid {{ grid-template-columns: 1fr; }}
      .reports-grid {{ grid-template-columns: 1fr; }}
      .kpi-grid {{ grid-template-columns: 1fr 1fr; }}
      .modal-container {{ width: 100vw; height: 100vh; border-radius: 0; }}
    }}
  </style>
</head>
<body>

  <!-- Header / Navbar -->
  <header class="navbar">
    <div class="container nav-content">
      <a href="./index.html" class="brand">
        <div class="brand-icon">⚡</div>
        <div class="brand-text">AlphaIgnitor3</div>
      </a>
      <div class="nas-badge">
        <span class="pulse-dot"></span>
        <span>ASUSTOR NAS Web Server</span>
      </div>
    </div>
  </header>

  <main class="container">
    <!-- Hero Section -->
    <section class="hero">
      <h1 class="hero-title">Analytics & Forecast Portal</h1>
      <p class="hero-subtitle">
        AI日次株価予測（Zero-Shot Ensemble）・バックテスト戦略分析・moomoo運用レポートの統合ポータルです。
      </p>

      <!-- KPI Summary Bar -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">最新レポート日付</div>
          <div class="kpi-value">{latest_daily['date'] if latest_daily else 'N/A'}</div>
          <div class="kpi-meta">{latest_daily['weekday'] + '曜日' if latest_daily else ''}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">アーカイブ蓄積日数</div>
          <div class="kpi-value">{total_days} <span style="font-size: 16px; font-weight: normal; color: var(--text-secondary);">日分</span></div>
          <div class="kpi-meta">Daily Forecast Archive</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">バックテスト運用分析</div>
          <div class="kpi-value" style="color: {bt_kpi_color};">{bt_kpi_val}</div>
          <div class="kpi-meta">{bt_kpi_meta}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">ポータル最終更新</div>
          <div class="kpi-value" style="font-size: 18px; margin-top: 4px;">{generated_at.split(' ')[0]}</div>
          <div class="kpi-meta">{generated_at.split(' ')[1]} {data.get("tz_name", "SGT")}</div>
        </div>
      </div>

      <!-- Featured Launchpad Cards -->
      <div class="featured-grid">
        <!-- Latest Daily Forecast Card -->
        {f'''
        <div class="feature-card">
          <div>
            <span class="feature-tag tag-latest">★ LATEST DAILY FORECAST</span>
            <h2 class="feature-title">最新予測レポート ({latest_daily['date']})</h2>
            <p class="feature-desc">
              Chronos / TimesFM / Multi-step アンサンブルによる最新の株価シグナル、売買推奨、方向感一致率、インタラクティブチャートを閲覧できます。
            </p>
          </div>
          <div class="feature-action">
            <a href="{latest_daily['path']}" target="_blank" class="btn btn-primary">
              レポートを開く (別タブ) ↗
            </a>
            <button onclick="openPreview('{latest_daily['path']}', '{latest_daily['date']} 日次予測レポート')" class="btn btn-secondary">
              クイックプレビュー
            </button>
          </div>
        </div>
        ''' if latest_daily else ''}

        <!-- Backtest Report Card -->
        {f'''
        <div class="feature-card">
          <div>
            <span class="feature-tag tag-backtest">STRATEGY BACKTEST</span>
            <h2 class="feature-title">売買戦略バックテスト & moomoo運用</h2>
            <p class="feature-desc">
              {bt_desc}
            </p>
          </div>
          <div class="feature-action">
            <a href="{backtest_info['path']}" target="_blank" class="btn btn-primary">
              バックテストを開く ↗
            </a>
            <button onclick="openPreview('{backtest_info['path']}', '売買戦略バックテスト レポート')" class="btn btn-secondary">
              クイックプレビュー
            </button>
          </div>
        </div>
        ''' if backtest_info else ''}
      </div>
    </section>

    <!-- Archive Section -->
    <section>
      <div class="section-header">
        <h2 class="section-title">
          <span>📅 日次予測レポート アーカイブ</span>
          <span style="font-size: 14px; font-weight: normal; color: var(--text-muted);">({total_days} 件)</span>
        </h2>
        <div class="search-box">
          <span class="search-icon">🔍</span>
          <input type="text" id="searchInput" class="search-input" placeholder="日付で絞り込み (例: 2026-09)..." oninput="filterReports()">
        </div>
      </div>

      <div class="reports-grid" id="reportsGrid">
        {"".join([f'''
        <div class="report-card" data-date="{r['date']}">
          <div class="card-top">
            <div class="card-date">
              {r['date']}
              <span class="card-weekday">({r['weekday']})</span>
            </div>
            {'<span class="badge-pill">NEW</span>' if i == 0 else ''}
          </div>
          <div class="card-meta">
            <div>サイズ: {r['size_str']}</div>
            <div>更新: {r['modified']}</div>
          </div>
          <div class="card-actions">
            <a href="{r['path']}" target="_blank" class="btn btn-primary">開く ↗</a>
            <button onclick="openPreview('{r['path']}', '{r['date']} 日次予測レポート')" class="btn btn-secondary">プレビュー</button>
          </div>
        </div>
        ''' for i, r in enumerate(daily_reports)])}
      </div>
    </section>
  </main>

  <!-- In-page Quick Preview Modal -->
  <div id="previewModal" class="modal-overlay" onclick="handleModalOverlayClick(event)">
    <div class="modal-container">
      <div class="modal-header">
        <div class="modal-title" id="modalTitle">レポートプレビュー</div>
        <div class="modal-actions">
          <a id="modalExternalLink" href="#" target="_blank" class="btn btn-secondary" style="padding: 6px 14px; font-size: 13px;">
            別タブで全画面表示 ↗
          </a>
          <button class="close-btn" onclick="closePreview()">&times;</button>
        </div>
      </div>
      <iframe id="previewIframe" class="modal-iframe" src="about:blank"></iframe>
    </div>
  </div>

  <footer>
    <p>AlphaIgnitor3 Daily Pipeline &copy; 2026 | Automated Zero-Shot Ensemble Forecasting System</p>
    <p style="margin-top: 4px; font-size: 12px;">Synced to ASUSTOR Drivestor 2 (AS1102T) Web Root</p>
  </footer>

  <script>
    const reports = {reports_json};

    function filterReports() {{
      const query = document.getElementById('searchInput').value.trim().toLowerCase();
      const cards = document.querySelectorAll('.report-card');
      cards.forEach(card => {{
        const date = card.getAttribute('data-date').toLowerCase();
        if (date.includes(query)) {{
          card.style.display = 'flex';
        }} else {{
          card.style.display = 'none';
        }}
      }});
    }}

    function openPreview(url, title) {{
      const modal = document.getElementById('previewModal');
      const iframe = document.getElementById('previewIframe');
      const titleEl = document.getElementById('modalTitle');
      const extLink = document.getElementById('modalExternalLink');

      titleEl.textContent = title || 'レポートプレビュー';
      extLink.href = url;
      iframe.src = url;
      modal.classList.add('active');
      document.body.style.overflow = 'hidden';
    }}

    function closePreview() {{
      const modal = document.getElementById('previewModal');
      const iframe = document.getElementById('previewIframe');
      iframe.src = 'about:blank';
      modal.classList.remove('active');
      document.body.style.overflow = '';
    }}

    function handleModalOverlayClick(event) {{
      if (event.target.id === 'previewModal') {{
        closePreview();
      }}
    }}

    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') {{
        closePreview();
      }}
    }});
  </script>
</body>
</html>
"""


def generate_portal(*, report_dir: Path = Path("report"), output: Path | None = None) -> Path:
    """Generate index.html portal in the report directory."""
    r_dir = Path(report_dir).resolve()
    out_path = (Path(output) if output else (r_dir / "index.html")).resolve()

    data = scan_reports(r_dir)
    html_content = render_portal_html(data)
    out_path.write_text(html_content, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate AlphaIgnitor3 Portal index.html"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report"),
        help="Path to report directory (default: report)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output index.html path (default: <report-dir>/index.html)",
    )
    args = parser.parse_args()

    out_path = generate_portal(report_dir=args.report_dir, output=args.output)
    print(f"Portal index.html generated successfully at: {out_path}")


if __name__ == "__main__":
    main()
