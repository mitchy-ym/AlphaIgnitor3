"""HTMLレンダリングモジュール（テーブルとページ全体）。"""
from __future__ import annotations

import html
import math

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
    tds = []
    if checkbox_col:
      tds.append(f'<td><input type="checkbox" class="row-check" data-ticker="{ticker_val}"></td>')
    for label, key in cols:
      v = row_dict.get(key)
      if v is None or (isinstance(v, float) and not np.isfinite(v)):
        s = ""
      else:
        s = str(v)
      if label.startswith("Day "):
        tds.append(f'<td class="day-horizon-col">{html.escape(s)}</td>')
      elif label in {"🟢", "🔴"}:
        tds.append(f'<td style="text-align:center">{html.escape(s)}</td>')
      else:
        tds.append(f"<td>{html.escape(s)}</td>")
    rows_html.append("<tr>" + "".join(tds) + "</tr>")

  id_attr = f' id="{html.escape(table_id)}"' if table_id else ""
  return f"<table{id_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"


def render_html(
    *,
    summary_dl: str,
    acc_html: str,
    all_tbl: pd.DataFrame,
    cols: list[tuple[str, str]],
    charts_html_block: str,
    max_horizon: int,
) -> str:
    """完全なHTMLレポートドキュメントを生成する。"""
    return f"""
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Daily Forecast Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    body {{
      font-family: 'Inter', Meiryo, "Meiryo UI", sans-serif;
      margin: 24px;
      color: #1e293b;
      background-color: #f8fafc;
      line-height: 1.5;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-family: 'Outfit', sans-serif;
      font-size: 28px;
      font-weight: 800;
      background: linear-gradient(135deg, #4f46e5, #06b6d4);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.02em;
    }}
    h2 {{
      margin: 0 0 14px 0;
      font-family: 'Outfit', sans-serif;
      font-size: 18px;
      font-weight: 600;
      color: #0f172a;
      letter-spacing: -0.01em;
    }}
    .sub {{
      color: #64748b;
      margin-bottom: 20px;
      font-size: 14px;
      font-weight: 500;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05), 0 0 0 1px rgba(0, 0, 0, 0.01);
      transition: box-shadow 0.3s ease;
    }}
    section:hover {{
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    }}
    dl {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 10px 14px;
      margin: 0;
      font-size: 13px;
    }}
    dt {{
      color: #64748b;
      font-weight: 600;
    }}
    dd {{
      color: #1e293b;
      font-weight: 500;
      margin: 0;
    }}
    /* Left-right split layout */
    .split-layout {{
      display: flex;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 20px;
    }}
    .split-layout:last-of-type {{
      margin-bottom: 0;
    }}
    .split-top-row {{
      align-items: stretch;
    }}
    .split-top-row .left-panel,
    .split-top-row .right-panel {{
      display: flex;
      flex-direction: column;
    }}
    .split-top-row section {{
      flex: 1;
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
    /* Table */
    .table-scroll {{
      overflow-x: auto;
      border-radius: 8px;
      border: 1px solid #e2e8f0;
    }}
    table {{
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #f1f5f9;
      padding: 10px 12px;
      text-align: left;
      white-space: nowrap;
      transition: background-color 0.2s ease;
    }}
    th {{
      font-weight: 600;
      background: #f8fafc;
      color: #475569;
      border-bottom: 2px solid #e2e8f0;
    }}
    th.sortable {{
      cursor: pointer;
      user-select: none;
    }}
    th.sortable:hover {{
      background: #f1f5f9;
      color: #0f172a;
    }}
    th.sortable .sort-ind {{
      margin-left: 6px;
      color: #4f46e5;
      font-weight: 700;
    }}
    td:first-child, th:first-child {{
      text-align: center;
      width: 36px;
      padding: 8px 4px;
    }}
    input.row-check {{
      cursor: pointer;
      width: 16px;
      height: 16px;
      accent-color: #4f46e5;
    }}
    tr:hover td {{
      background-color: #f8fafc;
    }}
    tr.checked-row td {{
      background-color: #e0e7ff !important;
    }}
    /* Charts panel */
    .charts-container {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .chart-block {{
      position: relative;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }}
    .chart-block h3 {{
      margin: 0 0 8px 0;
      font-family: 'Outfit', sans-serif;
      font-size: 14px;
      font-weight: 600;
      color: #0f172a;
    }}
    .chart-block img, .chart-block svg {{
      width: 100%;
      height: auto;
      border: 1px solid #f1f5f9;
      border-radius: 6px;
    }}
    /* Signal overlay */
    .signal-overlay {{
      position: absolute;
      top: 36px;
      right: 12px;
      z-index: 100;
      background: rgba(255, 255, 255, 0.98);
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 11px;
      line-height: 1.6;
      pointer-events: none;
      max-width: 230px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
      display: none;
      backdrop-filter: blur(4px);
    }}
    .signal-overlay .ov-badge {{
      font-family: 'Outfit', sans-serif;
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 6px;
      padding-bottom: 6px;
      border-bottom: 1px solid #f1f5f9;
    }}
    .signal-overlay .ov-cat {{
      font-weight: 700;
      color: #94a3b8;
      margin: 6px 0 3px;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .signal-overlay .ov-item {{
      padding: 2px 0;
      font-weight: 500;
    }}
    .sig-bull {{ color: #059669; }}
    .sig-bear {{ color: #dc2626; }}
    .sig-neutral {{ color: #475569; }}
    .note {{
      color: #64748b;
      font-size: 12px;
      font-weight: 400;
    }}
    #no-charts-msg {{
      padding: 24px 0;
      text-align: center;
    }}
    #check-all-master {{
      cursor: pointer;
      width: 16px;
      height: 16px;
      accent-color: #4f46e5;
    }}
    /* Sector filter dropdown */
    th.filterable {{
      cursor: pointer;
      user-select: none;
      position: relative;
    }}
    th.filterable .filter-ind {{
      color: #94a3b8;
    }}
    th.filterable.filter-active {{
      color: #4f46e5;
    }}
    th.filterable.filter-active .filter-ind {{
      color: #4f46e5;
      font-weight: 900;
    }}
    .sector-dropdown {{
      display: none;
      position: absolute;
      top: 100%;
      left: 0;
      z-index: 200;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
      min-width: 200px;
      max-height: 320px;
      overflow-y: auto;
      padding: 8px 0;
      font-size: 12px;
      font-weight: normal;
      white-space: nowrap;
    }}
    .sector-dropdown.open {{
      display: block;
    }}
    .sector-dropdown label {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      cursor: pointer;
      color: #334155;
      font-weight: 500;
    }}
    .sector-dropdown label:hover {{
      background: #f1f5f9;
      color: #0f172a;
    }}
    .sector-dropdown input[type=checkbox] {{
      accent-color: #4f46e5;
    }}
    .sector-dropdown .dd-divider {{
      border-top: 1px solid #f1f5f9;
      margin: 6px 0;
    }}
    .sector-dropdown .dd-actions {{
      display: flex;
      gap: 8px;
      padding: 4px 14px 6px;
    }}
    .sector-dropdown .dd-actions button {{
      font-family: 'Inter', sans-serif;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 8px;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      background: #ffffff;
      color: #475569;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .sector-dropdown .dd-actions button:hover {{
      background: #f1f5f9;
      color: #0f172a;
      border-color: #94a3b8;
    }}
    .day-horizon-col {{
      display: none;
    }}
    .toggle-btn {{
      font-family: 'Outfit', sans-serif;
      font-size: 12px;
      font-weight: 600;
      padding: 4px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #ffffff;
      color: #475569;
      cursor: pointer;
      margin-left: 12px;
      vertical-align: middle;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
      transition: all 0.2s ease;
    }}
    .toggle-btn:hover {{
      background: #f8fafc;
      border-color: #94a3b8;
      color: #0f172a;
    }}
    .toggle-btn.active {{
      background: #e0e7ff;
      border-color: #c7d2fe;
      color: #4f46e5;
    }}
    /* Directional accuracy panel */
    .acc-panel {{
      min-width: 0;
    }}
    .acc-panel h2 {{
      margin: 0 0 10px 0;
    }}
    .acc-table {{
      width: auto;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #e2e8f0;
    }}
    .acc-table th, .acc-table td {{
      border: 1px solid #e2e8f0;
      padding: 8px 12px;
      text-align: right;
    }}
    .acc-table th:first-child, .acc-table td:first-child {{
      text-align: left;
    }}
    .acc-table th:nth-child(2), .acc-table td:nth-child(2) {{
      text-align: center;
    }}
    .acc-table thead th {{
      background: #f8fafc;
      font-weight: 600;
      color: #475569;
      border-bottom: 2px solid #e2e8f0;
    }}
    .acc-table tfoot td {{
      background: #f1f5f9;
      font-weight: 700;
      color: #0f172a;
      border-top: 2px solid #cbd5e1;
    }}
    .acc-rate-hi {{
      color: #059669;
      font-weight: 700;
    }}
    .acc-rate-lo {{
      color: #dc2626;
      font-weight: 700;
    }}
    </style>
  </head>
<body>
  <h1>Daily Forecast Report</h1>
  <p class="sub">Ranking by Day {max_horizon} q0.5 from the shared multi-step forecast run</p>

  <div class="split-layout split-top-row">
    <div class="left-panel">
      <section>
        <h2>Summary</h2>
        <dl>{summary_dl}</dl>
      </section>
    </div>
    <div class="right-panel">
      <section>
        {acc_html}
        <p class="note">Directional accuracy is computed separately for Day 1 to Day 3 from the same shared multi-step forecast output.</p>
      </section>
    </div>
  </div>

  <div class="split-layout">
    <div class="left-panel">
      <section>
        <h2>All Tickers <span class="note" style="font-weight:normal;font-size:12px;">(click column to sort &middot; check to show chart)</span><button id="toggle-day-cols" class="toggle-btn">📅 Horizons</button></h2>
        <div class="table-scroll">{_render_table(all_tbl, cols, table_id="all-tickers", sortable=True, checkbox_col=True)}</div>
      </section>
    </div>
    <div class="right-panel">
      <section>
        <h2>Charts <span class="note" id="chart-count-label" style="font-weight:normal;font-size:12px;"></span></h2>
        <div class="charts-container" id="charts-container">
          {charts_html_block}
        </div>
      </section>
    </div>
  </div>

  <script>
  (function () {{
    const table = document.getElementById('all-tickers');
    if (!table) return;
    const tbody = table.tBodies && table.tBodies[0];
    if (!tbody) return;

    const ths = Array.from(table.querySelectorAll('thead th'));
    // Checkbox column is index 0; Ticker is index 1.
    // Server renders All Tickers sorted by 🟢 desc → Avg desc.
    // Reflect initial sort indicator on the 🟢 column (index 4).
    let state = {{ idx: 4, asc: false }};

    // ── Sector / Signal filter dropdowns ──────────────────────────────────
    let activeSectors = null; // null = no filter (show all)
    let activeSignals  = null;

    // Find th indexes by data-filterable attribute
    const sectorThIdx = ths.findIndex(th => th.getAttribute('data-filterable') === 'sector');
    const sectorTdIdx = sectorThIdx;
    const signalThIdx = ths.findIndex(th => th.getAttribute('data-filterable') === 'signal');
    const signalTdIdx = signalThIdx;

    // Collect unique values from a given column index
    function allColValues(tdIdx) {{
      const vals = new Set();
      Array.from(tbody.rows).forEach(r => {{
        const td = r.children[tdIdx];
        vals.add(td ? td.textContent.trim() : '');
      }});
      return Array.from(vals).sort();
    }}

    function allSectorValues() {{ return allColValues(sectorTdIdx); }}
    function allSignalValues() {{ return allColValues(signalTdIdx); }}

    function applyRowFilter() {{
      Array.from(tbody.rows).forEach(r => {{
        let show = true;
        if (activeSectors !== null) {{
          const td = r.children[sectorTdIdx];
          const sec = td ? td.textContent.trim() : '';
          if (!activeSectors.has(sec)) show = false;
        }}
        if (activeSignals !== null) {{
          const td = r.children[signalTdIdx];
          const sig = td ? td.textContent.trim() : '';
          if (!activeSignals.has(sig)) show = false;
        }}
        r.style.display = show ? '' : 'none';
      }});
    }}

    if (sectorThIdx >= 0) {{
      const sectorTh = ths[sectorThIdx];

      // Build dropdown DOM
      const dropdown = document.createElement('div');
      dropdown.className = 'sector-dropdown';

      function rebuildDropdown() {{
        dropdown.innerHTML = '';
        // Action buttons row
        const actions = document.createElement('div');
        actions.className = 'dd-actions';
        const btnAll = document.createElement('button');
        btnAll.textContent = 'Select All';
        btnAll.addEventListener('click', e => {{
          e.stopPropagation();
          dropdown.querySelectorAll('input[type=checkbox]').forEach(cb => {{ cb.checked = true; }});
        }});
        const btnNone = document.createElement('button');
        btnNone.textContent = 'Deselect All';
        btnNone.addEventListener('click', e => {{
          e.stopPropagation();
          dropdown.querySelectorAll('input[type=checkbox]').forEach(cb => {{ cb.checked = false; }});
        }});
        actions.appendChild(btnAll);
        actions.appendChild(btnNone);
        dropdown.appendChild(actions);
        const divider = document.createElement('div');
        divider.className = 'dd-divider';
        dropdown.appendChild(divider);

        const sectors = allSectorValues();
        sectors.forEach(sec => {{
          const lbl = document.createElement('label');
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.value = sec;
          cb.checked = activeSectors === null || activeSectors.has(sec);
          cb.addEventListener('click', e => e.stopPropagation());
          lbl.appendChild(cb);
          lbl.appendChild(document.createTextNode(sec || '(blank)'));
          dropdown.appendChild(lbl);
        }});
      }}

      sectorTh.style.position = 'relative';
      sectorTh.appendChild(dropdown);

      // Toggle open/close
      sectorTh.addEventListener('click', e => {{
        e.stopPropagation();
        const isOpen = dropdown.classList.contains('open');
        // Close any other open dropdowns first
        document.querySelectorAll('.sector-dropdown.open').forEach(d => d.classList.remove('open'));
        if (!isOpen) {{
          rebuildDropdown();
          dropdown.classList.add('open');
        }}
      }});

      // Apply button (clicking outside closes and applies)
      document.addEventListener('click', () => {{
        if (!dropdown.classList.contains('open')) return;
        dropdown.classList.remove('open');
        // Collect checked values
        const checked = Array.from(dropdown.querySelectorAll('input[type=checkbox]:checked')).map(cb => cb.value);
        const allVals = allSectorValues();
        if (checked.length === allVals.length) {{
          activeSectors = null;
        }} else {{
          activeSectors = new Set(checked);
        }}
        applyRowFilter();
        // Reflect active state on th
        sectorTh.classList.toggle('filter-active', activeSectors !== null);
      }});

      // Prevent closing when clicking inside dropdown
      dropdown.addEventListener('click', e => e.stopPropagation());
    }}

    // ── Signal filter dropdown ────────────────────────────────────────────
    if (signalThIdx >= 0) {{
      const signalTh = ths[signalThIdx];

      const sigDropdown = document.createElement('div');
      sigDropdown.className = 'sector-dropdown';

      function rebuildSigDropdown() {{
        sigDropdown.innerHTML = '';
        const actions = document.createElement('div');
        actions.className = 'dd-actions';
        const btnAll = document.createElement('button');
        btnAll.textContent = 'Select All';
        btnAll.addEventListener('click', e => {{
          e.stopPropagation();
          sigDropdown.querySelectorAll('input[type=checkbox]').forEach(cb => {{ cb.checked = true; }});
        }});
        const btnNone = document.createElement('button');
        btnNone.textContent = 'Deselect All';
        btnNone.addEventListener('click', e => {{
          e.stopPropagation();
          sigDropdown.querySelectorAll('input[type=checkbox]').forEach(cb => {{ cb.checked = false; }});
        }});
        actions.appendChild(btnAll);
        actions.appendChild(btnNone);
        sigDropdown.appendChild(actions);
        const divider = document.createElement('div');
        divider.className = 'dd-divider';
        sigDropdown.appendChild(divider);

        const _allSigs = allSignalValues();
        // Sort: Bull (U+1F7E2=0x1F7E2) → Neutral (U+1F7E1) → Bear (U+1F534)
        const _sigPriority = (s) => {{
          const cp = s.codePointAt(0);
          if (cp === 0x1F7E2) return 0;
          if (cp === 0x1F7E1) return 1;
          if (cp === 0x1F534) return 2;
          return 3;
        }};
        const sigs = _allSigs.slice().sort((a, b) => _sigPriority(a) - _sigPriority(b));
        sigs.forEach(sig => {{
          const lbl = document.createElement('label');
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.value = sig;
          cb.checked = activeSignals === null || activeSignals.has(sig);
          cb.addEventListener('click', e => e.stopPropagation());
          lbl.appendChild(cb);
          lbl.appendChild(document.createTextNode(sig || '(blank)'));
          sigDropdown.appendChild(lbl);
        }});
      }}

      signalTh.style.position = 'relative';
      signalTh.appendChild(sigDropdown);

      signalTh.addEventListener('click', e => {{
        e.stopPropagation();
        const isOpen = sigDropdown.classList.contains('open');
        document.querySelectorAll('.sector-dropdown.open').forEach(d => d.classList.remove('open'));
        if (!isOpen) {{
          rebuildSigDropdown();
          sigDropdown.classList.add('open');
        }}
      }});

      document.addEventListener('click', () => {{
        if (!sigDropdown.classList.contains('open')) return;
        sigDropdown.classList.remove('open');
        const checked = Array.from(sigDropdown.querySelectorAll('input[type=checkbox]:checked')).map(cb => cb.value);
        const allVals = allSignalValues();
        if (checked.length === allVals.length) {{
          activeSignals = null;
        }} else {{
          activeSignals = new Set(checked);
        }}
        applyRowFilter();
        signalTh.classList.toggle('filter-active', activeSignals !== null);
      }});

      sigDropdown.addEventListener('click', e => e.stopPropagation());
    }}

    // ── Chart visibility ──────────────────────────────────────────────────
    function updateCharts() {{
      const checked = new Set(
        Array.from(table.querySelectorAll('.row-check:checked')).map(cb => cb.dataset.ticker)
      );
      let visible = 0;
      document.querySelectorAll('#charts-container .chart-block').forEach(div => {{
        const show = checked.has(div.dataset.ticker);
        div.style.display = show ? '' : 'none';
        if (show) visible++;
      }});
      const noMsg = document.getElementById('no-charts-msg');
      if (noMsg) noMsg.style.display = visible === 0 ? '' : 'none';
      const label = document.getElementById('chart-count-label');
      if (label) label.textContent = visible > 0 ? `(${{visible}} selected)` : '';
      // Highlight checked rows
      Array.from(tbody.rows).forEach(r => {{
        const cb = r.querySelector('.row-check');
        r.classList.toggle('checked-row', !!(cb && cb.checked));
      }});
    }}

    // Delegate checkbox change events on the table
    table.addEventListener('change', e => {{
      if (e.target.classList.contains('row-check')) {{
        updateCharts();
        syncToggleBtn();
      }}
    }});

    // ── Master checkbox (select-all) ─────────────────────────────────────────
    const masterCb = document.getElementById('check-all-master');
    function syncToggleBtn() {{
      if (!masterCb) return;
      const all = Array.from(table.querySelectorAll('.row-check'));
      const checkedCount = all.filter(cb => cb.checked).length;
      if (checkedCount === 0) {{
        masterCb.checked = false;
        masterCb.indeterminate = false;
      }} else if (checkedCount === all.length) {{
        masterCb.checked = true;
        masterCb.indeterminate = false;
      }} else {{
        masterCb.checked = false;
        masterCb.indeterminate = true;
      }}
    }}
    if (masterCb) {{
      masterCb.addEventListener('change', () => {{
        const all = Array.from(table.querySelectorAll('.row-check'));
        all.forEach(cb => {{ cb.checked = masterCb.checked; }});
        updateCharts();
        syncToggleBtn();
      }});
    }}

    // ── Table sort ────────────────────────────────────────────────────────
    function cellText(tr, idx) {{
      const td = tr.children[idx];
      return (td && td.textContent ? td.textContent : '').trim();
    }}

    function parseValue(txt, sortType) {{
      const s = (txt || '').trim();
      if (!s) return null;
      if (sortType === 'forecast') {{
        // Format: '+0.66% [-4.1/+4.2]pp' — sort by the leading percentage (q0.5)
        const m = s.match(/^([+-]?[\\d.]+)%/);
        const v = m ? parseFloat(m[1]) : NaN;
        return Number.isFinite(v) ? v : null;
      }}
      if (sortType === 'number') {{
        const v = parseFloat(s);
        return Number.isFinite(v) ? v : null;
      }}
      // Fallback: try numeric
      const v = parseFloat(s.replace(/,/g, ''));
      if (Number.isFinite(v) && /^[+-]?\\d/.test(s)) return v;
      return s.toLowerCase();
    }}

    function compare(a, b) {{
      if (a === null && b === null) return 0;
      if (a === null) return 1;
      if (b === null) return -1;
      if (typeof a === 'number' && typeof b === 'number') return a - b;
      return String(a).localeCompare(String(b), 'en', {{ numeric: true, sensitivity: 'base' }});
    }}

    function clearIndicators() {{
      for (const th of ths) {{
        const ind = th.querySelector('.sort-ind');
        if (ind) ind.textContent = '';
      }}
    }}

    function setIndicator(th, asc) {{
      const ind = th.querySelector('.sort-ind');
      if (ind) ind.textContent = asc ? '↑' : '↓';
    }}

    function defaultDirection(sortType) {{
      if (sortType === 'forecast') return false;
      if (sortType === 'number') return false;
      return true;
    }}

    ths.forEach((th, idx) => {{
      if (idx === 0) return; // skip checkbox column
      if (th.getAttribute('data-filterable')) return; // skip filterable columns
      th.addEventListener('click', () => {{
        const sortType = th.getAttribute('data-sort-type') || 'string';
        const rows = Array.from(tbody.rows);

        let asc;
        if (state.idx === idx) {{
          asc = !state.asc;
        }} else {{
          asc = defaultDirection(sortType);
        }}
        state = {{ idx, asc }};

        rows.sort((ra, rb) => {{
          const va = parseValue(cellText(ra, idx), sortType);
          const vb = parseValue(cellText(rb, idx), sortType);
          const c = compare(va, vb);
          return asc ? c : -c;
        }});

        for (const r of rows) tbody.appendChild(r);

        clearIndicators();
        setIndicator(th, asc);
      }});
    }});

    // ── Horizon column toggle ────────────────────────────────────────────
    const toggleDayColsBtn = document.getElementById('toggle-day-cols');
    if (toggleDayColsBtn) {{
      toggleDayColsBtn.addEventListener('click', () => {{
        const isActive = toggleDayColsBtn.classList.toggle('active');
        const d = isActive ? 'table-cell' : 'none';
        table.querySelectorAll('.day-horizon-col').forEach(el => {{ el.style.display = d; }});
      }});
    }}

    // ── Initialisation ────────────────────────────────────────────────────
    // Check top 5 rows (q0.5 desc — server sort order)
    Array.from(tbody.rows).slice(0, 5).forEach(r => {{
      const cb = r.querySelector('.row-check');
      if (cb) cb.checked = true;
    }});

    updateCharts();
    syncToggleBtn();

    // Initial sort indicator: 🟢 ↓ (index 4 with checkbox col at 0)
    clearIndicators();
    if (ths[4]) setIndicator(ths[4], false);

    // ── Signal overlay on chart hover ─────────────────────────────────────
    document.querySelectorAll('.chart-block[data-signals]').forEach(function(block) {{
      const overlay = document.createElement('div');
      overlay.className = 'signal-overlay';
      block.appendChild(overlay);

      block.addEventListener('mouseenter', function() {{
        try {{
          const data = JSON.parse(block.dataset.signals);
          const v = data.verdict;
          const verdictText = v === 'bullish' ? '🟢 Bullish' : v === 'bearish' ? '🔴 Bearish' : '🟡 Neutral';
          const verdictCls  = v === 'bullish' ? 'sig-bull' : v === 'bearish' ? 'sig-bear' : 'sig-neutral';
          let h = '<div class="ov-badge ' + verdictCls + '">' + verdictText
                + ' <span style="font-weight:normal">🟢' + data.bull + ' 🔴' + data.bear + '</span></div>';
          // Group items by category
          const cats = {{}};
          (data.items || []).forEach(function(item) {{
            if (!cats[item.category]) cats[item.category] = [];
            cats[item.category].push(item);
          }});
          for (const cat in cats) {{
            h += '<div class="ov-cat">' + cat + '</div>';
            cats[cat].forEach(function(item) {{
              const icon = item.bullish === true ? '🟢' : item.bullish === false ? '🔴' : '🟡';
              const cls  = item.bullish === true ? 'sig-bull' : item.bullish === false ? 'sig-bear' : 'sig-neutral';
              h += '<div class="ov-item ' + cls + '">' + icon + ' ' + item.label + '</div>';
            }});
          }}
          overlay.innerHTML = h;
          overlay.style.display = 'block';
        }} catch(e) {{}}
      }});

      block.addEventListener('mouseleave', function() {{
        overlay.style.display = 'none';
      }});
    }});

  }})();
  </script>
</body>
</html>
"""
