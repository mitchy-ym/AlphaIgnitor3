  // Toast notification helper & resilient clipboard copy (works on HTTP LAN and mobile)
  function showCopyToast() {
    const toast = document.getElementById('copy-toast');
    if (toast) {
      toast.textContent = '✓ 発注指示をコピーしました';
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2500);
    }
  }

  function fallbackCopy(text) {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const success = document.execCommand('copy');
      document.body.removeChild(ta);
      if (success) {
        showCopyToast();
        return;
      }
    } catch (e) {}
    prompt('発注指示をコピーしてください:', text);
  }

  function copyOrderText(btn) {
    const text = btn.getAttribute('data-copy');
    if (!text) return;
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(showCopyToast).catch(() => {
        fallbackCopy(text);
      });
    } else {
      fallbackCopy(text);
    }
  }

  (function () {
    const table = document.getElementById('all-tickers');
    if (!table) return;
    const tbody = table.tBodies && table.tBodies[0];
    if (!tbody) return;

    const ths = Array.from(table.querySelectorAll('thead th'));
    const rows = Array.from(tbody.rows);

    // Initial state: default sort indicator on Avg or Bull
    let state = { idx: -1, asc: false };
    // Locate the 'Avg' or '🟢' column index
    const bullColIdx = ths.findIndex(th => th.textContent.includes('🟢'));
    if (bullColIdx >= 0) state = { idx: bullColIdx, asc: false };

    // ── Search & Filter Logic ─────────────────────────────────────────────
    let searchTerm = '';
    let activeSectors = null; // null = no filter (show all)
    let activeSignals = null;

    const sectorThIdx = ths.findIndex(th => th.getAttribute('data-filterable') === 'sector');
    const signalThIdx = ths.findIndex(th => th.getAttribute('data-filterable') === 'signal');
    const tickerThIdx = ths.findIndex(th => th.textContent.trim().startsWith('Ticker'));
    const nameThIdx   = ths.findIndex(th => th.textContent.trim().startsWith('Name'));

    function applyFilters() {
      let visibleCount = 0;
      rows.forEach(r => {
        let show = true;

        // Search text filter
        if (searchTerm) {
          const tText = (tickerThIdx >= 0 && r.children[tickerThIdx]) ? r.children[tickerThIdx].textContent.toLowerCase() : '';
          const nText = (nameThIdx >= 0 && r.children[nameThIdx]) ? r.children[nameThIdx].textContent.toLowerCase() : '';
          if (!tText.includes(searchTerm) && !nText.includes(searchTerm)) {
            show = false;
          }
        }

        // Sector filter
        if (show && activeSectors !== null && sectorThIdx >= 0) {
          const sec = r.children[sectorThIdx] ? r.children[sectorThIdx].textContent.trim() : '';
          if (!activeSectors.has(sec)) show = false;
        }

        // Signal filter
        if (show && activeSignals !== null && signalThIdx >= 0) {
          const sig = r.children[signalThIdx] ? r.children[signalThIdx].textContent.trim() : '';
          if (!activeSignals.has(sig)) show = false;
        }

        r.style.display = show ? '' : 'none';
        if (show) visibleCount++;
      });

      const badge = document.getElementById('row-count-badge');
      if (badge) badge.textContent = `${visibleCount} / ${rows.length} 銘柄`;
    }

    // Search input event
    const searchInput = document.getElementById('table-search');
    if (searchInput) {
      searchInput.addEventListener('input', () => {
        searchTerm = searchInput.value.trim().toLowerCase();
        applyFilters();
      });
    }

    // ── Sector dropdown filter ────────────────────────────────────────────
    function allColValues(idx) {
      const s = new Set();
      rows.forEach(r => {
        const td = r.children[idx];
        if (td) s.add(td.textContent.trim());
      });
      return Array.from(s).sort();
    }

    if (sectorThIdx >= 0) {
      const sectorTh = ths[sectorThIdx];
      const dd = document.createElement('div');
      dd.className = 'sector-dropdown';

      function rebuildSectorDd() {
        dd.innerHTML = `
          <div class="dd-actions">
            <button type="button" class="btn-dd-all">Select All</button>
            <button type="button" class="btn-dd-none">Deselect All</button>
          </div>
        `;
        dd.querySelector('.btn-dd-all').onclick = (e) => {
          e.stopPropagation();
          dd.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = true);
        };
        dd.querySelector('.btn-dd-none').onclick = (e) => {
          e.stopPropagation();
          dd.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = false);
        };

        allColValues(sectorThIdx).forEach(sec => {
          const lbl = document.createElement('label');
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.value = sec;
          cb.checked = activeSectors === null || activeSectors.has(sec);
          cb.onclick = (e) => e.stopPropagation();
          lbl.appendChild(cb);
          lbl.appendChild(document.createTextNode(sec || '(blank)'));
          dd.appendChild(lbl);
        });
      }

      sectorTh.style.position = 'relative';
      sectorTh.appendChild(dd);

      sectorTh.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = dd.classList.contains('open');
        document.querySelectorAll('.sector-dropdown.open').forEach(d => d.classList.remove('open'));
        if (!isOpen) {
          rebuildSectorDd();
          dd.classList.add('open');
        }
      });

      document.addEventListener('click', () => {
        if (!dd.classList.contains('open')) return;
        dd.classList.remove('open');
        const checked = Array.from(dd.querySelectorAll('input[type=checkbox]:checked')).map(c => c.value);
        const allVals = allColValues(sectorThIdx);
        activeSectors = (checked.length === allVals.length) ? null : new Set(checked);
        sectorTh.classList.toggle('filter-active', activeSectors !== null);
        applyFilters();
      });
      dd.addEventListener('click', (e) => e.stopPropagation());
    }

    // ── Signal dropdown filter ────────────────────────────────────────────
    if (signalThIdx >= 0) {
      const signalTh = ths[signalThIdx];
      const dd = document.createElement('div');
      dd.className = 'sector-dropdown';

      function rebuildSignalDd() {
        dd.innerHTML = `
          <div class="dd-actions">
            <button type="button" class="btn-sig-all">Select All</button>
            <button type="button" class="btn-sig-none">Deselect All</button>
          </div>
        `;
        dd.querySelector('.btn-sig-all').onclick = (e) => {
          e.stopPropagation();
          dd.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = true);
        };
        dd.querySelector('.btn-sig-none').onclick = (e) => {
          e.stopPropagation();
          dd.querySelectorAll('input[type=checkbox]').forEach(c => c.checked = false);
        };

        allColValues(signalThIdx).forEach(sig => {
          const lbl = document.createElement('label');
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.value = sig;
          cb.checked = activeSignals === null || activeSignals.has(sig);
          cb.onclick = (e) => e.stopPropagation();
          lbl.appendChild(cb);
          lbl.appendChild(document.createTextNode(sig || '(blank)'));
          dd.appendChild(lbl);
        });
      }

      signalTh.style.position = 'relative';
      signalTh.appendChild(dd);

      signalTh.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = dd.classList.contains('open');
        document.querySelectorAll('.sector-dropdown.open').forEach(d => d.classList.remove('open'));
        if (!isOpen) {
          rebuildSignalDd();
          dd.classList.add('open');
        }
      });

      document.addEventListener('click', () => {
        if (!dd.classList.contains('open')) return;
        dd.classList.remove('open');
        const checked = Array.from(dd.querySelectorAll('input[type=checkbox]:checked')).map(c => c.value);
        const allVals = allColValues(signalThIdx);
        activeSignals = (checked.length === allVals.length) ? null : new Set(checked);
        signalTh.classList.toggle('filter-active', activeSignals !== null);
        applyFilters();
      });
      dd.addEventListener('click', (e) => e.stopPropagation());
    }

    // ── Chart Visibility Sync ─────────────────────────────────────────────
    function updateCharts() {
      const checkedTickers = new Set(
        Array.from(table.querySelectorAll('.row-check:checked')).map(cb => cb.dataset.ticker)
      );
      let visible = 0;
      document.querySelectorAll('#charts-container .chart-block').forEach(div => {
        const show = checkedTickers.has(div.dataset.ticker);
        div.style.display = show ? '' : 'none';
        if (show) visible++;
      });
      const noMsg = document.getElementById('no-charts-msg');
      if (noMsg) noMsg.style.display = visible === 0 ? '' : 'none';
      const label = document.getElementById('chart-count-label');
      if (label) label.textContent = visible > 0 ? `(${visible} 銘柄表示中)` : '';

      rows.forEach(r => {
        const cb = r.querySelector('.row-check');
        r.classList.toggle('checked-row', !!(cb && cb.checked));
      });
    }

    table.addEventListener('change', e => {
      if (e.target.classList.contains('row-check')) {
        updateCharts();
        syncMasterCheckbox();
      }
    });

    // ── Master Checkbox ───────────────────────────────────────────────────
    const masterCb = document.getElementById('check-all-master');
    function syncMasterCheckbox() {
      if (!masterCb) return;
      const visibleCbs = Array.from(table.querySelectorAll('tbody tr')).filter(r => r.style.display !== 'none').map(r => r.querySelector('.row-check')).filter(Boolean);
      const checkedCount = visibleCbs.filter(cb => cb.checked).length;
      if (checkedCount === 0) {
        masterCb.checked = false;
        masterCb.indeterminate = false;
      } else if (checkedCount === visibleCbs.length) {
        masterCb.checked = true;
        masterCb.indeterminate = false;
      } else {
        masterCb.checked = false;
        masterCb.indeterminate = true;
      }
    }
    if (masterCb) {
      masterCb.addEventListener('change', () => {
        const visibleCbs = Array.from(table.querySelectorAll('tbody tr')).filter(r => r.style.display !== 'none').map(r => r.querySelector('.row-check')).filter(Boolean);
        visibleCbs.forEach(cb => cb.checked = masterCb.checked);
        updateCharts();
      });
    }

    // ── Quick Select Toolbar Buttons ──────────────────────────────────────
    const btnSelRec = document.getElementById('btn-sel-rec');
    if (btnSelRec) {
      btnSelRec.addEventListener('click', () => {
        let hasRec = false;
        rows.forEach(r => {
          const cb = r.querySelector('.row-check');
          const isRec = r.getAttribute('data-recommended') === '1';
          if (cb) {
            cb.checked = isRec;
            if (isRec) hasRec = true;
          }
        });
        // If no recommended, select top 3
        if (!hasRec) {
          rows.slice(0, 3).forEach(r => {
            const cb = r.querySelector('.row-check');
            if (cb) cb.checked = true;
          });
        }
        updateCharts();
        syncMasterCheckbox();
      });
    }

    const btnSelTop5 = document.getElementById('btn-sel-top5');
    if (btnSelTop5) {
      btnSelTop5.addEventListener('click', () => {
        let checked = 0;
        rows.forEach(r => {
          const cb = r.querySelector('.row-check');
          if (cb) {
            if (r.style.display !== 'none' && checked < 5) {
              cb.checked = true;
              checked++;
            } else {
              cb.checked = false;
            }
          }
        });
        updateCharts();
        syncMasterCheckbox();
      });
    }

    const btnSelAll = document.getElementById('btn-sel-all');
    if (btnSelAll) {
      btnSelAll.addEventListener('click', () => {
        rows.forEach(r => {
          const cb = r.querySelector('.row-check');
          if (cb && r.style.display !== 'none') cb.checked = true;
        });
        updateCharts();
        syncMasterCheckbox();
      });
    }

    const btnSelClear = document.getElementById('btn-sel-clear');
    if (btnSelClear) {
      btnSelClear.addEventListener('click', () => {
        rows.forEach(r => {
          const cb = r.querySelector('.row-check');
          if (cb) cb.checked = false;
        });
        updateCharts();
        syncMasterCheckbox();
      });
    }

    // ── Column Sorting ────────────────────────────────────────────────────
    function cellText(tr, idx) {
      const td = tr.children[idx];
      return (td && td.textContent ? td.textContent : '').trim();
    }

    function parseSortValue(txt, sortType) {
      const s = (txt || '').trim();
      if (!s) return null;
      if (sortType === 'forecast') {
        const m = s.match(/^([+-]?[\\d.]+)%/);
        const v = m ? parseFloat(m[1]) : NaN;
        return Number.isFinite(v) ? v : null;
      }
      if (sortType === 'number') {
        const v = parseFloat(s);
        return Number.isFinite(v) ? v : null;
      }
      const v = parseFloat(s.replace(/,/g, ''));
      if (Number.isFinite(v) && /^[+-]?\\d/.test(s)) return v;
      return s.toLowerCase();
    }

    function compareValues(a, b) {
      if (a === null && b === null) return 0;
      if (a === null) return 1;
      if (b === null) return -1;
      if (typeof a === 'number' && typeof b === 'number') return a - b;
      return String(a).localeCompare(String(b), 'en', { numeric: true, sensitivity: 'base' });
    }

    ths.forEach((th, idx) => {
      if (idx === 0) return; // Checkbox column
      if (th.getAttribute('data-filterable')) return;
      th.addEventListener('click', () => {
        const sortType = th.getAttribute('data-sort-type') || 'string';
        let asc = (state.idx === idx) ? !state.asc : (sortType === 'string');
        state = { idx, asc };

        rows.sort((ra, rb) => {
          const va = parseSortValue(cellText(ra, idx), sortType);
          const vb = parseSortValue(cellText(rb, idx), sortType);
          const c = compareValues(va, vb);
          return asc ? c : -c;
        });

        rows.forEach(r => tbody.appendChild(r));

        ths.forEach(t => {
          const ind = t.querySelector('.sort-ind');
          if (ind) ind.textContent = '';
        });
        const ind = th.querySelector('.sort-ind');
        if (ind) ind.textContent = asc ? '↑' : '↓';
      });
    });

    // ── Horizon Columns Toggle ────────────────────────────────────────────
    const toggleDayBtn = document.getElementById('toggle-day-cols');
    if (toggleDayBtn) {
      toggleDayBtn.addEventListener('click', () => {
        const isActive = toggleDayBtn.classList.toggle('active');
        const displayVal = isActive ? 'table-cell' : 'none';
        table.querySelectorAll('.day-horizon-col').forEach(el => { el.style.display = displayVal; });
      });
    }

    // ── Initial Selection on Page Load ────────────────────────────────────
    // Prioritize recommended buy rows!
    const recCheckboxes = Array.from(table.querySelectorAll('.row-check[data-recommended="1"]'));
    if (recCheckboxes.length > 0) {
      recCheckboxes.forEach(cb => cb.checked = true);
    } else {
      // Fallback: select top 3
      rows.slice(0, 3).forEach(r => {
        const cb = r.querySelector('.row-check');
        if (cb) cb.checked = true;
      });
    }

    updateCharts();
    syncMasterCheckbox();

    // ── Chart Signal Overlay (Hover + Tap for Mobile) ─────────────────────
    document.querySelectorAll('.chart-block[data-signals]').forEach(function(block) {
      const overlay = document.createElement('div');
      overlay.className = 'signal-overlay';
      block.appendChild(overlay);

      function renderOverlayContent() {
        try {
          const data = JSON.parse(block.dataset.signals);
          const v = data.verdict;
          const verdictText = v === 'bullish' ? '🟢 Bullish' : v === 'bearish' ? '🔴 Bearish' : '🟡 Neutral';
          const verdictCls  = v === 'bullish' ? 'sig-bull' : v === 'bearish' ? 'sig-bear' : 'sig-neutral';
          let h = '<div class="ov-badge ' + verdictCls + '">' + verdictText
                + ' <span style="font-weight:normal">🟢' + data.bull + ' 🔴' + data.bear + '</span></div>';
          const cats = {};
          (data.items || []).forEach(item => {
            if (!cats[item.category]) cats[item.category] = [];
            cats[item.category].push(item);
          });
          for (const cat in cats) {
            h += '<div class="ov-cat">' + cat + '</div>';
            cats[cat].forEach(item => {
              const icon = item.bullish === true ? '🟢' : item.bullish === false ? '🔴' : '🟡';
              const cls  = item.bullish === true ? 'sig-bull' : item.bullish === false ? 'sig-bear' : 'sig-neutral';
              h += '<div class="ov-item ' + cls + '">' + icon + ' ' + item.label + '</div>';
            });
          }
          overlay.innerHTML = h;
        } catch(e) {}
      }

      // Desktop hover
      block.addEventListener('mouseenter', () => {
        renderOverlayContent();
        overlay.style.display = 'block';
      });
      block.addEventListener('mouseleave', () => {
        overlay.style.display = 'none';
      });

      // Mobile touch toggle
      block.addEventListener('click', (e) => {
        if (overlay.style.display === 'block') {
          overlay.style.display = 'none';
        } else {
          renderOverlayContent();
          overlay.style.display = 'block';
        }
      });
    });

  })();
