"""SVGチャート生成モジュール。"""
from __future__ import annotations

import datetime as dt
import html
import math


def _svg_line_chart(
    *,
    dates: list[dt.date],
    closes: list[float],
    volumes: list[float] | None,
    asof: dt.date,
    forecast_dates: list[dt.date | None],
    q10s: list[float | None],
    q50s: list[float | None],
    q90s: list[float | None],
    forecast_opens: list[float | None],
    past_forecast_dates: list[dt.date] | None = None,
    past_forecast_q50_closes: list[float | None] | None = None,
    past_forecast_q10_closes: list[float | None] | None = None,
    past_forecast_q90_closes: list[float | None] | None = None,
    warmup_closes: list[float] | None = None,
    id_prefix: str = "",
) -> str:
    width = 560
    # Bottom room for x labels (5-trading-day spacing).
    # Reserve right margin for forecast price labels.
    ml, mr, mt, mb = 44, 82, 36, 75
    iw = width - ml - mr

    # Main (close) panel height matches previous layout.
    close_ih = 240 - mt - mb
    panel_gap = 8
    rsi_ih = 44
    macd_ih = max(int(round(close_ih / 3.0)), 1)
    ih_total = close_ih + panel_gap + rsi_ih + panel_gap + macd_ih
    height = int(mt + ih_total + mb)

    close_mt = mt
    close_mb = close_mt + close_ih
    rsi_mt = close_mb + panel_gap
    rsi_mb = rsi_mt + rsi_ih
    macd_mt = rsi_mb + panel_gap
    macd_mb = macd_mt + macd_ih

    axis_fs = 10

    pts = [(d, c) for d, c in zip(dates, closes) if isinstance(c, (int, float)) and math.isfinite(float(c))]
    if len(pts) < 2:
        return ""  # caller can handle

    d2, c2 = zip(*pts)
    yvals = list(map(float, c2))

    # Warmup closes (prepended for indicator stability, not drawn)
    _warmup = [float(v) for v in (warmup_closes or []) if math.isfinite(float(v))]
    _nwarm = len(_warmup)

    def fmt_day(d: dt.date) -> str:
        mon = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][d.month - 1]
        return f"{d.day:02d} {mon}."

    def fmt_price_with_pct(price: float, q_logret: float, step: float) -> str:
        p = fmt_tick(float(price), step)
        pct = (math.exp(float(q_logret)) - 1.0) * 100.0
        return f"{p}({pct:+.1f}%)"

    def nice_step(span: float, *, target_ticks: int = 5) -> float:
        span = float(span)
        if not math.isfinite(span) or span <= 0:
            return 1.0
        raw = span / max(target_ticks - 1, 1)
        exp = math.floor(math.log10(raw)) if raw > 0 else 0
        f = raw / (10 ** exp)
        if f <= 1:
            nf = 1
        elif f <= 2:
            nf = 2
        elif f <= 5:
            nf = 5
        else:
            nf = 10
        return float(nf) * (10 ** exp)

    def fmt_tick(v: float, step: float) -> str:
        v = float(v)
        if not math.isfinite(v):
            return ""
        if step >= 1:
            return f"${v:,.0f}"
        if step >= 0.1:
            return f"${v:,.1f}"
        return f"${v:,.2f}"

    def fmt_macd_tick(v: float, step: float) -> str:
        v = float(v)
        if not math.isfinite(v):
            return ""
        # Keep compact; MACD is typically small relative numbers.
        if step >= 1:
            return f"{v:+.0f}"
        if step >= 0.1:
            return f"{v:+.1f}"
        return f"{v:+.2f}"

    def ema(vals: list[float], span: int) -> list[float]:
        if not vals:
            return []
        a = 2.0 / (float(span) + 1.0)
        out = [float(vals[0])]
        for v in vals[1:]:
            out.append((a * float(v)) + ((1.0 - a) * out[-1]))
        return out

    # Pre-compute Bollinger Bands upper/lower to optionally expand yvals
    # Use warmup prefix so the band is stable from the first displayed point.
    _cv_early = _warmup + list(map(float, c2))
    _bb_period, _bb_k = 20, 2.0
    _bb_upper_e: list[float] = []
    _bb_lower_e: list[float] = []
    for _ie in range(len(_cv_early)):
        _we = _cv_early[max(0, _ie - _bb_period + 1):_ie + 1]
        _bm = sum(_we) / len(_we)
        _bs = math.sqrt(sum((_x - _bm) ** 2 for _x in _we) / len(_we)) if len(_we) > 1 else 0.0
        _bb_upper_e.append(_bm + _bb_k * _bs)
        _bb_lower_e.append(_bm - _bb_k * _bs)
    # Discard warmup prefix
    _bb_upper_e = _bb_upper_e[_nwarm:]
    _bb_lower_e = _bb_lower_e[_nwarm:]

    # Forecast (q*) is for y=log(C/O) on each forecast day.
    # For horizon h=0, use the actual open if available; for h>0, fallback to last actual close.
    n_horizons = len(forecast_dates)
    forecast_close10s_calc: list[float | None] = []
    forecast_close50s_calc: list[float | None] = []
    forecast_close90s_calc: list[float | None] = []
    _prev_close = float(c2[-1])
    for _h in range(n_horizons):
        _q10v = q10s[_h] if _h < len(q10s) else None
        _q50v = q50s[_h] if _h < len(q50s) else None
        _q90v = q90s[_h] if _h < len(q90s) else None
        _fo_raw = forecast_opens[_h] if _h < len(forecast_opens) else None
        if all(v is not None and math.isfinite(float(v)) for v in [_q10v, _q50v, _q90v]):
            _fo: float | None = None
            if _fo_raw is not None and math.isfinite(float(_fo_raw)):
                _fo = float(_fo_raw)
            elif math.isfinite(_prev_close) and _prev_close > 0:
                _fo = _prev_close
            if _fo is not None and _fo > 0:
                _c10 = _fo * math.exp(float(_q10v))
                _c50 = _fo * math.exp(float(_q50v))
                _c90 = _fo * math.exp(float(_q90v))
                forecast_close10s_calc.append(_c10)
                forecast_close50s_calc.append(_c50)
                forecast_close90s_calc.append(_c90)
                yvals.extend([_c10, _c50, _c90])
                _prev_close = _fo
                continue
        forecast_close10s_calc.append(None)
        forecast_close50s_calc.append(None)
        forecast_close90s_calc.append(None)

    # Expand yvals with BB bands and past forecast dots to auto-fit y range
    for _bv in _bb_upper_e + _bb_lower_e:
        if math.isfinite(_bv):
            yvals.append(_bv)
    if past_forecast_q50_closes:
        for _pv in past_forecast_q50_closes:
            if _pv is not None and math.isfinite(float(_pv)):
                yvals.append(float(_pv))

    ymin_raw = float(min(yvals))
    ymax_raw = float(max(yvals))
    if not math.isfinite(ymin_raw) or not math.isfinite(ymax_raw):
        return ""

    span = ymax_raw - ymin_raw
    if span <= 0:
        # Degenerate range: pad around the value.
        span = abs(ymax_raw) * 0.1
        if span <= 0:
            span = 1.0
        ymin_raw = ymax_raw - span
        ymax_raw = ymax_raw + span
        span = ymax_raw - ymin_raw

    step = nice_step(span, target_ticks=5)
    ymin = math.floor(ymin_raw / step) * step
    ymax = math.ceil(ymax_raw / step) * step

    # Avoid too many ticks; widen step if needed.
    while (ymax - ymin) / step > 6.01:
        step *= 2.0
        ymin = math.floor(ymin_raw / step) * step
        ymax = math.ceil(ymax_raw / step) * step

    if not math.isfinite(ymin) or not math.isfinite(ymax) or ymin == ymax:
        return ""

    def x_at(i: int, n: int) -> float:
        if n <= 1:
            return float(ml)
        return float(ml) + (iw * (i / (n - 1)))

    def y_at_close(v: float) -> float:
        v = float(v)
        return float(close_mt) + (close_ih * (1.0 - (v - ymin) / (ymax - ymin)))

    last_close = float(c2[-1])

    n = len(d2)
    # Reserve one extra x-slot per forecast horizon so they stay inside the frame.
    total_n = n + max(n_horizons, 1)
    path_cmds = []
    for i, v in enumerate(c2):
        x = x_at(i, total_n)
        y = y_at_close(float(v))
        path_cmds.append(("M" if i == 0 else "L", x, y))
    path_d = " ".join([f"{cmd}{x:.2f},{y:.2f}" for cmd, x, y in path_cmds])

    # Forecast x positions: one slot per horizon after last actual.
    x_last = x_at(n - 1, total_n)
    x_fcst_list = [x_at(n + _h, total_n) for _h in range(n_horizons)]
    x_right = x_at(total_n - 1, total_n)

    # MACD series (based on closes only) — computed over warmup+display, warmup sliced off
    close_vals = [float(v) for v in c2]
    _full_cv_macd = _warmup + close_vals
    _ema12_full = ema(_full_cv_macd, 12)
    _ema26_full = ema(_full_cv_macd, 26)
    ema12 = _ema12_full[_nwarm:]
    ema26 = _ema26_full[_nwarm:]
    macd_vals = [a - b for a, b in zip(ema12, ema26)]
    signal_vals = ema(macd_vals, 9)
    hist_vals = [m - s for m, s in zip(macd_vals, signal_vals)]

    # Forecast MACD: continue EMAs from last actual values
    _valid_fcst_h = [_h for _h in range(n_horizons) if forecast_close50s_calc[_h] is not None]
    fcst50_cls = [float(forecast_close50s_calc[_h]) for _h in _valid_fcst_h]
    fcst10_cls = [
        float(forecast_close10s_calc[_h]) if forecast_close10s_calc[_h] is not None else float(forecast_close50s_calc[_h])
        for _h in _valid_fcst_h
    ]
    fcst90_cls = [
        float(forecast_close90s_calc[_h]) if forecast_close90s_calc[_h] is not None else float(forecast_close50s_calc[_h])
        for _h in _valid_fcst_h
    ]

    def _cont_ema(last_v: float, vals: list[float], span: int) -> list[float]:
        _a = 2.0 / (float(span) + 1.0)
        _out: list[float] = []
        _p = last_v
        for _v in vals:
            _p = _a * _v + (1.0 - _a) * _p
            _out.append(_p)
        return _out

    def _fcst_macd(closes: list[float]) -> tuple[list[float], list[float], list[float]]:
        if not closes:
            return [], [], []
        _e12 = _cont_ema(ema12[-1], closes, 12)
        _e26 = _cont_ema(ema26[-1], closes, 26)
        _m = [_x - _y for _x, _y in zip(_e12, _e26)]
        _s = _cont_ema(signal_vals[-1], _m, 9)
        _h = [_x - _y for _x, _y in zip(_m, _s)]
        return _m, _s, _h

    fcst_macd50, fcst_sig50, fcst_hist_f50 = _fcst_macd(fcst50_cls)
    fcst_macd10, _, _ = _fcst_macd(fcst10_cls)
    fcst_macd90, _, _ = _fcst_macd(fcst90_cls)

    # === Technical indicators ===
    # All computed over warmup+display range, then warmup prefix sliced off.
    # Bollinger Bands
    _full_cv = _warmup + close_vals
    bb_mid: list[float] = []
    bb_upper: list[float] = []
    bb_lower: list[float] = []
    for _ib in range(len(_full_cv)):
        _wb = _full_cv[max(0, _ib - _bb_period + 1):_ib + 1]
        _bm = sum(_wb) / len(_wb)
        _bs = math.sqrt(sum((_x - _bm) ** 2 for _x in _wb) / len(_wb)) if len(_wb) > 1 else 0.0
        bb_mid.append(_bm)
        bb_upper.append(_bm + _bb_k * _bs)
        bb_lower.append(_bm - _bb_k * _bs)
    bb_mid = bb_mid[_nwarm:]
    bb_upper = bb_upper[_nwarm:]
    bb_lower = bb_lower[_nwarm:]

    # BB extension into forecast horizon (continuing rolling window with q0.5 closes)
    _bb_fcst_upper: list[float] = []
    _bb_fcst_lower: list[float] = []
    if fcst50_cls:
        _bb_win = list(close_vals[-(  _bb_period - 1):])
        for _fc in fcst50_cls:
            _wb = (_bb_win + [_fc])[-_bb_period:]
            _bm = sum(_wb) / len(_wb)
            _bs = math.sqrt(sum((_x - _bm) ** 2 for _x in _wb) / len(_wb)) if len(_wb) > 1 else 0.0
            _bb_fcst_upper.append(_bm + _bb_k * _bs)
            _bb_fcst_lower.append(_bm - _bb_k * _bs)
            _bb_win = (_bb_win + [_fc])[1:]

    ema20 = ema(_full_cv, 20)[_nwarm:]
    ema50 = ema(_full_cv, 50)[_nwarm:]

    _w52 = min(252, len(close_vals))
    high_52w = max(close_vals[-_w52:])
    low_52w = min(close_vals[-_w52:])

    # ATR proxy: close-to-close Wilder smoothing (no OHLC data available)
    _atr_period = 14
    _tr = [abs(close_vals[_ia] - close_vals[_ia - 1]) for _ia in range(1, len(close_vals))]
    current_atr: float = float("nan")
    if len(_tr) >= _atr_period:
        _atr_cur = sum(_tr[:_atr_period]) / _atr_period
        for _ia in range(_atr_period, len(_tr)):
            _atr_cur = (_atr_cur * (_atr_period - 1) + _tr[_ia]) / _atr_period
        current_atr = _atr_cur

    # RSI (14) with Wilder smoothing; growing window for first points so RSI starts from index 1
    def _calc_rsi(vals: list[float], period: int = 14) -> list[float]:
        if len(vals) < 2:
            return [50.0] * len(vals)
        _gains = [max(vals[_ir] - vals[_ir - 1], 0.0) for _ir in range(1, len(vals))]
        _losses = [max(vals[_ir - 1] - vals[_ir], 0.0) for _ir in range(1, len(vals))]
        _res: list[float] = [50.0]  # seed first point at neutral
        _ag = _gains[0]
        _al = _losses[0]
        for _ir in range(1, len(_gains)):
            _w = min(_ir + 1, period)
            _ag = (_ag * (_w - 1) + _gains[_ir]) / _w
            _al = (_al * (_w - 1) + _losses[_ir]) / _w
            _res.append(100.0 - 100.0 / (1.0 + (_ag / _al if _al > 0 else float("inf"))))
        return _res

    rsi_vals = _calc_rsi(_full_cv, 14)[_nwarm:]

    # Forecast RSI continuation from q0.5 closes
    fcst_rsi50: list[float] = []
    if fcst50_cls:
        _rp = 14
        _recent_c = close_vals[-(2 * _rp):]
        if len(_recent_c) >= _rp + 1:
            _rg2 = [max(_recent_c[_ir] - _recent_c[_ir - 1], 0.0) for _ir in range(1, len(_recent_c))]
            _rl2 = [max(_recent_c[_ir - 1] - _recent_c[_ir], 0.0) for _ir in range(1, len(_recent_c))]
            _rag = sum(_rg2[:_rp]) / _rp
            _ral = sum(_rl2[:_rp]) / _rp
            for _ir in range(_rp, len(_rg2)):
                _rag = (_rag * (_rp - 1) + _rg2[_ir]) / _rp
                _ral = (_ral * (_rp - 1) + _rl2[_ir]) / _rp
            _prev_rf = close_vals[-1]
            for _fc in fcst50_cls:
                _g = max(_fc - _prev_rf, 0.0)
                _l = max(_prev_rf - _fc, 0.0)
                _rag = (_rag * (_rp - 1) + _g) / _rp
                _ral = (_ral * (_rp - 1) + _l) / _rp
                fcst_rsi50.append(100.0 - 100.0 / (1.0 + (_rag / _ral if _ral > 0 else float("inf"))))
                _prev_rf = _fc

    # y_at functions for RSI panel
    def y_at_rsi(v: float) -> float:
        return float(rsi_mt) + (rsi_ih * (1.0 - (float(v) / 100.0)))

    # MACD panel range: keep 0 centered, avoid clipping (include forecast)
    _all_macd = macd_vals + signal_vals + hist_vals + fcst_macd50 + fcst_sig50 + fcst_hist_f50
    max_abs = max([abs(float(v)) for v in _all_macd if math.isfinite(float(v))] + [0.0])
    if not math.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1e-6
    pad = max_abs * 0.10
    max_abs_p = max_abs + pad
    macd_ymin = -max_abs_p
    macd_ymax = +max_abs_p
    macd_span = macd_ymax - macd_ymin
    macd_step = nice_step(macd_span, target_ticks=4)

    def y_at_macd(v: float) -> float:
        v = float(v)
        return float(macd_mt) + (macd_ih * (1.0 - (v - macd_ymin) / (macd_ymax - macd_ymin)))

    macd_path_cmds = []
    signal_path_cmds = []
    for i in range(0, n):
        x = x_at(i, total_n)
        y_m = y_at_macd(macd_vals[i])
        y_s = y_at_macd(signal_vals[i])
        macd_path_cmds.append(("M" if i == 0 else "L", x, y_m))
        signal_path_cmds.append(("M" if i == 0 else "L", x, y_s))
    macd_path_d = " ".join([f"{cmd}{x:.2f},{y:.2f}" for cmd, x, y in macd_path_cmds])
    signal_path_d = " ".join([f"{cmd}{x:.2f},{y:.2f}" for cmd, x, y in signal_path_cmds])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style><![CDATA[text { font-family: Inter, Meiryo, "Meiryo UI", sans-serif; }]]></style>',
        f'<defs>'
        f'<pattern id="{id_prefix}hatch-pos" patternUnits="userSpaceOnUse" width="3" height="3" patternTransform="rotate(45 0 0)">'
        f'<line x1="0" y1="0" x2="0" y2="3" stroke="#1565c0" stroke-width="3" stroke-opacity="0.85"/>'
        f'</pattern>'
        f'<pattern id="{id_prefix}hatch-neg" patternUnits="userSpaceOnUse" width="3" height="3" patternTransform="rotate(45 0 0)">'
        f'<line x1="0" y1="0" x2="0" y2="3" stroke="#d00" stroke-width="3" stroke-opacity="0.85"/>'
        f'</pattern>'
        f'<pattern id="{id_prefix}hatch-bb" patternUnits="userSpaceOnUse" width="4" height="4" patternTransform="rotate(45 0 0)">'
        f'<rect width="4" height="4" fill="none"/>'
        f'<line x1="0" y1="0" x2="0" y2="4" stroke="#90a4ae" stroke-width="2" stroke-opacity="0.45"/>'
        f'</pattern>'
        f'</defs>',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
    ]

    # Legend row (y=30): price-panel indicators
    _leg_y = 30
    _leg_sym_len = 14  # line symbol length
    _lx = float(ml)
    def _leg_line(x: float, col: str, dash: str = "") -> str:
        _da = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<line x1="{x:.1f}" y1="{_leg_y-3:.1f}" x2="{x+_leg_sym_len:.1f}" y2="{_leg_y-3:.1f}" stroke="{col}" stroke-width="1.5"{_da}/>'
    def _leg_band(x: float, col: str) -> str:
        return f'<rect x="{x:.1f}" y="{_leg_y-7:.1f}" width="{_leg_sym_len:.1f}" height="8" fill="{col}" fill-opacity="0.2" stroke="{col}" stroke-width="0.6" stroke-dasharray="2,2"/>'
    def _leg_dot(x: float, col: str) -> str:
        return f'<circle cx="{x+5:.1f}" cy="{_leg_y-3:.1f}" r="3" fill="none" stroke="{col}" stroke-width="1.2"/>'
    def _leg_txt(x: float, label: str) -> str:
        return f'<text x="{x+_leg_sym_len+2:.1f}" y="{_leg_y:.1f}" font-size="8" fill="#444">{html.escape(label)}</text>'
    # item widths: symbol(14) + gap(2) + text
    _items = [
        ("line",  "#111",     "",    "Close",    32),
        ("line",  "#ff6d00",  "",    "EMA20",    34),
        ("line",  "#7b1fa2",  "",    "EMA50",    34),
        ("band",  "#607d8b",  "",    "BB(20)",   36),
        ("line",  "#d00",     "",    "Fcst",     28),
        ("dot",   "#43a047",  "",    "Past",     26),
        ("line",  "#5c6bc0",  "",    "RSI",      22),
        ("line",  "#1565c0",  "",    "MACD",     30),
        ("line",  "#ef6c00",  "",    "Sig",      20),
    ]
    for _kind, _col, _dash, _lbl, _tw in _items:
        if _lx + _leg_sym_len + 2 + _tw > ml + iw + mr - 4:
            break
        if _kind == "line":
            svg.append(_leg_line(_lx, _col, _dash))
        elif _kind == "band":
            svg.append(_leg_band(_lx, _col))
        elif _kind == "dot":
            svg.append(_leg_dot(_lx, _col))
        svg.append(_leg_txt(_lx, _lbl))
        _lx += _leg_sym_len + 2 + _tw + 6

    # axes (price + rsi + macd)
    svg.append(f'<line x1="{ml}" y1="{close_mt}" x2="{ml}" y2="{close_mb}" stroke="#ddd"/>')
    svg.append(f'<line x1="{ml}" y1="{rsi_mt}" x2="{ml}" y2="{rsi_mb}" stroke="#ddd"/>')
    svg.append(f'<line x1="{ml}" y1="{rsi_mb}" x2="{ml+iw}" y2="{rsi_mb}" stroke="#ddd"/>')
    svg.append(f'<line x1="{ml}" y1="{macd_mt}" x2="{ml}" y2="{macd_mb}" stroke="#ddd"/>')
    svg.append(f'<line x1="{ml}" y1="{macd_mb}" x2="{ml+iw}" y2="{macd_mb}" stroke="#ddd"/>')
    # separators
    svg.append(f'<line x1="{ml}" y1="{close_mb}" x2="{ml+iw}" y2="{close_mb}" stroke="#eee"/>')
    svg.append(f'<line x1="{ml}" y1="{rsi_mb}" x2="{ml+iw}" y2="{rsi_mb}" stroke="#eee"/>')

    # vertical grid every 5 trading days (span both panels)
    for i in range(0, total_n):
        if i == 0:
            continue
        if i % 5 != 0 and i != (total_n - 1):
            continue
        x = x_at(i, total_n)
        stroke = "#ddd" if i == (total_n - 1) else "#eee"
        svg.append(f'<line x1="{x:.2f}" y1="{close_mt}" x2="{x:.2f}" y2="{macd_mb}" stroke="{stroke}"/>')

    # x labels: every 5 trading days (vertical text)
    label_y = macd_mb + 8
    for i in range(0, total_n):
        if i % 5 != 0 and i not in (0, n - 1, total_n - 1):
            continue

        d = None
        is_fcst_day = False
        if i < n:
            d = d2[i]
        elif n <= i < n + n_horizons and forecast_dates[i - n] is not None and (i - n) == n_horizons - 1:
            d = forecast_dates[i - n]
            is_fcst_day = True

        if d is None:
            continue

        x = x_at(i, total_n)
        txt = fmt_day(d)
        y = label_y
        if is_fcst_day:
            y = label_y + 12
        svg.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{axis_fs}" fill="#555" text-anchor="start" transform="rotate(90 {x:.2f} {y:.2f})">{html.escape(txt)}</text>'
        )

    # grid (horizontal) + y tick labels (price panel)
    tick = ymin
    # Numeric jitter guard for floating arithmetic
    for _ in range(0, 16):
        if tick > ymax + step * 0.25:
            break
        y = y_at_close(tick)
        stroke = "#ddd" if abs(tick - ymin) < 1e-9 or abs(tick - ymax) < 1e-9 else "#eee"
        svg.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{ml+iw}" y2="{y:.2f}" stroke="{stroke}"/>')
        svg.append(
            f'<text x="{ml-6}" y="{y+3:.2f}" font-size="{axis_fs}" fill="#555" text-anchor="end">{html.escape(fmt_tick(tick, step))}</text>'
        )
        tick += step

    # MACD panel baseline at 0 (only y label we show)
    y0 = y_at_macd(0.0)
    svg.append(f'<line x1="{ml}" y1="{y0:.2f}" x2="{ml+iw}" y2="{y0:.2f}" stroke="#ddd"/>')
    svg.append(
        f'<text x="{ml-6}" y="{y0+3:.2f}" font-size="{axis_fs}" fill="#555" text-anchor="end">0</text>'
    )

    # volume overlay (bars) inside the close panel (no axis labels)
    if volumes is not None:
        vol_vals = [float(v) if v is not None and math.isfinite(float(v)) else float("nan") for v in volumes]
        if len(vol_vals) == n:
            vmax = max([v for v in vol_vals if math.isfinite(v)] + [0.0])
            if vmax > 0:
                vol_h = float(close_ih) * 0.25
                dx = (x_at(1, total_n) - x_at(0, total_n)) if total_n > 1 else 8.0
                bar_w = max(1.0, float(dx) * 0.70)
                for i, v in enumerate(vol_vals):
                    # Skip the first day's bar (can look odd at the left edge).
                    if i == 0:
                        continue
                    if not math.isfinite(v) or v <= 0:
                        continue
                    x = x_at(i, total_n)
                    h = (float(v) / float(vmax)) * vol_h
                    y = float(close_mb) - h
                    x0 = x - (bar_w / 2.0)
                    # keep inside plot
                    if x0 < ml:
                        x0 = float(ml)
                    if x0 + bar_w > ml + iw:
                        bar_w = float(ml + iw) - x0
                    if bar_w <= 0:
                        continue
                    svg.append(
                        f'<rect x="{x0:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="#999" fill-opacity="0.18" stroke="none" />'
                    )

    # 52-week high / low horizontal dotted reference lines
    y52h = y_at_close(high_52w)
    y52l = y_at_close(low_52w)
    if close_mt < y52h < close_mb:
        svg.append(f'<line x1="{ml}" y1="{y52h:.2f}" x2="{ml+iw}" y2="{y52h:.2f}" stroke="#e53935" stroke-width="0.8" stroke-dasharray="2,4" opacity="0.6"/>')
        svg.append(f'<text x="{ml+2}" y="{y52h-2:.2f}" font-size="8" fill="#e53935" opacity="0.8">52W H</text>')
    if close_mt < y52l < close_mb:
        svg.append(f'<line x1="{ml}" y1="{y52l:.2f}" x2="{ml+iw}" y2="{y52l:.2f}" stroke="#43a047" stroke-width="0.8" stroke-dasharray="2,4" opacity="0.6"/>')
        svg.append(f'<text x="{ml+2}" y="{y52l+8:.2f}" font-size="8" fill="#43a047" opacity="0.8">52W L</text>')

    # Bollinger Bands (20 ± 2σ): filled band then dashed upper/lower
    _bb_valid = [(i, bb_upper[i], bb_lower[i]) for i in range(n)
                 if math.isfinite(bb_upper[i]) and math.isfinite(bb_lower[i])]
    if _bb_valid:
        _bb_up_pts = [(x_at(_i, total_n), y_at_close(_u)) for _i, _u, _l in _bb_valid]
        _bb_lo_pts = [(x_at(_i, total_n), y_at_close(_l)) for _i, _u, _l in _bb_valid]
        # Historical polygon: plain grey semi-transparent fill
        _poly = _bb_up_pts + list(reversed(_bb_lo_pts))
        _ps = " ".join(f"{_px:.2f},{_py:.2f}" for _px, _py in _poly)
        svg.append(f'<polygon points="{_ps}" fill="#90a4ae" fill-opacity="0.10" stroke="none"/>')
        # Forecast extension polygon: hatch fill
        if _bb_fcst_upper and _bb_fcst_lower:
            _fx_pts = [(x_fcst_list[_valid_fcst_h[_k]], y_at_close(_bb_fcst_upper[_k])) for _k in range(len(_bb_fcst_upper))]
            _fl_pts = [(x_fcst_list[_valid_fcst_h[_k]], y_at_close(_bb_fcst_lower[_k])) for _k in range(len(_bb_fcst_lower))]
            _fpoly = [_bb_up_pts[-1]] + _fx_pts + list(reversed(_fl_pts)) + [_bb_lo_pts[-1]]
            _fps = " ".join(f"{_px:.2f},{_py:.2f}" for _px, _py in _fpoly)
            svg.append(f'<polygon points="{_fps}" fill="url(#{id_prefix}hatch-bb)" stroke="none"/>')
        _du = " ".join(f"{'M' if _k==0 else 'L'}{_px:.2f},{_py:.2f}" for _k,(_px,_py) in enumerate(_bb_up_pts))
        _dl = " ".join(f"{'M' if _k==0 else 'L'}{_px:.2f},{_py:.2f}" for _k,(_px,_py) in enumerate(_bb_lo_pts))
        svg.append(f'<path d="{_du}" fill="none" stroke="#607d8b" stroke-width="0.8" stroke-dasharray="3,3"/>')
        svg.append(f'<path d="{_dl}" fill="none" stroke="#607d8b" stroke-width="0.8" stroke-dasharray="3,3"/>')
        # Extend dashed border lines into forecast region
        if _bb_fcst_upper and _bb_fcst_lower:
            _fdu_pts = [_bb_up_pts[-1]] + [(x_fcst_list[_valid_fcst_h[_k]], y_at_close(_bb_fcst_upper[_k])) for _k in range(len(_bb_fcst_upper))]
            _fdl_pts = [_bb_lo_pts[-1]] + [(x_fcst_list[_valid_fcst_h[_k]], y_at_close(_bb_fcst_lower[_k])) for _k in range(len(_bb_fcst_lower))]
            _fdu = " ".join(f"{'M' if _k==0 else 'L'}{_px:.2f},{_py:.2f}" for _k,(_px,_py) in enumerate(_fdu_pts))
            _fdl = " ".join(f"{'M' if _k==0 else 'L'}{_px:.2f},{_py:.2f}" for _k,(_px,_py) in enumerate(_fdl_pts))
            svg.append(f'<path d="{_fdu}" fill="none" stroke="#607d8b" stroke-width="0.8" stroke-dasharray="3,3"/>')
            svg.append(f'<path d="{_fdl}" fill="none" stroke="#607d8b" stroke-width="0.8" stroke-dasharray="3,3"/>')

    # EMA50 (purple) and EMA20 (orange) lines
    def _ema_path(ema_v: list[float]) -> str:
        _cmds = []
        for _i, _v in enumerate(ema_v[:n]):
            if math.isfinite(_v):
                _cmds.append(("M" if not _cmds else "L", x_at(_i, total_n), y_at_close(_v)))
        return " ".join(f"{c}{_x:.2f},{_y:.2f}" for c, _x, _y in _cmds)
    _p50 = _ema_path(ema50)
    _p20 = _ema_path(ema20)
    if _p50:
        svg.append(f'<path d="{_p50}" fill="none" stroke="#7b1fa2" stroke-width="1.0" opacity="0.75"/>')
    if _p20:
        svg.append(f'<path d="{_p20}" fill="none" stroke="#ff6d00" stroke-width="1.0" opacity="0.75"/>')

    # EMA forecast extensions (dashed, continuing from last actual value with q0.5 closes)
    if fcst50_cls and _valid_fcst_h:
        _fcst_ema50 = _cont_ema(ema50[-1], fcst50_cls, 50)
        _fcst_ema20 = _cont_ema(ema20[-1], fcst50_cls, 20)
        def _fcst_ema_path(last_val: float, fcst_vals: list[float]) -> str:
            _pts = [(x_last, y_at_close(last_val))] + [
                (x_fcst_list[_valid_fcst_h[_k]], y_at_close(fcst_vals[_k]))
                for _k in range(len(fcst_vals))
            ]
            return " ".join(f"{'M' if _k==0 else 'L'}{_px:.2f},{_py:.2f}" for _k,(_px,_py) in enumerate(_pts))
        _fp50 = _fcst_ema_path(ema50[-1], _fcst_ema50)
        _fp20 = _fcst_ema_path(ema20[-1], _fcst_ema20)
        svg.append(f'<path d="{_fp50}" fill="none" stroke="#7b1fa2" stroke-width="1.0" stroke-dasharray="4,3" opacity="0.75"/>')
        svg.append(f'<path d="{_fp20}" fill="none" stroke="#ff6d00" stroke-width="1.0" stroke-dasharray="4,3" opacity="0.75"/>')

    # actual line (draw after grid/bars so it stays on top)
    svg.append(f'<path d="{path_d}" fill="none" stroke="#111" stroke-width="1.5"/>')

    # MACD histogram (MACD - Signal)
    dx = (x_at(1, total_n) - x_at(0, total_n)) if total_n > 1 else 8.0
    bar_w = max(1.0, float(dx) * 0.70)
    for i in range(0, n):
        v = float(hist_vals[i])
        if not math.isfinite(v) or abs(v) < 1e-12:
            continue
        x = x_at(i, total_n)
        yv = y_at_macd(v)
        y_top = min(y0, yv)
        h = abs(y0 - yv)
        if h <= 0.5:
            continue
        x0 = x - (bar_w / 2.0)
        if x0 < ml:
            x0 = float(ml)
        bw = bar_w
        if x0 + bw > ml + iw:
            bw = float(ml + iw) - x0
        if bw <= 0:
            continue
        col = "#1565c0" if v >= 0 else "#d00"
        svg.append(
            f'<rect x="{x0:.2f}" y="{y_top:.2f}" width="{bw:.2f}" height="{h:.2f}" fill="{col}" fill-opacity="0.25" stroke="none" />'
        )

    # MACD lines (on top of histogram)
    svg.append(f'<path d="{macd_path_d}" fill="none" stroke="#1565c0" stroke-width="1.4"/>')
    svg.append(f'<path d="{signal_path_d}" fill="none" stroke="#ef6c00" stroke-width="1.4"/>')

    # Forecast MACD histogram (diagonal stripe hatching)
    if fcst_hist_f50:
        for _i, _v in enumerate(fcst_hist_f50):
            _v = float(_v)
            if not math.isfinite(_v) or abs(_v) < 1e-12:
                continue
            _x = x_fcst_list[_valid_fcst_h[_i]]
            _yv = y_at_macd(_v)
            _y_top = min(y0, _yv)
            _h_bar = abs(y0 - _yv)
            if _h_bar <= 0.5:
                continue
            _x0b = _x - (bar_w / 2.0)
            if _x0b < ml:
                _x0b = float(ml)
            _bw = bar_w
            if _x0b + _bw > ml + iw:
                _bw = float(ml + iw) - _x0b
            if _bw <= 0:
                continue
            _fill = f"url(#{id_prefix}hatch-pos)" if _v >= 0 else f"url(#{id_prefix}hatch-neg)"
            svg.append(
                f'<rect x="{_x0b:.2f}" y="{_y_top:.2f}" width="{_bw:.2f}" height="{_h_bar:.2f}" fill="{_fill}" stroke="none" />'
            )

    # Forecast MACD q0.5 dashed line (connects from last actual MACD)
    if fcst_macd50:
        _pts_m = [(x_last, y_at_macd(macd_vals[-1]))] + [
            (x_fcst_list[_valid_fcst_h[_i]], y_at_macd(_v)) for _i, _v in enumerate(fcst_macd50)
        ]
        _d_m = " ".join(f"{'M' if _i == 0 else 'L'}{_px:.2f},{_py:.2f}" for _i, (_px, _py) in enumerate(_pts_m))
        svg.append(f'<path d="{_d_m}" fill="none" stroke="#1565c0" stroke-width="1.4" stroke-dasharray="4,3"/>')

    # Forecast Signal q0.5 dashed line (connects from last actual Signal)
    if fcst_sig50:
        _pts_s = [(x_last, y_at_macd(signal_vals[-1]))] + [
            (x_fcst_list[_valid_fcst_h[_i]], y_at_macd(_v)) for _i, _v in enumerate(fcst_sig50)
        ]
        _d_s = " ".join(f"{'M' if _i == 0 else 'L'}{_px:.2f},{_py:.2f}" for _i, (_px, _py) in enumerate(_pts_s))
        svg.append(f'<path d="{_d_s}" fill="none" stroke="#ef6c00" stroke-width="1.4" stroke-dasharray="4,3"/>')

    # ── RSI panel ──────────────────────────────────────────────────────────
    # Reference lines at 70, 50, 30
    for _rv, _rc in ((70, "#e53935"), (50, "#aaa"), (30, "#43a047")):
        _ry = y_at_rsi(float(_rv))
        _stroke = "#eee" if _rv == 50 else "#f5f5f5"
        svg.append(f'<line x1="{ml}" y1="{_ry:.2f}" x2="{ml+iw}" y2="{_ry:.2f}" stroke="{_stroke}"/>')
        svg.append(f'<text x="{ml-6}" y="{_ry+3:.2f}" font-size="{axis_fs}" fill="{_rc}" text-anchor="end">{_rv}</text>')
    # RSI line (actual)
    _rsi_cmds: list[tuple[str, float, float]] = []
    _rsi_last_pt: tuple[float, float] | None = None
    for _i, _rv in enumerate(rsi_vals[:n]):
        if math.isfinite(_rv):
            _rx, _ry2 = x_at(_i, total_n), y_at_rsi(_rv)
            _rsi_cmds.append(("M" if not _rsi_cmds else "L", _rx, _ry2))
            _rsi_last_pt = (_rx, _ry2)
    if _rsi_cmds:
        _rsi_d = " ".join(f"{c}{_x:.2f},{_y:.2f}" for c, _x, _y in _rsi_cmds)
        svg.append(f'<path d="{_rsi_d}" fill="none" stroke="#5c6bc0" stroke-width="1.3"/>')
    # RSI forecast continuation (dashed) — starts exactly from the last drawn actual point
    if fcst_rsi50 and _rsi_last_pt is not None:
        _rsi_fcst_pts = [_rsi_last_pt] + [
            (x_fcst_list[_valid_fcst_h[_i]], y_at_rsi(_v)) for _i, _v in enumerate(fcst_rsi50)
        ]
        _rfd = " ".join(f"{'M' if _k==0 else 'L'}{_px:.2f},{_py:.2f}" for _k, (_px, _py) in enumerate(_rsi_fcst_pts))
        svg.append(f'<path d="{_rfd}" fill="none" stroke="#5c6bc0" stroke-width="1.3" stroke-dasharray="4,3"/>')
    # RSI panel label
    svg.append(f'<text x="{ml+2}" y="{rsi_mt+9:.2f}" font-size="9" fill="#5c6bc0" font-weight="bold">RSI(14)</text>')
    # ATR annotation (right side of RSI panel)
    if math.isfinite(current_atr):
        _atr_txt = f"ATR(14): {fmt_tick(current_atr, current_atr)}"
        svg.append(f'<text x="{ml+iw-2}" y="{rsi_mt+9:.2f}" font-size="9" fill="#888" text-anchor="end">{html.escape(_atr_txt)}</text>')

    # Past forecast overlay: dots showing historical q0.5 predicted closes vs actual
    if past_forecast_dates and past_forecast_q50_closes:
        for _pfd, _pfc50 in zip(past_forecast_dates, past_forecast_q50_closes):
            if _pfc50 is None or not math.isfinite(float(_pfc50)):
                continue
            # find index of this date in d2
            _pfi = next((_ii for _ii, _dd in enumerate(d2) if _dd == _pfd), None)
            if _pfi is None:
                continue
            _actual_c = float(c2[_pfi])
            _pred_c = float(_pfc50)
            _px_dot = x_at(_pfi, total_n)
            _py_pred = y_at_close(_pred_c)
            _py_act = y_at_close(_actual_c)
            # Draw connector line between predicted and actual
            svg.append(
                f'<line x1="{_px_dot:.2f}" y1="{_py_pred:.2f}" x2="{_px_dot:.2f}" y2="{_py_act:.2f}" '
                f'stroke="#9c27b0" stroke-width="0.8" opacity="0.5"/>'
            )
            # Predicted close dot: green if direction correct, red if wrong
            _prev_c = float(c2[max(_pfi - 1, 0)])
            _correct = (_pred_c >= _prev_c) == (_actual_c >= _prev_c)
            _dot_col = "#43a047" if _correct else "#e53935"
            svg.append(
                f'<circle cx="{_px_dot:.2f}" cy="{_py_pred:.2f}" r="3" fill="none" '
                f'stroke="{_dot_col}" stroke-width="1.2"/>'
            )

    # Forecast overlay: multi-horizon band (q0.1–q0.9) and q0.5 polyline.
    valid_h = [
        _h for _h in range(n_horizons)
        if (
            forecast_close10s_calc[_h] is not None
            and forecast_close50s_calc[_h] is not None
            and forecast_close90s_calc[_h] is not None
        )
    ]
    if valid_h:
        y_last = y_at_close(last_close)

        # Build upper/lower envelope polygon points (last actual → horizons).
        upper_pts = [(x_last, y_last)]
        lower_pts = [(x_last, y_last)]
        q50_pts = [(x_last, y_last)]
        for _h in valid_h:
            _x_h = x_fcst_list[_h]
            upper_pts.append((_x_h, y_at_close(float(forecast_close90s_calc[_h]))))
            lower_pts.append((_x_h, y_at_close(float(forecast_close10s_calc[_h]))))
            q50_pts.append((_x_h, y_at_close(float(forecast_close50s_calc[_h]))))

        # Polygon: upper-bound path forward, lower-bound path backward.
        poly_pts = upper_pts + list(reversed(lower_pts))
        poly_str = " ".join(f"{_px:.2f},{_py:.2f}" for _px, _py in poly_pts)
        svg.append(
            f'<polygon points="{poly_str}" fill="#d00" fill-opacity="0.18" stroke="none" />'
        )

        # q0.5 polyline
        q50_d = " ".join(
            f"{'M' if _i == 0 else 'L'}{_px:.2f},{_py:.2f}" for _i, (_px, _py) in enumerate(q50_pts)
        )
        svg.append(f'<path d="{q50_d}" fill="none" stroke="#d00" stroke-width="2" />')

        # Price labels for the last (furthest) horizon only.
        last_h = valid_h[-1]
        fc10_last = float(forecast_close10s_calc[last_h])
        fc50_last = float(forecast_close50s_calc[last_h])
        fc90_last = float(forecast_close90s_calc[last_h])
        q10_last = q10s[last_h]
        q50_last = q50s[last_h]
        q90_last = q90s[last_h]

        x_price = (ml + iw) + 2
        labels = [
            (y_at_close(fc10_last), fmt_price_with_pct(fc10_last, float(q10_last), step), "#111"),
            (y_at_close(fc50_last), fmt_price_with_pct(fc50_last, float(q50_last), step), "#d00"),
            (y_at_close(fc90_last), fmt_price_with_pct(fc90_last, float(q90_last), step), "#111"),
        ]
        labels.sort(key=lambda t: t[0])

        # Prevent overlaps by enforcing minimum vertical separation.
        label_fs = axis_fs
        min_sep = float(label_fs + 2)
        y_min = float(close_mt + 10)
        y_max = float(close_mt + close_ih - 2)
        adj: list[tuple[float, str, str]] = []
        prev_y = None
        for y, txt, col in labels:
            yy = float(y)
            if prev_y is not None and yy < prev_y + min_sep:
                yy = prev_y + min_sep
            yy = max(y_min, min(y_max, yy))
            adj.append((yy, txt, col))
            prev_y = yy

        # If clamping caused collisions at the bottom, shift up as needed.
        if adj and adj[-1][0] > y_max:
            overflow = adj[-1][0] - y_max
            adj = [(y - overflow, txt, col) for (y, txt, col) in adj]

        for y, txt, col in adj:
            svg.append(
                f'<text x="{x_price:.2f}" y="{y:.2f}" font-size="{label_fs}" fill="{col}" text-anchor="start">{html.escape(txt)}</text>'
            )

    svg.append("</svg>")
    return "".join(svg)

