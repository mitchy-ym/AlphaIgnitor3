from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import alphaignitor.common.trading_calendar as tc
from alphaignitor.common.day_store import read_day_partition
from alphaignitor.common.massive_splits import (
    adjustment_factor_for_date,
    get_api_key,
    load_or_fetch_splits,
)
from alphaignitor.pipeline._indicators import ema as _ema, sma as _sma, rsi as _rsi


ALLOWED_SPLIT_TYPES = {"forward_split", "reverse_split", "stock_dividend"}

# 埋め込みキャッシュ互換性用の panel ビルド署名。
# コバリエート定義や前処理ロジックを変更したら明示的に更新する。
PANEL_BUILD_SIGNATURE = "2026-03-08-window-hash-v1"

# 学習に使用する共変量リスト (25 features)
#
# 現行の TiREX + LightGBM では、価格リターン、トレンド、価格水準、
# マクロ、市場相対、オシレーターを入力として使用する。
# TiREX は各チャネルを埋め込み特徴として扱い、LightGBM は全銘柄の
# ウィンドウを横断的に学習するため、中長期モメンタムと市場平均との差分も採用する。
#
# Volatility、Volume、Sector 系の候補列はパネル parquet には計算・保持するが、
# 過去検証で寄与が弱い、または有害だったため現時点の学習入力からは外している。
# (_compute_features_for_ticker 参照) 将来の再評価時は COVARIATE_COLS に追加する。
COVARIATE_COLS = [
    # Return/Gaps  ← 最重要グループ (除外時 −0.48pp)
    "ret_cc_1d",
    "ret_oc_1d",
    "gap_co_1d",
    "ret_cc_5d",
    "ret_cc_20d",
    "ret_cc_60d",
    "ret_cc_120d",
    # Trend/Cross  ← 第2重要グループ (除外時 −0.32pp)
    "close_to_sma20",
    "close_to_sma60",
    "sma20_60_diff",
    "sma20_60_cross_dir",
    "macd_hist",
    "macd_cross_dir",
    # Price Level  ← 有益 (除外時 −0.07pp; 2026-02-19採用)
    "close_to_52w_high",
    # Macro Momentum ← 有益 (除外時 -0.17pp)
    "vix_ret_1d",
    "vix_ret_5d",
    # Macro Relative ← 有益 (除外時 -0.15pp)
    "spy_rel_ret_1d",
    "spy_rel_ret_5d",
    "spy_beta_20d",
    # Macro Volatility ← 有益 (除外時 -0.05pp)
    "spy_gk_vol_5d",
    "spy_gk_vol_20d",
    # Oscillators ← 有益 (影響 -0.06pp)
    "rsi_14",
    "adx_14",
    # Cross-Section ← TiREX (全銘柄横断学習) では有効
    "rel_ret_1d",   # 個別銘柄 vs 市場平均の1日相対リターン
    "rel_ret_5d",   # 個別銘柄 vs 市場平均の5日相対リターン
    # --- 以下は学習不使用 ---
    # Volatility : "atr_14_norm", "vol_cc_20d", "hl_range_1d", "upper_wick_1d", "lower_wick_1d"
    # Volume     : "vol_ratio_20d", "mfi_14"
    # Sector     : "sector_rel_ret_1d", "sector_rel_ret_5d", "sector_beta_20d" → 過去検証で有害
]


@dataclass(frozen=True)
class BuildConfig:
    start_date: dt.date
    end_date: dt.date
    download_dir: Path
    out_parquet: Path
    tickers_csv: Path
    max_tickers: int | None
    splits_cache_dir: Path


def _parse_iso_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(s).strip())
    except Exception as e:
        raise ValueError(f"invalid ISO date: {s}") from e


def _read_tickers(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, encoding="utf-8")
    for col in ["Ticker", "ticker", "symbol", "Symbol"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip().tolist()
            return [v for v in vals if v]
    raise ValueError(f"Ticker column not found in: {csv_path}")


def _read_day_aggs_for_date(day_root: Path, trade_date: dt.date) -> pd.DataFrame:
    return read_day_partition(day_root=day_root, trade_date=trade_date)


def _ensure_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"ticker", "open", "high", "low", "close", "volume"}
    lower_cols = {c.lower(): c for c in df.columns}
    missing = [c for c in needed if c not in lower_cols]
    if missing:
        raise ValueError(f"missing required columns in day_aggs: {missing} (cols={list(df.columns)})")

    # Normalize column names
    rename: dict[str, str] = {}
    for k in needed:
        rename[lower_cols[k]] = k
    # also support window_start -> window_start if present
    if "window_start" in lower_cols:
        rename[lower_cols["window_start"]] = "window_start"
    if "transactions" in lower_cols:
        rename[lower_cols["transactions"]] = "transactions"

    df = df.rename(columns=rename)
    return df


def _apply_split_adjustments(
    df: pd.DataFrame,
    *,
    trade_date: dt.date,
    splits_by_ticker: dict[str, list],
) -> pd.DataFrame:
    # Compute per-ticker factor for this trade_date.
    factors = {}
    for t, events in splits_by_ticker.items():
        factors[t] = adjustment_factor_for_date(events_asc=events, d=trade_date)

    f = df["ticker"].map(factors).astype("float64")
    # 未知のティッカーは factor=1 にフォールバック。
    f = f.fillna(1.0)
    # ゼロ以下の不正な係数はスキップ。
    invalid_mask = f <= 0
    if invalid_mask.any():
        bad_tickers = df.loc[invalid_mask, "ticker"].unique().tolist()
        warnings.warn(
            f"adjustment_factor <= 0 のティッカーが存在します。係数を 1.0 にリセットします: {bad_tickers}",
            stacklevel=2,
        )
        f = f.where(~invalid_mask, 1.0)

    for col in ["open", "high", "low", "close"]:
        df[col] = (df[col].astype("float64") * f).astype("float64")

    # volume adjustment for split scaling: keep dollar-volume comparable
    # If price is multiplied by f, share count should be divided by f.
    df["volume"] = (df["volume"].astype("float64") / f).round().astype("int64")

    return df


def _safe_log(x: pd.Series) -> pd.Series:
    x = x.astype("float64")
    return np.log(x.where(x > 0))


def _true_range(high: pd.Series, low: pd.Series, close_prev: pd.Series) -> pd.Series:
    a = (high - low).abs()
    b = (high - close_prev).abs()
    c = (low - close_prev).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    # Wilder's DMI/ADX
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(high, low, close.shift(1))

    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return adx


def _compute_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    # df columns: trade_date, open, high, low, close, volume
    df = df.sort_values("trade_date").reset_index(drop=True)

    o = df["open"].astype("float64")
    h = df["high"].astype("float64")
    lo = df["low"].astype("float64")
    c = df["close"].astype("float64")

    # Target y_t uses same-day O/C (this is what we predict for next day).
    df["y"] = np.log(c.where(c > 0) / o.where(o > 0))

    # Returns/gaps for row t must be based on t-1 and earlier -> shift by 1.
    ret_cc_1d_raw = np.log(c / c.shift(1))
    ret_oc_1d_raw = np.log(c / o)
    gap_co_1d_raw = np.log(o / c.shift(1))

    df["ret_cc_1d"] = ret_cc_1d_raw.shift(1)
    df["ret_oc_1d"] = ret_oc_1d_raw.shift(1)
    df["gap_co_1d"] = gap_co_1d_raw.shift(1)
    df["ret_cc_5d"] = (np.log(c / c.shift(5))).shift(1)
    df["ret_cc_20d"] = (np.log(c / c.shift(20))).shift(1)
    df["ret_cc_60d"] = (np.log(c / c.shift(60))).shift(1)
    df["ret_cc_120d"] = (np.log(c / c.shift(120))).shift(1)
    # 52週高値比率: min_periods=1 でウォームアップ中も「既存データ内の最高値」を使用
    df["close_to_52w_high"] = (c / c.rolling(252, min_periods=1).max() - 1.0).shift(1)

    sma20 = _sma(c, 20)
    sma60 = _sma(c, 60)
    df["close_to_sma20"] = (c / sma20 - 1.0).shift(1)
    df["close_to_sma60"] = (c / sma60 - 1.0).shift(1)

    sma_diff_raw = (sma20 - sma60)
    df["sma20_60_diff"] = sma_diff_raw.shift(1)

    # Cross dir for row t is based on sign change from t-2->t-1 in sma_diff.
    prev = sma_diff_raw.shift(1)
    prev2 = sma_diff_raw.shift(2)
    cross_up = (prev2 <= 0) & (prev > 0)
    cross_down = (prev2 >= 0) & (prev < 0)
    df["sma20_60_cross_dir"] = np.select([cross_up, cross_down], [1, -1], default=0).astype("int8")

    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    macd_hist_raw = macd - macd_signal
    df["macd_hist"] = macd_hist_raw.shift(1)

    mh_prev = macd_hist_raw.shift(1)
    mh_prev2 = macd_hist_raw.shift(2)
    macd_up = (mh_prev2 <= 0) & (mh_prev > 0)
    macd_down = (mh_prev2 >= 0) & (mh_prev < 0)
    df["macd_cross_dir"] = np.select([macd_up, macd_down], [1, -1], default=0).astype("int8")

    df["rsi_14"] = _rsi(c, 14).shift(1)
    df["adx_14"] = _adx(h, lo, c, 14).shift(1)

    # Garman-Klass volatility
    log_hl = np.log(h / lo)
    log_co = np.log(c / o)
    gk_raw = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    df["garman_klass_5d"] = gk_raw.rolling(5).mean().shift(1)
    df["garman_klass_20d"] = gk_raw.rolling(20).mean().shift(1)

    return df


def build_daily_panel(cfg: BuildConfig) -> Path:
    tickers = _read_tickers(cfg.tickers_csv)
    
    if cfg.max_tickers is not None:
        tickers = tickers[: int(cfg.max_tickers)]
        
    # Add macro tickers to the universe AFTER slicing
    macro_tickers = ["SPY", "QQQ", "VIX", "VIXY", "VIXM", "UVIX", "SVIX", "XLK", "XLY", "XLC", "XLV", "XLF", "XLP", "XLI", "XLRE", "XLB", "XLU", "XLE"]
    for t in macro_tickers:
        if t not in tickers:
            tickers.append(t)
    if not tickers:
        raise ValueError("No tickers in universe")

    print(f"universe_tickers: {len(tickers)}")

    splits_by_ticker: dict[str, list] = {}
    use_split_adjustment = str(os.environ.get("USE_SPLIT_ADJUSTMENT", "false")).strip().lower() in {
        "1", "true", "yes", "on"
    }
    if use_split_adjustment:
        api_key = get_api_key()
        for t in tqdm(tickers, desc="load splits", unit="ticker", dynamic_ncols=True, ascii=True):
            splits_by_ticker[t] = load_or_fetch_splits(
                ticker=t,
                cache_dir=cfg.splits_cache_dir,
                allowed_types=ALLOWED_SPLIT_TYPES,
                api_key=api_key,
            )

    date_list = list(
        tc.classify_us_stock_days(list(_date_range(cfg.start_date, cfg.end_date)))[0]
    )

    frames: list[pd.DataFrame] = []
    loaded_days = 0
    kept_rows = 0
    for d in tqdm(date_list, desc="load day_aggs", unit="day", dynamic_ncols=True, ascii=True):
        daily = _read_day_aggs_for_date(cfg.download_dir, d)
        if daily.empty:
            continue
        daily = _ensure_required_cols(daily)
        daily = daily[daily["ticker"].isin(tickers)].copy()
        if daily.empty:
            continue
        if use_split_adjustment:
            daily = _apply_split_adjustments(daily, trade_date=d, splits_by_ticker=splits_by_ticker)
        # Assign scalar to avoid index-alignment NaNs after filtering.
        daily["trade_date"] = d
        loaded_days += 1
        kept_rows += int(len(daily))
        frames.append(daily[["ticker", "trade_date", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise FileNotFoundError("No day_aggs files found/loaded in the date range")

    raw = pd.concat(frames, ignore_index=True)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date

    # ティッカーごとに特徴量を計算。
    out_frames: list[pd.DataFrame] = []
    ticker_list = [(t, g) for t, g in raw.groupby("ticker", sort=False)]
    for t, g in tqdm(ticker_list, desc="compute features", unit="ticker", dynamic_ncols=True, ascii=True):
        g2 = _compute_features_for_ticker(g)
        out_frames.append(g2)

    panel = pd.concat(out_frames, ignore_index=True)

    # クロスセクション超過リターン: 各日付の全銘柄平均との差分 (per-ticker loop 外で計算)
    # Jane Street 上位解法の cross-section 集計特徴量に相当
    panel["rel_ret_1d"] = (
        panel["ret_cc_1d"] - panel.groupby("trade_date")["ret_cc_1d"].transform("mean")
    )
    panel["rel_ret_5d"] = (
        panel["ret_cc_5d"] - panel.groupby("trade_date")["ret_cc_5d"].transform("mean")
    )

    # --- Macro Features ---
    # Extract SPY and VIX data
    spy_data = panel[panel["ticker"] == "SPY"].set_index("trade_date")
    vix_data = panel[panel["ticker"] == "VIX"].set_index("trade_date")
    
    # If VIX is not present, try VIXY or VIXM
    if vix_data.empty:
        for alt_vix in ["VIXY", "VIXM", "UVIX", "SVIX"]:
            vix_data = panel[panel["ticker"] == alt_vix].set_index("trade_date")
            if not vix_data.empty:
                break

    if not spy_data.empty:
        panel["SPY_ret_1d"] = panel["trade_date"].map(spy_data["ret_cc_1d"])
        panel["SPY_ret_5d"] = panel["trade_date"].map(spy_data["ret_cc_5d"])
        panel["spy_gk_vol_5d"] = panel["trade_date"].map(spy_data["garman_klass_5d"])
        panel["spy_gk_vol_20d"] = panel["trade_date"].map(spy_data["garman_klass_20d"])
        
        panel["spy_rel_ret_1d"] = panel["ret_cc_1d"] - panel["SPY_ret_1d"]
        panel["spy_rel_ret_5d"] = panel["ret_cc_5d"] - panel["SPY_ret_5d"]
        
        # Calculate Beta to SPY
        # NOTE: ret_cc_1d は _compute_features_for_ticker 内で既に .shift(1) 済み
        # （各行 t の値は log(C_{t-1} / C_{t-2})）。そのため rolling_cov / var_spy の
        # 結果に追加の .shift(1) は不要。行 t での spy_beta_20d は t-20..t-1 の
        # リターンのみを使用しており、未来情報は含まない。
        def rolling_cov(g):
            return g["ret_cc_1d"].rolling(20).cov(g["SPY_ret_1d"])
        
        cov_with_spy = panel.groupby("ticker", group_keys=False).apply(rolling_cov, include_groups=False)
        var_spy = panel["trade_date"].map(spy_data["ret_cc_1d"].rolling(20).var())
        panel["spy_beta_20d"] = cov_with_spy / var_spy

    if not vix_data.empty:
        panel["vix_ret_1d"] = panel["trade_date"].map(vix_data["ret_cc_1d"])
        panel["vix_ret_5d"] = panel["trade_date"].map(vix_data["ret_cc_5d"])
        panel["VIX_to_sma20"] = panel["trade_date"].map(vix_data["close_to_sma20"])

    # --- Sector Features ---
    # Load sector mapping
    try:
        stock_list_df = pd.read_csv(cfg.tickers_csv, encoding="utf-8")
        # Map sectors to ETFs (approximate mapping)
        sector_etf_map = {
            "Information Technology": "XLK",
            "Consumer Discretionary": "XLY",
            "Communication": "XLC",
            "Health Care": "XLV",
            "Financials": "XLF",
            "Consumer Staples": "XLP",
            "Industrials": "XLI",
            "Real Estate": "XLRE",
            "Materials": "XLB",
            "Utilities": "XLU",
            "Energy": "XLE"
        }
        
        # Create ticker -> ETF mapping
        ticker_to_etf = {}
        for _, row in stock_list_df.iterrows():
            ticker = row.get("Ticker") or row.get("ticker") or row.get("symbol") or row.get("Symbol")
            sector = row.get("Sector")
            if ticker and sector and sector in sector_etf_map:
                ticker_to_etf[ticker] = sector_etf_map[sector]
                
        # Calculate Sector relative to SPY
        if not spy_data.empty:
            panel["Sector_rel_to_SPY_5d"] = np.nan
            for etf in set(ticker_to_etf.values()):
                etf_data = panel[panel["ticker"] == etf].set_index("trade_date")
                if not etf_data.empty:
                    etf_rel_5d = etf_data["ret_cc_5d"] - spy_data["ret_cc_5d"]
                    # Find tickers belonging to this ETF
                    tickers_in_etf = [t for t, e in ticker_to_etf.items() if e == etf]
                    mask = panel["ticker"].isin(tickers_in_etf)
                    panel.loc[mask, "Sector_rel_to_SPY_5d"] = panel.loc[mask, "trade_date"].map(etf_rel_5d)
                    
    except Exception as e:
        tqdm.write(f"Warning: Could not compute sector features: {e}")

    # Drop rows that cannot be used (warmup NaNs).
    need_cols = [
        "ticker", "trade_date", "y", *COVARIATE_COLS,
        "VIX_to_sma20", "SPY_ret_1d", "SPY_ret_5d",
        "Sector_rel_to_SPY_5d"
    ]
    # Only keep columns that actually exist in the panel
    need_cols = [c for c in need_cols if c in panel.columns]
    panel = panel[need_cols]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaNs in the required columns (excluding the new macro columns for now to avoid dropping everything if they are missing)
    required_cols = ["ticker", "trade_date", "y", *COVARIATE_COLS]
    panel = panel.dropna(subset=required_cols, axis=0, how="any")

    if panel.empty:
        raise ValueError(
            "daily panel is empty after feature warmup/dropna. "
            "The selected window is likely too short for SMA60/rolling features. "
            "Try a longer date range (e.g., >= ~70 trading days; commonly 252)."
        )

    panel = panel.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    cfg.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cfg.out_parquet, index=False)
    return cfg.out_parquet


def _date_range(start_date: dt.date, end_date: dt.date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur = cur + dt.timedelta(days=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Day Aggregatesから学習用の日次パネル (ticker×trade_date) を生成します。\n"
            "- Splits endpointで株価調整（必須）\n"
            "- target: y=log(C/O)\n"
            "- covariates: 確定20特徴量（t-1までで確定）"
        )
    )
    p.add_argument("--start-date", type=str, required=True)
    p.add_argument("--end-date", type=str, required=True)
    p.add_argument(
        "--download-dir",
        type=Path,
        default=Path("aggs/us_stock_day"),
    )
    p.add_argument(
        "--tickers-csv",
        type=Path,
        default=Path("us_stock_list.csv"),
    )
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument(
        "--splits-cache-dir",
        type=Path,
        default=Path("aggs/splits_cache"),
    )
    p.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="出力parquet（未指定なら aggs/parquet/us_stock_daily_panel_<start>_<end>.parquet）",
    )
    return p.parse_args()


def run_build_daily_panel(
    *,
    start_date: dt.date | str,
    end_date: dt.date | str,
    download_dir: Path = Path("aggs/us_stock_day"),
    out_parquet: Path | None = None,
    tickers_csv: Path = Path("us_stock_list.csv"),
    max_tickers: int | None = None,
    splits_cache_dir: Path = Path("aggs/splits_cache"),
) -> Path:
    s_date = _parse_iso_date(str(start_date)) if not isinstance(start_date, dt.date) else start_date
    e_date = _parse_iso_date(str(end_date)) if not isinstance(end_date, dt.date) else end_date
    if e_date < s_date:
        raise ValueError("end-date must be >= start-date")

    target_parquet = out_parquet
    if target_parquet is None:
        target_parquet = Path("aggs/parquet") / f"us_stock_daily_panel_{s_date.isoformat()}_{e_date.isoformat()}.parquet"

    cfg = BuildConfig(
        start_date=s_date,
        end_date=e_date,
        download_dir=Path(download_dir),
        out_parquet=Path(target_parquet),
        tickers_csv=Path(tickers_csv),
        max_tickers=max_tickers,
        splits_cache_dir=Path(splits_cache_dir),
    )
    return build_daily_panel(cfg)


def main() -> int:
    args = parse_args()
    out = run_build_daily_panel(
        start_date=args.start_date,
        end_date=args.end_date,
        download_dir=Path(args.download_dir),
        out_parquet=Path(args.out_parquet) if args.out_parquet else None,
        tickers_csv=Path(args.tickers_csv),
        max_tickers=args.max_tickers,
        splits_cache_dir=Path(args.splits_cache_dir),
    )
    print("daily_panel:", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
