"""テクニカルインジケータ関数群。

build_daily_panel と report_daily_forecast の両方で使用する共通実装。
EMA / SMA / RSI / ボリンジャーバンドをここで一元管理する。
"""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """指数移動平均（EMA）を返す。span未満の期間はNaN。"""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """単純移動平均（SMA）を返す。window未満の期間はNaN。"""
    return series.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder方式のRSI（Relative Strength Index）を返す。"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ボリンジャーバンドを返す（上限, 中心MA, 下限）。

    Returns:
        upper: 上限バンド
        mid:   中心移動平均
        lower: 下限バンド
    """
    mid = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + std_multiplier * std
    lower = mid - std_multiplier * std
    return upper, mid, lower
