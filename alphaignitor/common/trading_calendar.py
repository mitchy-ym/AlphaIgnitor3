from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable


def _nth_weekday_of_month(*, year: int, month: int, weekday: int, n: int) -> date:
    if n < 1:
        raise ValueError("n must be >= 1")
    d = date(year, month, 1)
    days_ahead = (weekday - d.weekday()) % 7
    d = d + timedelta(days=days_ahead + 7 * (n - 1))
    if d.month != month:
        raise ValueError("weekday occurrence out of month")
    return d


def _last_weekday_of_month(*, year: int, month: int, weekday: int) -> date:
    if month == 12:
        d = date(year + 1, 1, 1)
    else:
        d = date(year, month + 1, 1)
    d = d - timedelta(days=1)
    while d.weekday() != weekday:
        d = d - timedelta(days=1)
    return d


def _easter_sunday_gregorian(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed_us_holiday(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def us_stock_market_holidays(year: int) -> frozenset[date]:
    holidays: set[date] = set()

    holidays.add(_observed_us_holiday(date(year, 1, 1)))
    holidays.add(_nth_weekday_of_month(year=year, month=1, weekday=0, n=3))
    holidays.add(_nth_weekday_of_month(year=year, month=2, weekday=0, n=3))

    easter = _easter_sunday_gregorian(year)
    holidays.add(easter - timedelta(days=2))

    holidays.add(_last_weekday_of_month(year=year, month=5, weekday=0))

    if year >= 2022:
        holidays.add(_observed_us_holiday(date(year, 6, 19)))

    holidays.add(_observed_us_holiday(date(year, 7, 4)))
    holidays.add(_nth_weekday_of_month(year=year, month=9, weekday=0, n=1))
    holidays.add(_nth_weekday_of_month(year=year, month=11, weekday=3, n=4))
    holidays.add(_observed_us_holiday(date(year, 12, 25)))

    return frozenset(holidays)


def is_us_stock_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in us_stock_market_holidays(d.year)


def classify_us_stock_days(date_list: list[date]) -> tuple[list[date], list[date], list[date]]:
    weekend_dates = [d for d in date_list if d.weekday() >= 5]
    holiday_dates = [d for d in date_list if d.weekday() < 5 and d in us_stock_market_holidays(d.year)]
    non_trading_dates = set(weekend_dates) | set(holiday_dates)
    trading_dates = [d for d in date_list if d not in non_trading_dates]
    return trading_dates, weekend_dates, holiday_dates


def previous_trading_day(d: date) -> date:
    cur = d
    while True:
        cur = cur - timedelta(days=1)
        if is_us_stock_trading_day(cur):
            return cur


def next_trading_days(start_exclusive: date, n: int) -> list[date]:
    out: list[date] = []
    cur = start_exclusive
    while len(out) < max(int(n), 0):
        cur = cur + timedelta(days=1)
        if is_us_stock_trading_day(cur):
            out.append(cur)
    return out


def trading_days_back_from(day_inclusive: date, n: int) -> list[date]:
    out: list[date] = []
    cur = day_inclusive
    while len(out) < max(int(n), 0):
        if is_us_stock_trading_day(cur):
            out.append(cur)
        cur = cur - timedelta(days=1)
    out.sort()
    return out


def to_et_date(value: object) -> date:
    """pd.Timestamp 等を America/New_York タイムゾーンの date に変換する。"""
    import pandas as pd  # pandas依存をこの関数の呼び出し時に限定する。
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    return ts.date()
