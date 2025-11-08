"""Utilities for caching Yahoo Finance requests across agents."""

from __future__ import annotations

from datetime import datetime
from threading import RLock
from typing import Dict, Tuple

import pandas as pd
import yfinance as yf

_download_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
_download_lock = RLock()

_history_cache: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}
_history_lock = RLock()

_PERIOD_ORDER = [
    "1d",
    "5d",
    "7d",
    "10d",
    "14d",
    "1mo",
    "60d",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
]
_PERIOD_RANK = {name: idx for idx, name in enumerate(_PERIOD_ORDER)}

# Maximum reliable period for each interval according to Yahoo Finance limitations.
_INTERVAL_MAX_PERIOD = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "90m": "60d",
}


def _normalize_period(period: str, interval: str) -> str:
    """Ensure requested period is compatible with the interval."""
    max_period = _INTERVAL_MAX_PERIOD.get(interval)
    if not max_period:
        return period

    period_rank = _PERIOD_RANK.get(period, max(_PERIOD_RANK.values()) + 1)
    max_rank = _PERIOD_RANK.get(max_period, max(_PERIOD_RANK.values()) + 1)
    if period_rank > max_rank:
        return max_period
    return period


def get_historical_data(symbol: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """
    Cached wrapper around ``yf.download`` using period/interval arguments.

    Returns a copy of the cached DataFrame so callers can mutate safely.
    """
    normalized_period = _normalize_period(period, interval)
    key = (symbol.upper(), normalized_period, interval)
    with _download_lock:
        cached = _download_cache.get(key)
    if cached is not None:
        return cached.copy()

    try:
        ticker = yf.Ticker(symbol)

        data = ticker.history(
            period=normalized_period,
            interval=interval,
            auto_adjust=False,
        )
    except TypeError as err:
        # Workaround for pandas/yfinance incompatibility when minute data is unavailable.
        if "Cannot convert numpy.ndarray to numpy.ndarray" in str(err):
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()

    if not data.empty:
        with _download_lock:
            _download_cache[key] = data
        return data.copy()

    return data


def get_price_history(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Cached wrapper around ``yf.download`` using start/end arguments.

    start/end are stored as ISO strings to create a stable cache key.
    """
    key = (symbol.upper(), start.isoformat(), end.isoformat(), interval)
    with _history_lock:
        cached = _history_cache.get(key)
    if cached is not None:
        return cached.copy()

    try:
        ticker = yf.Ticker(symbol)
        # hist_data = ticker.history(start=start_date, end=end_date)
        data = ticker.history(
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except TypeError as err:
        if "Cannot convert numpy.ndarray to numpy.ndarray" in str(err):
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()

    if not data.empty:
        with _history_lock:
            _history_cache[key] = data
        return data.copy()

    return data
