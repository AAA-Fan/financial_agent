"""Utilities for caching Alpha Vantage requests across agents."""

from __future__ import annotations

import os
from datetime import datetime
from io import StringIO
from threading import RLock
from typing import Dict, Tuple

import pandas as pd
import requests

_ALPHA_BASE_URL = "https://www.alphavantage.co/query"
_ALPHA_PERIOD_FUNCTION = {
    "daily": "TIME_SERIES_DAILY",
    "weekly": "TIME_SERIES_WEEKLY",
    "monthly": "TIME_SERIES_MONTHLY",
}
_INTERVAL_ALIASES = {
    "1d": "daily",
    "daily": "daily",
    "day": "daily",
    "d": "daily",
    "1w": "weekly",
    "1wk": "weekly",
    "weekly": "weekly",
    "week": "weekly",
    "w": "weekly",
    "1mo": "monthly",
    "monthly": "monthly",
    "month": "monthly",
    "m": "monthly",
}

_download_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
_download_lock = RLock()


def _get_api_key() -> str:
    """Load Alpha Vantage API key from environment."""
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "ALPHAVANTAGE_API_KEY is not configured. "
            "Add it to your environment or .env file."
        )
    return key


def _normalize_interval(interval: str) -> str:
    interval_key = interval.lower()
    if interval_key in _ALPHA_PERIOD_FUNCTION:
        return interval_key
    alias = _INTERVAL_ALIASES.get(interval_key)
    if alias:
        return alias
    raise ValueError(
        f"Unsupported Alpha Vantage period '{interval}'. "
        "Use one of daily, weekly, or monthly."
    )


def _fetch_alpha_series(symbol: str, interval: str) -> pd.DataFrame:
    """Retrieve a time series from Alpha Vantage and normalize the response."""
    params = {
        "function": _ALPHA_PERIOD_FUNCTION[interval],
        "symbol": symbol.upper(),
        "apikey": _get_api_key(),
        "datatype": "csv",
    }
    try:
        response = requests.get(_ALPHA_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Alpha Vantage request failed: {exc}") from exc

    payload = response.text.strip()
    if not payload:
        return pd.DataFrame()
    if payload.startswith("{"):
        # Alpha Vantage returns JSON for errors even when datatype=csv.
        raise RuntimeError(f"Alpha Vantage error: {payload}")

    frame = pd.read_csv(StringIO(payload))
    if frame.empty:
        return frame

    renamed = frame.rename(
        columns={
            "timestamp": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    renamed["Date"] = pd.to_datetime(renamed["Date"])
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")

    normalized = renamed.sort_values("Date").set_index("Date")
    return normalized


def get_historical_data(symbol: str, interval: str = "daily", days: int | None = None) -> pd.DataFrame:
    """
    Cached wrapper around Alpha Vantage TIME_SERIES_* endpoints.

    Args:
        symbol: Equity ticker symbol.
        period: One of ``daily``, ``weekly``, ``monthly`` (aliases supported).
        days: Optional number of most recent rows to return (useful for daily analysis).
    """
    normalized_interval = _normalize_interval(interval)
    key = (symbol.upper(), normalized_interval)
    with _download_lock:
        cached = _download_cache.get(key)
    if cached is None:
        data = _fetch_alpha_series(symbol, normalized_interval)
        with _download_lock:
            _download_cache[key] = data
    else:
        data = cached

    result = data.copy()
    if days is not None and days > 0:
        result = result.tail(days)
    return result


def get_price_history(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Provide a historical slice between ``start`` and ``end`` using cached data.

    Since Alpha Vantage does not support arbitrary intervals, ``interval`` is treated
    as a hint (mapped to daily/weekly/monthly via aliases).
    """
    period = _normalize_period(interval)
    data = get_historical_data(symbol, period=period)
    if data.empty:
        return data

    mask = (data.index >= start) & (data.index <= end)
    return data.loc[mask].copy()


if __name__ == "__main__":
    data = get_historical_data("AAPL", "daily", days=30)
    print(data.tail())
