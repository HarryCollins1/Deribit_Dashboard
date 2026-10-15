"""Historical volatility retrieval and timestamp-aligned independent reconciliation."""

import numpy as np
import pandas as pd

from deribit_dashboard.history import download_candles
from deribit_dashboard.models import milliseconds
from deribit_dashboard.pricing import YEAR_SECONDS


def realized_volatility(candles, window=24 * 30, interval_seconds=3600):
    if window < 2 or interval_seconds <= 0:
        raise ValueError("Need at least two returns and a positive interval")
    if not candles:
        raise ValueError("No candle observations")
    frame = pd.DataFrame(candles).sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    if frame.empty or (frame.close <= 0).any():
        raise ValueError("Positive candle closes required")
    returns = np.log(frame.close).diff()
    valid = frame.timestamp_ms.diff().eq(interval_seconds * 1000)
    returns = returns.where(valid)
    frame["own_realized_pct"] = (
        returns.rolling(window, min_periods=window).std(ddof=1)
        * np.sqrt(YEAR_SECONDS / interval_seconds)
        * 100
    )
    frame["missing_interval"] = ~valid
    # Exchange candle timestamps denote the open; realized returns are known at the close.
    frame["timestamp_ms"] += int(interval_seconds * 1000)
    return frame[["timestamp_ms", "own_realized_pct", "missing_interval"]]


async def volatility_history(client, currency, perpetual, start, end, window=24 * 30):
    """Hourly perpetual RV, published RV and paginated hourly DVOL. No forward filling."""
    candles = await download_candles(client, perpetual, start, end, "60")
    candles = [c for c in candles if c["timestamp_ms"] + 3600000 <= milliseconds(end)]
    if not candles:
        raise ValueError("No completed perpetual candles returned")
    own = realized_volatility(candles, window)
    published = await client.call("public/get_historical_volatility", currency=currency)
    rv = pd.DataFrame(published, columns=["timestamp_ms", "published_realized_pct"])
    stop, beginning, seen, data = milliseconds(end), milliseconds(start), set(), {}
    while stop > beginning:
        response = await client.call(
            "public/get_volatility_index_data",
            currency=currency,
            start_timestamp=beginning,
            end_timestamp=stop,
            resolution="3600",
        )
        for row in response.get("data", []):
            if beginning <= row[0] <= milliseconds(end):
                data[row[0]] = row[4]
        continuation = response.get("continuation")
        if continuation is None:
            break
        if continuation in seen or continuation >= stop:
            raise ValueError("Non-progressing DVOL continuation")
        seen.add(continuation)
        stop = continuation
    dvol = pd.DataFrame(sorted(data.items()), columns=["timestamp_ms", "dvol_pct"])
    merged = (
        own.merge(rv.drop_duplicates("timestamp_ms"), on="timestamp_ms", how="outer")
        .merge(dvol, on="timestamp_ms", how="outer")
        .sort_values("timestamp_ms")
    )
    merged = merged[merged.timestamp_ms.between(beginning, milliseconds(end))]
    merged["rv_difference_points"] = merged.own_realized_pct - merged.published_realized_pct
    merged["timestamp"] = pd.to_datetime(merged.timestamp_ms, unit="ms", utc=True)
    return merged


RV_EXPLANATION = (
    "Own RV = sample standard deviation of hourly perpetual log returns × sqrt(365.25×24) × 100, "
    "using the selected rolling number of returns. Missing hourly intervals invalidate the window. "
    "Published RV uses the exchange's index and methodology; perpetual funding/basis, sampling, "
    "window boundaries, annualization and sample versus population variance can cause differences. "
    "Own returns are timestamped at candle close; unfinished candles are excluded. "
    "Only exact timestamps are reconciled, without forward filling. DVOL is forward-looking implied "
    "volatility, whereas both realized series are backward-looking. Empty overlap is reported as missing."
)
