"""Historical BTC perpetual close paths for present-day strategy scenario replay.

The paths contain observed candle closes only. OHLC is retained for inspection,
but an intrabar high/low does not provide the time or sequence of barrier hits.
Repricing these paths with today's calibration is not a historical option backtest.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from deribit_dashboard.client import DeribitClient
from deribit_dashboard.history import RESOLUTIONS, download_candles
from deribit_dashboard.models import milliseconds, parse_time, utc_now
from deribit_dashboard.pricing import DAYS

_REQUIRED = ("timestamp_ms", "open", "high", "low", "close")
_LIMITATION = (
    "Barrier monitoring uses observed perpetual candle closes only; intrabar crossings may be "
    "missed. Perpetual trade prices are not the option settlement index. Repricing uses the "
    "selected current calibration, not historical option quotes, spreads, funding or fills."
)


@dataclass
class HistoricalPaths:
    times_years: np.ndarray
    spots: np.ndarray
    labels: pd.DataFrame
    diagnostics: dict
    source: str
    observations: pd.DataFrame = field(default_factory=pd.DataFrame)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if not math.isfinite(value) or value != int(value) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _configuration(horizon_days, resolution_minutes, max_paths, stride_steps):
    resolution_minutes = _integer(resolution_minutes, "resolution_minutes")
    resolution = "1D" if resolution_minutes == 1440 else str(resolution_minutes)
    if resolution not in RESOLUTIONS:
        raise ValueError("resolution_minutes must match an available Deribit candle resolution")
    if isinstance(horizon_days, bool) or not math.isfinite(horizon_days) or horizon_days <= 0:
        raise ValueError("horizon_days must be finite and positive")
    steps_float = horizon_days * 1440 / resolution_minutes
    steps = math.floor(steps_float)
    if steps < 1:
        raise ValueError("The horizon must contain at least one complete candle interval")
    if steps > 1_000_000:
        raise ValueError("A historical path may contain at most one million intervals")
    max_paths = _integer(max_paths, "max_paths")
    stride_steps = steps if stride_steps is None else _integer(stride_steps, "stride_steps")
    return steps, resolution_minutes, max_paths, stride_steps, resolution


def build_historical_paths(
    candles: pd.DataFrame,
    horizon_days: float,
    resolution_minutes: int,
    spot: float | None = None,
    max_paths: int = 30,
    *,
    stride_steps: int | None = None,
    as_of=None,
) -> HistoricalPaths:
    """Make fixed-horizon chronological scenarios from complete candle closes.

    ``timestamp_ms`` is the candle OPEN time. The observation becomes available at
    open + resolution, and must be <= ``as_of`` (default current UTC). Supplying
    ``spot`` rebases each close path by spot / its first close; None preserves
    historical absolute prices. No prices are interpolated or forward-filled.

    Fractional final intervals are omitted and explicitly reported. The last
    observation is at or before the requested horizon, never beyond it.
    Default windows share their boundary close, with no overlapping RETURN
    intervals. ``stride_steps < horizon_steps`` deliberately produces overlapping
    scenarios, flagged in diagnostics. Starts use a fixed chronological grid;
    missing/invalid data reject a window rather than shortening its horizon.
    The earliest valid windows are retained up to max_paths, regardless of returns.
    """
    steps, minutes, max_paths, stride, resolution = _configuration(
        horizon_days, resolution_minutes, max_paths, stride_steps
    )
    if spot is not None and (isinstance(spot, bool) or not math.isfinite(spot) or spot <= 0):
        raise ValueError("Rebased spot must be finite and positive")
    as_of = parse_time(as_of) if as_of is not None else utc_now()
    interval_ms = minutes * 60_000
    cutoff_ms = milliseconds(as_of)
    frame = candles.copy(deep=True)
    if frame.empty:
        for column in _REQUIRED:
            if column not in frame:
                frame[column] = pd.Series(dtype=float)
    missing_columns = set(_REQUIRED) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Candle columns missing: {sorted(missing_columns)}")
    # Mixing instruments or environments creates spurious returns at series joins.
    provenance = {}
    for column in ("instrument_name", "environment", "base_currency", "quote_currency"):
        if column in frame:
            values = frame[column].dropna().unique()
            if len(values) > 1:
                raise ValueError(f"Historical paths require one {column}")
            if len(values):
                provenance[column] = str(values[0])
    if provenance.get("base_currency", "BTC") != "BTC":
        raise ValueError("Historical barrier replay currently supports BTC perpetual paths")
    if "resolution" in frame and not frame.empty:
        supplied = set(frame["resolution"].dropna().astype(str))
        if supplied - {resolution}:
            raise ValueError("Candle resolution does not match resolution_minutes")

    timestamp = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    valid_timestamp = (
        np.isfinite(timestamp)
        & (timestamp >= 0)
        & (timestamp % 1 == 0)
        & ~frame["timestamp_ms"].map(lambda value: isinstance(value, bool))
    )
    # Conversion checks bounds before converting to int64 and forming UTC labels.
    datetime_values = pd.to_datetime(
        timestamp.where(valid_timestamp), unit="ms", utc=True, errors="coerce"
    )
    valid_timestamp &= datetime_values.notna()
    invalid_timestamp_rows = int((~valid_timestamp).sum())
    frame = frame.loc[valid_timestamp].copy()
    frame["timestamp_ms"] = timestamp.loc[valid_timestamp].astype("int64")
    # The final candle is rejected until its close has occurred. Historical end
    # dates also use this rule in download_historical_paths.
    incomplete = frame["timestamp_ms"] > cutoff_ms - interval_ms
    incomplete_rows = int(incomplete.sum())
    frame = frame.loc[~incomplete].copy()
    earliest = int(frame["timestamp_ms"].min()) if len(frame) else None
    latest = int(frame["timestamp_ms"].max()) if len(frame) else None
    # Remove every ambiguous duplicate, rather than choosing whichever was last.
    duplicate = frame["timestamp_ms"].duplicated(keep=False)
    duplicate_rows = int(duplicate.sum())
    duplicate_timestamps = int(frame.loc[duplicate, "timestamp_ms"].nunique())
    frame = frame.loc[~duplicate].sort_values("timestamp_ms").reset_index(drop=True)
    ohlc = frame[list(_REQUIRED[1:])].apply(pd.to_numeric, errors="coerce")
    valid_price = np.isfinite(ohlc).all(axis=1) & (ohlc > 0).all(axis=1)
    # Retained OHLC must be internally coherent even though only closes are used.
    valid_price &= (
        (ohlc["low"] <= ohlc["open"])
        & (ohlc["low"] <= ohlc["close"])
        & (ohlc["high"] >= ohlc["open"])
        & (ohlc["high"] >= ohlc["close"])
        & (ohlc["low"] <= ohlc["high"])
    )
    invalid_ohlc_rows = int((~valid_price).sum())
    frame.loc[:, list(_REQUIRED[1:])] = ohlc
    frame = frame.loc[valid_price].copy().set_index("timestamp_ms", drop=False)
    frame["candle_open_utc"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame["candle_close_utc"] = pd.to_datetime(
        frame["timestamp_ms"] + interval_ms, unit="ms", utc=True
    )
    differences = np.diff(frame["timestamp_ms"].to_numpy(dtype=np.int64))
    gaps = differences[differences > interval_ms]
    # Uneven timestamp cadence is not regularized into fabricated observations.
    irregular_intervals = int(np.count_nonzero(differences % interval_ms))
    missing_intervals = int(np.sum(np.maximum(gaps // interval_ms - 1, 0)))
    candidates = (
        0
        if earliest is None or latest - earliest < steps * interval_ms
        else ((latest - earliest - steps * interval_ms) // (stride * interval_ms) + 1)
    )
    diagnostics = {
        "input_rows": len(candles),
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "incomplete_or_after_cutoff_rows": incomplete_rows,
        "duplicate_rows_removed": duplicate_rows,
        "duplicate_timestamps_removed": duplicate_timestamps,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "usable_rows": len(frame),
        "missing_intervals": missing_intervals,
        "irregular_intervals": irregular_intervals,
        "candidate_windows": int(candidates),
        "examined_windows": 0,
        "rejected_incomplete_windows": 0,
        "unexamined_windows_after_limit": 0,
        "paths": 0,
        "horizon_days": float(horizon_days),
        "requested_horizon_days": float(horizon_days),
        "actual_horizon_days": steps * minutes / 1440,
        "unobserved_final_interval_days": max(0.0, horizon_days - steps * minutes / 1440),
        "horizon_steps": steps,
        "resolution_minutes": minutes,
        "stride_steps": stride,
        "overlapping_return_intervals": stride < steps,
        "shared_boundary_close": stride == steps,
        "rebased": spot is not None,
        "as_of_utc": as_of.isoformat(),
        "selection": "Earliest complete windows in chronological order; no return-based selection",
        "monitoring": "Observed perpetual candle closes only",
        "limitations": _LIMITATION,
        **provenance,
    }
    paths, labels, observations = [], [], []
    for candidate in range(int(candidates)):
        if len(paths) >= max_paths:
            diagnostics["unexamined_windows_after_limit"] = int(candidates) - candidate
            break
        diagnostics["examined_windows"] += 1
        start = earliest + candidate * stride * interval_ms
        expected = start + np.arange(steps + 1, dtype=np.int64) * interval_ms
        if not np.isin(expected, frame.index.to_numpy()).all():
            diagnostics["rejected_incomplete_windows"] += 1
            continue
        window = frame.loc[expected].copy()
        closes = window["close"].to_numpy(dtype=float)
        scale = float(spot / closes[0]) if spot is not None else 1.0
        path_index = len(paths)
        paths.append(closes * scale)
        labels.append(
            {
                "path": path_index,
                "start_utc": window.iloc[0]["candle_close_utc"],
                "end_utc": window.iloc[-1]["candle_close_utc"],
                "start_candle_open_utc": window.iloc[0]["candle_open_utc"],
                "historical_start": float(closes[0]),
                "historical_end": float(closes[-1]),
                "historical_return_pct": float((closes[-1] / closes[0] - 1) * 100),
                "rebase_factor": scale,
                "observations": steps + 1,
                **provenance,
            }
        )
        window["path"] = path_index
        window["step"] = np.arange(steps + 1)
        window["replayed_spot"] = closes * scale
        window["rebase_factor"] = scale
        observations.append(window.reset_index(drop=True))
    diagnostics["paths"] = len(paths)
    diagnostics["rows_used"] = (
        int(pd.concat(observations)["timestamp_ms"].nunique()) if observations else 0
    )
    diagnostics["unused_usable_rows"] = len(frame) - diagnostics["rows_used"]
    source = (
        f"Historical perpetual candle closes: {provenance.get('instrument_name', 'unspecified instrument')}"
        f" ({provenance.get('environment', 'caller-provided data')}); current-calibration replay"
    )
    return HistoricalPaths(
        times_years=np.arange(steps + 1, dtype=float) * minutes / (DAYS * 1440),
        spots=np.column_stack(paths) if paths else np.empty((steps + 1, 0)),
        labels=pd.DataFrame(labels),
        diagnostics=diagnostics,
        source=source,
        observations=pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(),
    )


async def download_historical_paths(
    client: DeribitClient,
    instrument_metadata: dict,
    start,
    end,
    horizon_days: float,
    resolution_minutes: int = 60,
    spot: float | None = None,
    max_paths: int = 30,
    *,
    stride_steps: int | None = None,
) -> HistoricalPaths:
    """Download a selected metadata-discovered BTC perpetual, then prepare paths.

    Call ``client.instruments('BTC')`` to populate the selector. The instrument
    name is read from that selection; no name or settlement convention is inferred.
    Candles closing after the requested end or current UTC are excluded.
    """
    if (
        instrument_metadata.get("base_currency") != "BTC"
        or instrument_metadata.get("kind") != "future"
        or instrument_metadata.get("settlement_period") != "perpetual"
        or not instrument_metadata.get("instrument_name")
    ):
        raise ValueError("Select a discovered BTC perpetual instrument")
    _, _, _, _, resolution = _configuration(
        horizon_days, resolution_minutes, max_paths, stride_steps
    )
    start, end = parse_time(start), parse_time(end)
    if start >= end:
        raise ValueError("Historical start must precede end")
    as_of = min(end, utc_now())
    if start >= as_of:
        raise ValueError("Historical start must precede the completed-data cutoff")
    name = instrument_metadata["instrument_name"]
    rows = await download_candles(client, name, start, as_of, resolution)
    frame = pd.DataFrame(rows)
    for key, value in {
        "instrument_name": name,
        "environment": client.settings.environment,
        "base_currency": "BTC",
        "quote_currency": instrument_metadata.get("quote_currency"),
        "resolution": resolution,
    }.items():
        frame[key] = value
    result = build_historical_paths(
        frame,
        horizon_days,
        resolution_minutes,
        spot,
        max_paths,
        stride_steps=stride_steps,
        as_of=as_of,
    )
    result.source = f"Deribit public/get_tradingview_chart_data; {result.source}"
    result.diagnostics.update(
        {
            "requested_start_utc": start.isoformat(),
            "requested_end_utc": end.isoformat(),
            "instrument_metadata": dict(instrument_metadata),
            "candle_timestamp_convention": "Exchange candle open time; observation time = open + resolution",
        }
    )
    return result
