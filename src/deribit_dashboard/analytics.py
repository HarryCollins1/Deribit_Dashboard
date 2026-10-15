"""Snapshot-relative analytics. IV remains in percentage points as sent by Deribit."""

import math

import numpy as np
import pandas as pd

from deribit_dashboard.models import Snapshot, parse_time

YEAR_SECONDS = 365.25 * 24 * 3600


def snapshot_frame(snapshot: Snapshot, stale_seconds: float = 120) -> pd.DataFrame:
    rows = []
    for name, instrument in snapshot.instruments.items():
        ticker = snapshot.tickers.get(name, {})
        perpetual = instrument.get("settlement_period") == "perpetual" or name.endswith("PERPETUAL")
        expiration_ms = instrument.get("expiration_timestamp")
        expiration = (
            None
            if perpetual or expiration_ms is None
            else pd.to_datetime(expiration_ms, unit="ms", utc=True)
        )
        exchange_time = pd.to_datetime(ticker["timestamp"], unit="ms", utc=True) if ticker else None
        age = (snapshot.captured_at - exchange_time).total_seconds() if exchange_time else math.nan
        row = {
            "instrument_name": name,
            "kind": instrument["kind"],
            "base_currency": instrument.get("base_currency"),
            "option_type": instrument.get("option_type"),
            "strike": instrument.get("strike"),
            "expiry": expiration,
            "is_perpetual": perpetual,
            "days_to_expiry": (expiration - snapshot.captured_at).total_seconds() / 86400
            if expiration is not None
            else math.nan,
            "exchange_time": exchange_time,
            "received_at": parse_time(snapshot.received_at[name]) if ticker else None,
            "quote_age_seconds": age,
            "missing": not bool(ticker),
            "clock_ahead": bool(ticker) and age < -2,
            "stale": not bool(ticker) or age > stale_seconds or age < -2,
            "settlement_currency": instrument.get("settlement_currency"),
            "quote_currency": instrument.get("quote_currency"),
            "contract_size": instrument.get("contract_size"),
            "state": ticker.get("state"),
        }
        for field in [
            "mark_price",
            "last_price",
            "best_bid_price",
            "best_ask_price",
            "best_bid_amount",
            "best_ask_amount",
            "index_price",
            "underlying_price",
            "mark_iv",
            "bid_iv",
            "ask_iv",
            "open_interest",
            "interest_rate",
            "current_funding",
            "funding_8h",
        ]:
            row[field] = ticker.get(field)
        row["underlying_index"] = ticker.get("underlying_index")
        row.update(
            {
                field: (ticker.get("greeks") or {}).get(field)
                for field in ["delta", "gamma", "theta", "vega", "rho"]
            }
        )
        row["volume_24h"] = (ticker.get("stats") or {}).get("volume")
        rows.append(row)
    frame = pd.DataFrame(rows)
    numeric = [
        "strike",
        "contract_size",
        "mark_price",
        "last_price",
        "best_bid_price",
        "best_ask_price",
        "index_price",
        "underlying_price",
        "mark_iv",
        "bid_iv",
        "ask_iv",
        "open_interest",
        "interest_rate",
        "current_funding",
        "funding_8h",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "volume_24h",
        "best_bid_amount",
        "best_ask_amount",
    ]
    for field in numeric:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["log_moneyness"] = np.log(
        (frame.strike / frame.underlying_price).where(
            (frame.strike > 0) & (frame.underlying_price > 0)
        )
    )
    frame["spread"] = frame.best_ask_price - frame.best_bid_price
    frame["crossed_quote"] = frame.spread < 0
    # Preserve the native premium unit. Convert only inverse coin-settled options.
    frame["premium_currency"] = np.where(
        frame.kind == "option", frame.settlement_currency, frame.quote_currency
    )
    stable = {"USD", "USDC", "USDT", "EURR"}
    inverse_option = (
        (frame.kind == "option")
        & (frame.settlement_currency == frame.base_currency)
        & ~frame.settlement_currency.isin(stable)
    )
    frame["mark_usd_indicative"] = np.where(
        inverse_option, frame.mark_price * frame.index_price, frame.mark_price
    )
    return frame


def option_points(frame: pd.DataFrame, side: str = "OTM", include_stale: bool = False):
    result = frame[(frame.kind == "option") & (frame.days_to_expiry > 0)].copy()
    if not include_stale:
        result = result[~result.stale]
    if side in {"call", "put"}:
        result = result[result.option_type == side]
    elif side == "OTM":
        result = result[
            ((result.option_type == "put") & (result.strike < result.underlying_price))
            | ((result.option_type == "call") & (result.strike >= result.underlying_price))
        ]
    else:
        raise ValueError("Side must be OTM, call, or put")
    return result[
        np.isfinite(result.log_moneyness) & np.isfinite(result.mark_iv) & (result.mark_iv > 0)
    ]


def surface_grid(points: pd.DataFrame, count: int = 61):
    """Linear interpolation in log(K/F) within each expiry; NaN outside observed support.

    Plotly connects expiry slices visually. This is a display interpolation, not an
    arbitrage-free volatility calibration or a source of calculated Greeks.
    """
    if points.empty:
        return np.array([]), np.array([]), np.empty((0, 0))
    x = np.linspace(points.log_moneyness.min(), points.log_moneyness.max(), count)
    y, slices = [], []
    for _, group in points.groupby("expiry", sort=True):
        smile = group.groupby("log_moneyness", sort=True).mark_iv.mean()
        if len(smile) < 2:
            continue
        z = np.interp(x, smile.index.to_numpy(), smile.to_numpy(), left=np.nan, right=np.nan)
        y.append(float(group.days_to_expiry.iloc[0]))
        slices.append(z)
    return x, np.array(y), np.array(slices) if slices else np.empty((0, len(x)))


def futures_curve(frame: pd.DataFrame):
    result = frame[frame.kind == "future"].copy()
    result["basis_pct"] = ((result.mark_price / result.index_price) - 1) * 100
    result.loc[result.index_price <= 0, "basis_pct"] = np.nan
    years = result.days_to_expiry / 365.25
    result["annualized_basis_pct"] = result.basis_pct / years.where(years > 0)
    # Perpetuals have funding, not an expiry-based annualized basis.
    result.loc[result.is_perpetual, "annualized_basis_pct"] = np.nan
    return result.sort_values("days_to_expiry", na_position="first")
