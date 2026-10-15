"""Deterministic synthetic data, explicitly labelled and separated from exchange data."""

import math
from datetime import timedelta

from deribit_dashboard.models import Snapshot, milliseconds, utc_now


def demo_snapshot(currency="BTC", captured_at=None, shift=0.0):
    if currency not in {"BTC", "ETH"}:
        raise ValueError("Synthetic markets are available for BTC and ETH")
    captured_at = captured_at or utc_now()
    spot = (100000 if currency == "BTC" else 3500) * (1 + shift)
    instruments, tickers, received = {}, {}, {}
    for days in [7, 30, 90, 180]:
        expiry = (captured_at + timedelta(days=days)).replace(
            hour=8, minute=0, second=0, microsecond=0
        )
        years = (expiry - captured_at).total_seconds() / (365.25 * 86400)
        forward = spot * math.exp(0.055 * years)
        code = expiry.strftime("%d%b%y").upper()
        future = f"{currency}-{code}"
        instruments[future] = {
            "instrument_name": future,
            "kind": "future",
            "base_currency": currency,
            "quote_currency": "USD",
            "counter_currency": "USD",
            "instrument_type": "reversed",
            "settlement_currency": currency,
            "contract_size": 10,
            "tick_size": 0.5,
            "expiration_timestamp": milliseconds(expiry),
            "settlement_period": "month",
        }
        tickers[future] = {
            "instrument_name": future,
            "timestamp": milliseconds(captured_at),
            "mark_price": forward,
            "index_price": spot,
            "last_price": forward,
            "best_bid_price": forward - 5,
            "best_ask_price": forward + 5,
            "open_interest": 25000000,
            "state": "open",
            "stats": {"volume": 1234},
        }
        for ratio in [0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25, 1.4]:
            strike = round((100000 if currency == "BTC" else 3500) * ratio)
            sigma = (
                0.50
                + 0.035 * math.sqrt(days / 30)
                - 0.12 * math.log(ratio)
                + 0.6 * math.log(ratio) ** 2
            )
            d1 = (math.log(forward / strike) + 0.5 * sigma**2 * years) / (sigma * math.sqrt(years))
            d2 = d1 - sigma * math.sqrt(years)

            def normal(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))

            density = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
            for side in ["call", "put"]:
                sign = 1 if side == "call" else -1
                name = f"{currency}-{code}-{strike}-{'C' if sign == 1 else 'P'}"
                price_usd = sign * (forward * normal(sign * d1) - strike * normal(sign * d2))
                mark = price_usd / spot
                instruments[name] = {
                    "instrument_name": name,
                    "kind": "option",
                    "base_currency": currency,
                    "quote_currency": currency,
                    "counter_currency": "USD",
                    "instrument_type": "reversed",
                    "settlement_currency": currency,
                    "contract_size": 1,
                    "tick_size": 0.0001,  # Synthetic fixture convention, not a live contract fallback
                    "expiration_timestamp": milliseconds(expiry),
                    "strike": strike,
                    "option_type": side,
                }
                tickers[name] = {
                    "instrument_name": name,
                    "timestamp": milliseconds(captured_at),
                    "mark_price": mark,
                    "last_price": mark,
                    "best_bid_price": max(mark * 0.97, 0),
                    "best_ask_price": mark * 1.03,
                    "index_price": spot,
                    "underlying_price": forward,
                    "underlying_index": future,
                    "interest_rate": 0,
                    "mark_iv": sigma * 100,
                    "bid_iv": sigma * 100 - 1,
                    "ask_iv": sigma * 100 + 1,
                    "best_bid_amount": 2,
                    "best_ask_amount": 3,
                    "open_interest": 200,
                    "state": "open",
                    "stats": {"volume": 12},
                    "greeks": {
                        "delta": normal(d1) - (1 if side == "put" else 0),
                        "gamma": density / (forward * sigma * math.sqrt(years)),
                        "vega": forward * density * math.sqrt(years) / 100,
                        "theta": -forward * density * sigma / (2 * math.sqrt(years)) / 365.25,
                        "rho": sign * strike * years * normal(sign * d2) / 100,
                    },
                }
    name = f"{currency}-PERPETUAL"
    instruments[name] = {
        "instrument_name": name,
        "kind": "future",
        "base_currency": currency,
        "quote_currency": "USD",
        "counter_currency": "USD",
        "instrument_type": "reversed",
        "settlement_currency": currency,
        "contract_size": 10,
        "tick_size": 0.5,
        "expiration_timestamp": 32503708800000,
        "settlement_period": "perpetual",
    }
    tickers[name] = {
        "instrument_name": name,
        "timestamp": milliseconds(captured_at),
        "mark_price": spot * 1.0004,
        "index_price": spot,
        "best_bid_price": spot,
        "best_ask_price": spot + 5,
        # Synthetic displayed inverse-contract depth, denominated in USD face amount.
        "best_bid_amount": 250000.0,
        "best_ask_amount": 300000.0,
        "funding_8h": 0.0001,
        "current_funding": 0.00005,
        "state": "open",
    }
    # These names/conventions belong only to the labelled synthetic fixture.
    name = f"{currency}_USDC"
    instruments[name] = {
        "instrument_name": name,
        "kind": "spot",
        "base_currency": currency,
        "quote_currency": "USDC",
        "counter_currency": "USDC",
        "settlement_currency": "USDC",
        "instrument_type": "linear",
        "tick_size": 0.01,
        "min_trade_amount": 0.001,
    }
    tickers[name] = {
        "instrument_name": name,
        "timestamp": milliseconds(captured_at),
        "mark_price": spot,
        "last_price": spot,
        "index_price": spot,
        "best_bid_price": spot * 0.9999,
        "best_ask_price": spot * 1.0001,
        "best_bid_amount": 2,
        "best_ask_amount": 3,
        "state": "open",
    }
    received = {name: captured_at.isoformat() for name in tickers}
    return Snapshot(
        currency=currency,
        environment="demo",
        source="synthetic",
        captured_at=captured_at,
        collection_started_at=captured_at,
        instruments=instruments,
        tickers=tickers,
        received_at=received,
    )


def demo_market_snapshot(captured_at=None):
    """One common-cutoff BTC/ETH market for cross-asset strategy exercises."""
    captured_at = captured_at or utc_now()
    parts = [demo_snapshot(currency, captured_at) for currency in ("BTC", "ETH")]
    return Snapshot(
        currency="ANY",
        environment="demo",
        source="synthetic",
        captured_at=captured_at,
        collection_started_at=captured_at,
        instruments={name: m for part in parts for name, m in part.instruments.items()},
        tickers={name: t for part in parts for name, t in part.tickers.items()},
        received_at={name: t for part in parts for name, t in part.received_at.items()},
    )
