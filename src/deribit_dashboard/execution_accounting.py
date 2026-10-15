"""Unit-preserving cash-flow formulas shared by hypothetical execution studies.

Callers validate instruments and choose the numeraire. These functions also accept
arrays, preserving the existing portfolio and barrier simulation calculations.
"""

import numpy as np


def perpetual_gain(contracts, contract_size, inverse, entry_price, exit_price):
    """Native BTC gain for inverse contracts; quote-currency gain for linear contracts."""
    change = (
        1 / np.asarray(entry_price) - 1 / np.asarray(exit_price)
        if inverse
        else np.asarray(exit_price) - np.asarray(entry_price)
    )
    return np.asarray(contracts) * contract_size * change


def perpetual_notional(contracts, contract_size, inverse, price):
    """Signed quote-currency face notional (inverse) or marked notional (linear)."""
    return np.asarray(contracts) * contract_size * (1.0 if inverse else np.asarray(price))


def funding_amount(signed_notional, period_rate, time_fraction):
    """Positive is funding paid by a long when the period rate is positive."""
    return np.asarray(signed_notional) * np.asarray(period_rate) * np.asarray(time_fraction)


def fee_from_notional(notional, fee_bps):
    """Nonnegative fee in the supplied notional's currency; no implicit conversion."""
    return np.abs(notional) * fee_bps / 10000
