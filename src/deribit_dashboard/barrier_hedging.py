"""Self-financing discrete delta hedge of one short barrier using a perpetual.

Dollar prices are per underlying coin. Contract sizes and inverse/linear settlement
come from captured metadata. Funding and execution friction are explicit assumptions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deribit_dashboard.execution_accounting import (
    fee_from_notional,
    funding_amount,
    perpetual_gain,
    perpetual_notional,
)
from deribit_dashboard.pricing import DAYS
from deribit_dashboard.strategies import DiscreteBarrierGrid


@dataclass(frozen=True)
class PerpetualHedge:
    instrument_name: str
    contract_size: float
    inverse: bool
    initial_basis_ratio: float = 1.0
    half_spread_bps: float = 0.0
    displayed_depth_notional: float = np.inf

    @classmethod
    def from_metadata(cls, metadata, initial_price, index_spot, cash_currency):
        if metadata.get("kind") != "future" or metadata.get("settlement_period") != "perpetual":
            raise ValueError("The dynamic hedge requires a discovered perpetual")
        if (
            metadata.get("base_currency") != "BTC"
            or metadata.get("quote_currency") != cash_currency
        ):
            raise ValueError(
                "Perpetual base/cash denomination must match the BTC option; no implicit FX"
            )
        size = metadata.get("contract_size")
        if (
            size is None
            or not np.isfinite([size, initial_price, index_spot]).all()
            or min(size, initial_price, index_spot) <= 0
        ):
            raise ValueError("Finite positive perpetual metadata and prices are required")
        inverse = metadata.get("settlement_currency") == metadata["base_currency"]
        if not inverse and metadata.get("settlement_currency") != cash_currency:
            raise ValueError("Unsupported perpetual settlement convention")
        return cls(metadata["instrument_name"], float(size), inverse, initial_price / index_spot)


def dynamic_hedge(
    option,
    barrier,
    times,
    spots,
    premium,
    perpetual,
    *,
    fee_bps=0.0,
    annual_funding=0.0,
    bump_fraction=0.005,
    impact_multiplier=0.0,
):
    """Hedge one SHORT barrier; positive P&L is surplus after closing its liability.

    Holdings chosen at time i earn only subsequent interval P&L. Observe barriers
    at their contractual dates, close the hedge on touch for knock-outs, and close
    all positions at the final observed time (marking any remaining liability).
    """
    times, spots = np.asarray(times, dtype=float), np.asarray(spots, dtype=float)
    if (
        times.ndim != 1
        or spots.ndim != 2
        or spots.shape[0] != len(times)
        or len(times) < 2
        or spots.shape[1] == 0
    ):
        raise ValueError("Supply time×path positive prices and matching elapsed years")
    if (
        not np.isfinite(times).all()
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or times[-1] > option.years + 1e-10
    ):
        raise ValueError("Hedge times must start at zero, increase, and stop by expiry")
    if not np.isfinite(spots).all() or np.any(spots <= 0) or not np.allclose(spots[0], option.spot):
        raise ValueError(
            "Hedge paths must start at the option spot; rebase historical scenarios explicitly"
        )
    if (
        not np.isfinite([premium, fee_bps, annual_funding, bump_fraction, impact_multiplier]).all()
        or min(premium, fee_bps, impact_multiplier) < 0
        or not 0 < bump_fraction < 0.05
    ):
        raise ValueError("Invalid hedge premium, cost, funding or delta bump")
    if (
        not np.isfinite([perpetual.contract_size, perpetual.initial_basis_ratio]).all()
        or perpetual.contract_size <= 0
        or perpetual.initial_basis_ratio <= 0
        or not np.isfinite(perpetual.half_spread_bps)
        or perpetual.half_spread_bps < 0
        or np.isnan(perpetual.displayed_depth_notional)
        or perpetual.displayed_depth_notional <= 0
    ):
        raise ValueError("Invalid perpetual metadata")
    monitoring = np.arange(0, option.years, barrier.monitoring_days / DAYS)
    monitoring = np.r_[monitoring, option.years]
    for observation in monitoring[monitoring <= times[-1] + 1e-12]:
        if np.min(abs(times - observation)) > 1e-10:
            raise ValueError("Hedge paths must include every contractual barrier observation")
    grid = DiscreteBarrierGrid(option, barrier, times * DAYS, grid_points=2401)
    count = spots.shape[1]
    cash = np.full(count, float(premium))
    holdings = np.zeros(count)
    hit = np.full(count, barrier.already_hit, dtype=bool)
    gains, funding, costs, interest = (np.zeros(count) for _ in range(4))
    peak_exposure = np.zeros(count)
    turnover = np.zeros(count)
    previous_future = spots[0] * perpetual.initial_basis_ratio
    for i, time in enumerate(times):
        s = spots[i]
        future = s * perpetual.initial_basis_ratio
        if i:
            dt = time - times[i - 1]
            accrual = cash * np.expm1(option.rate * dt)
            interest += accrual
            cash += accrual
            pnl = perpetual_gain(
                holdings, perpetual.contract_size, perpetual.inverse, previous_future, future
            ) * (s if perpetual.inverse else 1.0)
            fund = funding_amount(
                perpetual_notional(
                    holdings, perpetual.contract_size, perpetual.inverse, previous_future
                ),
                annual_funding,
                dt,
            )
            cash += pnl - fund
            gains += pnl
            funding += fund
        day = time * DAYS
        if np.any(np.isclose(day, grid.monitoring, atol=1e-8, rtol=0)):
            hit |= barrier.breached(s)
        if i == len(times) - 1:
            desired = np.zeros(count)
        else:
            bump = s * bump_fraction
            delta = (grid.price(s + bump, day, hit) - grid.price(s - bump, day, hit)) / (2 * bump)
            if perpetual.inverse:
                desired = delta * future / perpetual.contract_size
            else:
                desired = delta / (perpetual.contract_size * perpetual.initial_basis_ratio)
        change = np.abs(desired - holdings)
        notional = perpetual_notional(change, perpetual.contract_size, perpetual.inverse, future)
        fee = fee_from_notional(
            notional,
            (
                fee_bps
                + perpetual.half_spread_bps
                * (
                    1
                    + impact_multiplier
                    * np.maximum(notional / perpetual.displayed_depth_notional - 1, 0)
                )
            ),
        )
        costs += fee
        cash -= fee
        turnover += notional
        peak_exposure = np.maximum(
            peak_exposure,
            np.abs(desired) * perpetual.contract_size * (1.0 if perpetual.inverse else future),
        )
        holdings, previous_future = desired, future
    liability = grid.price(spots[-1], times[-1] * DAYS, hit)
    pnl = cash - liability
    result = pd.DataFrame(
        dict(
            path=np.arange(count),
            premium=premium,
            hedge_gains=gains,
            funding_paid=funding,
            transaction_cost=costs,
            cash_interest=interest,
            barrier_liability=liability,
            pnl=pnl,
            barrier_hit=hit,
            hedge_turnover=turnover,
            peak_hedge_notional=peak_exposure,
        )
    )
    result["reconciliation"] = (
        result.premium
        + result.hedge_gains
        - result.funding_paid
        - result.transaction_cost
        + result.cash_interest
        - result.barrier_liability
        - result.pnl
    )
    return result


def hedge_summary(results):
    error = results.pnl.to_numpy()
    return dict(
        paths=len(error),
        mean_pnl=float(error.mean()),
        hedge_rmse=float(np.sqrt(np.mean(error**2))),
        pnl_std=float(np.std(error, ddof=1)) if len(error) > 1 else np.nan,
        pnl_5pct=float(np.quantile(error, 0.05)),
        pnl_95pct=float(np.quantile(error, 0.95)),
        loss_fraction=float(np.mean(error < 0)),
        max_reconciliation_error=float(results.reconciliation.abs().max()),
    )
