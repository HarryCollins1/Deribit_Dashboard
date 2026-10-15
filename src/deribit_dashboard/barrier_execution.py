"""Captured top-of-book execution assumptions for simulated vanilla hedges.

Options are normalized to cash per underlying coin; position quantities are
underlying units. Deribit option amounts are underlying units, so contract counts
are quantities divided by metadata contract_size. Beyond the captured depth,
impact is an explicit scenario assumption, not reconstructed historical liquidity.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deribit_dashboard.execution_accounting import fee_from_notional
from deribit_dashboard.surface import Contract


@dataclass(frozen=True)
class StripExecution:
    strikes: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_depth: np.ndarray
    ask_depth: np.ndarray
    contract_sizes: np.ndarray
    tick_cash: np.ndarray
    spot: float
    fee_bps: float = 0.0
    impact_multiplier: float = 1.0
    inverse_quotes: np.ndarray | None = None

    def __post_init__(self):
        values = [
            self.strikes,
            self.bid,
            self.ask,
            self.bid_depth,
            self.ask_depth,
            self.contract_sizes,
            self.tick_cash,
        ]
        values = [np.asarray(v, float) for v in values]
        if any(v.ndim != 1 or v.shape != values[0].shape for v in values):
            raise ValueError("Execution inputs must contain one value per discovered strike")
        if not len(values[0]) or any(not np.isfinite(v).all() for v in values):
            raise ValueError("Execution metadata and quotes must be finite")
        if (
            np.any(values[1] < 0)
            or np.any(values[2] < values[1])
            or any(np.any(values[i] <= 0) for i in (0, 3, 4, 5, 6))
        ):
            raise ValueError(
                "Executable strips require positive depth, sizes, ticks and valid bid/ask"
            )
        if (
            not np.isfinite([self.spot, self.fee_bps, self.impact_multiplier]).all()
            or self.spot <= 0
            or min(self.fee_bps, self.impact_multiplier) < 0
        ):
            raise ValueError("Invalid execution fee or impact assumptions")
        for name, value in zip(
            ("strikes", "bid", "ask", "bid_depth", "ask_depth", "contract_sizes", "tick_cash"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        inverse = (
            np.zeros(len(values[0]), dtype=bool)
            if self.inverse_quotes is None
            else np.asarray(self.inverse_quotes, dtype=bool)
        )
        if inverse.shape != values[0].shape:
            raise ValueError("Quote denomination flags must match execution strikes")
        object.__setattr__(self, "inverse_quotes", inverse)

    @classmethod
    def from_snapshot(cls, snapshot, calls, spot, fee_bps=0.0, impact_multiplier=1.0):
        rows, inverse = [], []
        for row in calls.itertuples():
            metadata, quote = (
                snapshot.instruments[row.instrument_name],
                snapshot.tickers[row.instrument_name],
            )
            contract = Contract.from_metadata(metadata)
            inverse.append(contract.inverse)
            factor = contract.cash_factor(spot)
            bid, ask = float(quote["best_bid_price"]), float(quote["best_ask_price"])
            rows.append(
                (
                    row.strike,
                    bid * factor,
                    ask * factor,
                    quote.get("best_bid_amount", 0),
                    quote.get("best_ask_amount", 0),
                    contract.multiplier,
                    contract.tick_at((bid + ask) / 2) * factor,
                )
            )
        arrays = np.asarray(rows, dtype=float).T
        return cls(*arrays, spot, fee_bps, impact_multiplier, np.asarray(inverse))

    def trade(self, quantities, marks=None, spot=None, *, captured=False):
        """Signed cash debit and components for a vector/batch of call trades.

        Entry uses actual captured bid/ask, rounded outward to the captured tick.
        Later trades preserve captured relative spread, with one tick as a floor,
        around supplied model marks. Displayed depth remains fixed. Additional
        price impact = half-spread × max(|quantity| / side depth − 1, 0) × multiplier.
        Fees apply to traded option premium, not underlying notional.
        """
        q = np.asarray(quantities, float)
        if q.shape[-1] != len(self.strikes) or not np.isfinite(q).all():
            raise ValueError("Trade quantities must match execution strikes")
        if captured:
            mid = (self.bid + self.ask) / 2
            half = (self.ask - self.bid) / 2
            tick = self.tick_cash
        else:
            mid = np.asarray(marks, float)
            if mid.shape[-1] != len(self.strikes) or not np.isfinite(mid).all() or np.any(mid < 0):
                raise ValueError("Continuation execution needs nonnegative model marks")
            relative = (self.ask - self.bid) / np.maximum(self.bid + self.ask, 1e-15)
            ratio = np.asarray(self.spot if spot is None else spot, float) / self.spot
            if np.any(ratio <= 0):
                raise ValueError("Continuation spot must be positive")
            expanded_ratio = np.expand_dims(ratio, -1) if ratio.ndim else ratio
            tick = self.tick_cash * np.where(self.inverse_quotes, expanded_ratio, 1.0)
            half = np.maximum(mid * relative, tick / 2)
        depth = np.where(q >= 0, self.ask_depth, self.bid_depth)
        impact = half * np.maximum(np.abs(q) / depth - 1, 0) * self.impact_multiplier
        adverse = half + impact
        unrounded = np.maximum(0, mid + np.sign(q) * adverse)
        fill = (
            np.where(q >= 0, np.ceil(unrounded / tick - 1e-12), np.floor(unrounded / tick + 1e-12))
            * tick
        )
        fee = fee_from_notional(q * fill, self.fee_bps)
        slippage = q * (fill - mid)
        return dict(
            cash_debit=np.sum(q * fill + fee, axis=-1),
            spread_impact_cost=np.sum(slippage, axis=-1),
            fees=np.sum(fee, axis=-1),
            turnover=np.sum(np.abs(q) * fill, axis=-1),
            exceeds_displayed_depth=np.sum(np.abs(q) > depth, axis=-1),
        )

    def audit(self, weights):
        return pd.DataFrame(
            dict(
                strike=self.strikes,
                quantity_underlying=weights,
                contracts=weights / self.contract_sizes,
                bid_cash=self.bid,
                ask_cash=self.ask,
                bid_depth_underlying=self.bid_depth,
                ask_depth_underlying=self.ask_depth,
                contract_size=self.contract_sizes,
                tick_cash_at_capture=self.tick_cash,
            )
        )


def replication_entry(plan, execution=None):
    if execution is None:
        return dict(
            cost=plan.cost,
            fitted_cost=plan.cost,
            entry_fee=0.0,
            entry_spread_impact=0.0,
            quote_basis_adjustment=0.0,
            entry_depth_exceedances=0,
        )
    if not np.array_equal(plan.strikes, execution.strikes):
        raise ValueError("Execution strikes must match the fitted strip")
    trade = execution.trade(plan.weights, captured=True)
    midpoint_cost = plan.cash + plan.weights @ ((execution.bid + execution.ask) / 2)
    return dict(
        cost=float(plan.cash + trade["cash_debit"]),
        fitted_cost=plan.cost,
        entry_fee=float(trade["fees"]),
        entry_spread_impact=float(trade["spread_impact_cost"]),
        quote_basis_adjustment=float(midpoint_cost - plan.cost),
        entry_depth_exceedances=int(trade["exceeds_displayed_depth"]),
    )
