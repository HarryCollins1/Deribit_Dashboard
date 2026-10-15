"""Finite vanilla-strip approximations to hypothetical discrete up-barrier calls.

These are approximations, not exact barrier replication. A static strip depends
only on the terminal spot and cannot distinguish paths with different past hits.
The semi-static strip trades at its first monitored hit; its fit is performed at
the barrier boundary, so discrete overshoots leave an additional hedging error.

All amounts are cash per one underlying unit. Multipliers, signed position sizes,
currency conversion and transaction costs must be applied by the caller.
"""

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from deribit_dashboard.barrier_execution import replication_entry
from deribit_dashboard.pricing import DAYS, Option
from deribit_dashboard.strategies import Barrier, DiscreteBarrierGrid


@dataclass(frozen=True)
class Replication:
    method: str
    option: Option
    barrier: Barrier
    strikes: np.ndarray
    weights: np.ndarray
    vanilla_prices: np.ndarray
    cash: float
    cost: float
    fit_rmse: float
    terminal_rmse: float
    boundary_rmse: float
    terminal_max_error: float
    boundary_max_error: float
    ridge: float
    boundary_weight: float
    fit_samples: int
    assumptions: tuple[str, ...]


def _calls(spots, strikes, years, option):
    """Broadcast Black-Scholes call marks; strikes and spots use the same cash unit."""
    spots, strikes = np.asarray(spots, float), np.asarray(strikes, float)
    if years <= 1e-12:
        return np.maximum(spots - strikes, 0.0)
    forward = spots * exp((option.rate - option.dividend) * years)
    discount = exp(-option.rate * years)
    scale = option.volatility * sqrt(years)
    if scale == 0:
        return discount * np.maximum(forward - strikes, 0.0)
    d1 = np.log(forward / strikes) / scale + scale / 2
    return discount * (forward * ndtr(d1) - strikes * ndtr(d1 - scale))


def _validate(option, barrier, strikes, prices, ridge):
    if not isinstance(option, Option) or option.side != "call" or option.american:
        raise ValueError("Replication supports European calls only")
    if option.years <= 0:
        raise ValueError("Replication requires positive remaining maturity")
    if not isinstance(barrier, Barrier) or barrier.kind not in {"up-and-out", "up-and-in"}:
        raise ValueError("Replication supports zero-rebate up-and-out/in barriers only")
    strikes, prices = np.asarray(strikes, float), np.asarray(prices, float)
    if strikes.ndim != 1 or prices.shape != strikes.shape or len(strikes) < 2:
        raise ValueError("Supply matching discovered strikes and vanilla prices, at least two")
    if not np.isfinite(strikes).all() or np.any(strikes <= 0):
        raise ValueError("Discovered strikes must be finite and positive")
    if not np.isfinite(prices).all() or np.any(prices < 0):
        raise ValueError("Vanilla prices must be finite and nonnegative")
    if len(np.unique(strikes)) != len(strikes):
        raise ValueError("One vanilla price per distinct discovered strike is required")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge regularization must be finite and nonnegative")
    order = np.argsort(strikes)
    return strikes[order].copy(), prices[order].copy()


def _fit(option, barrier, strikes, vanilla_prices, ridge, method, boundary_weight):
    strikes, prices = _validate(option, barrier, strikes, vanilla_prices, ridge)
    if not np.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError("Boundary weight must be finite and nonnegative")
    if method == "semi-static" and boundary_weight == 0:
        raise ValueError("Semi-static approximation requires a positive boundary weight")
    target_index = np.flatnonzero(np.isclose(strikes, option.strike, rtol=1e-12, atol=0))
    if barrier.knock_in and len(target_index) != 1:
        raise ValueError("Knock-in complement requires the target call's discovered strike")

    # Equal-probability lognormal quantiles give a risk-neutral, normalized
    # terminal squared-error objective, rather than arbitrary strike-level weights.
    probabilities = (np.arange(601) + 0.5) / 601
    terminal_spots = option.spot * np.exp(
        (option.rate - option.dividend - 0.5 * option.volatility**2) * option.years
        + option.volatility * sqrt(option.years) * ndtri(probabilities)
    )
    terminal_calls = np.maximum(terminal_spots[:, None] - strikes[None, :], 0)
    terminal_target = np.maximum(terminal_spots - option.strike, 0)
    terminal_target *= terminal_spots < barrier.level
    known_hit = barrier.already_hit or bool(barrier.breached(option.spot))
    if known_hit:
        terminal_target[:] = 0
    # Coefficient zero is terminal cash / initial spot; the other coefficients
    # are vanilla call quantities. This avoids dollar-price conditioning issues.
    terminal_design = np.column_stack([np.ones(len(terminal_spots)), terminal_calls / option.spot])
    remaining_times = np.linspace(option.years / 48, option.years, 48)
    boundary_calls = np.array(
        [_calls(barrier.level, strikes, years, option) for years in remaining_times]
    )
    boundary_design = np.column_stack(
        [np.exp(-option.rate * remaining_times), boundary_calls / option.spot]
    )
    design = terminal_design / sqrt(len(terminal_spots))
    target = terminal_target / option.spot / sqrt(len(terminal_spots))
    if method == "semi-static":
        design = np.vstack([design, boundary_design * sqrt(boundary_weight / len(remaining_times))])
        target = np.r_[target, np.zeros(len(remaining_times))]
    if known_hit:
        coefficients = np.zeros(len(strikes) + 1)
    else:
        regularized = np.vstack([design, sqrt(ridge) * np.eye(len(strikes) + 1)])
        coefficients = np.linalg.lstsq(
            regularized, np.r_[target, np.zeros(len(strikes) + 1)], rcond=None
        )[0]
    terminal_errors = terminal_design @ coefficients * option.spot - terminal_target
    boundary_errors = boundary_design @ coefficients * option.spot
    terminal_rmse = float(np.sqrt(np.mean(terminal_errors**2)))
    boundary_rmse = float(np.sqrt(np.mean(boundary_errors**2)))
    weights = coefficients[1:].copy()
    cash = float(coefficients[0] * option.spot * exp(-option.rate * option.years))
    if barrier.knock_in:
        weights *= -1
        weights[target_index[0]] += 1
        cash *= -1
    fit_rmse = terminal_rmse
    if method == "semi-static":
        fit_rmse = sqrt(
            (terminal_rmse**2 + boundary_weight * boundary_rmse**2) / (1 + boundary_weight)
        )
    return Replication(
        method=method,
        option=option,
        barrier=barrier,
        strikes=strikes,
        weights=weights,
        vanilla_prices=prices,
        cash=cash,
        cost=float(cash + weights @ prices),
        fit_rmse=fit_rmse,
        terminal_rmse=terminal_rmse,
        boundary_rmse=boundary_rmse,
        terminal_max_error=float(np.max(np.abs(terminal_errors))),
        boundary_max_error=float(np.max(np.abs(boundary_errors))),
        ridge=float(ridge),
        boundary_weight=float(boundary_weight),
        fit_samples=len(terminal_spots),
        assumptions=(
            "Finite discovered-strike call strip and cash; approximation, not exact replication.",
            "All vanilla hedges have the target expiry; quantities are underlying-unit calls, not metadata contract counts.",
            "Initial cost uses supplied vanilla prices; cash earns the model's continuously compounded rate.",
            "Static fit targets the terminal truncated payoff and cannot reproduce prior barrier touches.",
            "Terminal RMSE uses 601 equally probable Black-Scholes terminal quantiles, not a realized hedging-error sample.",
            "Semi-static boundary fit uses 48 remaining maturities at exactly the barrier; discrete overshoots remain unhedged.",
            "Intermediate fair marks use constant-volatility Black-Scholes. Optional StripExecution adds captured entry bid/ask and scenario continuation spreads, depth impact and fees; the fit itself is frictionless.",
            "Semi-static knock-outs sell their strip at first monitored touch; knock-ins switch into the target vanilla call and carry the cash difference.",
        ),
    )


def fit_static(option, barrier, strikes, vanilla_prices, ridge=1e-8):
    """Fit a buy-and-hold cash + call strip to a terminal-payoff proxy.

    ``cash`` is the cash deposited today, ``weights`` are call quantities, and
    ``cost = cash + weights @ vanilla_prices``. All strikes must be discovered
    calls of the same expiry, quoted in the option's cash price units by caller.
    """
    return _fit(option, barrier, strikes, vanilla_prices, ridge, "static", 0.0)


def fit_semi_static(option, barrier, strikes, vanilla_prices, ridge=1e-8, boundary_weight=1.0):
    """Fit terminal payoff and touch-boundary values in normalized least squares.

    Boundary weight one gives equal aggregate weight to terminal and boundary
    mean squared errors. Reported RMSEs exclude the ridge penalty.
    """
    return _fit(option, barrier, strikes, vanilla_prices, ridge, "semi-static", boundary_weight)


def evaluate_replication(replication, times, spots, barrier_premium, *, execution=None):
    """Horizon P&L of hedging ONE short barrier with the fitted long strip.

    ``times`` are increasing elapsed years, including inception and every
    contractual monitoring date through the final observation; ``spots`` has
    shape (time, path). A horizon before expiry uses Black-Scholes continuation
    marks for vanilla calls and DiscreteBarrierGrid for the barrier liability.
    The output ``*_payoff`` columns then represent marked horizon values.

    The supplied premium is received at inception. Borrowing/lending is
    unrestricted at the option rate. Positive P&L benefits the barrier seller.
    Optional execution uses captured entry quotes and stated spread/depth/fee
    scenarios on later trades. Missing expiry observations are never fabricated.
    """
    times, spots = np.asarray(times, float), np.asarray(spots, float)
    if spots.ndim == 1:
        spots = spots[:, None]
    option, barrier = replication.option, replication.barrier
    if (
        times.ndim != 1
        or len(times) < 2
        or spots.ndim != 2
        or spots.shape[0] != len(times)
        or spots.shape[1] == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.isclose(times[0], 0, rtol=0, atol=1e-12)
        or times[-1] > option.years + 1e-10
    ):
        raise ValueError("Paths require increasing elapsed years from zero to at most expiry")
    if not np.isfinite(spots).all() or np.any(spots <= 0):
        raise ValueError("Path spots must be finite and positive")
    if not np.allclose(spots[0], option.spot, rtol=1e-10, atol=0):
        raise ValueError("Paths must start at the calibrated initial spot")
    if not np.isfinite(barrier_premium) or barrier_premium < 0:
        raise ValueError("Barrier premium must be finite and nonnegative")
    if replication.method not in {"static", "semi-static"}:
        raise ValueError("Unknown replication method")
    interval = barrier.monitoring_days / DAYS
    horizon = min(float(times[-1]), option.years)
    observation_times = np.unique(
        np.r_[0.0, np.arange(interval, option.years, interval), option.years]
    )
    observation_times = observation_times[observation_times <= horizon + 1e-10]
    monitor_indices = []
    for observation in observation_times:
        index = int(np.argmin(abs(times - observation)))
        if abs(times[index] - observation) > 1e-10:
            raise ValueError("Paths must include every contractual barrier observation")
        monitor_indices.append(index)
    breached = barrier.breached(spots[monitor_indices])
    if barrier.already_hit:
        breached[0] = True
    hit = breached.any(axis=0)
    first = np.array(monitor_indices)[breached.argmax(axis=0)]
    remaining = max(0.0, option.years - horizon)
    vanilla_payoff = _calls(spots[-1], option.strike, remaining, option)
    if remaining <= 1e-12:
        liability = vanilla_payoff * (hit if barrier.knock_in else ~hit)
    else:
        grid = DiscreteBarrierGrid(option, barrier, [horizon * DAYS])
        liability = grid.price(spots[-1], day=horizon * DAYS, hit=hit)
    final_calls = _calls(spots[-1, :, None], replication.strikes[None, :], remaining, option)
    hedge_payoff = replication.cash * exp(option.rate * horizon) + final_calls @ replication.weights
    entry = replication_entry(replication, execution)
    exit_cost = np.zeros(spots.shape[1])
    exit_cost_terminal = np.zeros(spots.shape[1])
    exit_fees = np.zeros(spots.shape[1])
    exit_depth_exceedances = np.zeros(spots.shape[1], dtype=int)
    target = np.isclose(replication.strikes, option.strike).astype(float)
    terminal_weights = np.broadcast_to(
        replication.weights, (spots.shape[1], len(replication.strikes))
    ).copy()
    if replication.method == "semi-static":
        for index in np.unique(first[hit]):
            selected = hit & (first == index)
            day, current_spots = times[index], spots[index, selected]
            remaining = max(0, option.years - day)
            carry = exp(option.rate * (horizon - day))
            prices = _calls(current_spots[:, None], replication.strikes[None, :], remaining, option)
            liquidation = replication.cash * exp(option.rate * day) + prices @ replication.weights
            if barrier.knock_in:
                vanilla_mark = _calls(current_spots, option.strike, remaining, option)
                hedge_payoff[selected] = (
                    vanilla_payoff[selected] + (liquidation - vanilla_mark) * carry
                )
            else:
                hedge_payoff[selected] = liquidation * carry
            after = target if barrier.knock_in else np.zeros_like(target)
            terminal_weights[selected] = after
            if execution is not None and remaining > 1e-12:
                trade = execution.trade(after - replication.weights, prices, current_spots)
                friction = trade["spread_impact_cost"] + trade["fees"]
                exit_cost[selected] += friction
                exit_cost_terminal[selected] += friction * carry
                exit_fees[selected] += trade["fees"]
                exit_depth_exceedances[selected] += trade["exceeds_displayed_depth"]
    if execution is not None and option.years - horizon > 1e-12:
        trade = execution.trade(-terminal_weights, final_calls, spots[-1])
        friction = trade["spread_impact_cost"] + trade["fees"]
        exit_cost += friction
        exit_cost_terminal += friction
        exit_fees += trade["fees"]
        exit_depth_exceedances += trade["exceeds_displayed_depth"]
    hedge_payoff -= exit_cost_terminal
    initial_funding = (barrier_premium - entry["cost"]) * exp(option.rate * horizon)
    replication_error = hedge_payoff - liability
    pnl = initial_funding + replication_error
    return pd.DataFrame(
        dict(
            path=np.arange(spots.shape[1]),
            valuation_years=horizon,
            barrier_hit=hit,
            first_hit_years=np.where(hit, times[first], np.nan),
            barrier_premium=float(barrier_premium),
            hedge_cost=entry["cost"],
            fitted_hedge_cost=replication.cost,
            entry_quote_basis_adjustment=entry["quote_basis_adjustment"],
            transaction_cost=entry["entry_fee"] + entry["entry_spread_impact"] + exit_cost,
            execution_fees=entry["entry_fee"] + exit_fees,
            unwind_cost=exit_cost,
            unwind_cost_at_horizon=exit_cost_terminal,
            depth_exceedances=entry["entry_depth_exceedances"] + exit_depth_exceedances,
            initial_funding_terminal=initial_funding,
            hedge_payoff=hedge_payoff,
            barrier_payoff=liability,
            replication_error=replication_error,
            pnl=pnl,
            reconciliation=pnl - initial_funding - hedge_payoff + liability,
        )
    )
