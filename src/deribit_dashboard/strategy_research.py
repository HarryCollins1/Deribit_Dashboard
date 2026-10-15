"""Explicit research assumptions for strategy sizing, smile dynamics and collateral.

These are scenario rules, not exchange margin requirements or calibrated forecasts.
All instrument quantities are resolved from captured metadata before these functions
are called. No UI or persistence dependencies live here.
"""

from dataclasses import dataclass, field, replace
from math import exp, sqrt

import numpy as np
import pandas as pd
from scipy.special import ndtr

from deribit_dashboard.pricing import DAYS, MODELS


@dataclass(frozen=True)
class SurfaceEvolution:
    mode: str = "fixed volatility"
    parallel_shock: float = 0.0
    skew_shock: float = 0.0
    term_shock: float = 0.0

    def __post_init__(self):
        if self.mode not in {"fixed volatility", "sticky strike", "sticky forward delta"}:
            raise ValueError("Unknown surface evolution rule")
        if not np.isfinite([self.parallel_shock, self.skew_shock, self.term_shock]).all():
            raise ValueError("Surface shocks must be finite")


@dataclass(frozen=True)
class MarginScenario:
    enabled: bool = False
    initial_margin_fraction: float = 0.2
    maintenance_margin_fraction: float = 0.1
    coin_collateral_fraction: float = 0.0
    collateral_asset: str = "BTC"
    collateral_haircut: float = 0.0
    liquidation_fee_bps: float = 10.0

    def __post_init__(self):
        values = [
            self.initial_margin_fraction,
            self.maintenance_margin_fraction,
            self.coin_collateral_fraction,
            self.collateral_haircut,
            self.liquidation_fee_bps,
        ]
        if not np.isfinite(values).all():
            raise ValueError("Margin assumptions must be finite")
        if not 0 <= self.maintenance_margin_fraction <= self.initial_margin_fraction <= 1:
            raise ValueError("Require 0 <= maintenance <= initial margin <= 1")
        if not 0 <= self.coin_collateral_fraction <= 1 or not 0 <= self.collateral_haircut < 1:
            raise ValueError("Invalid collateral allocation or haircut")
        if self.collateral_asset not in {"BTC", "ETH"} or self.liquidation_fee_bps < 0:
            raise ValueError("Invalid collateral asset or liquidation cost")


@dataclass(frozen=True)
class PathRegime:
    """One deterministic regime change embedded in every stochastic path.

    Jump and basis shocks are relative levels; subsequent volatility multipliers,
    correlation and annual funding rates replace the prior regime until another
    event. The option-volatility shock is a relative continuation-mark shock.
    """

    day: float
    spot_jumps: dict = field(default_factory=dict)
    volatility_multipliers: dict = field(default_factory=dict)
    correlation: float | None = None
    basis_shocks: dict = field(default_factory=dict)
    annual_funding_rates: dict = field(default_factory=dict)
    option_volatility_shock: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.day) or self.day <= 0:
            raise ValueError("Regime changes require a positive finite elapsed day")
        for values in (
            self.spot_jumps,
            self.volatility_multipliers,
            self.basis_shocks,
            self.annual_funding_rates,
        ):
            if set(values) - {"BTC", "ETH"} or not all(np.isfinite(v) for v in values.values()):
                raise ValueError("Regime inputs must be finite BTC/ETH values")
        if any(v <= -1 for v in (*self.spot_jumps.values(), *self.basis_shocks.values())):
            raise ValueError("Jump and basis shocks must exceed -100%")
        if any(v < 0 for v in self.volatility_multipliers.values()):
            raise ValueError("Path volatility multipliers must be nonnegative")
        if self.correlation is not None and (
            not np.isfinite(self.correlation) or abs(self.correlation) > 1
        ):
            raise ValueError("Regime correlation must be between -1 and 1")
        if not np.isfinite(self.option_volatility_shock) or self.option_volatility_shock < -1:
            raise ValueError("Option volatility shock must be at least -100%")


def surface_volatility(leg, spots, day, evolution, diagnostics=None):
    """Frozen initial smile/term references; hypothetical evolution, without refitting.

    Sticky forward delta retains the initial forward-call-delta smile. Its scalar
    fixed point is damped and convergence checked. Between maturities interpolate
    total variance; outside captured tenors/strikes use the edge vol and count it.
    Shocks are absolute decimal volatility: parallel + skew*log(K/F) + term*T.
    """
    option = leg.option
    spots = np.asarray(spots, dtype=float)
    remaining = max(option.years - day / DAYS, 0.0)
    vol = np.full_like(spots, option.volatility)
    diagnostics = diagnostics if diagnostics is not None else {}
    if remaining <= 1e-12:
        return vol
    if evolution.mode != "fixed volatility":
        if leg.model_name != "Black-Scholes" or option.american or not leg.surface_reference:
            raise ValueError("Surface evolution requires a captured Black-Scholes vanilla surface")
        tenors, estimates = [], []
        for maturity, strikes, sigmas in leg.surface_reference:
            strikes, sigmas = np.asarray(strikes), np.asarray(sigmas)
            if evolution.mode == "sticky strike":
                estimate = np.full_like(spots, np.interp(option.strike, strikes, sigmas))
                outside = np.full(
                    spots.shape, option.strike < strikes[0] or option.strike > strikes[-1]
                )
            else:
                original_forward = option.spot * exp((option.rate - option.dividend) * maturity)
                widths = np.maximum(sigmas * sqrt(maturity), 1e-12)
                delta = ndtr(np.log(original_forward / strikes) / widths + widths / 2)
                order = np.argsort(delta)
                target_forward = spots * exp((option.rate - option.dividend) * remaining)
                estimate = np.full_like(spots, option.volatility)
                for _ in range(100):
                    width = np.maximum(estimate * sqrt(remaining), 1e-12)
                    target_delta = ndtr(np.log(target_forward / option.strike) / width + width / 2)
                    mapped = np.interp(target_delta, delta[order], sigmas[order])
                    updated = (estimate + mapped) / 2
                    if np.max(np.abs(updated - estimate)) < 1e-8:
                        estimate = updated
                        break
                    estimate = updated
                else:
                    raise ValueError("Sticky-delta fixed point did not converge for this scenario")
                outside = (target_delta < delta.min()) | (target_delta > delta.max())
            diagnostics["surface_wing_edge_uses"] = diagnostics.get(
                "surface_wing_edge_uses", 0
            ) + int(outside.sum())
            tenors.append(maturity)
            estimates.append(estimate)
        tenors = np.asarray(tenors)
        index = int(np.searchsorted(tenors, remaining))
        if index == 0 or index == len(tenors):
            vol = estimates[0 if index == 0 else -1]
            if remaining < tenors[0] or remaining > tenors[-1]:
                diagnostics["surface_tenor_edge_uses"] = (
                    diagnostics.get("surface_tenor_edge_uses", 0) + spots.size
                )
        else:
            left, right = index - 1, index
            weight = (remaining - tenors[left]) / (tenors[right] - tenors[left])
            variance = (1 - weight) * estimates[left] ** 2 * tenors[left] + weight * estimates[
                right
            ] ** 2 * tenors[right]
            vol = np.sqrt(variance / remaining)
    forward = spots * exp((option.rate - option.dividend) * remaining)
    vol = (
        vol
        + evolution.parallel_shock
        + evolution.skew_shock * np.log(option.strike / forward)
        + evolution.term_shock * remaining
    )
    if np.any(vol < 0) or not np.isfinite(vol).all():
        raise ValueError("Surface evolution produced invalid volatility; reduce shocks")
    return vol


def scenario_factors(times, assets, scenario):
    """Paths through explicit fractional-return waypoints, or a named path shape.

    Waypoints are relative to inception, not chained returns. Piecewise log-linear
    paths include their exact event times. A jump occurs at its event time and then
    holds; barrier checks still occur only on contractual observation dates.
    """
    horizon = float(times[-1])
    shape = scenario.get("path_shape", "monotone")
    if shape not in {"monotone", "touch and retrace", "jump then hold", "waypoints"}:
        raise ValueError("Unknown stress path shape")
    factors = {}
    for asset in assets:
        end = 1 + float(scenario.get("spot_shocks", {}).get(asset, 0))
        event_day = float(scenario.get("event_day", horizon / 2))
        if not 0 < event_day <= horizon or not np.isfinite(end) or end <= 0:
            raise ValueError(
                "Stress events require positive spot levels and 0 < event day <= horizon"
            )
        if shape == "jump then hold":
            factors[asset] = np.where(times[:, None] >= event_day, end, 1.0)
            continue
        points = [(0.0, 1.0), (horizon, end)]
        if shape == "touch and retrace":
            peak = 1 + float(scenario.get("peak_shocks", {}).get(asset, 0))
            points.insert(1, (event_day, peak))
        elif shape == "waypoints":
            points = [(float(p["day"]), 1 + float(p.get(asset, 0))) for p in scenario["waypoints"]]
        x, y = np.array(points, dtype=float).T
        if (
            x[0] != 0
            or x[-1] != horizon
            or np.any(np.diff(x) <= 0)
            or np.any(y <= 0)
            or not np.isfinite(y).all()
        ):
            raise ValueError(
                "Waypoints must span day zero to horizon, increase strictly, and have positive finite levels"
            )
        factors[asset] = np.exp(np.interp(times, x, np.log(y)))[:, None]
    return factors


def size_pair(
    legs,
    target_first_notional,
    *,
    method="dollar",
    betas=None,
    annual_volatility=None,
    depth_fraction=1.0,
):
    """Size two signed legs to cancel a stated exposure; apply one common depth scale.

    Beta and volatility weights act on dollar delta. No beta is estimated here.
    Contract steps are rounded toward zero if captured; residuals are reported.
    Missing displayed depth rejects a depth-capped sizing request.
    """
    from deribit_dashboard.strategies import barrier_price

    if len(legs) != 2 or method not in {"dollar", "delta", "beta", "volatility"}:
        raise ValueError("Select exactly two legs and a dollar/delta/beta/volatility sizing rule")
    if not np.isfinite(target_first_notional) or target_first_notional <= 0:
        raise ValueError("Target first-leg notional must be positive")
    if depth_fraction is not None and (
        not np.isfinite(depth_fraction) or not 0 < depth_fraction <= 1
    ):
        raise ValueError("Depth fraction must be in (0, 1]")
    unit_notionals, exposures = [], []
    for leg in legs:
        notional = (
            leg.multiplier
            * (1 if leg.inverse and leg.kind in {"future", "perpetual"} else leg.spot)
            * leg.conversion_rate
        )
        unit_notionals.append(notional)
        if method == "dollar":
            exposure = notional
        else:
            if leg.option:
                bump = leg.spot * 0.0001
                price = (
                    (lambda o: barrier_price(o, leg.barrier))
                    if leg.barrier
                    else MODELS[leg.model_name].price
                )
                delta = (
                    price(replace(leg.option, spot=leg.spot + bump))
                    - price(replace(leg.option, spot=leg.spot - bump))
                ) / (2 * bump)
                exposure = delta * leg.spot * leg.multiplier * leg.conversion_rate
            elif leg.kind == "spot":
                exposure = notional
            elif leg.inverse:
                exposure = notional * leg.spot / leg.entry_price
            else:
                exposure = leg.multiplier * leg.entry_price * leg.conversion_rate
            if method == "beta":
                if leg.asset not in (betas or {}):
                    raise ValueError("Supply explicit beta assumptions for both assets")
                exposure *= betas[leg.asset]
            if method == "volatility":
                if leg.asset not in (annual_volatility or {}) or annual_volatility[leg.asset] <= 0:
                    raise ValueError(
                        "Supply positive annual volatility assumptions for both assets"
                    )
                exposure *= annual_volatility[leg.asset]
        if not np.isfinite(exposure) or abs(exposure) < 1e-10:
            raise ValueError("Cannot neutralize a zero or nonfinite exposure")
        exposures.append(exposure)
    first = (1 if legs[0].quantity >= 0 else -1) * target_first_notional / unit_notionals[0]
    quantities = np.array([first, -first * exposures[0] / exposures[1]])
    desired = quantities.copy()
    scale = 1.0
    if depth_fraction is not None:
        for leg, quantity in zip(legs, quantities, strict=True):
            if (
                leg.depth_contracts is None
                or not np.isfinite(leg.depth_contracts)
                or leg.depth_contracts < 0
            ):
                raise ValueError(
                    "Captured displayed depth is missing; disable the depth cap explicitly or refresh quotes"
                )
            if abs(quantity) > 0:
                scale = min(scale, depth_fraction * leg.depth_contracts / abs(quantity))
        quantities *= scale
    for i, leg in enumerate(legs):
        if leg.quantity_step is not None:
            if not np.isfinite(leg.quantity_step) or leg.quantity_step <= 0:
                raise ValueError("Captured quantity increment must be positive")
            quantities[i] = (
                np.sign(quantities[i])
                * np.floor(abs(quantities[i]) / leg.quantity_step + 1e-10)
                * leg.quantity_step
            )
        if leg.minimum_quantity is not None and 0 < abs(quantities[i]) < leg.minimum_quantity:
            quantities[i] = 0.0
    result = [replace(leg, quantity=float(q)) for leg, q in zip(legs, quantities, strict=True)]
    audit = pd.DataFrame(
        dict(
            instrument_name=[leg.instrument_name for leg in legs],
            desired_quantity=desired,
            sized_quantity=quantities,
            unit_cash_notional=unit_notionals,
            unit_target_exposure=exposures,
            signed_target_exposure=quantities * exposures,
            displayed_depth=[leg.depth_contracts for leg in legs],
            minimum_quantity=[leg.minimum_quantity for leg in legs],
            quantity_step=[leg.quantity_step for leg in legs],
            common_depth_scale=scale,
        )
    )
    return result, audit


def collateral_ledger(legs, times, factors, raw_equity, initial_equity, margin):
    """Conservative gross-notional scenario margin with explicit first-breach closeout.

    Derivatives reserve gross margin without spread offsets; long options do not
    reserve extra margin beyond their premium, spot has no borrowed leverage here.
    Coin collateral is part of initial equity, never extra capital. On liquidation
    all positions and collateral convert to cash at the observed node; no recovery
    is assumed between observations and negative balances remain visible.
    """
    equity = np.array(raw_equity, copy=True)
    times = np.asarray(times, dtype=float)
    if (
        equity.ndim != 2
        or equity.shape[0] != len(times)
        or not np.isfinite(equity).all()
        or not np.isfinite(times).all()
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Margin ledger requires finite equity and an increasing elapsed-day grid from zero"
        )
    count = equity.shape[1]
    if not margin.enabled:
        return equity, pd.DataFrame(), dict(liquidation_fraction=0.0, margin_enabled=False)
    collateral = np.zeros_like(equity)
    if margin.coin_collateral_fraction:
        collateral = (
            initial_equity * margin.coin_collateral_fraction * factors[margin.collateral_asset]
        )
        equity += collateral - initial_equity * margin.coin_collateral_fraction
    gross = np.zeros_like(equity)
    # Liquidation also converts the remaining coin collateral into reporting cash.
    exit_notional = np.array(collateral, copy=True)
    for leg in legs:
        amounts = abs(leg.quantity) * leg.multiplier * leg.conversion_rate
        notional = (
            np.full_like(equity, amounts)
            if leg.inverse and leg.kind in {"future", "perpetual"}
            else amounts * leg.spot * factors[leg.asset]
        )
        active = (
            np.ones(len(times), dtype=bool)
            if leg.years is None
            else times < leg.years * DAYS - 1e-8
        )
        notional = notional * active[:, None]
        if leg.barrier and not leg.barrier.knock_in:
            intervals = times / leg.barrier.monitoring_days
            monitored = np.isclose(intervals, np.round(intervals), rtol=0, atol=1e-8)
            monitored |= np.isclose(times, leg.years * DAYS, atol=1e-8)
            observations = leg.barrier.breached(leg.spot * factors[leg.asset]) & monitored[:, None]
            knocked_out = np.maximum.accumulate(observations, axis=0) | leg.barrier.already_hit
            notional = np.where(knocked_out, 0, notional)
        exit_notional += notional
        if leg.kind in {"future", "perpetual"} or (leg.option and leg.quantity < 0):
            gross += notional
    maintenance = margin.maintenance_margin_fraction * gross
    available = equity - margin.collateral_haircut * collateral
    initial_required = margin.initial_margin_fraction * gross[0]
    if np.any(available[0] < initial_required):
        raise ValueError(
            "Initial equity after collateral haircut is below the chosen initial margin requirement"
        )
    breached = available < maintenance
    # Nonpositive equity is always a breach, including accounts with no derivatives.
    breached |= equity <= 0
    rows = []
    for path in range(count):
        indices = np.flatnonzero(breached[:, path])
        hit = int(indices[0]) if len(indices) else None
        fee = (
            float(exit_notional[hit, path] * margin.liquidation_fee_bps / 10000)
            if hit is not None
            else 0.0
        )
        before = float(equity[hit, path]) if hit is not None else np.nan
        if hit is not None:
            equity[hit:, path] = before - fee
        rows.append(
            dict(
                path=f"path_{path}",
                liquidated=hit is not None,
                liquidation_day=float(times[hit]) if hit is not None else np.nan,
                equity_before_liquidation=before,
                liquidation_cost=fee,
                equity_after_liquidation=before - fee,
                initial_margin=float(initial_required[path]),
                minimum_margin_buffer=float(
                    np.min(
                        (available - maintenance)[
                            : hit + 1 if hit is not None else len(times), path
                        ]
                    )
                ),
                terminal_collateral_pnl=float(
                    collateral[hit if hit is not None else -1, path]
                    - initial_equity * margin.coin_collateral_fraction
                ),
                strategy_pnl_to_close=float(
                    raw_equity[hit if hit is not None else -1, path] - initial_equity
                ),
                pnl_reconciliation_error=float(
                    equity[-1, path]
                    - initial_equity
                    - (
                        raw_equity[hit if hit is not None else -1, path]
                        - initial_equity
                        + collateral[hit if hit is not None else -1, path]
                        - initial_equity * margin.coin_collateral_fraction
                        - fee
                    )
                ),
                closeout_reconciliation_error=(float(equity[hit, path]) - before + fee)
                if hit is not None
                else 0.0,
            )
        )
    ledger = pd.DataFrame(rows)
    return (
        equity,
        ledger,
        dict(
            margin_enabled=True,
            liquidation_fraction=float(ledger.liquidated.mean()),
            mean_liquidation_cost=float(ledger.liquidation_cost.mean()),
            max_closeout_reconciliation_error=float(
                ledger.closeout_reconciliation_error.abs().max()
            ),
            max_pnl_reconciliation_error=float(ledger.pnl_reconciliation_error.abs().max()),
        ),
    )
