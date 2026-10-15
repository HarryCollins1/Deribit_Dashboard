"""Conditional-model delta and variance hedges on held-out simulated paths.

Each decision sees only current spot, barrier history, and (for Heston) current
raw Euler variance. Independent nested paths price continuation. Full truncation
uses positive variance in coefficients while retaining the raw Markov state.
Local-volatility continuation queries the ORIGINAL calibrated clock and spots.

Heston variance is latent in real markets. The variance-aware experiment is an
oracle-state simulation, not an executable estimator of live instantaneous risk.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from deribit_dashboard.barrier_calibration import fit_local_vol
from deribit_dashboard.barrier_hedging import dynamic_hedge, hedge_summary
from deribit_dashboard.barrier_models import (
    HestonParameters,
    _monitor_times,
    simulate_gbm,
    simulate_heston,
    simulate_local_vol,
)
from deribit_dashboard.barrier_workbench import execution_for
from deribit_dashboard.pricing import DAYS
from deribit_dashboard.strategies import Barrier


@dataclass(frozen=True)
class ModelHedgeControls:
    barrier: float = 120000.0
    path_model: str = "Heston"
    paths: int = 16
    inner_paths: int = 256
    inner_steps: int = 30
    spot_bump: float = 0.01
    variance_bump: float = 0.0025
    max_vanilla_quantity: float = 10.0
    minimum_variance_sensitivity: float = 1.0
    add_vanilla_hedge: bool = True
    seed: int = 701

    def __post_init__(self):
        if self.path_model not in {"GBM", "Local vol", "Heston"}:
            raise ValueError("Select GBM, Local vol or Heston path dynamics")
        if (
            any(not isinstance(n, int) or n % 2 for n in (self.paths, self.inner_paths))
            or not 4 <= self.paths <= 128
            or not 64 <= self.inner_paths <= 4096
        ):
            raise ValueError("Use 4–128 even outer paths and 64–4,096 even inner paths")
        if not 10 <= self.inner_steps <= 250 or not 0 < self.spot_bump < 0.05:
            raise ValueError("Invalid conditional integration steps or spot bump")
        if (
            not np.isfinite(
                [
                    self.barrier,
                    self.variance_bump,
                    self.max_vanilla_quantity,
                    self.minimum_variance_sensitivity,
                ]
            ).all()
            or min(
                self.barrier,
                self.variance_bump,
                self.max_vanilla_quantity,
                self.minimum_variance_sensitivity,
            )
            <= 0
        ):
            raise ValueError("Barrier, variance bump and hedge stability controls must be positive")


def _conditional_samples(
    option,
    barrier,
    elapsed,
    spots,
    raw_variance,
    hit,
    model,
    parameters,
    local_vol,
    controls,
    *,
    spot_scale=1.0,
    variance_shift=0.0,
):
    """Return nested (outer state × inner path) discounted barrier/vanilla payoffs."""
    spots = np.asarray(spots, float) * spot_scale
    remaining = max(option.years - elapsed, 0)
    count, inner = len(spots), controls.inner_paths
    if remaining < 1e-12:
        vanilla = np.maximum(spots - option.strike, 0)[:, None] * np.ones((1, inner))
        active = hit if barrier.knock_in else ~hit
        return vanilla * active[:, None], vanilla
    observation = _monitor_times(remaining, barrier.monitoring_days)
    times = np.unique(np.r_[np.linspace(0, remaining, controls.inner_steps + 1), observation])
    times = times[np.r_[True, np.diff(times) > 1e-14]]
    rng = np.random.default_rng(controls.seed + 15485863 + int(round(elapsed * DAYS * 1e6)))
    shocks = rng.standard_normal((len(times) - 1, inner // 2))
    shocks = np.concatenate([shocks, -shocks], axis=1)
    other_shocks = rng.standard_normal((len(times) - 1, inner // 2))
    other_shocks = np.concatenate([other_shocks, -other_shocks], axis=1)
    s = np.broadcast_to(spots[:, None], (count, inner)).copy()
    v = np.broadcast_to((np.asarray(raw_variance) + variance_shift)[:, None], s.shape).copy()
    touched = np.broadcast_to(np.asarray(hit)[:, None], s.shape).copy() | barrier.breached(s)
    for step, (previous, current) in enumerate(zip(times[:-1], times[1:], strict=True)):
        dt = current - previous
        z = shocks[step][None, :]
        if model == "Heston":
            vol = np.sqrt(np.maximum(v, 0))
        elif model == "Local vol":
            vol = np.asarray(local_vol(elapsed + previous, s.ravel())).reshape(s.shape)
        else:
            vol = np.sqrt(max(option.volatility**2 + variance_shift, 0))
        s *= np.exp((option.rate - option.dividend - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z)
        if model == "Heston":
            other = other_shocks[step][None, :]
            zv = parameters.rho * z + np.sqrt(1 - parameters.rho**2) * other
            v += (
                parameters.kappa * (parameters.theta - np.maximum(v, 0)) * dt
                + parameters.xi * vol * np.sqrt(dt) * zv
            )
        if np.min(abs(observation - current)) < 1e-10:
            touched |= barrier.breached(s)
    vanilla = np.exp(-option.rate * remaining) * np.maximum(s - option.strike, 0)
    return vanilla * (touched if barrier.knock_in else ~touched), vanilla


def conditional_risk(
    option,
    barrier,
    elapsed,
    spots,
    raw_variance,
    hit,
    model,
    parameters,
    local_vol,
    controls,
    *,
    variance_risk=False,
):
    """Fixed-calibration price/delta and optional conditional-variance derivatives.

    The same independent inner shocks are reused for every bump. Sampling SE of
    delta describes nested estimator uncertainty; it excludes outer-path error.
    """
    base_args = (
        option,
        barrier,
        elapsed,
        spots,
        raw_variance,
        hit,
        model,
        parameters,
        local_vol,
        controls,
    )
    center = _conditional_samples(*base_args)
    low = _conditional_samples(*base_args, spot_scale=1 - controls.spot_bump)
    high = _conditional_samples(*base_args, spot_scale=1 + controls.spot_bump)
    h = np.asarray(spots)[:, None] * controls.spot_bump
    delta_samples = (high[0] - low[0]) / (2 * h)
    half = controls.inner_paths // 2
    pairs = (delta_samples[:, :half] + delta_samples[:, half:]) / 2
    result = dict(
        price=center[0].mean(axis=1),
        delta=delta_samples.mean(axis=1),
        delta_se=pairs.std(axis=1, ddof=1) / np.sqrt(half),
        vanilla_price=center[1].mean(axis=1),
        vanilla_delta=((high[1] - low[1]) / (2 * h)).mean(axis=1),
    )
    if variance_risk:
        if model != "Heston":
            raise ValueError("Conditional variance hedge is defined here only for Heston dynamics")
        low_v = _conditional_samples(*base_args, variance_shift=-controls.variance_bump)
        high_v = _conditional_samples(*base_args, variance_shift=controls.variance_bump)
        result["variance_sensitivity"] = (high_v[0] - low_v[0]).mean(axis=1) / (
            2 * controls.variance_bump
        )
        result["vanilla_variance_sensitivity"] = (high_v[1] - low_v[1]).mean(axis=1) / (
            2 * controls.variance_bump
        )
    return result


def _run_conditional_hedge(report, controls, sim, local, parameters, premium, add_vanilla):
    option, cfg, market = report.market.option, report.config, report.market
    barrier = Barrier(cfg.kind, controls.barrier, cfg.monitoring_days)
    observe = _monitor_times(option.years, cfg.monitoring_days)
    indices = np.array([np.argmin(abs(sim.times - t)) for t in observe])
    if not np.allclose(sim.times[indices], observe, atol=1e-10, rtol=0):
        raise ValueError("Outer paths must contain all monitoring observations")
    count = sim.spots.shape[1]
    execution = execution_for(market, cfg)
    target_index = int(np.flatnonzero(np.isclose(market.calls.strike, option.strike))[0])
    target = np.zeros(len(market.calls))
    target[target_index] = 1
    perp = market.perpetual
    cash = np.full(count, premium, float)
    holdings = np.zeros(count)
    vanilla_holdings = np.zeros(count)
    hit = np.zeros(count, dtype=bool)
    gains, vanilla_flows, funding, costs, interest = (np.zeros(count) for _ in range(5))
    quantity_caps = np.zeros(count, int)
    weak_sensitivity = np.zeros(count, int)
    max_delta_se = np.zeros(count)
    risk_rows = []
    previous_future = sim.spots[0] * perp.initial_basis_ratio
    for i, index in enumerate(indices):
        elapsed, spot = sim.times[index], sim.spots[index]
        future = spot * perp.initial_basis_ratio
        if i:
            dt = elapsed - observe[i - 1]
            accrual = cash * np.expm1(option.rate * dt)
            cash += accrual
            interest += accrual
            if perp.inverse:
                gain = holdings * perp.contract_size * (1 / previous_future - 1 / future) * spot
                fund = holdings * perp.contract_size * cfg.annual_funding * dt
            else:
                gain = holdings * perp.contract_size * (future - previous_future)
                fund = holdings * perp.contract_size * previous_future * cfg.annual_funding * dt
            gains += gain
            funding += fund
            cash += gain - fund
        hit |= barrier.breached(spot)
        variance = (
            sim.variances[index]
            if sim.variances is not None
            else np.full(count, option.volatility**2)
        )
        terminal = i == len(indices) - 1
        if terminal:
            vanilla = np.maximum(spot - option.strike, 0)
            liability = vanilla * (hit if barrier.knock_in else ~hit)
            cash += vanilla_holdings * vanilla
            vanilla_flows += vanilla_holdings * vanilla
            desired_vanilla = np.zeros(count)
            desired = np.zeros(count)
        else:
            risk = conditional_risk(
                option,
                barrier,
                elapsed,
                spot,
                variance,
                hit,
                controls.path_model,
                parameters,
                local,
                controls,
                variance_risk=add_vanilla,
            )
            max_delta_se = np.maximum(max_delta_se, risk["delta_se"])
            desired_vanilla = np.zeros(count)
            if add_vanilla:
                sensitivity = risk["vanilla_variance_sensitivity"]
                valid = np.abs(sensitivity) >= controls.minimum_variance_sensitivity
                weak_sensitivity += ~valid
                ratio = np.divide(
                    risk["variance_sensitivity"], sensitivity, out=np.zeros(count), where=valid
                )
                quantity_caps += np.abs(ratio) > controls.max_vanilla_quantity
                desired_vanilla = np.clip(
                    ratio, -controls.max_vanilla_quantity, controls.max_vanilla_quantity
                )
                if not barrier.knock_in:
                    desired_vanilla[hit] = 0
            delta = risk["delta"] - desired_vanilla * risk["vanilla_delta"]
            desired = (
                delta * future / perp.contract_size
                if perp.inverse
                else delta / (perp.contract_size * perp.initial_basis_ratio)
            )
            change_vanilla = desired_vanilla - vanilla_holdings
            mark = risk["vanilla_price"]
            if execution is None:
                debit = change_vanilla * mark
                friction = np.zeros(count)
            else:
                # Only the discovered target call trades. Other marks are irrelevant
                # because their quantities are identically zero.
                marks = np.zeros((count, len(target)))
                marks[:, target_index] = mark
                trade = execution.trade(
                    change_vanilla[:, None] * target, marks, spot, captured=i == 0
                )
                debit = trade["cash_debit"]
                friction = trade["spread_impact_cost"] + trade["fees"]
            cash -= debit
            vanilla_flows -= debit - friction
            costs += friction
            for path in range(count):
                risk_rows.append(
                    dict(
                        path=path,
                        elapsed_days=elapsed * DAYS,
                        spot=spot[path],
                        raw_variance=variance[path],
                        barrier_hit=hit[path],
                        barrier_delta=risk["delta"][path],
                        delta_sampling_se=risk["delta_se"][path],
                        vanilla_quantity=desired_vanilla[path],
                        residual_delta=delta[path],
                    )
                )
        notional = np.abs(desired - holdings) * perp.contract_size * (1 if perp.inverse else future)
        spread = perp.half_spread_bps if cfg.market_execution else 0
        friction = (
            notional
            * (
                cfg.fee_bps
                + spread
                * (
                    1
                    + cfg.impact_multiplier
                    * np.maximum(notional / perp.displayed_depth_notional - 1, 0)
                )
            )
            / 10000
        )
        cash -= friction
        costs += friction
        holdings, vanilla_holdings, previous_future = desired, desired_vanilla, future
    pnl = cash - liability
    result = pd.DataFrame(
        dict(
            path=np.arange(count),
            premium=premium,
            hedge_gains=gains,
            vanilla_net_cashflow=vanilla_flows,
            funding_paid=funding,
            transaction_cost=costs,
            cash_interest=interest,
            barrier_liability=liability,
            pnl=pnl,
            barrier_hit=hit,
            maximum_delta_sampling_se=max_delta_se,
            vanilla_quantity_cap_events=quantity_caps,
            weak_variance_sensitivity_events=weak_sensitivity,
        )
    )
    result["reconciliation"] = (
        premium + gains + vanilla_flows - funding - costs + interest - liability - pnl
    )
    return result, pd.DataFrame(risk_rows)


def model_hedge_study(report, controls=None):
    """Compare BS delta, conditional-model delta and optional Heston variance hedge.

    Every method receives the SAME selected-model MC premium and sees the SAME
    independent evaluation paths. The pricing experiment's seed is not reused.
    """
    controls = controls or ModelHedgeControls(barrier=report.config.probability_level)
    option, cfg = report.market.option, report.config
    if option.years * DAYS / cfg.monitoring_days > 120:
        raise ValueError("Conditional hedge study is bounded to 120 monitoring intervals")
    monitoring_count = int(np.ceil(option.years * DAYS / cfg.monitoring_days))
    evaluation_count = 8 if controls.add_vanilla_hedge and controls.path_model == "Heston" else 3
    projected_work = (
        controls.paths
        * controls.inner_paths
        * (controls.inner_steps + monitoring_count)
        * monitoring_count
        * evaluation_count
    )
    if projected_work > 300_000_000:
        raise ValueError(
            "Reduce outer paths, conditional paths or steps: conditional hedge study is bounded to 300 million path-step evaluations"
        )
    local = fit_local_vol(report.market.analysis.surface, option)
    parameters = HestonParameters(**report.diagnostics["heston_parameters"])
    inputs = dict(
        paths=controls.paths,
        steps=max(cfg.steps, controls.inner_steps),
        seed=controls.seed,
        monitoring_days=cfg.monitoring_days,
    )
    if controls.path_model == "Heston":
        sim = simulate_heston(option, parameters, retain_variance=True, **inputs)
    elif controls.path_model == "Local vol":
        sim = simulate_local_vol(option, local, **inputs)
    else:
        sim = simulate_gbm(option, **inputs)
    # Use independent inception continuation paths for all methods' common premium.
    initial_controls = replace(
        controls, inner_paths=max(controls.inner_paths, 1024), seed=controls.seed + 104729
    )
    inception = conditional_risk(
        option,
        Barrier(cfg.kind, controls.barrier, cfg.monitoring_days),
        0,
        np.array([option.spot]),
        np.array([parameters.v0]),
        np.array([False]),
        controls.path_model,
        parameters,
        local,
        initial_controls,
    )
    premium = float(inception["price"][0])
    observe = _monitor_times(option.years, cfg.monitoring_days)
    indices = [np.argmin(abs(sim.times - t)) for t in observe]
    bs = dynamic_hedge(
        option,
        Barrier(cfg.kind, controls.barrier, cfg.monitoring_days),
        sim.times[indices],
        sim.spots[indices],
        premium,
        report.market.perpetual
        if cfg.market_execution
        else replace(report.market.perpetual, half_spread_bps=0),
        fee_bps=cfg.fee_bps,
        annual_funding=cfg.annual_funding,
        impact_multiplier=cfg.impact_multiplier,
    )
    rows = [bs.assign(method="Black-Scholes delta", path_model=controls.path_model)]
    audits = []
    for add_vanilla in (
        [False, True] if controls.add_vanilla_hedge and controls.path_model == "Heston" else [False]
    ):
        result, audit = _run_conditional_hedge(
            report, controls, sim, local, parameters, premium, add_vanilla
        )
        method = (
            "Conditional model delta + vanilla variance hedge"
            if add_vanilla
            else "Conditional model delta"
        )
        rows.append(result.assign(method=method, path_model=controls.path_model))
        audits.append(audit.assign(method=method))
    details = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(
        [
            dict(
                method=method,
                path_model=controls.path_model,
                common_premium=premium,
                **hedge_summary(group),
            )
            for method, group in details.groupby("method", sort=False)
        ]
    )
    return details, summary, pd.concat(audits, ignore_index=True)
