"""Numerical convergence and separately labelled monitoring-contract studies.

The GBM bridge conditions on exact lognormal endpoints. For an upper barrier,
interval survival is 1-exp(-2 log(B/S_i) log(B/S_j)/(sigma² dt)). Conditional
independence of Brownian bridges permits multiplication over intervals. It is
exact for continuous GBM monitoring, not an approximation offered for Heston or LV.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deribit_dashboard.barrier_calibration import fit_local_vol
from deribit_dashboard.barrier_models import (
    HestonParameters,
    _check_option,
    antithetic_estimate,
    bump_greeks,
    discounted_payoffs,
    simulate_gbm,
)
from deribit_dashboard.pricing import DAYS
from deribit_dashboard.strategies import Barrier, DiscreteBarrierGrid


def gbm_bridge_samples(paths, option, barrier, kind="up-and-out"):
    """Conditional payoff and touch probability samples for continuous GBM.

    Samples are conditional expectations, not Bernoulli hit indicators. This
    reduces sampling variance. Zero volatility uses deterministic endpoint hits.
    """
    if paths.model != "GBM" or kind not in {
        "up-and-out",
        "up-and-in",
        "down-and-out",
        "down-and-in",
    }:
        raise ValueError("Continuous bridge is supported only for GBM zero-rebate barriers")
    if not np.isfinite(barrier) or barrier <= 0:
        raise ValueError("Barrier must be finite and positive")
    _check_option(paths, option)
    sign = 1 if kind.startswith("up") else -1
    distance = sign * np.log(barrier / paths.spots)
    hit = np.any(distance <= 0, axis=0)
    survival = np.ones(paths.spots.shape[1])
    if option.volatility > 0:
        denominator = option.volatility**2 * np.diff(paths.times)[:, None]
        exponent = -2 * np.maximum(distance[:-1], 0) * np.maximum(distance[1:], 0) / denominator
        survival = np.prod(-np.expm1(exponent), axis=0)
    survival[hit] = 0
    vanilla = np.exp(-option.rate * option.years) * np.maximum(
        option.sign * (paths.spots[-1] - option.strike), 0
    )
    active = 1 - survival if kind.endswith("-in") else survival
    return vanilla * active, 1 - survival


@dataclass(frozen=True)
class ConvergenceControls:
    barrier: float = 120000.0
    models: tuple = ("GBM", "Local vol", "Heston")
    paths: int = 1024
    steps: int = 60
    seed: int = 42
    path_counts: tuple = (256, 1024, 4096)
    step_counts: tuple = (30, 60, 120)
    seeds: tuple = (11, 29, 47)
    bump_fractions: tuple = (0.005, 0.01, 0.02)
    grid_points: tuple = (801, 1601, 3201)
    monitoring_days: tuple = (0.25, 1.0, 2.0)

    def __post_init__(self):
        if not self.models or set(self.models) - {"GBM", "Local vol", "Heston"}:
            raise ValueError("Choose GBM, Local vol and/or Heston")
        if not np.isfinite(self.barrier) or self.barrier <= 0:
            raise ValueError("Choose a positive finite convergence barrier")
        if any(
            not isinstance(n, int) or n < 128 or n > 16384 or n % 2
            for n in (self.paths, *self.path_counts)
        ):
            raise ValueError("Convergence paths must be even counts between 128 and 16,384")
        if any(
            not isinstance(n, int) or not 10 <= n <= 1500 for n in (self.steps, *self.step_counts)
        ):
            raise ValueError("Convergence steps must be between 10 and 1,500")
        if any(not 0 < b < 0.05 for b in self.bump_fractions) or any(
            not 0.05 <= d <= 30 for d in self.monitoring_days
        ):
            raise ValueError("Invalid convergence bump or monitoring interval")
        if any(not isinstance(n, int) or not 401 <= n <= 6401 for n in self.grid_points):
            raise ValueError("Grid sizes must be integers between 401 and 6,401")
        if (
            sum(
                map(
                    len,
                    (
                        self.path_counts,
                        self.step_counts,
                        self.seeds,
                        self.bump_fractions,
                        self.grid_points,
                        self.monitoring_days,
                    ),
                )
            )
            > 30
        ):
            raise ValueError("At most 30 convergence perturbations per model")


def barrier_convergence(report, controls=None):
    """Return detailed trials plus empirical summaries with calibration held fixed.

    Changes in price/gamma across trials are numerical sensitivity diagnostics,
    not certified bias estimates. Cross-trial differences are correlated and no
    invalid independent-SE significance test is imposed. Monitoring changes the
    contract; it is reported separately from numerical convergence.
    """
    controls = controls or ConvergenceControls(barrier=report.config.probability_level)
    option, config = report.market.option, report.config
    if option.years * DAYS / min((config.monitoring_days, *controls.monitoring_days)) > 1500:
        raise ValueError("Convergence permits at most 1,500 monitoring observations")
    local = fit_local_vol(report.market.analysis.surface, option)
    parameters = HestonParameters(**report.diagnostics["heston_parameters"])
    baseline = dict(
        paths=controls.paths,
        steps=controls.steps,
        seed=controls.seed,
        relative_bump=config.spot_bump_fraction,
        monitoring_days=config.monitoring_days,
    )
    trials = [("baseline", "baseline", baseline)]
    for dimension, field, values in (
        ("path count", "paths", controls.path_counts),
        ("integration steps", "steps", controls.step_counts),
        ("random seed", "seed", controls.seeds),
        ("spot bump", "relative_bump", controls.bump_fractions),
        ("monitoring contract", "monitoring_days", controls.monitoring_days),
    ):
        trials.extend((dimension, value, dict(baseline, **{field: value})) for value in values)
    rows = []
    for model in controls.models:
        for dimension, value, inputs in trials:
            estimate = bump_greeks(
                option,
                controls.barrier,
                model=model,
                parameters=parameters,
                local_vol=local,
                kind=config.kind,
                **inputs,
            )
            rows.append(
                dict(
                    model=model,
                    dimension=dimension,
                    input_value=str(value),
                    paths=inputs["paths"],
                    steps=inputs["steps"],
                    seed=inputs["seed"],
                    **{k: v for k, v in estimate.items() if k != "model"},
                    contract="discrete",
                    grid_points=np.nan,
                    changed_contract=dimension == "monitoring contract",
                )
            )
    for points in controls.grid_points:
        barrier = Barrier(config.kind, controls.barrier, config.monitoring_days)
        grid = DiscreteBarrierGrid(option, barrier, [0], grid_points=points)
        h = option.spot * config.spot_bump_fraction
        prices = grid.price(np.array([option.spot - h, option.spot, option.spot + h]), 0)
        rows.append(
            dict(
                model="GBM grid",
                dimension="spatial grid",
                input_value=str(points),
                price=float(prices[1]),
                price_se=np.nan,
                delta=float((prices[2] - prices[0]) / (2 * h)),
                gamma=float((prices[2] - 2 * prices[1] + prices[0]) / h**2),
                gamma_se=np.nan,
                grid_points=points,
                monitoring_days=config.monitoring_days,
                contract="discrete",
                changed_contract=False,
            )
        )
    paths = simulate_gbm(
        option,
        paths=max(controls.path_counts or (controls.paths,)),
        steps=controls.steps,
        seed=controls.seed,
        monitoring_days=config.monitoring_days,
    )
    bridge, touch = gbm_bridge_samples(paths, option, controls.barrier, config.kind)
    discrete = discounted_payoffs(
        paths, option, controls.barrier, config.kind, config.monitoring_days
    )
    value, se = antithetic_estimate(bridge)
    touch_mean, touch_se = antithetic_estimate(touch)
    difference, difference_se = antithetic_estimate(bridge - discrete)
    rows.append(
        dict(
            model="GBM bridge",
            dimension="monitoring contract",
            input_value="continuous",
            contract="continuous",
            changed_contract=True,
            price=value,
            price_se=se,
            touch_probability=touch_mean,
            touch_probability_se=touch_se,
            paired_price_difference_to_discrete=difference,
            paired_difference_se=difference_se,
            paths=paths.spots.shape[1],
            steps=len(paths.times) - 1,
        )
    )
    result = pd.DataFrame(rows)
    result["price_ci_low"] = result.price - 1.96 * result.price_se
    result["price_ci_high"] = result.price + 1.96 * result.price_se
    result["change_from_baseline"] = np.nan
    for model in controls.models:
        baseline_price = result.loc[
            (result.model == model) & (result.dimension == "baseline"), "price"
        ].iloc[0]
        result.loc[result.model == model, "change_from_baseline"] = (
            result.loc[result.model == model, "price"] - baseline_price
        )
    summaries = []
    for (model, dimension), group in result.groupby(["model", "dimension"], sort=False):
        summaries.append(
            dict(
                model=model,
                dimension=dimension,
                trials=len(group),
                price_range=float(group.price.max() - group.price.min()),
                empirical_price_std=float(group.price.std()) if len(group) > 1 else np.nan,
                gamma_range=float(group.gamma.max() - group.gamma.min()),
                maximum_sampling_se=float(group.price_se.max()),
                changed_contract=bool(group.changed_contract.any()),
            )
        )
    return result, pd.DataFrame(summaries)
