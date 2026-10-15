"""Risk-neutral Monte Carlo for hypothetical, discretely monitored barriers.

The same simulated paths evaluate every barrier level. Antithetic pairs are the
independent sampling units for standard errors. Sampling uncertainty does not
include Heston/local-volatility time-discretization or calibration/model error.

Heston uses log-spot Euler and the full-truncation variance scheme of Lord,
Koekkoek and van Dijk: https://repub.eur.nl/pub/7738 . Parameters are explicit
inputs; this module does not claim to calibrate them to an option surface.
"""

from dataclasses import dataclass, replace
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr

from deribit_dashboard.pricing import DAYS, Option

BARRIER_KINDS = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")


@dataclass(frozen=True)
class HestonParameters:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self):
        if not np.isfinite([self.v0, self.kappa, self.theta, self.xi, self.rho]).all():
            raise ValueError("Heston parameters must be finite")
        if min(self.v0, self.kappa, self.theta, self.xi) < 0 or abs(self.rho) > 1:
            raise ValueError(
                "Heston variance, mean reversion and vol-of-vol must be nonnegative; |rho| <= 1"
            )


@dataclass
class SimulationPaths:
    times: np.ndarray
    spots: np.ndarray
    model: str
    diagnostics: dict
    variances: np.ndarray | None = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.spots = np.asarray(self.spots, dtype=float)
        if self.times.ndim != 1 or not len(self.times) or self.spots.ndim != 2:
            raise ValueError("Times must be one-dimensional; spots must be time × path")
        if self.spots.shape[0] != len(self.times) or self.spots.shape[1] < 2:
            raise ValueError("Spot observations must match times and contain at least two paths")
        if not np.isfinite(self.times).all() or not np.isfinite(self.spots).all():
            raise ValueError("Path values must be finite")
        if self.times[0] != 0 or np.any(np.diff(self.times) <= 0) or np.any(self.spots <= 0):
            raise ValueError("Times must start at zero and increase; spots must be positive")
        if self.diagnostics.get("antithetic") and (
            self.spots.shape[1] < 4 or self.spots.shape[1] % 2
        ):
            raise ValueError("Antithetic paths require an even count of at least four")
        if self.variances is not None:
            self.variances = np.asarray(self.variances, float)
            if self.variances.shape != self.spots.shape or not np.isfinite(self.variances).all():
                raise ValueError(
                    "Conditional variance states must match time×path spots and be finite"
                )


def _monitor_times(years, monitoring_days):
    if not np.isfinite(monitoring_days) or monitoring_days <= 0:
        raise ValueError("Monitoring interval must be finite and positive")
    interval = monitoring_days / DAYS
    if years / interval > 10000:
        raise ValueError("At most 10,000 barrier observations are supported")
    return np.unique(np.r_[0.0, np.arange(interval, years, interval), years])


def _setup(option, paths, steps, seed, monitoring_days):
    if option.american:
        raise ValueError("Path pricing supports European exercise only")
    if not isinstance(paths, int) or paths < 4 or paths % 2:
        raise ValueError("Supply an even path count of at least four")
    if not isinstance(steps, int) or not 1 <= steps <= 10000:
        raise ValueError("Integration steps must be an integer between 1 and 10,000")
    observations = _monitor_times(option.years, monitoring_days)
    times = np.unique(np.r_[np.linspace(0, option.years, steps + 1), observations])
    # Floating-point duplicates from uniform and calendar grids add no economic time.
    keep = np.r_[True, np.diff(times) > 1e-14]
    times = times[keep]
    if len(times) * paths > 30_000_000:
        raise ValueError("Reduce paths or steps: at most 30 million time-path cells are supported")
    rng = np.random.default_rng(seed)
    half = paths // 2
    independent = rng.standard_normal((len(times) - 1, half))
    shocks = np.concatenate((independent, -independent), axis=1)
    diagnostics = dict(
        antithetic=True,
        independent_pairs=half,
        paths=paths,
        seed=seed,
        requested_steps=steps,
        integration_steps=len(times) - 1,
        monitoring_days=monitoring_days,
        initial_spot=option.spot,
        rate=option.rate,
        dividend=option.dividend,
        risk_neutral=True,
        monitoring="Discrete inception, scheduled calendar intervals and expiry; no between-observation touches",
        error_note="Standard errors measure sampling error only; discretization, calibration and model error are excluded",
    )
    return times, shocks, rng, diagnostics


def simulate_gbm(option: Option, paths=4096, steps=120, seed=42, monitoring_days=1.0):
    """Exact lognormal transitions on a grid containing all monitoring dates."""
    times, shocks, _, diagnostics = _setup(option, paths, steps, seed, monitoring_days)
    dt = np.diff(times)[:, None]
    increments = (option.rate - option.dividend - 0.5 * option.volatility**2) * dt
    increments = increments + option.volatility * np.sqrt(dt) * shocks
    spots = option.spot * np.exp(np.vstack((np.zeros(paths), np.cumsum(increments, axis=0))))
    diagnostics.update(volatility=option.volatility, scheme="Exact GBM lognormal transitions")
    return SimulationPaths(times, spots, "GBM", diagnostics)


def simulate_heston(
    option: Option,
    parameters: HestonParameters,
    paths=4096,
    steps=120,
    seed=42,
    monitoring_days=1.0,
    retain_variance=False,
):
    """Heston full-truncation Euler variance with positive log-spot updates.

    dS/S=(r-q)dt+sqrt(v)dW_S; dv=kappa(theta-v)dt+xi sqrt(v)dW_v.
    The stored raw Euler variance can be negative; v+ enters both drift and
    diffusion on the following step. rho is the spot/variance shock correlation.
    """
    if not isinstance(parameters, HestonParameters):
        raise ValueError("Supply explicit HestonParameters")
    times, z_spot, rng, diagnostics = _setup(option, paths, steps, seed, monitoring_days)
    half = paths // 2
    z_other = rng.standard_normal((len(times) - 1, half))
    z_other = np.concatenate((z_other, -z_other), axis=1)
    z_var = parameters.rho * z_spot + sqrt(max(0.0, 1 - parameters.rho**2)) * z_other
    spots = np.full((len(times), paths), option.spot, dtype=float)
    variance = np.full(paths, parameters.v0, dtype=float)
    variance_paths = np.empty_like(spots) if retain_variance else None
    if variance_paths is not None:
        variance_paths[0] = variance
    negatives = 0
    for i, dt in enumerate(np.diff(times)):
        positive = np.maximum(variance, 0)
        spots[i + 1] = spots[i] * np.exp(
            (option.rate - option.dividend - 0.5 * positive) * dt
            + np.sqrt(positive * dt) * z_spot[i]
        )
        variance = variance + parameters.kappa * (parameters.theta - positive) * dt
        variance = variance + parameters.xi * np.sqrt(positive * dt) * z_var[i]
        negatives += int(np.count_nonzero(variance < 0))
        if variance_paths is not None:
            variance_paths[i + 1] = variance
    diagnostics.update(
        scheme="Heston full-truncation Euler variance and log-spot Euler",
        parameters=parameters.__dict__.copy(),
        feller_condition=bool(2 * parameters.kappa * parameters.theta >= parameters.xi**2),
        negative_raw_variance_fraction=negatives / max(1, (len(times) - 1) * paths),
        mean_terminal_variance=float(np.maximum(variance, 0).mean()),
        parameter_status="User supplied parameters; no Heston calibration performed",
    )
    return SimulationPaths(times, spots, "Heston", diagnostics, variance_paths)


def simulate_local_vol(
    option: Option,
    local_vol,
    paths=4096,
    steps=120,
    seed=42,
    monitoring_days=1.0,
):
    """Log-Euler under sigma_local(t_years, spot_array); callable owns calibration.

    The callable must return finite, nonnegative decimal annualized volatility.
    No extrapolation or clipping is supplied here; invalid surface queries fail.
    """
    if not callable(local_vol):
        raise ValueError("Supply a local_vol(time_years, spot_array) callable")
    times, shocks, _, diagnostics = _setup(option, paths, steps, seed, monitoring_days)
    spots = np.full((len(times), paths), option.spot, dtype=float)
    min_vol, max_vol = np.inf, -np.inf
    for i, dt in enumerate(np.diff(times)):
        volatility = np.asarray(local_vol(float(times[i]), spots[i]), dtype=float)
        if (
            volatility.shape not in ((), (paths,))
            or not np.isfinite(volatility).all()
            or np.any(volatility < 0)
        ):
            raise ValueError(
                "Local volatility must return finite nonnegative scalar or one value per path"
            )
        min_vol, max_vol = (
            min(min_vol, float(volatility.min())),
            max(max_vol, float(volatility.max())),
        )
        spots[i + 1] = spots[i] * np.exp(
            (option.rate - option.dividend - 0.5 * volatility**2) * dt
            + volatility * sqrt(dt) * shocks[i]
        )
    diagnostics.update(
        scheme="Local-volatility log-Euler",
        minimum_sampled_local_volatility=min_vol if len(times) > 1 else np.nan,
        maximum_sampled_local_volatility=max_vol if len(times) > 1 else np.nan,
        parameter_status="Caller supplied local-volatility function; inspect its calibration and domain separately",
    )
    return SimulationPaths(times, spots, "Local volatility", diagnostics)


def _monitor_indices(paths, monitoring_days):
    requested = _monitor_times(float(paths.times[-1]), monitoring_days)
    indices = np.searchsorted(paths.times, requested)
    indices = np.clip(indices, 0, len(paths.times) - 1)
    previous = np.maximum(0, indices - 1)
    indices = np.where(
        abs(paths.times[previous] - requested) < abs(paths.times[indices] - requested),
        previous,
        indices,
    )
    if not np.allclose(paths.times[indices], requested, rtol=0, atol=1e-12):
        raise ValueError(
            "Paths do not contain the requested monitoring dates; simulate with matching monitoring_days"
        )
    return indices


def antithetic_estimate(samples, paired=True):
    """Mean and sampling SE; first-half paths pair with the second half.

    Pair averages, rather than correlated individual observations, provide the
    independent samples. ``paired=False`` accepts ordinary independent samples.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) < 2:
        raise ValueError("Supply a finite one-dimensional sample with at least two observations")
    if paired:
        if len(samples) < 4 or len(samples) % 2:
            raise ValueError("Antithetic samples require an even count of at least four")
        half = len(samples) // 2
        independent = (samples[:half] + samples[half:]) / 2
    else:
        independent = samples
    return float(np.mean(independent)), float(np.std(independent, ddof=1) / sqrt(len(independent)))


def _estimate(samples, paths):
    return antithetic_estimate(samples, paired=bool(paths.diagnostics.get("antithetic")))


def _events(paths, threshold, monitoring_days, direction):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be finite and positive")
    if direction not in {"up", "down"}:
        raise ValueError("Direction must be 'up' or 'down'")
    observations = paths.spots[_monitor_indices(paths, monitoring_days)]
    if direction == "up":
        return paths.spots[-1] >= threshold, np.any(observations >= threshold, axis=0)
    return paths.spots[-1] <= threshold, np.any(observations <= threshold, axis=0)


def probabilities(paths: SimulationPaths, threshold, monitoring_days=1.0, direction="up"):
    """Risk-neutral terminal-beyond and discrete-touch estimates at a fixed level.

    Normal 95% confidence intervals are approximate and unreliable for rare or
    unobserved events. In particular, zero observed hits do not prove zero risk.
    """
    finish, touch = _events(paths, threshold, monitoring_days, direction)
    finish_mean, finish_se = _estimate(finish.astype(float), paths)
    touch_mean, touch_se = _estimate(touch.astype(float), paths)
    return dict(
        threshold=float(threshold),
        direction=direction,
        finish_probability=finish_mean,
        finish_probability_se=finish_se,
        finish_probability_ci_low=max(0.0, finish_mean - 1.96 * finish_se),
        finish_probability_ci_high=min(1.0, finish_mean + 1.96 * finish_se),
        touch_probability=touch_mean,
        touch_probability_se=touch_se,
        probability_se=touch_se,
        touch_probability_ci_low=max(0.0, touch_mean - 1.96 * touch_se),
        touch_probability_ci_high=min(1.0, touch_mean + 1.96 * touch_se),
        finish_paths=int(finish.sum()),
        touch_paths=int(touch.sum()),
        total_paths=len(finish),
        independent_sampling_units=len(finish) // 2
        if paths.diagnostics.get("antithetic")
        else len(finish),
        confidence_note="Approximate normal 95% interval; unreliable for rare/unobserved events; sampling error only",
        probability_measure="Risk-neutral; not a forecast of real-world frequency",
    )


def _check_option(paths, option):
    if option.american:
        raise ValueError("Barrier path pricing supports European exercise only")
    if not np.isclose(paths.times[-1], option.years, rtol=0, atol=1e-12):
        raise ValueError("Path maturity does not match option expiry")
    if not np.allclose(paths.spots[0], option.spot, rtol=1e-12, atol=0):
        raise ValueError("Path starting spot does not match option")
    for name in ("rate", "dividend"):
        if name in paths.diagnostics and not np.isclose(
            paths.diagnostics[name], getattr(option, name), rtol=0, atol=1e-12
        ):
            raise ValueError(f"Path {name} does not match option")
    if paths.model == "GBM" and "volatility" in paths.diagnostics:
        if not np.isclose(paths.diagnostics["volatility"], option.volatility, rtol=0, atol=1e-12):
            raise ValueError("GBM path volatility does not match option")


def _payoffs(paths, option, barrier, kind, monitoring_days, already_hit=False):
    if kind not in BARRIER_KINDS:
        raise ValueError("Unknown barrier type")
    _, hit = _events(paths, barrier, monitoring_days, "up" if kind.startswith("up") else "down")
    hit = hit | bool(already_hit)
    active = hit if kind.endswith("-in") else ~hit
    return (
        exp(-option.rate * option.years)
        * np.maximum(option.sign * (paths.spots[-1] - option.strike), 0)
        * active
    )


def discounted_payoffs(
    paths, option, barrier, kind="up-and-out", monitoring_days=1.0, already_hit=False
):
    """Per-path discounted zero-rebate payoff for common-random-number comparisons."""
    _check_option(paths, option)
    return _payoffs(paths, option, barrier, kind, monitoring_days, already_hit)


def evaluate_paths(
    paths: SimulationPaths,
    option: Option,
    barriers: list[float],
    kind="up-and-out",
    monitoring_days=1.0,
    already_hit=False,
):
    """Cash premium per underlying unit for zero-rebate barriers on COMMON paths.

    Option strike, side and barrier level may vary across calls while the model,
    expiry, carry and initial spot must match the simulated path distribution.
    """
    _check_option(paths, option)
    rows = []
    vanilla = exp(-option.rate * option.years) * np.maximum(
        option.sign * (paths.spots[-1] - option.strike), 0
    )
    vanilla_price, vanilla_se = _estimate(vanilla, paths)
    for level in barriers:
        samples = _payoffs(paths, option, level, kind, monitoring_days, already_hit)
        price, se = _estimate(samples, paths)
        row = probabilities(
            paths, level, monitoring_days, "up" if kind.startswith("up") else "down"
        )
        row.update(
            barrier=float(level),
            barrier_type=kind,
            model=paths.model,
            price=price,
            price_se=se,
            price_ci_low=max(0.0, price - 1.96 * se),
            price_ci_high=price + 1.96 * se,
            vanilla_price=vanilla_price,
            vanilla_price_se=vanilla_se,
            monitoring_days=monitoring_days,
            already_hit=bool(already_hit),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def black_scholes_finish_probability(option, threshold, direction="up"):
    """Analytic risk-neutral probability of ending beyond the threshold under GBM."""
    if not np.isfinite(threshold) or threshold <= 0 or direction not in {"up", "down"}:
        raise ValueError("Positive threshold and up/down direction required")
    if option.years == 0 or option.volatility == 0:
        terminal = option.spot * exp((option.rate - option.dividend) * option.years)
        return float(terminal >= threshold if direction == "up" else terminal <= threshold)
    z = (
        log(option.spot / threshold)
        + (option.rate - option.dividend - option.volatility**2 / 2) * option.years
    ) / (option.volatility * sqrt(option.years))
    return float(ndtr(z if direction == "up" else -z))


def black_scholes_touch_probability(option, threshold, direction="up"):
    """Analytic CONTINUOUS GBM touch probability, an upper bound on discrete touch.

    This reflection-principle result is a benchmark only; discrete simulations
    intentionally do not apply a continuous-monitoring bridge correction.
    """
    if not np.isfinite(threshold) or threshold <= 0 or direction not in {"up", "down"}:
        raise ValueError("Positive threshold and up/down direction required")
    sign = 1 if direction == "up" else -1
    distance = sign * log(threshold / option.spot)
    if distance <= 0:
        return 1.0
    if option.years == 0:
        return 0.0
    if option.volatility == 0:
        return black_scholes_finish_probability(option, threshold, direction)
    drift = sign * (option.rate - option.dividend - option.volatility**2 / 2)
    scale = option.volatility * sqrt(option.years)
    first = ndtr((drift * option.years - distance) / scale)
    log_second = 2 * drift * distance / option.volatility**2 + log_ndtr(
        (-drift * option.years - distance) / scale
    )
    return float(np.clip(first + exp(min(0.0, float(log_second))), 0, 1))


def bump_greeks(
    option,
    barrier,
    *,
    model="GBM",
    parameters=None,
    local_vol=None,
    paths=4096,
    steps=120,
    seed=42,
    relative_bump=0.01,
    kind="up-and-out",
    monitoring_days=1.0,
    already_hit=False,
):
    """Spot delta/gamma using common-random-number central differences.

    GBM/Heston paths scale with initial spot; local-volatility paths must instead
    be regenerated under the same shocks because sigma_local depends on spot.
    Sampling SEs use each paired path's complete bumped difference. Barrier
    discontinuities can make gamma extremely noisy; inspect its SE and bump size.
    """
    if not np.isfinite(relative_bump) or not 0 < relative_bump < 0.5:
        raise ValueError("Relative spot bump must be between zero and 0.5")
    controls = dict(paths=paths, steps=steps, seed=seed, monitoring_days=monitoring_days)
    if model == "GBM":
        original = simulate_gbm(option, **controls)
    elif model == "Heston":
        original = simulate_heston(option, parameters, **controls)
    elif model in {"Local volatility", "Local vol"}:
        original = simulate_local_vol(option, local_vol, **controls)
    else:
        raise ValueError("Model must be GBM, Heston or Local volatility")
    samples = []
    for multiplier in (1 - relative_bump, 1.0, 1 + relative_bump):
        bumped = replace(option, spot=option.spot * multiplier)
        if multiplier == 1:
            simulated = original
        elif original.model == "Local volatility":
            simulated = simulate_local_vol(bumped, local_vol, **controls)
        else:
            diagnostics = dict(original.diagnostics, initial_spot=bumped.spot)
            simulated = SimulationPaths(
                original.times, original.spots * multiplier, original.model, diagnostics
            )
        samples.append(_payoffs(simulated, bumped, barrier, kind, monitoring_days, already_hit))
    h = option.spot * relative_bump
    delta, delta_se = _estimate((samples[2] - samples[0]) / (2 * h), original)
    gamma, gamma_se = _estimate((samples[2] - 2 * samples[1] + samples[0]) / h**2, original)
    price, price_se = _estimate(samples[1], original)
    return dict(
        price=price,
        price_se=price_se,
        delta=delta,
        delta_se=delta_se,
        gamma=gamma,
        gamma_se=gamma_se,
        gamma_ci_low=gamma - 1.96 * gamma_se,
        gamma_ci_high=gamma + 1.96 * gamma_se,
        spot_bump=h,
        relative_bump=relative_bump,
        model=original.model,
        monitoring_days=monitoring_days,
        greek_note="Finite-difference model Greeks at fixed parameters; MC sampling and finite-bump errors remain",
    )
