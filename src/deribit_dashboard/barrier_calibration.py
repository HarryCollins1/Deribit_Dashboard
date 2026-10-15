"""Smooth Dupire dynamics and Heston calibration for the barrier experiment.

The arbitrage-constrained library surface remains unchanged. A smooth SVI approximation
is needed for Dupire derivatives; approximation, clipping and wing use are audited.
Forward term structures are standardized to the selected option's constant carry.
"""

from dataclasses import dataclass, field, replace
from math import exp, sqrt

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

from deribit_dashboard.pricing import BlackScholes
from deribit_dashboard.surface import normalized_call


def svi(k, parameters):
    a, b, rho, m, width = parameters
    z = np.asarray(k) - m
    root = np.sqrt(z * z + width * width)
    return a + b * (rho * z + root), b * (rho + z / root), b * width**2 / root**3


@dataclass
class DupireLocalVol:
    times: np.ndarray
    parameters: np.ndarray
    spot: float
    carry: float
    k_min: float
    k_max: float
    min_vol: float = 0.01
    max_vol: float = 3.0
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    counters: dict = field(default_factory=lambda: dict(evaluations=0, wing_uses=0, clipped=0))

    def variance(self, time, k):
        t = max(float(time), 1e-9)
        if t <= self.times[0]:
            w, wk, wkk = svi(k, self.parameters[0])
            wt = w / self.times[0]
            scale = t / self.times[0]
            w, wk, wkk = w * scale, wk * scale, wkk * scale
        else:
            upper = min(int(np.searchsorted(self.times, t, side="right")), len(self.times) - 1)
            lower = max(0, upper - 1)
            if lower == upper:
                w, wk, wkk = svi(k, self.parameters[lower])
                wt = w / self.times[lower]
                w, wk, wkk = (v * t / self.times[lower] for v in (w, wk, wkk))
            else:
                dt = self.times[upper] - self.times[lower]
                alpha = (t - self.times[lower]) / dt
                left, right = svi(k, self.parameters[lower]), svi(k, self.parameters[upper])
                w, wk, wkk = (a + alpha * (b - a) for a, b in zip(left, right, strict=True))
                wt = (right[0] - left[0]) / dt
        denominator = (1 - np.asarray(k) * wk / (2 * w)) ** 2
        denominator -= wk**2 * (1 / w + 0.25) / 4
        denominator += wkk / 2
        valid = (wt > 0) & (denominator > 1e-8)
        variance = np.divide(wt, denominator, out=np.full_like(w, np.nan), where=valid)
        return variance, wt, denominator

    def __call__(self, time, spots):
        prices = np.asarray(spots, dtype=float)
        if np.any(prices <= 0) or not np.isfinite(prices).all():
            raise ValueError("Local volatility requires finite positive spot")
        k = np.log(prices / (self.spot * exp(self.carry * float(time))))
        outside = (k < self.k_min) | (k > self.k_max)
        raw, _, _ = self.variance(time, np.clip(k, self.k_min, self.k_max))
        bad = ~np.isfinite(raw) | (raw < self.min_vol**2) | (raw > self.max_vol**2)
        self.counters["evaluations"] += prices.size
        self.counters["wing_uses"] += int(np.count_nonzero(outside))
        self.counters["clipped"] += int(np.count_nonzero(bad))
        # Numerical regularization is disclosed in the report, never hidden.
        raw = np.nan_to_num(raw, nan=self.min_vol**2, posinf=self.max_vol**2)
        return np.sqrt(np.clip(raw, self.min_vol**2, self.max_vol**2))

    def diagnostics(self):
        rows = []
        for t in np.linspace(max(1e-6, self.times[0] / 20), self.times[-1], 40):
            for k, value, wt, denominator in zip(
                np.linspace(self.k_min, self.k_max, 61),
                *self.variance(t, np.linspace(self.k_min, self.k_max, 61)),
                strict=True,
            ):
                rows.append((t, k, value, wt, denominator))
        frame = pd.DataFrame(
            rows, columns=["years", "log_moneyness", "local_variance", "w_time", "density_factor"]
        )
        frame["regularized"] = ~np.isfinite(frame.local_variance) | ~frame.local_variance.between(
            self.min_vol**2, self.max_vol**2
        )
        return frame


def fit_local_vol(surface, option, min_vol=0.01, max_vol=3.0):
    if not 0 < min_vol < max_vol:
        raise ValueError("Local-volatility floor must be positive and below its cap")
    points = surface.points()
    parameters, reports, times = [], [], []
    for time, group in points.groupby("years", sort=True):
        k, target = group.log_moneyness.to_numpy(), group.total_variance.to_numpy()
        initial = [max(target.min() / 2, 1e-7), max(target.mean(), 0.01), -0.3, 0.0, 0.1]
        fit = least_squares(
            lambda p: (svi(k, p)[0] - target) / max(target.mean(), 1e-6),
            initial,
            bounds=([1e-10, 0, -0.99, -2, 0.001], [5, 5, 0.99, 2, 3]),
            max_nfev=800,
        )
        if not fit.success:
            raise ValueError(f"SVI smoothing failed at maturity {time:g}")
        parameters.append(fit.x)
        times.append(time)
        fitted = svi(k, fit.x)[0]
        reports.append(
            dict(
                years=time,
                total_variance_rmse=float(np.sqrt(np.mean((fitted - target) ** 2))),
                **dict(zip(["a", "b", "rho", "m", "width"], fit.x, strict=True)),
            )
        )
    if not times:
        raise ValueError("Local volatility requires fitted smiles")
    return DupireLocalVol(
        np.asarray(times),
        np.asarray(parameters),
        option.spot,
        option.rate - option.dividend,
        float(points.log_moneyness.min()),
        float(points.log_moneyness.max()),
        min_vol,
        max_vol,
        pd.DataFrame(reports),
    )


_NODES, _WEIGHTS = leggauss(160)
_U = (_NODES + 1) * 100
_QUAD = _WEIGHTS * 100


def heston_calls(option, parameters, strikes=None):
    """Heston characteristic-function quadrature; cash calls per underlying unit.

    Finite integration domain [0,200] with 160 Gauss-Legendre points. The MC
    vanilla cross-check separately exposes time-discretization/calibration error.
    """
    strikes = np.atleast_1d(option.strike if strikes is None else strikes).astype(float)
    if np.any(strikes <= 0):
        raise ValueError("Strikes must be positive")
    p, t = parameters, option.years
    if t == 0:
        return np.maximum(option.spot - strikes, 0)
    if p.xi < 1e-7:
        average = (
            p.v0
            if p.kappa == 0
            else p.theta + (p.v0 - p.theta) * (-np.expm1(-p.kappa * t)) / (p.kappa * t)
        )
        return np.array(
            [
                BlackScholes().price(
                    replace(option, strike=k, volatility=sqrt(max(average, 0)), side="call")
                )
                for k in strikes
            ]
        )

    def characteristic(u):
        iu = 1j * u
        beta = p.kappa - p.rho * p.xi * iu
        d = np.sqrt(beta**2 + p.xi**2 * (u * u + iu))
        g = (beta - d) / (beta + d)
        exponential = np.exp(-d * t)
        c = iu * (option.rate - option.dividend) * t + p.kappa * p.theta / p.xi**2 * (
            (beta - d) * t - 2 * np.log((1 - g * exponential) / (1 - g))
        )
        variance = (beta - d) / p.xi**2 * (1 - exponential) / (1 - g * exponential)
        return np.exp(c + variance * p.v0)

    phases = np.exp(-1j * np.log(strikes / option.spot)[:, None] * _U)
    p2 = 0.5 + np.sum(_QUAD * np.real(phases * characteristic(_U) / (1j * _U)), axis=1) / np.pi
    p1 = (
        0.5
        + np.sum(
            _QUAD
            * np.real(
                phases
                * characteristic(_U - 1j)
                / (1j * _U * exp((option.rate - option.dividend) * t))
            ),
            axis=1,
        )
        / np.pi
    )
    return option.spot * exp(-option.dividend * t) * p1 - strikes * exp(-option.rate * t) * p2


def calibrate_heston(surface, option, initial=None):
    from deribit_dashboard.barrier_models import HestonParameters

    initial = initial or HestonParameters(
        option.volatility**2, 2.0, option.volatility**2, 0.6, -0.5
    )
    # Sample all observed expiries and strikes; standardize their forward carry.
    points = surface.points()
    groups = []
    for time, frame in points.groupby("years", sort=True):
        frame = frame.iloc[np.linspace(0, len(frame) - 1, min(11, len(frame))).astype(int)]
        f = option.spot * exp((option.rate - option.dividend) * time)
        strikes = np.exp(frame.log_moneyness.to_numpy()) * f
        target = (
            exp(-option.rate * time)
            * f
            * normalized_call(strikes / f, frame.total_variance.to_numpy())
        )
        groups.append((replace(option, years=time), strikes, target))

    def residual(x):
        p = HestonParameters(*x)
        return np.concatenate(
            [(heston_calls(o, p, strikes) - target) / option.spot for o, strikes, target in groups]
        )

    fit = least_squares(
        residual,
        [initial.v0, initial.kappa, initial.theta, max(initial.xi, 0.02), initial.rho],
        bounds=([0.0001, 0.05, 0.0001, 0.01, -0.98], [4, 20, 4, 3, 0.98]),
        max_nfev=150,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    if not fit.success or not np.isfinite(fit.fun).all():
        raise ValueError(
            "Heston calibration did not converge; review market inputs or use explicit parameters"
        )
    parameters = HestonParameters(*fit.x)
    rows = []
    for o, strikes, targets in groups:
        prices = heston_calls(o, parameters, strikes)
        rows.extend(
            dict(years=o.years, strike=k, target=target, heston=price, residual=price - target)
            for k, target, price in zip(strikes, targets, prices, strict=True)
        )
    diagnostics = dict(
        method="Heston characteristic-function least squares to fitted total-variance surface targets",
        evaluations=fit.nfev,
        price_rmse=float(np.sqrt(np.mean(fit.fun**2)) * option.spot),
        feller_satisfied=bool(2 * parameters.kappa * parameters.theta >= parameters.xi**2),
        near_parameter_bound=bool(np.any(fit.active_mask)),
        carry_convention="constant carry implied by the selected expiry; other expiry forwards standardized",
    )
    return parameters, pd.DataFrame(rows), diagnostics
