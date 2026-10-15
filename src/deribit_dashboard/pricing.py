"""Pluggable cash-numeraire models. Volatility is decimal/year (normal: cash/sqrt(year))."""

from dataclasses import dataclass, replace
from math import exp, log, sqrt
from typing import Protocol

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

DAYS = 365.25
YEAR_SECONDS = DAYS * 86400


@dataclass(frozen=True)
class Option:
    spot: float
    strike: float
    years: float
    volatility: float
    rate: float = 0.0
    dividend: float = 0.0  # continuously compounded dividend yield
    side: str = "call"
    american: bool = False

    def __post_init__(self):
        if not all(
            np.isfinite(v)
            for v in (self.spot, self.strike, self.years, self.volatility, self.rate, self.dividend)
        ):
            raise ValueError("Non-finite model input")
        if min(self.spot, self.strike) <= 0 or min(self.years, self.volatility) < 0:
            raise ValueError("Positive spot/strike and nonnegative time/volatility required")
        if self.side not in ("call", "put"):
            raise ValueError("Unknown option side")

    @property
    def sign(self):
        return 1 if self.side == "call" else -1

    @property
    def forward(self):
        return self.spot * exp((self.rate - self.dividend) * self.years)


class Model(Protocol):
    def price(self, option: Option) -> float: ...


def intrinsic(o):
    return max(o.sign * (o.spot - o.strike), 0.0)


class BlackScholes:
    def price(self, o):
        if o.american:
            raise ValueError("Use Binomial for American exercise")
        if o.years == 0:
            return intrinsic(o)
        f, d = o.forward, exp(-o.rate * o.years)
        v = o.volatility * sqrt(o.years)
        if v == 0:
            return d * max(o.sign * (f - o.strike), 0.0)
        d1 = log(f / o.strike) / v + v / 2
        return d * o.sign * (f * ndtr(o.sign * d1) - o.strike * ndtr(o.sign * (d1 - v)))

    def greeks(self, o):
        if o.years <= 0 or o.volatility <= 0 or o.american:
            return finite_greeks(self, o)
        v = o.volatility * sqrt(o.years)
        d1 = log(o.forward / o.strike) / v + v / 2
        d2 = d1 - v
        density = exp(-d1 * d1 / 2) / sqrt(2 * np.pi)
        dq, dr = exp(-o.dividend * o.years), exp(-o.rate * o.years)
        return dict(
            delta=dq * o.sign * ndtr(o.sign * d1),
            gamma=dq * density / (o.spot * v),
            vega=o.spot * dq * density * sqrt(o.years),
            theta=(
                -o.spot * dq * density * o.volatility / (2 * sqrt(o.years))
                - o.sign * o.rate * o.strike * dr * ndtr(o.sign * d2)
                + o.sign * o.dividend * o.spot * dq * ndtr(o.sign * d1)
            )
            / DAYS,
        )


class Bachelier:
    def price(self, o):
        if o.american:
            raise ValueError("Use Binomial for American exercise")
        if o.years == 0:
            return intrinsic(o)
        v = o.volatility * sqrt(o.years)
        x = o.sign * (o.forward - o.strike)
        if v == 0:
            return exp(-o.rate * o.years) * max(x, 0)
        z = x / v
        return exp(-o.rate * o.years) * (x * ndtr(z) + v * exp(-z * z / 2) / sqrt(2 * np.pi))

    def greeks(self, o):
        if o.years <= 0 or o.volatility <= 0:
            return finite_greeks(self, o)
        v = o.volatility * sqrt(o.years)
        z = (o.forward - o.strike) / v
        density = exp(-z * z / 2) / sqrt(2 * np.pi)
        d = exp(-o.rate * o.years)
        carry = exp((o.rate - o.dividend) * o.years)
        delta_f = o.sign * ndtr(o.sign * z)
        return dict(
            delta=d * carry * delta_f,
            gamma=d * carry * carry * density / v,
            vega=d * sqrt(o.years) * density,
            theta=(
                o.rate * self.price(o)
                - d
                * (
                    delta_f * o.forward * (o.rate - o.dividend)
                    + density * o.volatility / (2 * sqrt(o.years))
                )
            )
            / DAYS,
        )


class Binomial:
    """CRR lattice, continuous dividend yield, optional early exercise."""

    def __init__(self, steps=300):
        if steps < 2:
            raise ValueError("At least two tree steps required")
        self.steps = steps

    def price(self, o):
        if o.years == 0:
            return intrinsic(o)
        if o.volatility == 0:
            times = np.linspace(0, o.years, self.steps + 1) if o.american else np.array([o.years])
            return float(
                np.max(
                    np.exp(-o.rate * times)
                    * np.maximum(
                        o.sign * (o.spot * np.exp((o.rate - o.dividend) * times) - o.strike), 0
                    )
                )
            )
        dt = o.years / self.steps
        u = exp(o.volatility * sqrt(dt))
        down = 1 / u
        p = (exp((o.rate - o.dividend) * dt) - down) / (u - down)
        if not 0 <= p <= 1:
            raise ValueError("Invalid tree probability; increase steps or volatility")
        values = np.maximum(
            o.sign * (o.spot * u ** np.arange(-self.steps, self.steps + 1, 2) - o.strike), 0
        )
        for n in range(self.steps - 1, -1, -1):
            values = exp(-o.rate * dt) * ((1 - p) * values[:-1] + p * values[1:])
            if o.american:
                values = np.maximum(
                    values,
                    np.maximum(o.sign * (o.spot * u ** np.arange(-n, n + 1, 2) - o.strike), 0),
                )
        return float(values[0])


MODELS: dict[str, Model] = {
    "Black-Scholes": BlackScholes(),
    "Bachelier": Bachelier(),
    "Binomial": Binomial(),
}


def register_model(name: str, model: Model):
    if not name or not callable(getattr(model, "price", None)):
        raise ValueError("A model name and price(Option) are required")
    MODELS[name] = model


def implied_vol(price, option, model=None):
    model = model or MODELS["Black-Scholes"]
    if not np.isfinite(price) or price < 0 or option.years <= 0:
        raise ValueError("IV requires finite nonnegative price and positive maturity")
    # Trees can reject very small volatilities when carry dominates a lattice step.
    low = (
        0.0
        if not isinstance(model, Binomial)
        else max(
            1e-8, abs(option.rate - option.dividend) * sqrt(option.years / model.steps) * 1.001
        )
    )
    floor = model.price(replace(option, volatility=low))
    if price < floor - 1e-9:
        raise ValueError("Price below model lower bound")
    if abs(price - floor) <= 1e-10:
        return low
    high = max(1.0, option.spot if isinstance(model, Bachelier) else 1.0)
    for _ in range(20):
        if model.price(replace(option, volatility=high)) >= price:
            return brentq(
                lambda v: model.price(replace(option, volatility=v)) - price, low, high, xtol=1e-11
            )
        high *= 2
    raise ValueError("Price outside model IV bounds")


def finite_greeks(model, o, coin=False):
    if o.years <= 0:
        raise ValueError("Greeks undefined at expiry")
    # A larger spot bump regularizes the piecewise-linear tree delta/gamma.
    h = o.spot * (0.01 if isinstance(model, Binomial) else 1e-4)
    hv = max(o.volatility * 1e-4, 1e-6)
    ht = min(1e-5, o.years / 4)

    def evaluate(option):
        return model.price(option) / option.spot if coin else model.price(option)

    p = evaluate(o)
    up, down = evaluate(replace(o, spot=o.spot + h)), evaluate(replace(o, spot=o.spot - h))
    lo = max(0, o.volatility - hv)
    return dict(
        delta=(up - down) / (2 * h),
        gamma=(up - 2 * p + down) / h**2,
        vega=(
            evaluate(replace(o, volatility=o.volatility + hv)) - evaluate(replace(o, volatility=lo))
        )
        / (o.volatility + hv - lo),
        theta=(evaluate(replace(o, years=o.years - ht)) - evaluate(replace(o, years=o.years + ht)))
        / (2 * ht * DAYS),
    )


def risk(model, option, multiplier=1.0):
    """Cash and coin price derivatives, holding carry and model volatility fixed."""
    p = model.price(option)
    fd = finite_greeks(model, option)
    g = model.greeks(option) if hasattr(model, "greeks") else fd
    s = option.spot
    coin = dict(
        delta=g["delta"] / s - p / s**2,
        gamma=g["gamma"] / s - 2 * g["delta"] / s**2 + 2 * p / s**3,
        vega=g["vega"] / s,
        theta=g["theta"] / s,
    )
    result = {"cash_value": multiplier * p, "coin_value": multiplier * p / s}
    for unit, values in (("cash", g), ("coin", coin)):
        result.update({f"{unit}_{k}": multiplier * v for k, v in values.items()})
        result[f"{unit}_theta_per_gamma"] = (
            values["theta"] / values["gamma"] if abs(values["gamma"]) > 1e-14 else np.nan
        )
    coin_fd = finite_greeks(model, option, coin=True)
    result.update({f"fd_coin_{k}": multiplier * v for k, v in coin_fd.items()})
    result.update({f"fd_coin_error_{k}": multiplier * (coin[k] - coin_fd[k]) for k in coin})
    result.update({f"fd_cash_{k}": multiplier * v for k, v in fd.items()})
    result.update({f"fd_error_{k}": multiplier * (g[k] - fd[k]) for k in g})
    return result


def aggregate_portfolio(rows, quantities):
    """Rows must share base/cash currency; quantities are signed metadata contract counts."""
    if len(rows) != len(quantities) or not np.isfinite(quantities).all():
        raise ValueError("One finite signed quantity per risk row required")
    result = {}
    for unit in ("cash", "coin"):
        for key in ("value", "delta", "gamma", "vega", "theta"):
            field = f"{unit}_{key}"
            result[field] = sum(q * r[field] for q, r in zip(quantities, rows, strict=True))
        gamma = result[f"{unit}_gamma"]
        result[f"{unit}_theta_per_gamma"] = (
            result[f"{unit}_theta"] / gamma if abs(gamma) > 1e-14 else np.nan
        )
    return result
