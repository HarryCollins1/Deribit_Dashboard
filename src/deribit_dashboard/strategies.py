"""Metadata-resolved paper strategies, discrete barriers and time-series risk metrics.

No execution takes place. Positions are self-financed in an explicitly selected cash
numeraire. Non-cash balances at settlement are immediately converted at the modeled
spot; fixed external FX assumptions must be supplied by the caller.
"""

from dataclasses import dataclass, field, replace
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr

from deribit_dashboard.execution_accounting import funding_amount, perpetual_gain
from deribit_dashboard.models import milliseconds
from deribit_dashboard.pricing import DAYS, MODELS, YEAR_SECONDS, Option
from deribit_dashboard.strategy_research import (
    MarginScenario,
    PathRegime,
    SurfaceEvolution,
    collateral_ledger,
    scenario_factors,
    surface_volatility,
)

BARRIER_TYPES = ("none", "up-and-out", "up-and-in", "down-and-out", "down-and-in")
SUPPORTED_ASSETS = frozenset({"BTC", "ETH"})


@dataclass(frozen=True)
class Barrier:
    kind: str
    level: float
    monitoring_days: float = 1.0
    already_hit: bool = False

    def __post_init__(self):
        if self.kind not in BARRIER_TYPES[1:]:
            raise ValueError("Unknown barrier type")
        if (
            not np.isfinite([self.level, self.monitoring_days]).all()
            or min(self.level, self.monitoring_days) <= 0
        ):
            raise ValueError("Barrier level and monitoring interval must be positive")

    def breached(self, spot):
        return (
            np.asarray(spot) >= self.level
            if self.kind.startswith("up")
            else np.asarray(spot) <= self.level
        )

    @property
    def knock_in(self):
        return self.kind.endswith("-in")


@dataclass(frozen=True)
class StrategyLeg:
    instrument_name: str
    asset: str
    kind: str
    quantity: float
    multiplier: float
    spot: float
    entry_price: float
    cash_currency: str
    native_cash_currency: str
    conversion_rate: float
    years: float | None = None
    inverse: bool = False
    option: Option | None = None
    model_name: str = "Black-Scholes"
    barrier: Barrier | None = None
    quote_source: str = "midpoint"
    depth_contracts: float | None = None
    quantity_step: float | None = None
    surface_reference: tuple = ()
    minimum_quantity: float | None = None


@dataclass(frozen=True)
class SimulationConfig:
    initial_equity: float = 100000.0
    horizon_days: float = 30.0
    steps: int = 30
    paths: int = 128
    seed: int = 42
    annual_drift: dict = field(default_factory=lambda: {"BTC": 0.0, "ETH": 0.0})
    annual_volatility: dict = field(default_factory=lambda: {"BTC": 0.6, "ETH": 0.7})
    correlation: float = 0.5
    risk_free_rate: float = 0.0
    minimum_acceptable_return: float = 0.0
    entry_cost_bps: float = 0.0
    annual_funding_rates: dict = field(default_factory=dict)
    surface_evolution: SurfaceEvolution = field(default_factory=SurfaceEvolution)
    margin: MarginScenario = field(default_factory=MarginScenario)
    regimes: tuple[PathRegime, ...] = ()


@dataclass
class StrategySimulation:
    equity: pd.DataFrame
    path_metrics: pd.DataFrame
    summary: dict
    leg_details: pd.DataFrame
    barrier_hits: pd.DataFrame
    assumptions: list[str]
    margin_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict)


def _positive(value, name):
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _displayed_depth(ticker, multiplier):
    amounts = (ticker.get("best_bid_amount"), ticker.get("best_ask_amount"))
    if any(value is None or not np.isfinite(value) or value < 0 for value in amounts):
        return None
    return float(min(amounts)) / multiplier


def _cash_unit(metadata, instruments):
    base = metadata["base_currency"]
    for key in ("counter_currency", "quote_currency"):
        if metadata.get(key) and metadata[key] != base:
            return metadata[key]
    units = {
        m["quote_currency"]
        for m in instruments.values()
        if m.get("base_currency") == base
        and m.get("kind") == "future"
        and m.get("quote_currency") not in (None, base)
    }
    if len(units) != 1:
        raise ValueError("Cannot determine cash denomination from captured instrument metadata")
    return units.pop()


def required_cash_currencies(snapshot, rows):
    """Cash denominations that require explicit reporting conversion assumptions."""
    return sorted(
        {
            _cash_unit(snapshot.instruments[row["instrument_name"]], snapshot.instruments)
            for row in rows
        }
    )


def build_strategy_legs(
    snapshot,
    rows,
    calibrated_options=None,
    *,
    cash_currency="USD",
    conversion_rates=None,
    max_quote_age_seconds=60.0,
):
    """Resolve selected, discovered instruments; errors are explicit, never dropped.

    Spot quantity is base coin. Derivative quantity is contract count; contract_size
    comes only from metadata. FX values are reporting-cash units per native-cash unit.
    Options require a surface-calibrated price_selection row supplied by the caller.
    """
    calibrated_options = calibrated_options or {}
    conversions = dict(conversion_rates or {})
    conversions[cash_currency] = 1.0
    legs = []
    if not np.isfinite(max_quote_age_seconds) or max_quote_age_seconds < 0:
        raise ValueError("Invalid maximum quote age")
    for row in rows:
        name = row["instrument_name"]
        m = snapshot.instruments.get(name)
        if m is None:
            raise ValueError(f"{name}: instrument is absent from discovered metadata")
        base, kind = m["base_currency"], m["kind"]
        if base not in SUPPORTED_ASSETS or kind not in {"spot", "future", "option"}:
            raise ValueError(
                f"{name}: only BTC/ETH spot, futures, perpetuals and options supported"
            )
        quantity = float(row["quantity"])
        if not np.isfinite(quantity):
            raise ValueError(f"{name}: quantity must be finite")
        ticker = snapshot.tickers.get(name, {})
        age = (milliseconds(snapshot.captured_at) - ticker.get("timestamp", 0)) / 1000
        if not -2 <= age <= max_quote_age_seconds or ticker.get("state", "open") != "open":
            raise ValueError(f"{name}: missing, stale or closed quote")
        bid, ask = ticker.get("best_bid_price"), ticker.get("best_ask_price")
        if bid is None or ask is None or not np.isfinite([bid, ask]).all() or not 0 < bid <= ask:
            raise ValueError(f"{name}: a positive uncrossed two-sided quote is required")
        midpoint = (bid + ask) / 2
        spot = _positive(midpoint if kind == "spot" else ticker.get("index_price"), "index price")
        native = _cash_unit(m, snapshot.instruments)
        if native in SUPPORTED_ASSETS:
            raise ValueError(
                f"{name}: coin-quoted crosses require dynamic FX and are not supported"
            )
        if native not in conversions:
            raise ValueError(f"{name}: supply explicit {native} to {cash_currency} conversion")
        fx = _positive(conversions[native], "cash conversion rate")
        multiplier = (
            1.0 if kind == "spot" else _positive(m.get("contract_size"), "metadata contract size")
        )
        inverse, years, option = False, None, None
        model_name, source = "Black-Scholes", "captured midpoint"
        if kind == "option":
            calibrated = calibrated_options.get(name)
            if calibrated is None:
                raise ValueError(f"{name}: no usable surface-calibrated option")
            option, model_name = calibrated["option"], calibrated["model_name"]
            if model_name not in MODELS:
                raise ValueError(f"{name}: unknown pricing model")
            if not isinstance(option, Option) or option.years <= 0:
                raise ValueError(f"{name}: unexpired calibrated Option required")
            if option.side != m["option_type"] or option.strike != m["strike"]:
                raise ValueError(f"{name}: calibration does not match instrument")
            years = (
                (m["expiration_timestamp"] - milliseconds(snapshot.captured_at))
                / 1000
                / YEAR_SECONDS
            )
            if not np.isclose(years, option.years) or not np.isclose(spot, option.spot):
                raise ValueError(f"{name}: calibration does not match snapshot time or index")
            quote, settle = m.get("quote_currency"), m.get("settlement_currency")
            inverse = quote == settle == base
            if not inverse and not (quote == settle == native):
                raise ValueError(f"{name}: unsupported option settlement convention")
            midpoint = MODELS[model_name].price(option)
            source = "surface-calibrated model"
        elif kind == "future":
            settlement = m.get("settlement_currency")
            inverse = settlement == base and m.get("quote_currency") != base
            if not inverse and settlement != m.get("quote_currency"):
                raise ValueError(f"{name}: unsupported future settlement convention")
            if m.get("settlement_period") == "perpetual":
                kind = "perpetual"
            else:
                expiry = m.get("expiration_timestamp")
                if expiry is None:
                    raise ValueError(f"{name}: future expiry metadata is missing")
                years = (expiry - milliseconds(snapshot.captured_at)) / 1000 / YEAR_SECONDS
                if years <= 0:
                    raise ValueError(f"{name}: expired future")
        barrier = None
        barrier_type = row.get("barrier_type", "none")
        if barrier_type not in BARRIER_TYPES:
            raise ValueError(f"{name}: unknown barrier type")
        if barrier_type != "none":
            if kind != "option" or model_name != "Black-Scholes" or option.american:
                raise ValueError("Hypothetical barriers require European Black-Scholes option legs")
            barrier = Barrier(
                barrier_type,
                float(row["barrier_level"]),
                float(row.get("monitoring_days", 1.0)),
                bool(row.get("barrier_hit", False)),
            )
            if years * DAYS / barrier.monitoring_days > 1500:
                raise ValueError("Barrier monitoring is limited to 1,500 dates per contract")
        legs.append(
            StrategyLeg(
                name,
                base,
                kind,
                quantity,
                multiplier,
                spot,
                midpoint,
                cash_currency,
                native,
                fx,
                years,
                inverse,
                option,
                model_name,
                barrier,
                source,
                # Deribit ticker amounts use base amount for spot/options/linear
                # futures and quote face amount for inverse futures. Convert to
                # the strategy's contracts using captured contract_size.
                _displayed_depth(ticker, multiplier),
                float(m["amount_step"]) / multiplier if m.get("amount_step") else None,
                calibrated_options.get(name, {}).get("surface_reference", ()),
                float(m["min_trade_amount"]) / multiplier if m.get("min_trade_amount") else None,
            )
        )
    if not legs:
        raise ValueError("Select at least one strategy leg")
    return legs


def _vanilla_prices(option, model_name, spots, days, vol_multiplier=1.0):
    spots = np.asarray(spots, dtype=float)
    t = max(0.0, option.years - days / DAYS)
    vol = option.volatility * vol_multiplier
    if t <= 1e-12:
        return np.maximum(option.sign * (spots - option.strike), 0)
    f = spots * exp((option.rate - option.dividend) * t)
    discount = exp(-option.rate * t)
    if np.all(np.asarray(vol) == 0):
        if option.american:
            return np.array(
                [
                    MODELS[model_name].price(
                        replace(option, spot=float(s), years=t, volatility=vol)
                    )
                    for s in spots.reshape(-1)
                ]
            ).reshape(spots.shape)
        return discount * np.maximum(option.sign * (f - option.strike), 0)
    if model_name == "Black-Scholes" and not option.american:
        v = np.maximum(vol * sqrt(t), 1e-15)
        d1 = np.log(f / option.strike) / v + v / 2
        return (
            discount
            * option.sign
            * (f * ndtr(option.sign * d1) - option.strike * ndtr(option.sign * (d1 - v)))
        )
    if model_name == "Bachelier" and not option.american:
        v = vol * sqrt(t)
        x = option.sign * (f - option.strike)
        z = x / v
        return discount * (x * ndtr(z) + v * np.exp(-z * z / 2) / sqrt(2 * np.pi))
    return np.array(
        [
            MODELS[model_name].price(replace(option, spot=float(s), years=t, volatility=vol))
            for s in spots.reshape(-1)
        ]
    ).reshape(spots.shape)


def _monitoring_dates(option, barrier):
    expiry = option.years * DAYS
    return np.unique(
        np.r_[0.0, np.arange(barrier.monitoring_days, expiry, barrier.monitoring_days), expiry]
    )


class DiscreteBarrierGrid:
    """Backward lognormal expectation on a log-price grid, at fixed monitoring dates.

    Gaussian convolution integrates future value between observations. No Brownian
    bridge or continuous-monitoring correction is applied. Grid/truncation error is
    numerical; up/down in+out partition the analytic European vanilla exactly.
    """

    def __init__(
        self, option, barrier, valuation_days=(0.0,), *, grid_points=1601, vol_multiplier=1.0
    ):
        if option.american:
            raise ValueError("Barrier grid supports European exercise only")
        if grid_points < 401 or vol_multiplier < 0 or not np.isfinite(vol_multiplier):
            raise ValueError("Invalid barrier numerical controls")
        self.option, self.barrier, self.vol_multiplier = option, barrier, vol_multiplier
        expiry = option.years * DAYS
        self.monitoring = _monitoring_dates(option, barrier)
        times = np.unique(
            np.round(np.r_[self.monitoring, [d for d in valuation_days if 0 <= d <= expiry]], 10)
        )
        vol = option.volatility * vol_multiplier
        width = max(
            2.0,
            8 * vol * sqrt(option.years) + abs(option.rate - option.dividend) * option.years,
            abs(log(barrier.level / option.spot)) + 1.0,
        )
        self.grid = np.linspace(log(option.spot) - width, log(option.spot) + width, grid_points)
        # Put the discontinuity exactly on a node; avoid a moving barrier boundary.
        self.grid += log(barrier.level) - self.grid[np.argmin(abs(self.grid - log(barrier.level)))]
        self.spots = np.exp(self.grid)
        dx = self.grid[1] - self.grid[0]
        values = np.maximum(option.sign * (self.spots - option.strike), 0)
        values = np.where(barrier.breached(self.spots), 0.0, values)
        self.curves = {round(expiry, 10): values.copy()}
        for index in range(len(times) - 2, -1, -1):
            day, next_day = times[index : index + 2]
            dt = (next_day - day) / DAYS
            scale = vol * sqrt(dt) / dx
            smoothed = (
                gaussian_filter1d(values, scale, mode="nearest", truncate=8)
                if scale > 1e-8
                else values
            )
            shift = (option.rate - option.dividend - vol * vol / 2) * dt
            values = exp(-option.rate * dt) * np.interp(self.grid + shift, self.grid, smoothed)
            vanilla = _vanilla_prices(option, "Black-Scholes", self.spots, day, vol_multiplier)
            values = np.clip(values, 0, vanilla)
            if np.any(np.isclose(day, self.monitoring, rtol=0, atol=1e-8)):
                values = np.where(barrier.breached(self.spots), 0.0, values)
            self.curves[round(day, 10)] = values.copy()

    def price(self, spots, day=0.0, hit=False):
        spots = np.asarray(spots, dtype=float)
        if np.any(spots <= 0):
            raise ValueError("Barrier valuation requires positive spot")
        key = round(float(day), 10)
        if key not in self.curves:
            raise ValueError("Valuation date was not included in barrier grid")
        if np.any((np.log(spots) < self.grid[0]) | (np.log(spots) > self.grid[-1])):
            raise ValueError(
                "Simulated spot left the barrier valuation grid; reduce scenario range"
            )
        vanilla = _vanilla_prices(self.option, "Black-Scholes", spots, day, self.vol_multiplier)
        out = np.minimum(
            vanilla, np.maximum(0, np.interp(np.log(spots), self.grid, self.curves[key]))
        )
        hit = np.asarray(hit) | self.barrier.already_hit
        if np.any(np.isclose(day, self.monitoring, rtol=0, atol=1e-8)):
            hit = hit | self.barrier.breached(spots)
        out = np.where(hit, 0.0, out)
        return vanilla - out if self.barrier.knock_in else out


def barrier_price(option, barrier, *, grid_points=1601):
    """One-unit hypothetical premium, under the declared discrete monitoring schedule."""
    return float(DiscreteBarrierGrid(option, barrier, grid_points=grid_points).price(option.spot))


def strategy_metrics(equity, elapsed_days, risk_free_rate=0.0, minimum_acceptable_return=0.0):
    """Metrics for ONE ordered equity path, including its starting account equity.

    Annual drift/variance use actual elapsed intervals. On equal intervals the
    Sharpe denominator is sample return SD × sqrt(intervals/year). Sortino uses
    uncentered downside RMS relative to the continuously compounded annual MAR.
    Ratios are undefined after nonpositive equity, with <2 returns, or zero risk.
    """
    values, days = np.asarray(equity, dtype=float), np.asarray(elapsed_days, dtype=float)
    if values.ndim != 1 or len(values) != len(days) or len(values) < 2:
        raise ValueError("Supply matching ordered equity and elapsed-day observations")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(days).all()
        or np.any(np.diff(days) <= 0)
        or values[0] <= 0
    ):
        raise ValueError("Finite equity, positive initial equity and increasing times required")
    if not np.isfinite([risk_free_rate, minimum_acceptable_return]).all():
        raise ValueError("Risk-free rate and minimum acceptable return must be finite")
    duration = (days[-1] - days[0]) / DAYS
    peaks = np.maximum.accumulate(values)
    drawdown = (peaks - values) / peaks
    bankrupt = bool(np.any(values <= 0))
    metrics = dict(
        initial_equity=float(values[0]),
        final_equity=float(values[-1]),
        pnl=float(values[-1] - values[0]),
        total_return=float(values[-1] / values[0] - 1),
        max_drawdown=float(np.max(drawdown)),
        minimum_equity=float(np.min(values)),
        bankrupt=bankrupt,
        observations=len(values),
        elapsed_days=float(days[-1] - days[0]),
        annualized_return=np.nan,
        annualized_volatility=np.nan,
        sharpe_ratio=np.nan,
        sortino_ratio=np.nan,
    )
    if bankrupt:
        return metrics
    exponent = log(values[-1] / values[0]) / duration
    metrics["annualized_return"] = exp(exponent) - 1 if abs(exponent) < 700 else np.nan
    returns = np.diff(values) / values[:-1]
    dt = np.diff(days) / DAYS
    if len(returns) < 2:
        return metrics
    mean = np.sum(returns) / duration
    variance_time = duration - np.sum(dt**2) / duration
    variance = np.sum((returns - mean * dt) ** 2) / variance_time
    metrics["annualized_volatility"] = float(sqrt(variance))
    excess = returns - np.expm1(risk_free_rate * dt)
    excess_mean = np.sum(excess) / duration
    excess_risk = sqrt(np.sum((excess - excess_mean * dt) ** 2) / variance_time)
    if excess_risk > 1e-12:
        metrics["sharpe_ratio"] = float(excess_mean / excess_risk)
    target_excess = returns - np.expm1(minimum_acceptable_return * dt)
    downside = sqrt(np.sum(np.minimum(target_excess, 0) ** 2) / duration)
    if downside > 1e-12:
        metrics["sortino_ratio"] = float(np.sum(target_excess) / duration / downside)
    return metrics


def _time_grid(legs, horizon_days, steps):
    times = list(np.linspace(0, horizon_days, steps + 1))
    for leg in legs:
        if leg.years is not None and leg.years * DAYS < horizon_days:
            times.append(leg.years * DAYS)
        if leg.barrier:
            times.extend(t for t in _monitoring_dates(leg.option, leg.barrier) if t <= horizon_days)
    return np.unique(np.round(times, 10))


def _check_legs(legs):
    if not legs or len(legs) > 20:
        raise ValueError("Use between 1 and 20 strategy legs")
    if len({leg.cash_currency for leg in legs}) != 1:
        raise ValueError("Every leg must use the same explicit reporting cash currency")
    for asset in {leg.asset for leg in legs}:
        spots = [leg.spot * leg.conversion_rate for leg in legs if leg.asset == asset]
        # Different market/index prices are preserved; paths share proportional moves.
        if max(spots) / min(spots) > 1.25:
            raise ValueError(f"{asset}: inconsistent converted underlying prices across legs")


def _entry_value(leg):
    if leg.kind in {"future", "perpetual"}:
        return 0.0
    unit = barrier_price(leg.option, leg.barrier) if leg.barrier else leg.entry_price
    return leg.quantity * leg.multiplier * unit * leg.conversion_rate


def describe_strategy(legs):
    _check_legs(legs)
    rows = []
    for leg in legs:
        entry_value = _entry_value(leg)
        notional = (
            abs(leg.quantity)
            * leg.multiplier
            * (1.0 if leg.inverse and leg.kind in {"future", "perpetual"} else leg.spot)
            * leg.conversion_rate
        )
        rows.append(
            dict(
                instrument_name=leg.instrument_name,
                asset=leg.asset,
                product=leg.kind,
                quantity=leg.quantity,
                quantity_unit="base coin" if leg.kind == "spot" else "contracts",
                multiplier=leg.multiplier,
                inverse=leg.inverse,
                model=leg.model_name if leg.option else "",
                barrier=leg.barrier.kind if leg.barrier else "none",
                barrier_level=leg.barrier.level if leg.barrier else np.nan,
                monitoring_days=leg.barrier.monitoring_days if leg.barrier else np.nan,
                entry_price=leg.entry_price,
                entry_cash_value=entry_value,
                gross_cash_notional=notional,
                native_cash_currency=leg.native_cash_currency,
                reporting_currency=leg.cash_currency,
                cash_conversion_rate=leg.conversion_rate,
                quote_source=leg.quote_source,
                displayed_depth_contracts=leg.depth_contracts,
                quantity_step=leg.quantity_step,
                minimum_quantity=leg.minimum_quantity,
                initial_surface_tenors=len(leg.surface_reference),
            )
        )
    return pd.DataFrame(rows)


def _path_equity(
    legs,
    times,
    factors,
    initial_equity,
    entry_cost_bps=0.0,
    funding=None,
    vol_multiplier=1.0,
    *,
    evolution=None,
    regimes=(),
    basis_shocks=None,
    diagnostics=None,
):
    """Mark all legs at every time, freezing settlement values at exact expiry dates."""
    funding = funding or {}
    evolution = evolution or SurfaceEvolution()
    diagnostics = diagnostics if diagnostics is not None else {}
    basis_shocks = basis_shocks or {}
    if any(leg.barrier for leg in legs) and (
        evolution != SurfaceEvolution() or any(r.option_volatility_shock for r in regimes)
    ):
        raise ValueError(
            "Barrier continuation uses fixed calibrated volatility; use Barrier comparison for evolving-volatility models"
        )
    if (
        any(leg.option and leg.model_name != "Black-Scholes" for leg in legs)
        and evolution != SurfaceEvolution()
    ):
        raise ValueError("Surface evolution scenarios require Black-Scholes vanilla pricing")
    count = next(iter(factors.values())).shape[1]
    initial_values = [_entry_value(leg) for leg in legs]
    details = describe_strategy(legs)
    costs = float(details.gross_cash_notional.sum()) * entry_cost_bps / 10000
    equity = np.full((len(times), count), initial_equity - sum(initial_values) - costs)
    hits_report = []
    for leg, initial_value in zip(legs, initial_values, strict=True):
        spots = leg.spot * factors[leg.asset]
        values = np.zeros_like(spots)
        hit = np.full(count, bool(leg.barrier and leg.barrier.already_hit))
        grid = (
            DiscreteBarrierGrid(leg.option, leg.barrier, times, vol_multiplier=vol_multiplier)
            if leg.barrier
            else None
        )
        frozen = None
        for i, day in enumerate(times):
            s = spots[i]
            if frozen is not None:
                values[i] = frozen
                continue
            if leg.barrier and np.any(np.isclose(day, grid.monitoring, rtol=0, atol=1e-8)):
                hit |= leg.barrier.breached(s)
            if leg.kind == "spot":
                value = leg.quantity * leg.multiplier * s * leg.conversion_rate
            elif leg.kind == "option":
                multiplier = vol_multiplier
                pricing_option = leg.option
                for regime in regimes:
                    if day >= regime.day:
                        multiplier = vol_multiplier * (1 + regime.option_volatility_shock)
                if evolution != SurfaceEvolution() and not grid:
                    vol = surface_volatility(leg, s, day, evolution, diagnostics)
                    pricing_option = replace(leg.option, volatility=1.0)
                    multiplier = vol * multiplier
                price = (
                    grid.price(s, min(day, leg.years * DAYS), hit)
                    if grid
                    else _vanilla_prices(pricing_option, leg.model_name, s, day, multiplier)
                )
                value = leg.quantity * leg.multiplier * price * leg.conversion_rate
            else:
                remaining = max(0, 1 - day / (leg.years * DAYS)) if leg.years is not None else 1.0
                future = s * exp(log(leg.entry_price / leg.spot) * remaining)
                basis = basis_shocks.get(leg.asset, 0.0)
                for regime in regimes:
                    if day >= regime.day and leg.asset in regime.basis_shocks:
                        basis = regime.basis_shocks[leg.asset]
                future *= (1 + basis) ** remaining
                value = (
                    perpetual_gain(
                        leg.quantity, leg.multiplier, leg.inverse, leg.entry_price, future
                    )
                    * (s if leg.inverse else 1.0)
                    * leg.conversion_rate
                )
            values[i] = value
            if leg.years is not None and day >= leg.years * DAYS - 1e-8:
                frozen = np.array(value, copy=True)
        if leg.kind == "perpetual":
            rates = np.full(len(times) - 1, funding.get(leg.asset, 0.0), dtype=float)
            for regime in regimes:
                if leg.asset in regime.annual_funding_rates:
                    rates[times[:-1] >= regime.day] = regime.annual_funding_rates[leg.asset]
            base_notional = leg.quantity * leg.multiplier * leg.conversion_rate
            notional = np.full_like(spots, base_notional) if leg.inverse else base_notional * spots
            funding_cost = np.cumsum(
                funding_amount(notional[:-1], rates[:, None], np.diff(times)[:, None] / DAYS),
                axis=0,
            )
            values[1:] -= funding_cost
        equity += values
        if leg.barrier:
            hits_report.append(
                dict(
                    instrument_name=leg.instrument_name,
                    barrier=leg.barrier.kind,
                    paths_hit=int(np.count_nonzero(hit)),
                    hit_fraction=float(np.mean(hit)),
                )
            )
    # Include the pre-trade account as the initial observation; costs occur at first update.
    equity[0] = initial_equity
    return equity, pd.DataFrame(hits_report), costs


def _assumptions(config, currency):
    return [
        f"Equity and all P&L are in {currency}; external cash FX conversion rates are fixed explicit inputs.",
        "Correlated geometric Brownian spot scenarios with optional explicit jump/regime dates; drift, volatility and correlation are user assumptions, not calibrated forecasts.",
        f"Option evolution: {config.surface_evolution}. Sticky-strike/delta use captured initial smile/term references; no refitting or claim of arbitrage-preserving scenario dynamics. Tenor/wing edge extensions are counted.",
        "Barrier overlays are hypothetical European Black-Scholes, zero rebate, observed at inception, each stated monitoring interval and expiry; no between-observation barrier hits.",
        "Barrier-hit summaries describe the full underlying path, including dates after any scenario liquidation; liquidation stops portfolio P&L exposure at its recorded closeout date.",
        "Barrier continuation values use a 1,601-node log-price grid and Gaussian backward integration; finite-grid/truncation error remains.",
        "Dated futures log-basis converges to zero at expiry; explicit basis shocks and funding changes occur at scenario event dates. No exchange daily settlement ledger is modeled.",
        "Option/future expiry occurs on the time grid at simulated index spot (not exchange TWAP); native settlement is immediately converted to reporting cash and held without interest thereafter.",
        f"One-time entry friction is {config.entry_cost_bps:g} bps of gross underlying notional. Margin scenario: {config.margin}. Scenario margin uses conservative gross derivative exposure with no spread offsets, not exchange-certified margin rules. Liquidation closes all holdings at the first observed breach, charges the chosen fee, then holds cash.",
        f"Perpetual annual funding inputs: {config.annual_funding_rates}; absent assets mean zero funding. No spot borrow or collateral yield is modeled.",
        "Sharpe and Sortino use within-path equity returns, actual elapsed intervals and explicit risk-free/MAR inputs. Nonpositive equity or zero denominators produce undefined ratios.",
    ]


def simulate_strategy(legs, config=None):
    config = config or SimulationConfig()
    _check_legs(legs)
    _positive(config.initial_equity, "initial equity")
    _positive(config.horizon_days, "horizon")
    if not 2 <= config.steps <= 366 or not 1 <= config.paths <= 2048 or config.horizon_days > 366:
        raise ValueError("Use 2–366 steps, 1–2,048 paths and at most 366 days")
    if (
        not np.isfinite(
            [
                config.correlation,
                config.risk_free_rate,
                config.minimum_acceptable_return,
                config.entry_cost_bps,
            ]
        ).all()
        or abs(config.correlation) > 1
        or config.entry_cost_bps < 0
    ):
        raise ValueError("Invalid correlation, rates or entry costs")
    assets = sorted({leg.asset for leg in legs})
    if config.margin.enabled and config.margin.coin_collateral_fraction:
        assets = sorted(set(assets) | {config.margin.collateral_asset})
    for asset in assets:
        if asset not in config.annual_drift or asset not in config.annual_volatility:
            raise ValueError(f"Provide drift and volatility for {asset}")
        if (
            not np.isfinite(
                [
                    config.annual_drift[asset],
                    config.annual_volatility[asset],
                    config.annual_funding_rates.get(asset, 0.0),
                ]
            ).all()
            or config.annual_volatility[asset] < 0
        ):
            raise ValueError("Invalid path drift, volatility or funding")
    times = _time_grid(legs, config.horizon_days, config.steps)
    regimes = tuple(sorted(config.regimes, key=lambda r: r.day))
    if any(r.day > config.horizon_days for r in regimes) or len({r.day for r in regimes}) != len(
        regimes
    ):
        raise ValueError("Regime dates must be unique and inside the simulation horizon")
    times = np.unique(np.r_[times, [r.day for r in regimes]])
    if len(times) > 1501:
        raise ValueError("Reduce barrier monitoring frequency or scenario horizon")
    if (
        any(leg.model_name == "Binomial" and leg.option for leg in legs)
        and config.paths * len(times) > 6000
    ):
        raise ValueError(
            "Binomial scenario repricing is limited to 6,000 path-time observations; reduce paths/steps"
        )
    rng = np.random.default_rng(config.seed)
    z = rng.standard_normal((len(times) - 1, config.paths, len(assets)))
    correlations = np.full(len(times) - 1, config.correlation, dtype=float)
    for regime in regimes:
        if regime.correlation is not None:
            correlations[times[:-1] >= regime.day] = regime.correlation
    if len(assets) == 2:
        z[:, :, 1] = (
            correlations[:, None] * z[:, :, 0]
            + np.sqrt(np.maximum(0, 1 - correlations**2))[:, None] * z[:, :, 1]
        )
    factors = {}
    dt = np.diff(times)[:, None] / DAYS
    for index, asset in enumerate(assets):
        vol, drift = (
            np.full((len(times) - 1, 1), config.annual_volatility[asset], dtype=float),
            config.annual_drift[asset],
        )
        for regime in regimes:
            if asset in regime.volatility_multipliers:
                vol[times[:-1] >= regime.day] = (
                    config.annual_volatility[asset] * regime.volatility_multipliers[asset]
                )
        increments = (drift - vol * vol / 2) * dt + vol * np.sqrt(dt) * z[:, :, index]
        for regime in regimes:
            if asset in regime.spot_jumps:
                increments[np.isclose(times[1:], regime.day), :] += np.log1p(
                    regime.spot_jumps[asset]
                )
        factors[asset] = np.exp(np.vstack([np.zeros(config.paths), np.cumsum(increments, axis=0)]))
    diagnostics = {}
    equity, hits, costs = _path_equity(
        legs,
        times,
        factors,
        config.initial_equity,
        config.entry_cost_bps,
        config.annual_funding_rates,
        evolution=config.surface_evolution,
        regimes=regimes,
        diagnostics=diagnostics,
    )
    equity, margin_ledger, margin_summary = collateral_ledger(
        legs, times, factors, equity, config.initial_equity, config.margin
    )
    metrics = pd.DataFrame(
        [
            dict(
                path=f"path_{i}",
                **strategy_metrics(
                    equity[:, i], times, config.risk_free_rate, config.minimum_acceptable_return
                ),
            )
            for i in range(config.paths)
        ]
    )
    summary = dict(
        reporting_currency=legs[0].cash_currency,
        paths=config.paths,
        seed=config.seed,
        time_observations=len(times),
        horizon_days=config.horizon_days,
        initial_equity=config.initial_equity,
        entry_cost=costs,
        probability_of_loss=float(np.mean(metrics.pnl < 0)),
        bankruptcy_fraction=float(metrics.bankrupt.mean()),
        mean_terminal_pnl=float(metrics.pnl.mean()),
        median_terminal_pnl=float(metrics.pnl.median()),
        terminal_pnl_5pct=float(metrics.pnl.quantile(0.05)),
        terminal_pnl_95pct=float(metrics.pnl.quantile(0.95)),
        worst_max_drawdown=float(metrics.max_drawdown.max()),
        median_max_drawdown=float(metrics.max_drawdown.median()),
        median_sharpe=float(metrics.sharpe_ratio.median()),
        median_sortino=float(metrics.sortino_ratio.median()),
        paths_with_defined_sharpe=int(metrics.sharpe_ratio.notna().sum()),
        paths_with_defined_sortino=int(metrics.sortino_ratio.notna().sum()),
        **margin_summary,
    )
    return StrategySimulation(
        pd.DataFrame(equity, index=pd.Index(times, name="elapsed_days"), columns=metrics.path),
        metrics,
        summary,
        describe_strategy(legs),
        hits,
        _assumptions(config, legs[0].cash_currency),
        margin_ledger,
        diagnostics,
    )


def stress_strategy(
    legs,
    scenarios,
    *,
    initial_equity=100000.0,
    entry_cost_bps=0.0,
    annual_funding_rates=None,
    surface_evolution=None,
    margin=None,
):
    """Deterministic explicit stress paths; scenarios are never return samples.

    Volatility shock is a relative change in each model's native calibrated vol.
    Initial entry premium is unshocked. Barrier status follows the explicit path
    at contractual observation dates, including expiry when crossed by the horizon.
    """
    _check_legs(legs)
    margin = margin or MarginScenario()
    evolution = surface_evolution or SurfaceEvolution()
    _positive(initial_equity, "initial equity")
    if not np.isfinite(entry_cost_bps) or entry_cost_bps < 0:
        raise ValueError("Entry costs must be nonnegative")
    if not all(np.isfinite(value) for value in (annual_funding_rates or {}).values()):
        raise ValueError("Funding rates must be finite")
    results = []
    for scenario in scenarios:
        days = _positive(float(scenario.get("days", 1.0)), "stress horizon")
        if days > 366:
            raise ValueError("Stress horizon is limited to 366 days")
        vol_multiplier = 1 + float(scenario.get("volatility_shock", 0.0))
        if not np.isfinite(vol_multiplier) or vol_multiplier < 0:
            raise ValueError("Relative volatility shock must be at least -100%")
        times = _time_grid(legs, days, max(2, int(np.ceil(days))))
        if scenario.get("path_shape") == "waypoints":
            times = np.unique(np.r_[times, [p["day"] for p in scenario["waypoints"]]])
        elif scenario.get("path_shape", "monotone") != "monotone":
            times = np.unique(np.r_[times, scenario.get("event_day", days / 2)])
        if times[0] != 0 or times[-1] != days or len(times) > 1501:
            raise ValueError(
                "Stress path event dates must fit the selected horizon and observation limit"
            )
        assets = {leg.asset for leg in legs}
        if margin.enabled and margin.coin_collateral_fraction:
            assets.add(margin.collateral_asset)
        factors = scenario_factors(times, assets, scenario)
        funding = dict(annual_funding_rates or {})
        funding.update(scenario.get("annual_funding_rates", {}))
        if not all(np.isfinite(v) for v in funding.values()):
            raise ValueError("Scenario funding rates must be finite")
        basis = scenario.get("basis_shocks", {})
        if not all(np.isfinite(v) and v > -1 for v in basis.values()):
            raise ValueError("Basis shocks must be finite and greater than -100%")
        diagnostics = {}
        equity, hits, costs = _path_equity(
            legs,
            times,
            factors,
            initial_equity,
            entry_cost_bps,
            funding,
            vol_multiplier,
            evolution=evolution,
            basis_shocks=basis,
            diagnostics=diagnostics,
        )
        equity, ledger, margin_summary = collateral_ledger(
            legs, times, factors, equity, initial_equity, margin
        )
        results.append(
            dict(
                scenario=scenario["name"],
                days=days,
                btc_shock=scenario.get("spot_shocks", {}).get("BTC", 0.0),
                eth_shock=scenario.get("spot_shocks", {}).get("ETH", 0.0),
                volatility_shock=scenario.get("volatility_shock", 0.0),
                pnl=float(equity[-1, 0] - initial_equity),
                final_equity=float(equity[-1, 0]),
                minimum_equity=float(equity[:, 0].min()),
                max_drawdown=float(strategy_metrics(equity[:, 0], times)["max_drawdown"]),
                bankrupt=bool(np.any(equity[:, 0] <= 0)),
                barrier_legs_hit=int(hits.paths_hit.sum()) if not hits.empty else 0,
                entry_cost=costs,
                reporting_currency=legs[0].cash_currency,
                path_shape=scenario.get("path_shape", "monotone"),
                liquidated=bool(ledger.liquidated.iloc[0]) if not ledger.empty else False,
                liquidation_day=float(ledger.liquidation_day.iloc[0])
                if not ledger.empty
                else np.nan,
                liquidation_cost=float(ledger.liquidation_cost.iloc[0])
                if not ledger.empty
                else 0.0,
                surface_edge_uses=sum(diagnostics.values()),
            )
        )
    return pd.DataFrame(results)
