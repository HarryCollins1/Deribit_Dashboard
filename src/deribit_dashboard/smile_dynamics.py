"""Fixed-maturity smile movement using the current constrained calibration.

No market fetches, refitting, portfolio preparation or UI dependencies occur here.
Cash values are per underlying unit, with coin value defined as cash value / spot.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.special import ndtr

from deribit_dashboard.pricing import BlackScholes, Option

SCENARIOS = ("Initial", "Sticky strike", "Sticky forward delta")


@dataclass(frozen=True)
class SmileDynamicsConfig:
    instrument_name: str
    forward_shock: float = 0.05
    bump_fraction: float = 0.001
    grid_points: int = 61

    def __post_init__(self):
        if not isinstance(self.instrument_name, str) or not self.instrument_name.strip():
            raise ValueError("Choose a fitted option instrument")
        try:
            shock, bump = float(self.forward_shock), float(self.bump_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("Forward shock and bump fraction must be finite numbers") from exc
        if isinstance(self.forward_shock, bool) or not np.isfinite(shock) or shock <= -1:
            raise ValueError("Forward shock must be finite and greater than -100%")
        if isinstance(self.bump_fraction, bool) or not np.isfinite(bump) or not 0 < bump <= 0.05:
            raise ValueError("Bump fraction must be finite, positive and at most 5%")
        if (
            isinstance(self.grid_points, bool)
            or not isinstance(self.grid_points, (int, np.integer))
            or not 5 <= self.grid_points <= 501
        ):
            raise ValueError("Choose an integer from 5 to 501 curve grid points")
        object.__setattr__(self, "forward_shock", shock)
        object.__setattr__(self, "bump_fraction", bump)
        object.__setattr__(self, "grid_points", int(self.grid_points))


@dataclass
class SmileDynamicsResult:
    curves: pd.DataFrame
    summary: pd.DataFrame
    diagnostics: pd.DataFrame
    explanation: str


def _forward_call_d1(forward, strike, years, volatility):
    width = volatility * sqrt(years)
    if width <= 0 or not np.isfinite(width):
        raise ValueError("Positive finite implied volatility is needed for the delta coordinate")
    return log(forward / strike) / width + width / 2


def _forward_call_delta(forward, strike, years, volatility):
    return float(ndtr(_forward_call_d1(forward, strike, years, volatility)))


def _coin_greeks(price, delta, gamma, spot):
    return delta / spot - price / spot**2, gamma / spot - 2 * delta / spot**2 + 2 * price / spot**3


def _shape_checks(frame, rate, years):
    """Finite-grid call checks, with cash-scale tolerances; no global certificate."""
    frame = frame.sort_values("strike")
    if frame.empty:
        return dict(call_bounds_failures=0, monotonicity_failures=0, convexity_failures=0)
    strikes = frame.strike.to_numpy(dtype=float)
    prices = frame.call_price_cash.to_numpy(dtype=float)
    forwards = frame.forward.to_numpy(dtype=float)
    discount = exp(-rate * years)
    tolerance = 1e-7 * max(1.0, float(np.max(forwards)) * discount)
    lower = discount * np.maximum(forwards - strikes, 0.0)
    upper = discount * forwards
    slopes = np.diff(prices) / np.diff(strikes)
    return dict(
        call_bounds_failures=int(
            np.count_nonzero((prices < lower - tolerance) | (prices > upper + tolerance))
        ),
        monotonicity_failures=int(np.count_nonzero(np.diff(prices) > tolerance)),
        convexity_failures=int(np.count_nonzero(np.diff(slopes) < -1e-7 * discount)),
    )


def compare_smile_dynamics(analysis, config):
    """Compare fixed-time sticky-strike and sticky-forward-call-delta scenarios.

    Both spot and forward scale by the same factor, preserving inferred F/S carry.
    At fixed maturity, a monotone initial call-delta/IV smile gives the exact mapping
    K_reference = K * F_initial / F_new. This identity is verified through the delta
    coordinate, not extended to aging, arbitrary carry changes or an empirical law.
    """
    if not isinstance(config, SmileDynamicsConfig):
        raise ValueError("Use a SmileDynamicsConfig for the selected instrument")
    surface = analysis.surface
    if surface is None:
        raise ValueError("A successful constrained surface fit is required")
    selected = analysis.chain[analysis.chain.instrument_name == config.instrument_name]
    if len(selected) != 1 or selected.iloc[0].fit_status != "fitted":
        raise ValueError("Select one option with fit_status='fitted' in the current analysis")
    row = selected.iloc[0]
    expiry = int(row.expiry)
    indices = np.flatnonzero(np.asarray(surface.expiries) == expiry)
    if len(indices) != 1:
        raise ValueError("The selected expiry is absent from the fitted surface")
    index = int(indices[0])
    years = float(surface.times[index])
    initial_forward = float(surface.forwards[index])
    initial_spot, strike, rate = float(row.spot), float(row.strike), float(surface.rate)
    if (
        not np.isfinite([years, initial_forward, initial_spot, strike, rate]).all()
        or min(years, initial_forward, initial_spot, strike) <= 0
        or not np.isclose(float(row.years), years, rtol=0, atol=1e-10)
        or row.side not in {"call", "put"}
    ):
        raise ValueError(
            "The selected option requires finite positive time, spot, strike and forward"
        )
    x = np.asarray(surface.x, dtype=float)
    if (
        x.ndim != 1
        or len(x) < 3
        or not np.isfinite(x).all()
        or x[0] <= 0
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError("The calibrated strike grid must be finite, positive and increasing")
    source_knots = x * initial_forward
    low, high = float(source_knots[0]), float(source_knots[-1])
    factor = 1 + config.forward_shock
    shocked_spot, shocked_forward = initial_spot * factor, initial_forward * factor
    if not np.isfinite([shocked_spot, shocked_forward, low * factor, high * factor]).all():
        raise ValueError("The specified forward shock overflows the price or strike domain")
    carry_yield = rate - log(initial_forward / initial_spot) / years
    model = BlackScholes()

    @lru_cache(maxsize=20000)
    def source_iv(reference_strike):
        if not low <= reference_strike <= high:
            # Permit only roundoff at a fitted endpoint, never actual extrapolation.
            if not np.isclose(
                reference_strike, np.clip(reference_strike, low, high), rtol=1e-12, atol=0
            ):
                raise ValueError("outside initial fitted strike support; no extrapolation")
        reference_strike = float(np.clip(reference_strike, low, high))
        vol = float(surface.volatility(reference_strike, years))
        if not np.isfinite(vol) or vol <= 0:
            raise ValueError("initial fitted IV is not positive and finite")
        return vol

    validation_grid = np.unique(
        np.r_[np.geomspace(low, high, max(257, 4 * config.grid_points + 1)), source_knots]
    )
    # The geometric sampling grid and original knots can represent the same
    # strike a few floating-point units apart. IV inversion noise at that scale
    # must not be mistaken for a reversal of the delta coordinate.
    separated = np.r_[True, np.diff(validation_grid) > 1e-12 * validation_grid[1:]]
    validation_grid = validation_grid[separated]
    delta_coordinates = []
    delta_validation_error = ""
    try:
        for k in validation_grid:
            delta_coordinates.append(
                _forward_call_d1(initial_forward, float(k), years, source_iv(float(k)))
            )
        # N is strictly increasing analytically but saturates to 0/1 in floating
        # point in short-dated wings. Check its d1 argument to avoid false ambiguity.
        delta_map_verified = bool(np.all(np.diff(delta_coordinates) < 0))
        if not delta_map_verified:
            delta_validation_error = (
                "initial forward-call-delta map is not strictly decreasing on the validation grid"
            )
    except ValueError as exc:
        delta_map_verified = False
        delta_validation_error = str(exc)

    def point(mode, cash_strike, spot):
        forward = initial_forward * spot / initial_spot
        reference = (
            cash_strike * initial_spot / spot if mode == "Sticky forward delta" else cash_strike
        )
        result = dict(
            scenario=mode,
            expiry=expiry,
            strike=cash_strike,
            spot=spot,
            forward=forward,
            years=years,
            side=row.side,
            log_moneyness=log(cash_strike / forward),
            reference_strike=reference,
            iv_percent=np.nan,
            price_cash=np.nan,
            price_coin=np.nan,
            call_price_cash=np.nan,
            forward_call_delta=np.nan,
            delta_mapping_error=np.nan,
            status="ok",
        )
        try:
            if mode == "Sticky forward delta" and not delta_map_verified:
                raise ValueError(delta_validation_error)
            vol = source_iv(float(reference))
            delta = _forward_call_delta(forward, cash_strike, years, vol)
            if mode == "Sticky forward delta":
                initial_delta = _forward_call_delta(initial_forward, reference, years, vol)
                error = abs(delta - initial_delta)
                if error > 1e-10:
                    raise ValueError("forward-call-delta identity did not reconcile")
            else:
                error = 0.0
            option = Option(spot, cash_strike, years, vol, rate, carry_yield, row.side)
            price = float(model.price(option))
            call_price = (
                price
                if row.side == "call"
                else price + exp(-rate * years) * (forward - cash_strike)
            )
            result.update(
                iv_percent=vol * 100,
                price_cash=price,
                price_coin=price / spot,
                call_price_cash=call_price,
                forward_call_delta=delta,
                delta_mapping_error=error,
            )
        except (ValueError, OverflowError) as exc:
            result["status"] = str(exc)
        return result

    grid = np.geomspace(min(low, low * factor), max(high, high * factor), config.grid_points)
    curve_rows = []
    for mode in SCENARIOS:
        spot = initial_spot if mode == "Initial" else shocked_spot
        curve_rows.extend(point(mode, float(k), spot) for k in grid)
    curves = pd.DataFrame(curve_rows)
    initial_point = point("Initial", strike, initial_spot)
    if initial_point["status"] != "ok":
        raise ValueError(f"Selected fitted option cannot be priced: {initial_point['status']}")
    initial_price = initial_point["price_cash"]
    initial_coin = initial_point["price_coin"]

    def stencil(mode, spot, h, center):
        down, up = point(mode, strike, spot - h), point(mode, strike, spot + h)
        errors = [p["status"] for p in (down, up) if p["status"] != "ok"]
        if errors:
            return {}, "; ".join(sorted(set(errors)))
        values = {}
        for unit in ("cash", "coin"):
            below, middle, above = (
                down[f"price_{unit}"],
                center[f"price_{unit}"],
                up[f"price_{unit}"],
            )
            values[f"total_delta_{unit}"] = (above - below) / (2 * h)
            values[f"total_gamma_{unit}"] = (above - 2 * middle + below) / h**2
        return values, "ok"

    summary_rows = []
    for mode in SCENARIOS:
        spot = initial_spot if mode == "Initial" else shocked_spot
        center = point(mode, strike, spot)
        summary = dict(
            center,
            mode=mode,
            instrument_name=config.instrument_name,
            pnl_cash=center["price_cash"] - initial_price,
            pnl_coin=center["price_coin"] - initial_coin,
            fd_bump_cash=spot * config.bump_fraction,
            fd_half_bump_cash=spot * config.bump_fraction / 2,
            fd_status="not evaluated",
            half_step_status="not evaluated",
            fd_crossed_knot_count=0,
            half_step_crossed_knot_count=0,
            gamma_knot_sensitive=False,
        )
        for unit in ("cash", "coin"):
            for greek in ("delta", "gamma"):
                summary[f"fixed_iv_{greek}_{unit}"] = np.nan
                summary[f"total_{greek}_{unit}"] = np.nan
                summary[f"total_{greek}_{unit}_half_step"] = np.nan
                summary[f"{greek}_{unit}_half_step_change"] = np.nan
        if center["status"] == "ok":
            option = Option(
                spot, strike, years, center["iv_percent"] / 100, rate, carry_yield, row.side
            )
            greeks = model.greeks(option)
            coin_delta, coin_gamma = _coin_greeks(
                center["price_cash"], greeks["delta"], greeks["gamma"], spot
            )
            summary.update(
                fixed_iv_delta_cash=greeks["delta"],
                fixed_iv_gamma_cash=greeks["gamma"],
                fixed_iv_delta_coin=coin_delta,
                fixed_iv_gamma_coin=coin_gamma,
            )
            if mode == "Initial":
                summary["fd_status"] = summary["half_step_status"] = (
                    "not applicable: initial snapshot has no movement rule"
                )
            else:
                h = summary["fd_bump_cash"]
                values, summary["fd_status"] = stencil(mode, spot, h, center)
                half, summary["half_step_status"] = stencil(mode, spot, h / 2, center)
                summary.update(values)
                summary.update({f"{key}_half_step": value for key, value in half.items()})
                for unit in ("cash", "coin"):
                    for greek in ("delta", "gamma"):
                        key = f"total_{greek}_{unit}"
                        if key in values and key in half:
                            summary[f"{greek}_{unit}_half_step_change"] = half[key] - values[key]
                if mode == "Sticky forward delta":
                    for step, field in (
                        (h, "fd_crossed_knot_count"),
                        (h / 2, "half_step_crossed_knot_count"),
                    ):
                        mapped = strike * initial_spot / np.array([spot + step, spot - step])
                        summary[field] = int(
                            np.count_nonzero(
                                (source_knots[1:-1] >= mapped[0])
                                & (source_knots[1:-1] <= mapped[1])
                            )
                        )
                    summary["gamma_knot_sensitive"] = summary["fd_crossed_knot_count"] > 0
        else:
            summary["fd_status"] = summary["half_step_status"] = center["status"]
        summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)

    coverage = curves.pivot(index="strike", columns="scenario", values="status")
    common_strikes = coverage.index[(coverage == "ok").all(axis=1)]
    diagnostic_rows = []
    for mode in SCENARIOS:
        part = curves[curves.scenario == mode]
        covered = part[part.status == "ok"]
        common = covered[covered.strike.isin(common_strikes)]
        counts = _shape_checks(covered, rate, years)
        common_counts = _shape_checks(common, rate, years)
        selected_row = summary[summary["mode"] == mode].iloc[0]
        diagnostic_rows.append(
            dict(
                scenario=mode,
                expiry=expiry,
                total_points=len(part),
                covered_points=len(covered),
                unsupported_points=len(part) - len(covered),
                coverage_fraction=len(covered) / len(part),
                common_domain_points=len(common),
                **counts,
                **{f"common_{key}": value for key, value in common_counts.items()},
                delta_map_verified=delta_map_verified,
                delta_map_points=len(validation_grid),
                delta_map_evaluated_points=len(delta_coordinates),
                delta_map_failure=delta_validation_error,
                selected_option_status=selected_row.status,
                selected_fd_status=selected_row.fd_status,
                sampled_shape_checked=len(covered) >= 3,
                calendar_tested=False,
                check_scope="finite sampled call-price grid; not a global arbitrage certificate",
            )
        )

    explanation = (
        "These are hypothetical movements of the currently fitted constrained smile, not new "
        "observations or a refit. Independent raw-quote IV and exchange IV are not movement inputs. "
        "Remaining maturity and the cash rate stay fixed; spot and forward move by the same fraction, "
        "preserving their F/S carry. Sticky strike retains initial fitted IV at each strike. Sticky "
        "forward delta retains the positive undiscounted Black forward-call-delta coordinate N(d1), "
        "including for puts; this is neither premium-adjusted nor coin delta nor a finish probability. "
        "At fixed maturity and unchanged IV level, mapping K_reference=K*F_initial/F_new preserves "
        "that coordinate. The initial delta mapping is screened for strict monotonicity on a dense "
        "grid, and each mapped point's delta identity is checked. This fixed-time identity does not "
        "assert equivalence when maturity ages or carry changes. Unsupported mapped strikes remain "
        "visible as unavailable; there is no extrapolation or tenor interpolation in this panel. "
        "Cash prices and Greeks are per underlying unit, without contract multipliers. Coin price "
        "is V/S; coin delta is V_S/S-V/S² and coin gamma is V_SS/S-2V_S/S²+2V/S³. All delta/gamma "
        "derivatives use cash spot as the variable. Fixed-IV Greeks are Black–Scholes analytic "
        "sensitivities at each scenario's local IV. Total Greeks reapply that scenario's mapping "
        "on each spot bump while holding carry fixed; the initial snapshot has no uniquely defined "
        "total movement Greek. Full and half-size central bumps and crossed fit knots are reported. "
        "The original call interpolant is piecewise linear, so sticky-delta total gamma can be "
        "near zero between knots or strongly bump-dependent near them; it is not a smooth analytic "
        "surface gamma. Finite-grid call bounds, monotonicity and convexity checks are diagnostics, "
        "not a global arbitrage guarantee; calendar arbitrage is not tested for this one expiry. "
        "This is a vanilla scenario comparison, not an empirical forecast or barrier valuation."
    )
    return SmileDynamicsResult(curves, summary, pd.DataFrame(diagnostic_rows), explanation)
