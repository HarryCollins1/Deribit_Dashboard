"""Auditable quote hygiene, parity forwards and constrained total-variance calibration.

The returned interpolant is a normalized call-price surface, not unconstrained IV
interpolation. Convex linear strike interpolation and linear maturity interpolation
preserve butterfly and normalized calendar constraints inside the fitted domain.
"""

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.special import ndtr

from deribit_dashboard.models import milliseconds
from deribit_dashboard.pricing import YEAR_SECONDS, Option, implied_vol

RULES = {
    "missing_ticker": "No ticker captured",
    "metadata": "Missing/invalid contract size, tick size, or settlement/quote convention",
    "expired": "Expiry at or before snapshot cutoff",
    "invalid_spot": "Nonpositive or nonfinite index price",
    "one_sided": "Missing/nonfinite bid or ask",
    "zero_bid": "Bid <= 0",
    "crossed": "Bid > ask",
    "stale": "Exchange quote age exceeds selected seconds",
    "clock_ahead": "Exchange timestamp more than 2 seconds after cutoff",
    "closed": "Ticker state is not open",
    "wide": "Ask minus bid exceeds selected multiple of midpoint",
}

PARITY_ESTIMATORS = ("weighted_mean", "weighted_median", "huber")


def _weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2, side="left"))
    if index + 1 < len(order) and np.isclose(
        cumulative[index], cumulative[-1] / 2, rtol=1e-12, atol=0
    ):
        return float((values[order[index]] + values[order[index + 1]]) / 2)
    return float(values[order[index]])


def parity_estimate(estimates, half_spreads, estimator="weighted_mean", huber_delta=1.345):
    """Pool parity forwards and retain every pair, including robust downweights.

    Precision weights are inverse squared executable pair spreads. Huber uses
    IRLS around a weighted-median start, with scale=max(weighted MAD × 1.4826,
    weighted median bid/ask half-width). The scale floor prevents zero MAD from
    eliminating all but identical observations. Weighted-median diagnostics use
    the original precision weights; they are not a median sampling-error formula.
    """
    values, half = np.asarray(estimates, float), np.asarray(half_spreads, float)
    if estimator not in PARITY_ESTIMATORS:
        raise ValueError(f"Unknown parity estimator: {estimator}")
    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("Huber tuning must be finite and positive")
    if (
        values.ndim != 1
        or not len(values)
        or values.shape != half.shape
        or not np.isfinite(values).all()
        or not np.isfinite(half).all()
        or np.any(half < 0)
    ):
        raise ValueError("Supply finite pair estimates and nonnegative half-spreads")
    full = np.maximum(2 * half, 1e-12)
    precision = (full.min() / full) ** 2
    precision /= precision.sum()
    median = _weighted_median(values, precision)
    mean = float(precision @ values)
    mad = _weighted_median(np.abs(values - median), precision)
    scale = max(1.4826 * mad, _weighted_median(half, precision), 1e-12)
    factors = np.ones(len(values))
    converged = True
    location = median if estimator in {"weighted_median", "huber"} else mean
    if estimator == "huber":
        converged = False
        for _ in range(200):
            residual = np.abs(values - location)
            factors = np.minimum(1.0, huber_delta * scale / np.maximum(residual, 1e-300))
            weights = precision * factors
            updated = float(weights @ values / weights.sum())
            if abs(updated - location) <= 1e-10 * max(1.0, abs(location)):
                location, converged = updated, True
                break
            location = updated
        factors = np.minimum(
            1.0, huber_delta * scale / np.maximum(np.abs(values - location), 1e-300)
        )
    weights = precision * factors
    weights /= weights.sum()
    return dict(
        forward=float(location),
        estimator=estimator,
        weighted_mean_forward=mean,
        weighted_median_forward=median,
        pair_dispersion=float(np.std(values)),
        weighted_pair_dispersion=float(np.sqrt(weights @ (values - location) ** 2)),
        effective_pair_count=float(1 / np.sum(weights**2)),
        spread_uncertainty=float(np.sqrt(np.sum((weights * half) ** 2))),
        maximum_pair_weight=float(weights.max()),
        robust_scale=float(scale),
        downweighted_pairs=int(np.count_nonzero(factors < 1 - 1e-10)),
        estimator_converged=converged,
        precision_weights=precision,
        normalized_weights=weights,
        robust_weight_factors=factors,
    )


@dataclass(frozen=True)
class Contract:
    multiplier: float
    tick: float
    inverse: bool
    steps: tuple

    @classmethod
    def from_metadata(cls, m):
        size, tick = float(m["contract_size"]), float(m["tick_size"])
        if not np.isfinite([size, tick]).all() or min(size, tick) <= 0:
            raise ValueError("Invalid contract size/tick")
        base, quote, settlement = m["base_currency"], m["quote_currency"], m["settlement_currency"]
        inverse = quote == base == settlement
        if not inverse and (quote != settlement or quote == base):
            raise ValueError("Unsupported quote/settlement convention")
        steps = tuple(
            (float(x["above_price"]), float(x["tick_size"])) for x in m.get("tick_size_steps", [])
        )
        if any(not np.isfinite([a, t]).all() or a < 0 or t <= 0 for a, t in steps):
            raise ValueError("Invalid tick ladder")
        return cls(size, tick, inverse, tuple(sorted(steps)))

    def tick_at(self, price):
        return max([self.tick] + [tick for threshold, tick in self.steps if price > threshold])

    def cash_factor(self, spot):
        return spot if self.inverse else 1.0


def chain_audit(snapshot, base, settlement, stale_seconds=60.0, max_relative_width=1.0):
    if stale_seconds < 0 or max_relative_width <= 0:
        raise ValueError("Invalid hygiene limits")
    rows = []
    for name, m in snapshot.instruments.items():
        if (
            m["kind"] != "option"
            or m["base_currency"] != base
            or m.get("settlement_currency") != settlement
        ):
            continue
        t = snapshot.tickers.get(name, {})
        reasons = []
        contract = None
        try:
            contract = Contract.from_metadata(m)
        except (KeyError, TypeError, ValueError):
            reasons.append("metadata")
        if not t:
            reasons.append("missing_ticker")
        years = (
            (m["expiration_timestamp"] - milliseconds(snapshot.captured_at)) / 1000 / YEAR_SECONDS
        )
        spot = t.get("index_price") or np.nan
        bid, ask = t.get("best_bid_price"), t.get("best_ask_price")
        age = (milliseconds(snapshot.captured_at) - t.get("timestamp", 0)) / 1000
        if years <= 0:
            reasons.append("expired")
        if not np.isfinite(spot) or spot <= 0:
            reasons.append("invalid_spot")
        if bid is None or ask is None or not np.isfinite([bid, ask]).all():
            reasons.append("one_sided")
            bid, ask = np.nan, np.nan
        else:
            if bid <= 0:
                reasons.append("zero_bid")
            if bid > ask:
                reasons.append("crossed")
            if ask - bid > max_relative_width * (bid + ask) / 2:
                reasons.append("wide")
        if age > stale_seconds:
            reasons.append("stale")
        if age < -2:
            reasons.append("clock_ahead")
        if t.get("state", "open") != "open":
            reasons.append("closed")
        factor = contract.cash_factor(spot) if contract else np.nan
        rows.append(
            dict(
                instrument_name=name,
                expiry=m["expiration_timestamp"],
                years=years,
                strike=m["strike"],
                side=m["option_type"],
                spot=spot,
                bid=bid * factor,
                ask=ask * factor,
                mid=(bid + ask) / 2 * factor,
                quote_age_seconds=age,
                liquidity=min(t.get("best_bid_amount") or 0, t.get("best_ask_amount") or 0),
                minimum_displayed_best_quote_size=min(
                    t.get("best_bid_amount") or 0, t.get("best_ask_amount") or 0
                ),
                exchange_iv=t.get("mark_iv", np.nan),
                reasons=tuple(reasons),
                accepted=not reasons,
            )
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No options for selected base and settlement currency")
    remaining = set(frame.index)
    report = []
    for rule, description in RULES.items():
        affected = {i for i, r in enumerate(rows) if rule in r["reasons"]}
        removed = affected & remaining
        report.append(
            dict(
                rule=rule,
                definition=description,
                flagged=len(affected),
                flagged_fraction=len(affected) / len(rows),
                removed=len(removed),
                removed_fraction=len(removed) / len(rows),
            )
        )
        remaining -= removed
    return frame, pd.DataFrame(report)


def parity_forwards(
    snapshot,
    chain,
    rate=0.0,
    stale_seconds=60,
    *,
    estimator="weighted_mean",
    huber_delta=1.345,
    return_pairs=False,
):
    if estimator not in PARITY_ESTIMATORS:
        raise ValueError(f"Unknown parity estimator: {estimator}")
    if not np.isfinite([rate, huber_delta]).all() or huber_delta <= 0:
        raise ValueError("Finite rate and positive Huber tuning required")
    rows, pair_rows = [], []
    eligible = chain[chain.accepted]
    for expiry, group in chain.groupby("expiry"):
        good = eligible[eligible.expiry == expiry]
        estimates, halves, lowers, uppers, details = [], [], [], [], []
        for strike, pair in good.groupby("strike"):
            calls, puts = pair[pair.side == "call"], pair[pair.side == "put"]
            if len(calls) != 1 or len(puts) != 1:
                continue
            c, p = calls.iloc[0], puts.iloc[0]
            discount = exp(-rate * c.years)
            call_price, put_price = (
                c.get("calibration_price", c.mid),
                p.get("calibration_price", p.mid),
            )
            estimates.append(strike + (call_price - put_price) / discount)
            lowers.append(strike + (c.bid - p.ask) / discount)
            uppers.append(strike + (c.ask - p.bid) / discount)
            halves.append((c.ask - c.bid + p.ask - p.bid) / discount / 2)
            details.append(
                dict(
                    expiry=expiry,
                    strike=strike,
                    call_instrument=c.instrument_name,
                    put_instrument=p.instrument_name,
                    call_price=call_price,
                    put_price=put_price,
                    pair_forward=estimates[-1],
                    parity_low=lowers[-1],
                    parity_high=uppers[-1],
                    pair_half_spread=halves[-1],
                )
            )
        estimate = parity_estimate(estimates, halves, estimator, huber_delta) if estimates else None
        f = estimate["forward"] if estimate else np.nan
        if estimate:
            for index, detail in enumerate(details):
                detail.update(
                    estimator=estimator,
                    selected_forward=f,
                    residual=estimates[index] - f,
                    precision_weight=estimate["precision_weights"][index],
                    normalized_weight=estimate["normalized_weights"][index],
                    robust_weight_factor=estimate["robust_weight_factors"][index],
                )
                pair_rows.append(detail)
        futures = []
        for name, m in snapshot.instruments.items():
            if m["kind"] != "future" or m.get("expiration_timestamp") != expiry:
                continue
            reference = snapshot.instruments[group.iloc[0].instrument_name]
            if m["base_currency"] != reference["base_currency"] or m.get(
                "settlement_currency"
            ) != reference.get("settlement_currency"):
                continue
            t = snapshot.tickers.get(name, {})
            age = (milliseconds(snapshot.captured_at) - t.get("timestamp", 0)) / 1000
            b, a = t.get("best_bid_price"), t.get("best_ask_price")
            if b and a and np.isfinite([b, a]).all() and 0 < b <= a and -2 <= age <= stale_seconds:
                futures.append((b + a) / 2)
        future = float(np.median(futures)) if futures else np.nan
        rows.append(
            dict(
                expiry=expiry,
                years=group.years.iloc[0],
                pairs=len(estimates),
                unmatched_quotes=len(good) - 2 * len(estimates),
                forward=f,
                parity_low=max(lowers) if lowers else np.nan,
                parity_high=min(uppers) if uppers else np.nan,
                estimator=estimator,
                weighted_mean_forward=estimate["weighted_mean_forward"] if estimate else np.nan,
                weighted_median_forward=estimate["weighted_median_forward"] if estimate else np.nan,
                pair_dispersion=estimate["pair_dispersion"] if estimate else np.nan,
                weighted_pair_dispersion=estimate["weighted_pair_dispersion"]
                if estimate
                else np.nan,
                effective_pair_count=estimate["effective_pair_count"] if estimate else 0.0,
                spread_uncertainty=estimate["spread_uncertainty"] if estimate else np.nan,
                maximum_pair_weight=estimate["maximum_pair_weight"] if estimate else np.nan,
                robust_scale=estimate["robust_scale"] if estimate else np.nan,
                downweighted_pairs=estimate["downweighted_pairs"] if estimate else 0,
                estimator_converged=estimate["estimator_converged"] if estimate else False,
                traded_future=future,
                future_count=len(futures),
                basis=f - future,
                basis_bps=(f / future - 1) * 10000 if future > 0 else np.nan,
            )
        )
    frame = pd.DataFrame(rows)
    return (frame, pd.DataFrame(pair_rows)) if return_pairs else frame


def independently_invert(chain, forwards, rate):
    frame = chain.copy()
    frame["own_iv"] = np.nan
    frame["log_moneyness"] = np.nan
    frame["total_variance"] = np.nan
    frame["iv_status"] = "hygiene exclusion"
    for i, row in frame[frame.accepted].iterrows():
        f = forwards.loc[forwards.expiry == row.expiry, "forward"].iloc[0]
        if not np.isfinite(f) or f <= 0:
            frame.loc[i, "iv_status"] = "no valid parity forward"
            continue
        try:
            option = Option(
                f * exp(-rate * row.years), row.strike, row.years, 0.5, rate=rate, side=row.side
            )
            iv = implied_vol(row.get("calibration_price", row.mid), option)
            if iv <= 0:
                raise ValueError("zero total variance")
            frame.loc[i, ["own_iv", "log_moneyness", "total_variance"]] = (
                iv,
                log(row.strike / f),
                iv * iv * row.years,
            )
            frame.loc[i, "iv_status"] = "included"
        except ValueError as exc:
            frame.loc[i, "iv_status"] = str(exc)
    frame["iv_difference_points"] = frame.own_iv * 100 - frame.exchange_iv
    return frame


def normalized_call(x, w):
    root = np.sqrt(w)
    d1 = -np.log(x) / root + root / 2
    return ndtr(d1) - x * ndtr(d1 - root)


@dataclass
class Surface:
    expiries: np.ndarray
    times: np.ndarray
    forwards: np.ndarray
    x: np.ndarray
    calls: np.ndarray
    rate: float
    report: dict

    def price(self, strike, years, side="call"):
        if not self.times[0] <= years <= self.times[-1]:
            raise ValueError("Maturity outside fitted domain; extrapolation disabled")
        f = float(np.exp(np.interp(years, self.times, np.log(self.forwards))))
        x = strike / f
        if not self.x[0] - 1e-12 <= x <= self.x[-1] + 1e-12:
            raise ValueError("Strike outside common fitted moneyness domain")
        x = float(np.clip(x, self.x[0], self.x[-1]))
        c = np.interp(years, self.times, [np.interp(x, self.x, row) for row in self.calls])
        if side not in ("call", "put"):
            raise ValueError("Unknown option side")
        return exp(-self.rate * years) * f * (c if side == "call" else c - 1 + x)

    def volatility(self, strike, years):
        f = float(np.exp(np.interp(years, self.times, np.log(self.forwards))))
        return implied_vol(
            self.price(strike, years),
            Option(f * exp(-self.rate * years), strike, years, 0.5, rate=self.rate),
        )

    def points(self):
        rows = []
        for expiry, t, f, calls in zip(
            self.expiries, self.times, self.forwards, self.calls, strict=True
        ):
            for x, c in zip(self.x, calls, strict=True):
                iv = implied_vol(float(c), Option(1, float(x), 1, 0.5))
                rows.append(
                    dict(
                        expiry=expiry,
                        years=t,
                        strike=x * f,
                        log_moneyness=log(x),
                        total_variance=iv * iv,
                        fitted_iv=iv / sqrt(t),
                    )
                )
        return pd.DataFrame(rows)


def fit_surface(chain, forwards, rate=0.0, nodes=21):
    """Minimize total-variance residuals with exact linear price-shape constraints.

    The common domain is the intersection of observed expiry strike/F ranges.
    Failed/sparse expiries and quotes outside this domain are explicitly counted.
    """
    usable = chain[chain.iv_status == "included"]
    groups = [(e, g) for e, g in usable.groupby("expiry") if g.strike.nunique() >= 3]
    if not groups:
        raise ValueError("Need at least three distinct invertible strikes in one expiry")
    low = max(g.log_moneyness.min() for _, g in groups)
    high = min(g.log_moneyness.max() for _, g in groups)
    if high - low < 1e-4:
        raise ValueError("Expiries have no common strike/forward domain")
    x = np.exp(np.linspace(low, high, nodes))
    n = len(groups)
    observed_w, weights = np.empty((n, nodes)), np.empty((n, nodes))
    for j, (_, g) in enumerate(groups):
        aggregated = g.groupby("log_moneyness").agg(
            total_variance=("total_variance", "mean"), width=("ask", "mean")
        )
        observed_w[j] = np.interp(np.log(x), aggregated.index, aggregated.total_variance)
        # Precision from relative executable spreads, not exchange IV.
        spread = g.assign(relative=(g.ask - g.bid) / g.mid).groupby("log_moneyness").relative.mean()
        weights[j] = 1 / np.maximum(np.interp(np.log(x), spread.index, spread), 1e-6) ** 2
    weights /= weights.max()
    # Decision variables are normalized calls; constraints are therefore linear.
    # A weighted projection supplies the starting point for exact variance fitting.
    constraints, lower, upper = [], [], []

    def add(coeffs, lo, hi):
        row = np.zeros(n * nodes)
        for index, value in coeffs.items():
            row[index] = value
        constraints.append(row)
        lower.append(lo)
        upper.append(hi)

    dx = np.diff(x)
    for j in range(n):
        for k in range(nodes - 1):
            add({j * nodes + k: -1 / dx[k], j * nodes + k + 1: 1 / dx[k]}, -1, 0)
        for k in range(nodes - 2):
            add(
                {
                    j * nodes + k: 1 / dx[k],
                    j * nodes + k + 1: -1 / dx[k] - 1 / dx[k + 1],
                    j * nodes + k + 2: 1 / dx[k + 1],
                },
                0,
                np.inf,
            )
        if j:
            for k in range(nodes):
                add({j * nodes + k: 1, (j - 1) * nodes + k: -1}, 0, np.inf)
    matrix = np.array(constraints)
    # Fit total variance directly. Nonlinear call constraints use analytic Jacobians.
    scale = float(observed_w.max())

    def call_values(v):
        return normalized_call(x, scale * v.reshape(n, nodes)).ravel()

    def call_jacobian(v):
        variance = scale * v.reshape(n, nodes)
        root = np.sqrt(variance)
        d1 = -np.log(x) / root + root / 2
        derivative = np.exp(-d1 * d1 / 2) / np.sqrt(2 * np.pi) / (2 * root)
        return matrix * derivative.ravel()[None, :] * scale

    def objective(v):
        diff = scale * v.reshape(n, nodes) - observed_w
        return float(np.sum(weights * diff**2)), (2 * weights * diff * scale).ravel()

    constraint = NonlinearConstraint(
        lambda v: matrix @ call_values(v), lower, upper, jac=call_jacobian
    )
    initial = np.ones(n * nodes)
    result = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=Bounds(np.full(n * nodes, 1e-8 / scale), np.full(n * nodes, 100.0 / scale)),
        constraints=[constraint],
        options={"maxiter": 1500, "ftol": 1e-12},
    )
    fitted_calls = call_values(result.x)
    residual = matrix @ fitted_calls
    violation = max(
        float(np.max(np.array(lower) - residual)), float(np.max(residual - np.array(upper))), 0
    )
    if not result.success or violation > 1e-7:
        raise ValueError(f"Surface constraints failed: {result.message}; violation={violation:g}")
    expiries = np.array([e for e, _ in groups])
    selected = forwards.set_index("expiry").loc[expiries]
    within = usable.expiry.isin(expiries) & usable.log_moneyness.between(low - 1e-12, high + 1e-12)
    surface = Surface(
        expiries,
        selected.years.to_numpy(),
        selected.forward.to_numpy(),
        x,
        fitted_calls.reshape(n, nodes),
        rate,
        dict(
            optimizer=result.message,
            max_constraint_violation=violation,
            input_quotes=len(chain),
            hygiene_dropped=int((~chain.accepted).sum()),
            iv_dropped=int((chain.accepted & (chain.iv_status != "included")).sum()),
            domain_or_sparse_dropped=int((~within).sum()),
            fitted_quotes=int(within.sum()),
            fitted_expiries=n,
            total_expiries=int(chain.expiry.nunique()),
            objective="spread-weighted total-variance least squares on common log-moneyness nodes",
        ),
    )
    return surface
