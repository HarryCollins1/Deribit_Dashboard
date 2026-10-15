"""Descriptive downside measures and moving time-block uncertainty for research.

The resampling unit is a consecutive run of clock bins containing whole option
cross-sections, never an individual option/trade. Stationarity and weak dependence
remain assumptions; missing observations and inadequate time span are explicit.
"""

from math import ceil

import numpy as np
import pandas as pd


def moving_block_interval(
    times,
    values,
    *,
    bin_seconds=3600.0,
    block_seconds=86400.0,
    samples=2000,
    confidence=0.95,
    seed=42,
    minimum_blocks=5,
):
    
    if (
        not np.isfinite([bin_seconds, block_seconds, confidence]).all()
        or min(bin_seconds, block_seconds) <= 0
        or not 0 < confidence < 1
    ):
        raise ValueError("Use positive time bins/blocks and confidence inside (0, 1)")
    if (
        not isinstance(samples, int)
        or samples < 100
        or not isinstance(minimum_blocks, int)
        or minimum_blocks < 2
    ):
        raise ValueError("At least 100 bootstrap samples and two time blocks required")
    values = np.asarray(values, float)
    times = pd.to_datetime(list(times), utc=True)
    if values.ndim != 1 or len(values) != len(times):
        raise ValueError("One outcome per timezone-aware observation time required")
    result = dict(
        mean=np.nan,
        lower=np.nan,
        upper=np.nan,
        standard_error=np.nan,
        observations=len(values),
        time_bins=0,
        nonempty_bins=0,
        approximate_time_blocks=0,
        populated_time_blocks=0,
        bin_seconds=bin_seconds,
        requested_block_seconds=block_seconds,
        effective_block_seconds=ceil(block_seconds / bin_seconds) * bin_seconds,
        bootstrap_samples=samples,
        confidence=confidence,
        seed=seed,
        status="unavailable",
        reason="no common observations",
    )
    if not len(values):
        return result
    if times.isna().any() or not np.isfinite(values).all():
        result["reason"] = "missing/nonfinite times or outcomes retained; no complete-case interval"
        return result
    result["mean"] = float(values.mean())
    raw_bins = np.floor(times.as_unit("ns").asi8 / 1e9 / bin_seconds).astype(np.int64)
    bins = raw_bins - raw_bins.min()
    n, length = int(bins.max()) + 1, ceil(block_seconds / bin_seconds)
    totals = np.bincount(bins, weights=values, minlength=n)
    counts = np.bincount(bins, minlength=n)
    result.update(
        time_bins=n,
        nonempty_bins=int(np.count_nonzero(counts)),
        approximate_time_blocks=n // length,
        populated_time_blocks=len(np.unique(bins // length)),
    )
    if n // length < minimum_blocks or result["populated_time_blocks"] < minimum_blocks:
        result["reason"] = "insufficient populated time blocks for dependence-aware uncertainty"
        return result
    prefix_total, prefix_count = np.r_[0.0, np.cumsum(totals)], np.r_[0, np.cumsum(counts)]
    rng = np.random.default_rng(seed)
    # Keep memory O(samples), even for a long chronology.
    bootstrap_total, bootstrap_count = np.zeros(samples), np.zeros(samples)
    full, remainder = divmod(n, length)
    for width in [length] * full + ([remainder] if remainder else []):
        starts = rng.integers(0, n - length + 1, size=samples)
        bootstrap_total += prefix_total[starts + width] - prefix_total[starts]
        bootstrap_count += prefix_count[starts + width] - prefix_count[starts]
    if np.any(bootstrap_count == 0):
        result["reason"] = "empty resampled opportunities in a sparse chronology"
        return result
    estimates = bootstrap_total / bootstrap_count
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, [tail, 1 - tail])
    result.update(
        lower=float(lower),
        upper=float(upper),
        standard_error=float(estimates.std(ddof=1)),
        status="estimated",
        reason="conditional on fixed policy and block assumptions",
    )
    return result


def downside_metrics(values):
    """Native P&L downside; no annualized ratios or fictitious account capital."""
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        return dict(
            loss_fraction=np.nan,
            pnl_5pct=np.nan,
            expected_shortfall_5pct=np.nan,
            worst_pnl=np.nan,
            downside_rms=np.nan,
        )
    cutoff = float(np.quantile(values, 0.05))
    return dict(
        loss_fraction=float(np.mean(values < 0)),
        pnl_5pct=cutoff,
        expected_shortfall_5pct=float(values[values <= cutoff].mean()),
        worst_pnl=float(values.min()),
        downside_rms=float(np.sqrt(np.mean(np.minimum(values, 0) ** 2))),
    )


def outcome_curve(trades):
    """Completed/marked outcome curve; not an intrahorizon account-equity curve."""
    if trades.empty:
        return pd.DataFrame(
            columns=["time", "pnl_native", "cumulative_pnl_native", "drawdown_native"]
        )
    curve = (
        trades.groupby("evaluation_time", sort=True)
        .net_pnl_native.agg(lambda values: values.sum() if np.isfinite(values).all() else np.nan)
        .rename_axis("time")
        .reset_index(name="pnl_native")
    )
    # Missing outcomes propagate; never turn a missing trade P&L into a zero gain.
    curve["cumulative_pnl_native"] = curve.pnl_native.cumsum(skipna=False)
    peak = np.maximum.accumulate(np.r_[0.0, curve.cumulative_pnl_native.to_numpy()])[1:]
    curve["drawdown_native"] = curve.cumulative_pnl_native - peak
    return curve
