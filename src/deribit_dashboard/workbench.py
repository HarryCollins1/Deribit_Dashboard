"""Application services shared by CLI/notebooks and the thin dashboard view."""

import asyncio
from dataclasses import dataclass, field, replace
from datetime import UTC
from math import log, sqrt

import numpy as np
import pandas as pd

from deribit_dashboard.client import DeribitClient
from deribit_dashboard.collector import capture_snapshot
from deribit_dashboard.market import quote_market, replay
from deribit_dashboard.pricing import MODELS, YEAR_SECONDS, Option, implied_vol, risk
from deribit_dashboard.surface import (
    Contract,
    chain_audit,
    fit_surface,
    independently_invert,
    parity_forwards,
)
from deribit_dashboard.volatility import volatility_history


@dataclass
class Analysis:
    chain: pd.DataFrame
    filters: pd.DataFrame
    forwards: pd.DataFrame
    surface: object
    segments: pd.DataFrame
    fit_error: str | None = None
    parity_pairs: pd.DataFrame = field(default_factory=pd.DataFrame)

    def outcomes(self):
        counts = (
            self.chain.fit_status.value_counts().rename_axis("status").reset_index(name="quotes")
        )
        counts["fraction_of_chain"] = counts.quotes / len(self.chain)
        return counts


def selections(snapshot):
    return sorted(
        {
            (m["base_currency"], m.get("settlement_currency", ""))
            for m in snapshot.instruments.values()
            if m["kind"] == "option"
        }
    )


def analyze(
    snapshot,
    base,
    settlement,
    rate=0.0,
    stale_seconds=60,
    max_relative_width=1.0,
    expiries=None,
    strike_range=None,
    *,
    parity_estimator="weighted_mean",
    huber_delta=1.345,
    quote_mark="mid",
):
    if quote_mark not in {"bid", "mid", "ask"}:
        raise ValueError("Calibration quote mark must be bid, mid or ask")
    chain, filters = chain_audit(snapshot, base, settlement, stale_seconds, max_relative_width)
    # Keep captured midpoint and executable spreads intact for hygiene/precision.
    chain["calibration_price"] = chain[quote_mark]
    chain["calibration_quote_mark"] = quote_mark
    selected = pd.Series(True, index=chain.index)
    if expiries is not None:
        selected &= chain.expiry.isin(expiries)
    if strike_range is not None:
        selected &= chain.strike.between(*strike_range)
    removed = int((chain.accepted & ~selected).sum())
    filters.loc[len(filters)] = [
        "runtime_selection",
        "Outside selected expiry/strike range",
        int((~selected).sum()),
        float((~selected).mean()),
        removed,
        removed / len(chain),
    ]
    for i in chain.index[~selected]:
        chain.at[i, "reasons"] += ("runtime_selection",)
    chain["accepted"] &= selected
    forwards, pairs = parity_forwards(
        snapshot,
        chain,
        rate,
        stale_seconds,
        estimator=parity_estimator,
        huber_delta=huber_delta,
        return_pairs=True,
    )
    chain = independently_invert(chain, forwards, rate)
    surface, error = None, None
    chain["fitted_price"] = np.nan
    chain["fitted_iv"] = np.nan
    chain["fit_status"] = chain.iv_status
    try:
        surface = fit_surface(chain, forwards, rate)
        for i, row in chain[chain.iv_status == "included"].iterrows():
            if row.expiry not in surface.expiries:
                chain.loc[i, "fit_status"] = "sparse expiry"
                continue
            try:
                chain.loc[i, "fitted_price"] = surface.price(row.strike, row.years, row.side)
                chain.loc[i, "fitted_iv"] = surface.volatility(row.strike, row.years)
                chain.loc[i, "fit_status"] = "fitted"
            except ValueError:
                chain.loc[i, "fit_status"] = "outside common domain"
    except ValueError as exc:
        error = str(exc)
    chain["price_residual"] = chain.fitted_price - chain.calibration_price
    chain["midpoint_price_residual"] = chain.fitted_price - chain.mid
    chain["iv_residual"] = chain.fitted_iv - chain.own_iv
    chain["moneyness_segment"] = pd.cut(
        chain.log_moneyness,
        [-np.inf, -0.1, 0.1, np.inf],
        labels=["below forward", "near forward", "above forward"],
    )
    chain["liquidity_segment"] = pd.cut(
        chain.liquidity, [-np.inf, 0, 1, np.inf], labels=["no displayed size", "small", "larger"]
    )
    segments = chain.groupby(
        ["expiry", "moneyness_segment", "liquidity_segment"], observed=True
    ).agg(
        quotes=("instrument_name", "size"),
        fitted=("fitted_price", "count"),
        mean_residual=("price_residual", "mean"),
        rmse=(
            "price_residual",
            lambda x: np.sqrt(np.mean(x.dropna() ** 2)) if x.notna().any() else np.nan,
        ),
    )
    if surface is not None:
        surface.report.update(parity_estimator=parity_estimator, calibration_quote_mark=quote_mark)
    return Analysis(chain, filters, forwards, surface, segments.reset_index(), error, pairs)


@dataclass
class CalibrationSensitivity:
    summary: pd.DataFrame
    forwards: pd.DataFrame
    quotes: pd.DataFrame
    scenarios: dict[str, Analysis]
    explanation: str


def quote_perturbation_sensitivity(
    snapshot,
    base,
    settlement,
    rate=0.0,
    stale_seconds=60,
    max_relative_width=1.0,
    expiries=None,
    strike_range=None,
    *,
    parity_estimator="weighted_mean",
    huber_delta=1.345,
):
    """Refit bid/midpoint/ask marks while preserving all captured quotes and audit rows.

    Every eligible option is marked simultaneously at its own bid, midpoint or
    ask. Each run re-estimates parity forwards and independently inverts/fits IV.
    These are calibration perturbations, not executable whole-chain portfolios,
    confidence intervals or guaranteed price bounds. Report ranges only when all
    three fits succeed for an instrument; failed observations remain counted.
    """
    scenarios = {
        mark: analyze(
            snapshot,
            base,
            settlement,
            rate,
            stale_seconds,
            max_relative_width,
            expiries,
            strike_range,
            parity_estimator=parity_estimator,
            huber_delta=huber_delta,
            quote_mark=mark,
        )
        for mark in ("bid", "mid", "ask")
    }
    summaries, forwards = [], []
    mid_forward = scenarios["mid"].forwards.set_index("expiry").forward
    quotes = (
        scenarios["mid"]
        .chain[["instrument_name", "expiry", "strike", "side", "bid", "mid", "ask"]]
        .copy()
        .set_index("instrument_name")
    )
    for mark, result in scenarios.items():
        chain = result.chain
        hygiene = ~chain.accepted
        iv_failed = chain.accepted & (chain.iv_status != "included")
        fit_failed = (
            chain.accepted & (chain.iv_status == "included") & (chain.fit_status != "fitted")
        )
        fitted = chain.fit_status == "fitted"
        summaries.append(
            dict(
                quote_mark=mark,
                input_quotes=len(chain),
                hygiene_dropped=int(hygiene.sum()),
                iv_dropped=int(iv_failed.sum()),
                fit_dropped=int(fit_failed.sum()),
                fitted_quotes=int(fitted.sum()),
                fitted_expiries=int(chain.loc[fitted, "expiry"].nunique()),
                max_constraint_violation=result.surface.report["max_constraint_violation"]
                if result.surface is not None
                else np.nan,
                fit_error=result.fit_error,
            )
        )
        forward = result.forwards.copy()
        forward.insert(0, "quote_mark", mark)
        reference = forward.expiry.map(mid_forward)
        forward["mid_reference_forward"] = reference
        forward["forward_change"] = forward.forward - reference
        forward["forward_change_bps"] = (
            forward.forward / reference.where(reference > 0) - 1
        ) * 10000
        forwards.append(forward)
        source = chain.set_index("instrument_name")
        for column in ("own_iv", "fitted_iv", "fitted_price", "fit_status"):
            quotes[f"{mark}_{column}"] = source[column]
    success = pd.concat([quotes[f"{mark}_fit_status"].eq("fitted") for mark in scenarios], axis=1)
    quotes["successful_fit_scenarios"] = success.sum(axis=1)
    quotes["all_scenarios_fitted"] = success.all(axis=1)
    for field_name, scale, label in (
        ("fitted_price", 1.0, "fitted_price_range"),
        ("fitted_iv", 100.0, "fitted_iv_range_points"),
    ):
        values = quotes[[f"{mark}_{field_name}" for mark in scenarios]]
        quotes[label] = ((values.max(axis=1) - values.min(axis=1)) * scale).where(
            quotes.all_scenarios_fitted
        )
    return CalibrationSensitivity(
        pd.DataFrame(summaries),
        pd.concat(forwards, ignore_index=True),
        quotes.reset_index(),
        scenarios,
        "Bid/midpoint/ask marks perturb all eligible quotes simultaneously; parity forwards, "
        "independent IV and the constrained surface are recomputed. Original spreads and hygiene "
        "remain fixed. Ranges require all three fits; failed quotes are counted. This is a "
        "calibration sensitivity experiment, not a statistical confidence band or a tradable bound.",
    )


def price_selection(
    analysis,
    snapshot,
    names,
    model_name,
    rate=0.0,
    dividend=0.0,
    american=False,
    use_parity_carry=True,
):
    if analysis.surface is None:
        raise ValueError(analysis.fit_error)
    model = MODELS[model_name]
    results = []
    for name in names:
        row = analysis.chain.set_index("instrument_name").loc[name]
        if row.fit_status != "fitted":
            continue
        contract = Contract.from_metadata(snapshot.instruments[name])
        f = analysis.forwards.set_index("expiry").loc[row.expiry, "forward"]
        q = rate - log(f / row.spot) / row.years
        option = Option(row.spot, row.strike, row.years, row.fitted_iv, rate, q, row.side)
        # Convert the European surface price to each model's native volatility units.
        vol = implied_vol(row.fitted_price, option, model)
        option = replace(
            option, volatility=vol, american=american, dividend=q if use_parity_carry else dividend
        )
        values = risk(model, option, contract.multiplier)
        results.append(
            dict(
                instrument_name=name,
                model_name=model_name,
                model_volatility=vol,
                carry_yield=option.dividend,
                **values,
                option=option,
            )
        )
    return results


def quote_selection(analysis, snapshot, priced, horizon_seconds):
    row = analysis.chain.set_index("instrument_name").loc[priced["instrument_name"]]
    group = analysis.chain[
        (analysis.chain.expiry == row.expiry) & (analysis.chain.fit_status == "fitted")
    ]
    residual_rmse = sqrt(float(np.mean(group.price_residual**2)))
    contract = Contract.from_metadata(snapshot.instruments[priced["instrument_name"]])
    vega = priced["cash_vega"] / contract.multiplier
    iv_rmse = sqrt(float(np.mean(group.iv_residual**2)))
    # Local normal/lognormal mapping sigma_normal ≈ S*sigma_lognormal.
    if priced["model_name"] == "Bachelier":
        iv_rmse *= row.spot
    return quote_market(
        priced["cash_value"] / contract.multiplier,
        vega,
        residual_rmse,
        iv_rmse,
        row.years,
        horizon_seconds,
        row.bid,
        row.ask,
        row.liquidity,
        snapshot.instruments[priced["instrument_name"]],
        row.spot,
    )


def mock_replay(
    priced, quote, metadata, spot_move, vol_move, horizon_seconds, contracts, fee, inventory_limit
):
    """Two explicit mock market orders, with model-repriced future fair values."""
    option = priced["option"]
    model = MODELS[priced["model_name"]]
    dt = horizon_seconds / YEAR_SECONDS
    if 2 * dt >= option.years:
        raise ValueError("Simulation horizon must end before option expiry")
    first = replace(
        option,
        spot=option.spot * (1 + spot_move),
        volatility=max(0, option.volatility * (1 + vol_move)),
        years=option.years - dt,
    )
    second = replace(first, spot=option.spot, years=option.years - 2 * dt)
    p0, p1, p2 = [model.price(o) for o in (option, first, second)]
    events = [
        dict(
            bid=quote["bid"],
            ask=quote["ask"],
            fair_before=p0,
            fair_after=p1,
            spot=option.spot,
            next_spot=first.spot,
            hedge_delta=priced["cash_delta"] / metadata["contract_size"],
            taker_side="buy",
            trade_price=quote["ask"],
            contracts=contracts,
        ),
        dict(
            bid=quote["bid"],
            ask=quote["ask"],
            fair_before=p1,
            fair_after=p2,
            spot=first.spot,
            next_spot=second.spot,
            hedge_delta=priced["cash_delta"] / metadata["contract_size"],
            taker_side="sell",
            trade_price=quote["bid"],
            contracts=contracts,
        ),
    ]
    return replay(events, Contract.from_metadata(metadata).multiplier, fee, inventory_limit)


def load_live_chain(settings):
    return asyncio.run(capture_snapshot(settings))


def load_volatility(settings, currency, perpetual, start, end, window):
    async def fetch():
        async with DeribitClient(settings) as client:
            return await volatility_history(client, currency, perpetual, start, end, window)

    return asyncio.run(fetch())


def cash_currency_label(snapshot, base, settlement):
    """Read the cash unit for chart labels; ambiguous historical metadata stays 'cash'."""
    options = [
        m
        for m in snapshot.instruments.values()
        if m["kind"] == "option"
        and m["base_currency"] == base
        and m.get("settlement_currency") == settlement
    ]
    units = set()
    for metadata in options:
        if settlement != base and metadata.get("quote_currency") == settlement:
            units.add(settlement)
        elif metadata.get("counter_currency"):
            units.add(metadata["counter_currency"])
    if not units:
        units = {
            m["quote_currency"]
            for m in snapshot.instruments.values()
            if m["kind"] == "future"
            and m["base_currency"] == base
            and m.get("settlement_currency") == settlement
            and m.get("quote_currency")
            and m["quote_currency"] != base
        }
    return next(iter(units)) if len(units) == 1 else "cash units"


def plot_lines(
    frame,
    x,
    ys,
    title,
    group=None,
    *,
    x_label=None,
    y_label=None,
    x_scale=1.0,
    x_offset=0.0,
    y_scale=1.0,
    zero_line=False,
    highlights=None,
):
    """Static chart with display-only unit conversions; source frames are never changed."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    names = {
        "forward": "Option-implied forward",
        "traded_future": "Traded future midpoint",
        "total_variance": "Fitted total variance",
        "own_iv": "Observed midpoint IV",
        "fitted_iv": "Surface-fitted IV",
        "iv_percent": "Fitted IV",
        "iv_difference_points": "Own IV minus exchange mark IV",
        "cumulative_pnl": "Cumulative mock P&L",
        "own_realized_pct": "Own realized volatility (perpetual)",
        "published_realized_pct": "Exchange realized volatility",
        "dvol_pct": "DVOL implied volatility",
    }
    groups = list(frame.groupby(group)) if group else [(None, frame)]
    # Reserve room below the axes so a long expiry legend never covers the data.
    legend_rows = (len(groups) * len(ys) + 1) // 2
    fig, ax = plt.subplots(figsize=(10, 4.5 + 0.22 * legend_rows), layout="constrained")
    for index, (label, rows) in enumerate(groups):
        rows = rows.sort_values(x)
        x_values = rows[x] if x == "timestamp" else rows[x] * x_scale + x_offset
        for y in ys:
            legend = names.get(y, y.replace("_", " ").capitalize())
            if group == "expiry":
                expiry = pd.to_datetime(label, unit="ms", utc=True).strftime("%d %b %Y, %H:%M UTC")
                legend = expiry if len(ys) == 1 else f"{expiry} — {legend}"
            elif group:
                legend = f"{label} — {legend}"
            style = {"color": f"C{index % 10}"} if group else {}
            # Preserve both call/put observations without connecting duplicate strikes.
            observed = y in {"own_iv", "iv_difference_points"}
            ax.plot(
                x_values,
                rows[y] * y_scale,
                marker="o" if observed else ".",
                markersize=4,
                linestyle="none" if observed else "-",
                label=legend,
                **style,
            )
            if highlights is not None and x in highlights and y in highlights:
                marked = highlights[highlights[group] == label] if group else highlights
                ax.scatter(
                    marked[x] * x_scale + x_offset,
                    marked[y] * y_scale,
                    marker="*",
                    s=140,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                    **style,
                )
    ax.set(xlabel=x_label or x, ylabel=y_label or names.get(ys[0], ys[0]), title=title)
    if group:
        ax.legend(
            fontsize="small",
            title="Expiry (UTC)" if group == "expiry" else group.replace("_", " ").capitalize(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
        )
    else:
        ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    if x == "log_moneyness" and frame[x].min() <= 0 <= frame[x].max():
        ax.axvline(0, color="0.5", linestyle=":", linewidth=1)
        ax.text(
            0,
            0.97,
            "Strike = forward",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="0.35",
        )
    if zero_line:
        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
    if x == "timestamp":
        locator = mdates.AutoDateLocator(tz=UTC)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=UTC))
    elif x == "index":
        ax.set_xticks(sorted(frame[x].unique() * x_scale + x_offset))
    if ("own_iv" in ys and y_scale == 100) or "dvol_pct" in ys:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    elif "forward" in ys or "cumulative_pnl" in ys or "equity" in (y_label or "").lower():
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.4g}"))
    ax.grid(alpha=0.25)
    return fig
