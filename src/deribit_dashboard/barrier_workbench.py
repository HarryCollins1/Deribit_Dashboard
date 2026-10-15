"""Barrier experiment orchestration; UI-independent market preparation and reports."""

import asyncio
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from deribit_dashboard.barrier_calibration import calibrate_heston, fit_local_vol, heston_calls
from deribit_dashboard.barrier_execution import StripExecution, replication_entry
from deribit_dashboard.barrier_hedging import PerpetualHedge, dynamic_hedge, hedge_summary
from deribit_dashboard.barrier_history import download_historical_paths
from deribit_dashboard.barrier_models import (
    HestonParameters,
    SimulationPaths,
    antithetic_estimate,
    black_scholes_finish_probability,
    black_scholes_touch_probability,
    discounted_payoffs,
    evaluate_paths,
    probabilities,
    simulate_gbm,
    simulate_heston,
    simulate_local_vol,
)
from deribit_dashboard.barrier_replication import evaluate_replication, fit_semi_static, fit_static
from deribit_dashboard.client import DeribitClient
from deribit_dashboard.models import milliseconds
from deribit_dashboard.pricing import DAYS, BlackScholes
from deribit_dashboard.strategies import Barrier, barrier_price
from deribit_dashboard.workbench import analyze, cash_currency_label, price_selection


@dataclass(frozen=True)
class BarrierExperiment:
    barriers: tuple = (110000.0, 115000.0, 120000.0, 125000.0, 130000.0)
    kind: str = "up-and-out"
    monitoring_days: float = 1.0
    paths: int = 4096
    steps: int = 120
    seed: int = 42
    spot_bump_fraction: float = 0.01
    probability_level: float = 120000.0
    hedge_paths: int = 256
    fee_bps: float = 1.0
    annual_funding: float = 0.0
    fit_heston: bool = True
    heston_parameters: HestonParameters | None = None
    market_execution: bool = True
    impact_multiplier: float = 1.0

    def __post_init__(self):
        if (
            not 1 <= len(self.barriers) <= 15
            or len(set(self.barriers)) != len(self.barriers)
            or not np.isfinite(self.barriers).all()
            or min(self.barriers) <= 0
        ):
            raise ValueError("Supply 1–15 distinct positive barrier levels")
        if self.kind not in {"up-and-out", "up-and-in"}:
            raise ValueError("This comparison supports zero-rebate up-and-out/in calls")
        if not 0.05 <= self.monitoring_days <= 30 or not 0 < self.spot_bump_fraction < 0.05:
            raise ValueError("Invalid observation interval or Greek bump")
        if not 128 <= self.paths <= 16384 or self.paths % 2 or not 10 <= self.steps <= 1500:
            raise ValueError("Use an even count of 128–16,384 paths and 10–1,500 integration steps")
        if not 2 <= self.hedge_paths <= 2048 or self.hedge_paths % 2:
            raise ValueError("Use an even count of 2–2,048 hedge paths")
        if (
            not np.isfinite(
                [self.fee_bps, self.annual_funding, self.probability_level, self.impact_multiplier]
            ).all()
            or self.fee_bps < 0
            or self.probability_level <= 0
            or self.impact_multiplier < 0
        ):
            raise ValueError("Invalid funding, fee or probability threshold")


@dataclass
class BarrierMarket:
    option: object
    analysis: object
    instrument_name: str
    currency: str
    calls: pd.DataFrame
    perpetual: PerpetualHedge
    perpetual_metadata: dict
    snapshot_id: str
    strip_execution: StripExecution | None = None
    execution_errors: tuple = ()


@dataclass
class BarrierReport:
    market: BarrierMarket
    config: BarrierExperiment
    prices: pd.DataFrame
    vanilla: pd.DataFrame
    probabilities: pd.DataFrame
    replication_costs: pd.DataFrame
    replication_weights: pd.DataFrame
    hedge_metrics: pd.DataFrame
    path_results: pd.DataFrame
    diagnostics: dict
    plans: dict = field(default_factory=dict)


def prepare_barrier_market(snapshot, instrument_name, perpetual_name, rate=0.0, stale_seconds=60.0):
    metadata = snapshot.instruments[instrument_name]
    if (
        metadata.get("base_currency") != "BTC"
        or metadata.get("kind") != "option"
        or metadata.get("option_type") != "call"
    ):
        raise ValueError("Select a discovered BTC call")
    analysis = analyze(
        snapshot, "BTC", metadata["settlement_currency"], rate=rate, stale_seconds=stale_seconds
    )
    priced = price_selection(analysis, snapshot, [instrument_name], "Black-Scholes", rate=rate)
    if not priced:
        raise ValueError("Selected call failed chain hygiene or lies outside the fitted surface")
    option = priced[0]["option"]
    if option.years * DAYS > 366:
        raise ValueError("Barrier experiments are limited to one year")
    currency = cash_currency_label(snapshot, "BTC", metadata["settlement_currency"])
    calls = analysis.chain[
        (analysis.chain.expiry == metadata["expiration_timestamp"])
        & (analysis.chain.side == "call")
        & (analysis.chain.fit_status == "fitted")
    ].sort_values("strike")
    perp = snapshot.instruments[perpetual_name]
    quote = snapshot.tickers.get(perpetual_name, {})
    age = (milliseconds(snapshot.captured_at) - quote.get("timestamp", 0)) / 1000
    bid, ask = quote.get("best_bid_price"), quote.get("best_ask_price")
    if not -2 <= age <= stale_seconds or bid is None or ask is None or not 0 < bid <= ask:
        raise ValueError("Perpetual requires a fresh, positive two-sided quote at capture")
    hedge = PerpetualHedge.from_metadata(perp, (bid + ask) / 2, option.spot, currency)
    displayed = min(quote.get("best_bid_amount") or 0, quote.get("best_ask_amount") or 0)
    execution_errors = []
    if not np.isfinite(displayed) or displayed <= 0:
        execution_errors.append("Perpetual execution needs positive captured bid and ask size")
        displayed = np.inf
    hedge = replace(
        hedge,
        half_spread_bps=(ask - bid) / (ask + bid) * 10000,
        displayed_depth_notional=displayed if hedge.inverse else displayed * (bid + ask) / 2,
    )
    try:
        execution = StripExecution.from_snapshot(snapshot, calls, option.spot)
    except (ValueError, TypeError, KeyError) as exc:
        execution = None
        execution_errors.append(str(exc))
    return BarrierMarket(
        option,
        analysis,
        instrument_name,
        currency,
        calls,
        hedge,
        perp,
        snapshot.snapshot_id,
        execution,
        tuple(execution_errors),
    )


def _monitor_subset(times, spots, interval_days, max_paths=None):
    horizon = times[-1]
    scheduled = np.arange(0, horizon + 1e-12, interval_days / DAYS)
    indices = []
    for time in scheduled:
        index = int(np.argmin(abs(times - time)))
        if abs(times[index] - time) > 1e-10:
            raise ValueError(
                "Price paths do not contain the chosen barrier monitoring times; choose a compatible candle resolution"
            )
        indices.append(index)
    indices = sorted(set(indices + [len(times) - 1]))
    selected = spots
    if max_paths is not None and spots.shape[1] > max_paths:
        half = min(max_paths // 2, spots.shape[1] // 2)
        selected = spots[:, np.r_[np.arange(half), np.arange(half) + spots.shape[1] // 2]]
    return times[indices], selected[indices]


def _hedges(market, config, plans, premiums, times, spots, source):
    all_results, summaries = [], []
    execution = execution_for(market, config)
    for level in sorted(config.barriers):
        barrier = Barrier(config.kind, level, config.monitoring_days)
        for method in ("static", "semi-static", "dynamic perp"):
            if method == "dynamic perp":
                result = dynamic_hedge(
                    market.option,
                    barrier,
                    times,
                    spots,
                    premiums[level],
                    market.perpetual
                    if config.market_execution
                    else replace(market.perpetual, half_spread_bps=0),
                    fee_bps=config.fee_bps,
                    annual_funding=config.annual_funding,
                    impact_multiplier=config.impact_multiplier,
                )
            else:
                result = evaluate_replication(
                    plans[(level, method)], times, spots, premiums[level], execution=execution
                )
            result.insert(0, "method", method)
            result.insert(0, "barrier", level)
            result.insert(0, "source", source)
            all_results.append(result)
            summaries.append(
                dict(source=source, barrier=level, method=method, **hedge_summary(result))
            )
    return pd.concat(all_results, ignore_index=True), pd.DataFrame(summaries)


def execution_for(market, config):
    if not config.market_execution:
        return None
    if market.execution_errors:
        raise ValueError(
            "; ".join(market.execution_errors)
            + "; explicitly disable captured execution for a frictionless research run"
        )
    if market.strip_execution is None:
        raise ValueError(
            "Captured strip execution metadata unavailable; explicitly disable market execution for frictionless research"
        )
    return replace(
        market.strip_execution, fee_bps=config.fee_bps, impact_multiplier=config.impact_multiplier
    )


def run_barrier_comparison(market, config=None):
    config = config or BarrierExperiment()
    o = market.option
    if o.years * DAYS / config.monitoring_days > 1500:
        raise ValueError("Reduce monitoring frequency: at most 1,500 observations")
    local = fit_local_vol(market.analysis.surface, o)
    local_grid = local.diagnostics()
    initial = config.heston_parameters or HestonParameters(
        o.volatility**2, 2.0, o.volatility**2, 0.6, -0.5
    )
    if config.fit_heston:
        parameters, heston_fit, heston_diagnostics = calibrate_heston(
            market.analysis.surface, o, initial
        )
    else:
        parameters, heston_fit = initial, pd.DataFrame()
        heston_diagnostics = dict(
            method="Explicit user parameters; not calibrated", parameters=asdict(parameters)
        )
    controls = dict(
        paths=config.paths,
        steps=config.steps,
        seed=config.seed,
        monitoring_days=config.monitoring_days,
    )
    paths = {
        "Black-Scholes": simulate_gbm(o, **controls),
        "Local vol": simulate_local_vol(o, local, **controls),
        "Heston": simulate_heston(o, parameters, **controls),
    }
    vanilla_rows, probability_rows, price_rows = [], [], []
    vanilla = BlackScholes().price(o)
    for label, sim in paths.items():
        estimates = evaluate_paths(
            sim, o, list(config.barriers), config.kind, config.monitoring_days
        )
        probability_rows.append(
            dict(
                model=label, **probabilities(sim, config.probability_level, config.monitoring_days)
            )
        )
        vanilla_rows.append(
            dict(
                model=label,
                vanilla_price=float(estimates.vanilla_price.iloc[0]),
                sampling_se=float(estimates.vanilla_price_se.iloc[0]),
                surface_target=vanilla,
                residual_to_surface=float(estimates.vanilla_price.iloc[0] - vanilla),
            )
        )
        bump = o.spot * config.spot_bump_fraction
        bumps = []
        for multiplier in (1 - config.spot_bump_fraction, 1 + config.spot_bump_fraction):
            bumped_option = replace(o, spot=o.spot * multiplier)
            if label == "Local vol":
                bumped_paths = simulate_local_vol(bumped_option, local, **controls)
            else:
                bumped_paths = SimulationPaths(
                    sim.times,
                    sim.spots * multiplier,
                    sim.model,
                    dict(sim.diagnostics, initial_spot=bumped_option.spot),
                )
            bumps.append((bumped_option, bumped_paths))
        for row in estimates.to_dict("records"):
            level = row["barrier"]
            center = discounted_payoffs(sim, o, level, config.kind, config.monitoring_days)
            low, high = [
                discounted_payoffs(p, op, level, config.kind, config.monitoring_days)
                for op, p in bumps
            ]
            delta, delta_se = antithetic_estimate((high - low) / (2 * bump))
            gamma, gamma_se = antithetic_estimate((high - 2 * center + low) / bump**2)
            row.update(
                model=label,
                delta=delta,
                delta_se=delta_se,
                gamma=gamma,
                gamma_se=gamma_se,
                spot_bump=bump,
                barrier_distance_pct=(level / o.spot - 1) * 100,
            )
            price_rows.append(row)
    prices = pd.DataFrame(price_rows)
    premiums = dict(
        zip(
            prices[prices.model == "Black-Scholes"].barrier,
            prices[prices.model == "Black-Scholes"].price,
            strict=True,
        )
    )
    costs, weights, plans = [], [], {}
    execution = execution_for(market, config)
    for level in sorted(config.barriers):
        barrier = Barrier(config.kind, level, config.monitoring_days)
        grid_price = barrier_price(o, barrier, grid_points=3201)
        prices.loc[
            (prices.model == "Black-Scholes") & (prices.barrier == level), "bs_grid_crosscheck"
        ] = grid_price
        for method, fitter in (("static", fit_static), ("semi-static", fit_semi_static)):
            plan = fitter(
                o, barrier, market.calls.strike.to_numpy(), market.calls.fitted_price.to_numpy()
            )
            plans[(level, method)] = plan
            entry = replication_entry(plan, execution)
            costs.append(
                dict(
                    barrier=level,
                    method=method,
                    **entry,
                    cash_deposit=plan.cash,
                    terminal_fit_rmse=plan.terminal_rmse,
                    boundary_fit_rmse=plan.boundary_rmse,
                    bs_barrier_premium=premiums[level],
                    premium_minus_cost=premiums[level] - entry["cost"],
                )
            )
            weights.extend(
                dict(
                    barrier=level,
                    method=method,
                    strike=k,
                    weight=w,
                    vanilla_unit_price=p,
                    instrument_name=name,
                )
                for k, w, p, name in zip(
                    plan.strikes,
                    plan.weights,
                    plan.vanilla_prices,
                    market.calls.instrument_name,
                    strict=True,
                )
            )
    path_results, hedge_metrics = [], []
    for label, sim in paths.items():
        times, spots = _monitor_subset(
            sim.times, sim.spots, config.monitoring_days, min(config.paths, config.hedge_paths)
        )
        details, metrics = _hedges(market, config, plans, premiums, times, spots, label)
        path_results.append(details)
        hedge_metrics.append(metrics)
    diagnostics = dict(
        snapshot_id=market.snapshot_id,
        option=asdict(o),
        cash_currency=market.currency,
        heston_parameters=asdict(parameters),
        heston_calibration=heston_diagnostics,
        heston_targets=heston_fit.to_dict("records"),
        heston_vanilla_quadrature=float(heston_calls(o, parameters)[0]),
        local_svi=local.calibration.to_dict("records"),
        local_regularized_grid_fraction=float(local_grid.regularized.mean()),
        local_runtime=local.counters,
        local_extension="Linear total variance from zero to first expiry; frozen local volatility beyond observed moneyness wings. Floor 1%, cap 300%; nonpositive Dupire values regularized and counted.",
        forward_convention="Other smile maturities standardized to the selected expiry's constant carry",
        simulation={label: sim.diagnostics for label, sim in paths.items()},
        bs_analytic_finish_probability=black_scholes_finish_probability(
            o, config.probability_level
        ),
        bs_continuous_touch_benchmark=black_scholes_touch_probability(o, config.probability_level),
        hedge_convention="One short barrier per BTC; every hedge receives the same BS MC premium. Continuation marks and main comparison perp deltas use BS. Perp basis fixed, fees/funding explicit; hedge at monitoring dates. No liquidation/margin model.",
        replication_convention="Finite call-strip approximations, not exact replication; static cannot reproduce touch history; semi-static has boundary and overshoot errors. With market_execution, entry uses captured bid/ask, metadata ticks and depth. Subsequent execution preserves captured relative spreads and depth around BS marks; impact=halfspread×max(quantity/depth−1,0)×multiplier. This is an explicit scenario beyond top-of-book, not historical execution. Vanilla fee is bps of traded premium; perp fee is bps of notional. Expiry settlement has no transaction fee.",
        execution_enabled=config.market_execution,
        execution_metadata=market.strip_execution.audit(np.zeros(len(market.calls))).to_dict(
            "records"
        )
        if market.strip_execution
        else [],
    )
    return BarrierReport(
        market,
        config,
        prices,
        pd.DataFrame(vanilla_rows),
        pd.DataFrame(probability_rows),
        pd.DataFrame(costs),
        pd.DataFrame(weights),
        pd.concat(hedge_metrics, ignore_index=True),
        pd.concat(path_results, ignore_index=True),
        diagnostics,
        plans,
    )


def replay_historical(report, historical):
    if historical.spots.shape[1] == 0:
        raise ValueError("No complete historical windows; inspect the data-quality counts")
    cfg = report.config
    times, spots = _monitor_subset(historical.times_years, historical.spots, cfg.monitoring_days)
    bs = report.prices[report.prices.model == "Black-Scholes"]
    premiums = dict(zip(bs.barrier, bs.price, strict=True))
    details, metrics = _hedges(
        report.market, cfg, report.plans, premiums, times, spots, "Historical close-path replay"
    )
    if not historical.labels.empty:
        details = details.merge(
            historical.labels,
            left_on="path",
            right_index=True,
            how="left",
            suffixes=("", "_historical"),
        )
    return details, metrics


def load_barrier_history(settings, market, start, end, resolution_minutes=60, max_paths=12):
    async def load():
        async with DeribitClient(replace(settings, currency="BTC")) as client:
            metadata = await client.instrument(market.perpetual.instrument_name)
            return await download_historical_paths(
                client,
                metadata,
                start,
                end,
                market.option.years * DAYS,
                resolution_minutes,
                market.option.spot,
                max_paths,
            )

    return asyncio.run(load())


def barrier_figure(report):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    currency = report.market.currency
    for label, group in report.prices.groupby("model", sort=False):
        group = group.sort_values("barrier_distance_pct")
        x = group.barrier_distance_pct
        for ax, y, se in (
            (axes[0, 0], "price", "price_se"),
            (axes[0, 1], "gamma", "gamma_se"),
            (axes[1, 0], "touch_probability", "touch_probability_se"),
        ):
            ax.errorbar(x, group[y], yerr=1.96 * group[se], marker="o", capsize=3, label=label)
    hedges = report.hedge_metrics[report.hedge_metrics.source == "Heston"]
    for method, group in hedges.groupby("method", sort=False):
        group = group.sort_values("barrier")
        axes[1, 1].plot(
            (group.barrier / report.market.option.spot - 1) * 100,
            group.hedge_rmse,
            marker="o",
            label=method,
        )
    labels = [
        ("Barrier price", f"Premium ({currency} per BTC)"),
        ("Spot gamma; fixed calibration", f"Gamma (BTC² / {currency})"),
        ("Risk-neutral monitored touch probability", "Probability (fraction)"),
        ("Hedge error on Heston paths", f"RMSE of seller P&L ({currency} per BTC)"),
    ]
    for ax, (title, ylabel) in zip(axes.flat, labels, strict=True):
        ax.set_title(title)
        ax.set_xlabel("Barrier distance above initial spot (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    return fig
