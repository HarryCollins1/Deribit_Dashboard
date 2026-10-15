"""Chronological, out-of-sample option markouts on recorded full-chain snapshots.

The fit and quotes use only the signal snapshot. Future bid/ask observations evaluate
conditional frozen-vol repricing and hypothetical execution; they do not prove fills.
"""

from bisect import bisect_left
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from deribit_dashboard.models import milliseconds
from deribit_dashboard.pricing import MODELS, YEAR_SECONDS
from deribit_dashboard.relative_value import evaluate_relative_value
from deribit_dashboard.relative_value_execution import (
    DepthBudget,
    TradingEvaluationConfig,
    finish_trade,
    prepare_trade,
)
from deribit_dashboard.surface import Contract, chain_audit
from deribit_dashboard.workbench import (
    analyze,
    cash_currency_label,
    price_selection,
    quote_selection,
)


@dataclass(frozen=True)
class WalkForwardConfig:
    base: str = "BTC"
    settlement: str = "BTC"
    model: str = "Black-Scholes"
    rate: float = 0.0
    stale_seconds: float = 60.0
    max_relative_width: float = 1.0
    horizon_seconds: float = 60.0
    tolerance_seconds: float = 30.0
    contracts: float = 1.0
    fee_per_contract: float = 0.0
    max_depth_fraction: float = 1.0
    parity_estimator: str = "weighted_mean"

    def __post_init__(self):
        numbers = [
            self.rate,
            self.stale_seconds,
            self.max_relative_width,
            self.horizon_seconds,
            self.tolerance_seconds,
            self.contracts,
            self.fee_per_contract,
            self.max_depth_fraction,
        ]
        if self.base not in {"BTC", "ETH"} or self.model not in MODELS:
            raise ValueError("Choose BTC/ETH and a registered pricing model")
        if (
            not np.isfinite(numbers).all()
            or min(self.stale_seconds, self.tolerance_seconds, self.fee_per_contract) < 0
        ):
            raise ValueError(
                "Walk-forward controls must be finite; ages, tolerance and fees nonnegative"
            )
        if (
            min(
                self.max_relative_width,
                self.horizon_seconds,
                self.contracts,
                self.max_depth_fraction,
            )
            <= 0
            or self.max_depth_fraction > 1
        ):
            raise ValueError(
                "Use positive width, horizon, contracts and depth fraction at most one"
            )


@dataclass
class WalkForwardReport:
    observations: pd.DataFrame
    folds: pd.DataFrame
    coverage: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: dict


def select_horizon_capture(ordered, target_ms, tolerance_seconds):
    """Choose by capture cutoff alone, never quote quality or outcome."""
    index = bisect_left(ordered, target_ms, key=lambda s: milliseconds(s.captured_at))
    if index < len(ordered):
        capture = ordered[index]
        if milliseconds(capture.captured_at) <= target_ms + tolerance_seconds * 1000:
            return capture
    return None


def evaluate_walk_forward(snapshots, config=WalkForwardConfig()):
    """Each earlier capture trains once; select first capture at/after its target horizon.

    Selection uses capture times alone, never future prices or future fit quality. A
    later quote must itself be at/after the target and before its capture cutoff. No
    substituted later observation, interpolated quote or fabricated settlement is used.
    """
    if not 2 <= len(snapshots) <= 200:
        raise ValueError("Select 2–200 recorded snapshots")
    if len({s.environment for s in snapshots}) != 1:
        raise ValueError("Walk-forward inputs must share one environment")
    ordered = sorted(snapshots, key=lambda s: s.captured_at)
    if len({s.captured_at for s in ordered}) != len(ordered):
        raise ValueError(
            "One snapshot per cutoff is required; combine simultaneous currencies first"
        )
    if len({s.snapshot_id for s in ordered}) != len(ordered):
        raise ValueError("Snapshot IDs must be unique")
    units = {cash_currency_label(s, config.base, config.settlement) for s in ordered}
    if len(units) != 1 or "cash units" in units:
        raise ValueError("All snapshots require the same unambiguous cash denomination")
    cash_currency = next(iter(units))
    rows, folds = [], []
    future_audits = {}
    identity_fields = (
        "base_currency",
        "quote_currency",
        "counter_currency",
        "settlement_currency",
        "contract_size",
        "strike",
        "expiration_timestamp",
        "option_type",
    )
    for i, signal in enumerate(ordered[:-1]):
        target_ms = milliseconds(signal.captured_at) + config.horizon_seconds * 1000
        later = select_horizon_capture(ordered, target_ms, config.tolerance_seconds)
        fold = dict(
            signal_snapshot=signal.snapshot_id,
            signal_time=signal.captured_at,
            target_time=pd.to_datetime(target_ms, unit="ms", utc=True),
            evaluation_snapshot=later.snapshot_id if later else None,
        )
        names = [
            n
            for n, m in signal.instruments.items()
            if m.get("kind") == "option"
            and m.get("base_currency") == config.base
            and m.get("settlement_currency") == config.settlement
        ]
        if not names:
            folds.append(dict(**fold, status="no selected options", selected_quotes=0, evaluated=0))
            continue
        try:
            analysis = analyze(
                signal,
                config.base,
                config.settlement,
                config.rate,
                config.stale_seconds,
                config.max_relative_width,
                parity_estimator=config.parity_estimator,
            )
        except ValueError as exc:
            rows.extend(
                dict(**fold, instrument_name=n, status="training unavailable", reason=str(exc))
                for n in names
            )
            folds.append(
                dict(**fold, status="training unavailable", selected_quotes=len(names), evaluated=0)
            )
            continue
        fold["fit_error"] = analysis.fit_error
        eligible = analysis.chain[analysis.chain.fit_status == "fitted"].instrument_name.tolist()
        priced, pricing_errors = {}, {}
        for name in eligible:
            try:
                values = price_selection(analysis, signal, [name], config.model, rate=config.rate)
                if values:
                    priced[name] = values[0]
                else:
                    pricing_errors[name] = "Selected model did not return a price"
            except ValueError as exc:
                pricing_errors[name] = str(exc)
        if later is not None and later.snapshot_id not in future_audits:
            try:
                audited, _ = chain_audit(
                    later,
                    config.base,
                    config.settlement,
                    config.stale_seconds,
                    config.max_relative_width,
                )
                future_audits[later.snapshot_id] = audited.set_index("instrument_name")
            except ValueError:
                future_audits[later.snapshot_id] = pd.DataFrame()
        evaluated = 0
        for _, entry in analysis.chain.iterrows():
            name = entry.instrument_name
            row = dict(
                **fold,
                instrument_name=name,
                expiry=entry.expiry,
                side=entry.side,
                strike=entry.strike,
                cash_currency=cash_currency,
                requested_contracts=config.contracts,
                status="evaluated",
                reason="",
            )
            rows.append(row)
            if entry.fit_status != "fitted":
                if analysis.surface is None and entry.iv_status == "included":
                    row.update(
                        status="training fit unavailable",
                        reason=analysis.fit_error or "No fitted surface",
                    )
                else:
                    row.update(status="training excluded", reason=entry.fit_status)
                continue
            if name not in priced:
                row.update(status="model price unavailable", reason=pricing_errors[name])
                continue
            p = priced[name]
            contract = Contract.from_metadata(signal.instruments[name])
            try:
                quote = quote_selection(analysis, signal, p, config.horizon_seconds)
            except ValueError as exc:
                row.update(status="quote unavailable", reason=str(exc))
                continue
            row.update(
                entry_fair=quote["theoretical"],
                entry_bid=entry.bid,
                entry_ask=entry.ask,
                proposed_bid=quote["bid"],
                proposed_ask=quote["ask"],
                entry_volatility=p["model_volatility"],
                entry_spot=entry.spot,
                multiplier=contract.multiplier,
            )
            if later is None:
                row.update(
                    status="missing horizon capture",
                    reason="No capture in the requested forward horizon/tolerance",
                )
                continue
            audited = future_audits[later.snapshot_id]
            if name not in audited.index:
                row.update(
                    status="missing later instrument", reason="No matching quote or metadata"
                )
                continue
            if any(
                signal.instruments[name].get(f) != later.instruments[name].get(f)
                for f in identity_fields
            ):
                row.update(
                    status="metadata changed",
                    reason="Contract identity or units differ between captures",
                )
                continue
            observation = audited.loc[name]
            if not observation.accepted:
                row.update(status="later quote excluded", reason=", ".join(observation.reasons))
                continue
            stamp = later.tickers[name]["timestamp"]
            if stamp < target_ms or stamp > milliseconds(later.captured_at):
                row.update(
                    status="later timestamp outside horizon",
                    reason="Quote is before the target or after its capture cutoff",
                )
                continue
            elapsed = (stamp - milliseconds(signal.captured_at)) / 1000
            remaining = p["option"].years - elapsed / YEAR_SECONDS
            if remaining <= 0:
                row.update(
                    status="expiry settlement unavailable",
                    reason="An expired quote cannot supply an index settlement observation",
                )
                continue
            # Entry intent is frozen without looking at later liquidity. Later capacity
            # gates this full-size close; it never retroactively resizes the initial trade.
            entry_depth_contracts = max(0.0, entry.liquidity) / contract.multiplier
            size = min(config.contracts, entry_depth_contracts * config.max_depth_fraction)
            row.update(hypothetical_contracts=size, entry_depth_contracts=entry_depth_contracts)
            if size <= 0:
                row.update(
                    status="no entry displayed depth",
                    reason="Cannot size a liquidity-limited hypothetical order",
                )
                continue
            later_depth = max(0.0, observation.liquidity) / contract.multiplier
            if later_depth * config.max_depth_fraction + 1e-12 < size:
                row.update(
                    status="insufficient later displayed depth",
                    reason="Frozen entry size cannot be fully closed at selected depth cap",
                    later_depth_contracts=later_depth,
                )
                continue
            option = replace(p["option"], spot=float(observation.spot), years=remaining)
            conditional = MODELS[config.model].price(option)
            mid = float(observation.mid)
            amount = size * contract.multiplier
            fees = 2 * size * config.fee_per_contract
            signed_spot_change = float(observation.spot - entry.spot)
            delta_gain = p["cash_delta"] / contract.multiplier * signed_spot_change * amount
            row.update(
                elapsed_seconds=elapsed,
                evaluation_capture_elapsed_seconds=(
                    later.captured_at - signal.captured_at
                ).total_seconds(),
                evaluation_quote_age_seconds=(milliseconds(later.captured_at) - stamp) / 1000,
                evaluation_quote_time=pd.to_datetime(stamp, unit="ms", utc=True),
                later_spot=float(observation.spot),
                later_bid=float(observation.bid),
                later_ask=float(observation.ask),
                later_mid=mid,
                conditional_frozen_vol_value=conditional,
                conditional_repricing_error=conditional - mid,
                midpoint_change=mid - float(entry.mid),
                hypothetical_bid_markout=(mid - quote["bid"]) * amount,
                hypothetical_ask_markout=(quote["ask"] - mid) * amount,
                hypothetical_long_close_pnl=(float(observation.bid) - quote["bid"]) * amount - fees,
                hypothetical_short_close_pnl=(quote["ask"] - float(observation.ask)) * amount
                - fees,
                hypothetical_long_delta_hedged_pnl=(float(observation.bid) - quote["bid"]) * amount
                - fees
                - delta_gain,
                hypothetical_short_delta_hedged_pnl=(quote["ask"] - float(observation.ask)) * amount
                - fees
                + delta_gain,
                round_trip_fees=fees,
                later_depth_contracts=later_depth,
            )
            evaluated += 1
        folds.append(
            dict(
                **fold,
                status="evaluated" if evaluated else "no evaluated quotes",
                selected_quotes=len(analysis.chain),
                evaluated=evaluated,
            )
        )
    observations = pd.DataFrame(rows)
    if observations.empty:
        observations = pd.DataFrame(columns=["status", "instrument_name"])
    coverage = observations.status.value_counts().rename_axis("status").reset_index(name="quotes")
    coverage["fraction"] = coverage.quotes / len(observations) if len(observations) else 0.0
    successful = observations[observations.status == "evaluated"]
    metrics = dict(
        selected_observations=len(observations),
        evaluated_observations=len(successful),
        evaluated_fraction=len(successful) / len(observations) if len(observations) else 0.0,
    )
    if not successful.empty:
        metrics.update(
            conditional_repricing_rmse=float(
                np.sqrt(np.mean(successful.conditional_repricing_error**2))
            ),
            conditional_repricing_bias=float(successful.conditional_repricing_error.mean()),
            mean_hypothetical_long_close_pnl=float(successful.hypothetical_long_close_pnl.mean()),
            mean_hypothetical_short_close_pnl=float(successful.hypothetical_short_close_pnl.mean()),
        )
    diagnostics = dict(
        environment=ordered[0].environment,
        cash_currency=cash_currency,
        input_snapshots=len(ordered),
        training_folds=len(ordered) - 1,
        final_snapshot_role="evaluation only; no future capture is available to assess it as training",
        fit_information="Only the signal capture. Later captures are never fitted or used for selection/sizing.",
        evaluation="Conditional fixed-vol repricing uses the realized later spot; this is not an ex-ante spot forecast.",
        execution="Both maker-side entries are separate hypothetical scenarios, not observed fills. Later full-size closes require displayed depth.",
        hedge="A frictionless cash-index delta benchmark; no actual perpetual execution or funding is inferred.",
        statistics="Overlapping horizons and options are dependent observations; no independent-sample significance claim or portfolio Sharpe is reported.",
        missing="All selected quotes have an outcome. No future quote substitution or fabricated settlement.",
    )
    return WalkForwardReport(
        observations, pd.DataFrame(folds), coverage, pd.DataFrame([metrics]), diagnostics
    )


@dataclass
class TradingEvaluationReport:
    observations: pd.DataFrame
    trades: pd.DataFrame
    ledger: pd.DataFrame
    folds: pd.DataFrame
    coverage: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: dict
    fits: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    parity_pairs: pd.DataFrame = field(default_factory=pd.DataFrame)


def evaluate_trading(
    snapshots,
    config=TradingEvaluationConfig(),
    *,
    instrument_names=None,
    entry_snapshot_ids=None,
    signal_provider=None,
):
    """Evaluate independent signals with captured option and perpetual cash flows.

    All captures may produce entries, including the final capture. Missing future
    captures never suppress an entry: outstanding quantities and conservative marks
    remain visible. Explicit entry IDs can define a research window independently
    of outcome availability. Exit choice uses the same chronological horizon helper
    as the original markout evaluator, whose public API and defaults remain intact.
    An optional benchmark provider receives only the current signal snapshot and
    must return one observation per selected instrument at that exact cutoff.
    """
    from deribit_dashboard.models import Snapshot

    ordered = sorted(
        (Snapshot.from_dict(snapshot.to_dict()) for snapshot in snapshots),
        key=lambda snapshot: snapshot.captured_at,
    )
    if not ordered:
        raise ValueError("Select at least one recorded snapshot")
    if len({snapshot.environment for snapshot in ordered}) != 1:
        raise ValueError("Trading inputs must share one environment")
    if len({snapshot.captured_at for snapshot in ordered}) != len(ordered):
        raise ValueError("One snapshot per cutoff is required")
    if len({snapshot.snapshot_id for snapshot in ordered}) != len(ordered):
        raise ValueError("Snapshot IDs must be unique")
    input_ids = {snapshot.snapshot_id for snapshot in ordered}
    entry_ids = input_ids if entry_snapshot_ids is None else set(entry_snapshot_ids)
    if not entry_ids or entry_ids - input_ids:
        raise ValueError("Choose nonempty entry IDs from the captured snapshots")
    names = None if instrument_names is None else set(instrument_names)
    if names is not None and (not names or not all(isinstance(name, str) for name in names)):
        raise ValueError("Choose nonempty option instrument names")
    budget = DepthBudget(config.max_depth_fraction)
    rows, folds, trades, fits, calibration, pairs = [], [], [], [], [], []
    scheduled, missing = {}, []
    for snapshot in ordered:
        # Oldest trades close first; closes precede new entries at a shared cutoff.
        for trade in scheduled.pop(snapshot.snapshot_id, []):
            finish_trade(trade, ordered, snapshot, config, budget)
        if snapshot.snapshot_id not in entry_ids:
            continue
        candidates = {
            name
            for name, metadata in snapshot.instruments.items()
            if metadata.get("kind") == "option"
            and metadata.get("base_currency") == "BTC"
            and metadata.get("settlement_currency") == config.signal_config.settlement
        }
        selected = candidates if names is None else candidates & names
        for name in sorted((names or set()) - candidates):
            rows.append(
                dict(
                    signal_snapshot=snapshot.snapshot_id,
                    signal_time=snapshot.captured_at,
                    instrument_name=name,
                    signal="PASS",
                    status="selection unavailable",
                    reason="selected option not captured in this settlement at this cutoff",
                    trade_status="no trade",
                    execution="hypothetical",
                )
            )
        if not selected:
            folds.append(
                dict(
                    signal_snapshot=snapshot.snapshot_id,
                    signal_time=snapshot.captured_at,
                    status="no selected options",
                    selected_observations=len(names or []),
                )
            )
            continue
        provider = evaluate_relative_value if signal_provider is None else signal_provider
        report = provider(snapshot, config.signal_config, instrument_names=selected)
        if (
            set(report.signals.instrument_name) != selected
            or report.signals.instrument_name.duplicated().any()
            or not report.signals.signal_snapshot.eq(snapshot.snapshot_id).all()
            or not report.signals.signal_time.eq(snapshot.captured_at).all()
        ):
            raise ValueError(
                "Signal provider must preserve instruments and the exact signal cutoff"
            )
        fits.append(report.fits.assign(signal_snapshot=snapshot.snapshot_id))
        calibration.append(report.calibration.assign(signal_snapshot=snapshot.snapshot_id))
        pairs.append(report.parity_pairs.assign(signal_snapshot=snapshot.snapshot_id))
        fold = dict(
            signal_snapshot=snapshot.snapshot_id,
            signal_time=snapshot.captured_at,
            status="evaluated",
            selected_observations=len(report.signals),
            entered=0,
        )
        for signal in report.signals.to_dict("records"):
            signal.update(trade_status="no trade", execution="hypothetical")
            if signal["signal"] == "PASS":
                rows.append(signal)
                continue
            try:
                trade = prepare_trade(snapshot, signal, config, budget)
            except ValueError as exc:
                signal.update(trade_status="entry rejected", entry_reason=str(exc))
                rows.append(signal)
                continue
            # No exit data has entered prepare_trade. Exit allocation follows only
            # after sizes, entry prices, initial delta and ledger cash flows are frozen.
            rows.append(trade.row)
            trades.append(trade)
            fold["entered"] += 1
            later = select_horizon_capture(
                ordered, trade.row["target_ms"], config.tolerance_seconds
            )
            if later is None:
                missing.append(trade)
            else:
                scheduled.setdefault(later.snapshot_id, []).append(trade)
        folds.append(fold)
    for trade in missing:
        finish_trade(trade, ordered, None, config, budget)
    observations = pd.DataFrame(rows)
    if observations.empty:
        observations = pd.DataFrame(columns=["trade_status", "instrument_name", "signal"])
    trade_rows = pd.DataFrame([trade.row for trade in trades])
    ledger = pd.DataFrame([event for trade in trades for event in trade.ledger])
    if not ledger.empty:
        ledger = ledger.sort_values(["time", "trade_id", "event"], kind="stable").reset_index(
            drop=True
        )
    coverage = (
        observations.trade_status.value_counts()
        .rename_axis("trade_status")
        .reset_index(name="observations")
    )
    coverage["fraction"] = coverage.observations / len(observations) if len(observations) else 0.0
    metrics = []
    if not trade_rows.empty:
        for (native, cash), group in trade_rows.groupby(["native_currency", "cash_currency"]):
            metrics.append(
                dict(
                    native_currency=native,
                    cash_currency=cash,
                    hypothetical_trades=len(group),
                    closed_trades=int(group.trade_status.eq("closed hypothetically").sum()),
                    outstanding_trades=int(group.trade_status.eq("outstanding exposure").sum()),
                    conservative_net_pnl_native=float(group.net_pnl_native.sum()),
                    option_fees_native=float(group.option_fees_native.sum()),
                    hedge_fees_native=float(group.hedge_fees_native.sum()),
                    funding_gap_seconds=float(group.funding_gap_seconds.sum()),
                    maximum_reconciliation_error_native=float(
                        group.reconciliation_error_native.abs().max()
                    ),
                )
            )
    return TradingEvaluationReport(
        observations,
        trade_rows,
        ledger,
        pd.DataFrame(folds),
        coverage,
        pd.DataFrame(metrics),
        dict(
            environment=ordered[0].environment,
            input_snapshots=len(ordered),
            entry_snapshots=len(entry_ids),
            selected_observations=len(observations),
            hypothetical_trades=len(trades),
            execution="Hypothetical captured top-of-book executions, never actual fills or orders. Sizes use only entry-time quotes and rules; exits cannot resize them.",
            depth="Shared instrument/event/side budgets across trades. Closes precede new entries; older trades then instrument names determine priority. Repeated captures cannot replenish unchanged quote events.",
            hedge="One initial native-numeraire delta hedge using the selected captured BTC perpetual. No rehedging. Rounded lot residual delta is explicit; each leg closes independently to available depth.",
            cash="Incremental native-settlement cash ledger. BTC P&L converted at the marking index is indicative cash value, not an executed FX conversion. No implicit cross-settlement hedges.",
            funding="Left-held current_funding expressed per eight hours. Only already-captured observations reconstruct each following interval. Missing/expired rates use a separately disclosed adverse cost reserve; funding_8h is not a substitute.",
            marking="Remaining long options: zero. Short options: conservative upper bound. Remaining hedges: adverse as-of bid/ask stress and close-fee reserve. Stale references remain labelled; hedge marks are scenarios, not guaranteed worst-case loss bounds or fills.",
            missing="Missing exits, changed contracts, stale prices, short depth and expiry settlement gaps retain entered trades and outstanding exposure. No favorable later substitute. Final captures may enter and remain outstanding.",
            capital="Assumes sufficient native collateral to fund premiums and fees. No collateral FX P&L, borrowing charges, interest, margin or liquidation simulation. Not a live execution system.",
            statistics="Accounting and coverage only in Phase 3. No alpha, independent-sample significance, portfolio Sharpe or final-test inference is established.",
        ),
        fits=pd.concat(fits, ignore_index=True) if fits else pd.DataFrame(),
        calibration=pd.concat(calibration, ignore_index=True) if calibration else pd.DataFrame(),
        parity_pairs=pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(),
    )
