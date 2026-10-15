"""Chronological policy selection and held-out BTC relative-value research.

Thresholds are selected using validation cohorts only, then frozen before test
valuation or outcome evaluation. Missing exits and model failures stay visible.
"""

from dataclasses import dataclass, field, replace
from datetime import timedelta

import numpy as np
import pandas as pd

from deribit_dashboard.experiments import fingerprint
from deribit_dashboard.models import Snapshot, parse_time
from deribit_dashboard.relative_value import RelativeValueReport, evaluate_relative_value
from deribit_dashboard.relative_value_benchmarks import (
    STRATEGIES,
    classify_report,
    evaluate_benchmarks,
)
from deribit_dashboard.relative_value_execution import (
    DepthBudget,
    TradingEvaluationConfig,
    prepare_trade,
)
from deribit_dashboard.research_statistics import (
    downside_metrics,
    moving_block_interval,
    outcome_curve,
)
from deribit_dashboard.walk_forward import evaluate_trading


@dataclass(frozen=True)
class RelativeValueResearchConfig:
    training_end: object
    validation_end: object
    test_end: object
    trading_config: TradingEvaluationConfig = field(default_factory=TradingEvaluationConfig)
    threshold_grid_cash: tuple = (0.0, 10.0, 25.0, 50.0, 100.0)
    minimum_validation_trades: int = 5
    spread_factors: tuple = (1.0, 1.5, 2.0)
    strength_boundaries_cash: tuple = (0.0, 25.0, 100.0, 500.0)
    bootstrap_bin_seconds: float = 3600.0
    bootstrap_block_seconds: float = 86400.0
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    minimum_time_blocks: int = 5
    seed: int = 42

    def __post_init__(self):
        for key in ("training_end", "validation_end", "test_end"):
            object.__setattr__(self, key, parse_time(getattr(self, key)))
        if not self.training_end < self.validation_end < self.test_end:
            raise ValueError("Declare increasing chronological training/validation/test boundaries")
        if isinstance(self.trading_config, dict):
            object.__setattr__(
                self, "trading_config", TradingEvaluationConfig(**self.trading_config)
            )
        if not isinstance(self.trading_config, TradingEvaluationConfig):
            raise ValueError("A TradingEvaluationConfig is required")
        for key in ("threshold_grid_cash", "spread_factors", "strength_boundaries_cash"):
            values = tuple(sorted(set(getattr(self, key))))
            if not values or not np.isfinite(values).all():
                raise ValueError(
                    "Finite nonempty thresholds, spreads and signal-strength boundaries required"
                )
            object.__setattr__(self, key, values)
        if (
            min(self.threshold_grid_cash) < 0
            or min(self.strength_boundaries_cash) < 0
            or min(self.spread_factors) < 1
            or 1.0 not in self.spread_factors
        ):
            raise ValueError("Use nonnegative thresholds/bins and spread factors including 1")
        if (
            not np.isfinite(
                [self.bootstrap_bin_seconds, self.bootstrap_block_seconds, self.confidence]
            ).all()
            or self.bootstrap_bin_seconds <= 0
            or not 0 < self.confidence < 1
        ):
            raise ValueError("Invalid bootstrap clock bins or confidence")
        if (
            self.bootstrap_block_seconds
            < self.trading_config.horizon_seconds + self.trading_config.tolerance_seconds
        ):
            raise ValueError("Bootstrap block must span at least horizon plus tolerance")
        for key, lower in (
            ("minimum_validation_trades", 1),
            ("bootstrap_samples", 100),
            ("minimum_time_blocks", 2),
            ("seed", 0),
        ):
            value = getattr(self, key)
            if not isinstance(value, int) or value < lower:
                raise ValueError(f"Invalid {key}")


@dataclass
class RelativeValueResearchReport:
    protocol: pd.DataFrame
    selection: pd.DataFrame
    validation_scores: pd.DataFrame
    valuations: pd.DataFrame
    observations: pd.DataFrame
    trades: pd.DataFrame
    ledger: pd.DataFrame
    metrics: pd.DataFrame
    coverage: pd.DataFrame
    uncertainty: pd.DataFrame
    signal_strength: pd.DataFrame
    spread_sensitivity: pd.DataFrame
    outcome_curves: pd.DataFrame
    fits: pd.DataFrame
    calibration: pd.DataFrame
    parity_pairs: pd.DataFrame
    validation_trades: pd.DataFrame
    validation_observations: pd.DataFrame
    validation_ledger: pd.DataFrame
    diagnostics: dict


def _concat(frames):
    frames = list(frames)
    usable = [frame for frame in frames if not frame.empty]
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame(columns=columns)


def _period(snapshot, config):
    if snapshot.captured_at < config.training_end:
        return "training", config.training_end
    if snapshot.captured_at < config.validation_end:
        return "validation", config.validation_end
    return "test", config.test_end


def _empty_report(snapshot, names, config):
    records = [
        dict(
            signal_snapshot=snapshot.snapshot_id,
            signal_time=snapshot.captured_at,
            instrument_name=name,
            signal="PASS",
            status="quote excluded",
            reason="selected BTC option unavailable at this cutoff",
            cash_currency=None,
            native_currency=config.signal_config.settlement,
        )
        for name in sorted(names or [])
    ]
    columns = [
        "signal_snapshot",
        "signal_time",
        "instrument_name",
        "signal",
        "status",
        "reason",
        "cash_currency",
        "native_currency",
    ]
    return RelativeValueReport(
        pd.DataFrame(records, columns=columns), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    )


def _common_entry(snapshot, reports, config):
    common = {}
    by_model = {
        model: report.signals.set_index("instrument_name") for model, report in reports.items()
    }
    names = sorted(set().union(*(set(frame.index) for frame in by_model.values())))
    for name in names:
        reasons, capacities = [], []
        for model in STRATEGIES:
            frame = by_model[model]
            if name not in frame.index or frame.loc[name, "status"] != "valued":
                reason = frame.loc[name, "reason"] if name in frame.index else "missing observation"
                reasons.append(f"{model}: {reason}")
                continue
            # Probe both directions independently before any policy consumes depth.
            # The resulting common cap is available at signal time, never exit time.
            for side in ("BUY", "SELL"):
                signal = dict(frame.loc[name], instrument_name=name, signal=side)
                try:
                    trade = prepare_trade(
                        snapshot, signal, config, DepthBudget(config.max_depth_fraction)
                    )
                    capacities.append(abs(trade.option_quantity))
                except ValueError as exc:
                    reasons.append(f"{model} {side}: {exc}")
        common[name] = dict(
            common_eligible=not reasons,
            comparison_contract_cap=min(capacities) if not reasons else np.nan,
            comparison_reason="; ".join(reasons),
        )
    return common


def _prepare(snapshots, config, names, cache, protocol, anchor=None):
    previous_training = None
    for snapshot in snapshots:
        period, boundary = _period(snapshot, config)
        label_end = snapshot.captured_at + timedelta(
            seconds=config.trading_config.horizon_seconds + config.trading_config.tolerance_seconds
        )
        protocol_status = "eligible" if label_end < boundary else "purged boundary"
        candidates = {
            name
            for name, m in snapshot.instruments.items()
            if m.get("kind") == "option"
            and m.get("base_currency") == "BTC"
            and m.get("settlement_currency") == config.trading_config.signal_config.settlement
        }
        selected = candidates if names is None else candidates & names
        relative = (
            evaluate_relative_value(
                snapshot, config.trading_config.signal_config, instrument_names=selected
            )
            if selected
            else _empty_report(snapshot, names, config.trading_config)
        )
        frozen = previous_training if period == "training" else anchor
        frozen_report = (
            cache[frozen.snapshot_id]["reports"]["relative_value"] if frozen is not None else None
        )
        reports = evaluate_benchmarks(
            snapshot,
            config.trading_config.signal_config,
            relative_report=relative,
            frozen_snapshot=frozen,
            frozen_report=frozen_report,
        )
        if names is not None:
            absent = names - selected
            if absent and selected:
                empty = _empty_report(snapshot, absent, config.trading_config).signals
                reports = {
                    model: replace(
                        report, signals=pd.concat([report.signals, empty], ignore_index=True)
                    )
                    for model, report in reports.items()
                }
        common = _common_entry(snapshot, reports, config.trading_config)
        cache[snapshot.snapshot_id] = dict(
            reports=reports,
            common=common,
            period=period,
            protocol_status=protocol_status,
            requested_names=names,
        )
        protocol.append(
            dict(
                signal_snapshot=snapshot.snapshot_id,
                signal_time=snapshot.captured_at,
                period=period,
                boundary=boundary,
                prospective_label_end=label_end,
                protocol_status=protocol_status,
                selected_observations=len(common),
                common_observations=sum(item["common_eligible"] for item in common.values()),
                frozen_snapshot=frozen.snapshot_id if frozen else None,
            )
        )
        if period == "training":
            previous_training = snapshot


def _stress_snapshots(snapshots, factor):
    if factor == 1:
        return snapshots
    stressed = [Snapshot.from_dict(snapshot.to_dict()) for snapshot in snapshots]
    for snapshot in stressed:
        for name, quote in snapshot.tickers.items():
            metadata = snapshot.instruments[name]
            if metadata.get("base_currency") != "BTC" or not (
                metadata.get("kind") == "option" or metadata.get("settlement_period") == "perpetual"
            ):
                continue
            bid, ask = quote.get("best_bid_price"), quote.get("best_ask_price")
            if (
                isinstance(bid, (int, float))
                and isinstance(ask, (int, float))
                and np.isfinite([bid, ask]).all()
                and 0 < bid <= ask
            ):
                mid, half = (bid + ask) / 2, (ask - bid) * factor / 2
                quote["best_bid_price"], quote["best_ask_price"] = max(0.0, mid - half), mid + half
    return stressed


def _run_policy(snapshots, entry_ids, cache, model, threshold, config, factor=1.0):
    stressed = _stress_snapshots(snapshots, factor)

    def provider(snapshot, signal_config, *, instrument_names):
        prepared = cache[snapshot.snapshot_id]
        source = prepared["reports"][model]
        records = []
        for raw in source.signals.to_dict("records"):
            if raw["instrument_name"] not in instrument_names:
                continue
            row = dict(raw, model_status=raw["status"], model_reason=raw["reason"])
            if row["status"] == "valued":
                quote = snapshot.tickers[row["instrument_name"]]
                factor_cash = (
                    row["peer_reference_spot"]
                    if snapshot.instruments[row["instrument_name"]]["quote_currency"] == "BTC"
                    else 1.0
                )
                row.update(
                    market_bid_quote=quote["best_bid_price"],
                    market_ask_quote=quote["best_ask_price"],
                    market_bid_cash=quote["best_bid_price"] * factor_cash,
                    market_ask_cash=quote["best_ask_price"] * factor_cash,
                )
            records.append(row)
        report = classify_report(
            replace(source, signals=pd.DataFrame(records)), signal_config, threshold
        )
        rows = []
        for raw in report.signals.to_dict("records"):
            common = prepared["common"][raw["instrument_name"]]
            eligible = common["common_eligible"] and prepared["protocol_status"] == "eligible"
            row = dict(
                raw,
                **common,
                eligible_for_comparison=eligible,
                protocol_status=prepared["protocol_status"],
            )
            if not eligible:
                row.update(
                    signal="PASS",
                    status="comparison excluded",
                    reason=common["comparison_reason"] or prepared["protocol_status"],
                )
            rows.append(row)
        return replace(report, signals=pd.DataFrame(rows))

    report = evaluate_trading(
        stressed,
        config.trading_config,
        entry_snapshot_ids=entry_ids,
        instrument_names=cache[entry_ids[0]]["requested_names"],
        signal_provider=provider,
    )
    # Requested absent options were not in evaluate_trading's current universe.
    observed = report.observations.to_dict("records")
    present = {(row["signal_snapshot"], row["instrument_name"]) for row in observed}
    for snapshot_id in entry_ids:
        for raw in cache[snapshot_id]["reports"][model].signals.to_dict("records"):
            key = snapshot_id, raw["instrument_name"]
            if key not in present:
                observed.append(dict(raw, trade_status="no trade", execution="hypothetical"))
    rows = []
    for raw in observed:
        prepared = cache[raw["signal_snapshot"]]
        common = prepared["common"].get(
            raw["instrument_name"],
            dict(
                common_eligible=False,
                comparison_contract_cap=np.nan,
                comparison_reason="not captured",
            ),
        )
        eligible = common["common_eligible"] and prepared["protocol_status"] == "eligible"
        traded = raw.get("trade_status") in {"closed hypothetically", "outstanding exposure"}
        outcome = raw.get("net_pnl_native", np.nan) if traded else 0.0
        rows.append(
            dict(
                raw,
                **common,
                eligible_for_comparison=eligible,
                opportunity_pnl_native=outcome if eligible else np.nan,
                strategy=model,
                period=prepared["period"],
                spread_factor=factor,
                native_currency=config.trading_config.signal_config.settlement,
                protocol_status=prepared["protocol_status"],
            )
        )
    report.observations = pd.DataFrame(rows)
    report.trades = report.trades.assign(
        strategy=model, period=cache[entry_ids[0]]["period"], spread_factor=factor
    )
    report.ledger = report.ledger.assign(
        strategy=model, period=cache[entry_ids[0]]["period"], spread_factor=factor
    )
    return report


def _summarize(observations, trades):
    rows, curves = [], []
    if observations.empty:
        return pd.DataFrame(), pd.DataFrame()
    for keys, group in observations.groupby(
        ["period", "strategy", "spread_factor", "native_currency"], dropna=False
    ):
        period, model, factor, native = keys
        eligible = group[group.eligible_for_comparison]
        selected_trades = (
            trades[
                (trades.strategy == model)
                & (trades.period == period)
                & (trades.spread_factor == factor)
            ]
            if not trades.empty
            else trades
        )
        pnl = eligible.opportunity_pnl_native.to_numpy(float)
        complete = np.isfinite(pnl).all()
        net = float(pnl.sum()) if complete and len(pnl) else np.nan
        curve = outcome_curve(selected_trades)
        if not curve.empty:
            curves.append(
                curve.assign(
                    period=period, strategy=model, spread_factor=factor, native_currency=native
                )
            )
        costs = {
            column: float(selected_trades[column].sum()) if not selected_trades.empty else 0.0
            for column in (
                "option_fees_native",
                "hedge_fees_native",
                "captured_rate_funding_paid_native",
                "funding_gap_reserve_native",
                "residual_close_fee_reserve_native",
                "spread_cost_native",
                "option_premium_turnover_native",
                "hedge_notional_turnover_native",
            )
        }
        rows.append(
            dict(
                period=period,
                strategy=model,
                spread_factor=factor,
                native_currency=native,
                selected_observations=len(group),
                common_observations=len(eligible),
                coverage=len(eligible) / len(group) if len(group) else 0,
                exclusions=len(group) - len(eligible),
                hypothetical_trades=len(selected_trades),
                no_position_observations=len(eligible) - len(selected_trades),
                closed_trades=int(selected_trades.trade_status.eq("closed hypothetically").sum())
                if not selected_trades.empty
                else 0,
                outstanding_trades=int(
                    selected_trades.trade_status.eq("outstanding exposure").sum()
                )
                if not selected_trades.empty
                else 0,
                funding_gap_seconds=float(selected_trades.funding_gap_seconds.sum())
                if not selected_trades.empty
                else 0,
                net_pnl_native=net,
                mean_pnl_per_common_observation=net / len(pnl) if len(pnl) else np.nan,
                max_outcome_drawdown_native=(
                    -float(curve.drawdown_native.min()) if complete else np.nan
                )
                if not curve.empty
                else (0.0 if len(pnl) else np.nan),
                **costs,
                **downside_metrics(pnl),
            )
        )
    return pd.DataFrame(rows), _concat(curves)


def _interval_context(observations):
    outstanding = (
        int(observations.trade_status.eq("outstanding exposure").sum())
        if not observations.empty
        else 0
    )
    gaps = (
        float(observations.get("funding_gap_seconds", pd.Series(dtype=float)).sum())
        if not observations.empty
        else 0.0
    )
    return dict(
        outstanding_trades=outstanding,
        funding_gap_seconds=gaps,
        outcome_basis="includes conservative scenario marks on outstanding exposure"
        if outstanding
        else "closed hypothetical outcomes and no-position zeros",
    )


def evaluate_research(snapshots, config, *, instrument_names=None):
    """Select on validation only, then assess fixed policies on an untouched test period."""
    ordered = sorted(
        (Snapshot.from_dict(s.to_dict()) for s in snapshots), key=lambda s: s.captured_at
    )
    if not ordered or len({s.environment for s in ordered}) != 1:
        raise ValueError("Research needs snapshots from one environment")
    if len({s.snapshot_id for s in ordered}) != len(ordered) or len(
        {s.captured_at for s in ordered}
    ) != len(ordered):
        raise ValueError("Research capture IDs and cutoffs must be unique")
    ordered = [snapshot for snapshot in ordered if snapshot.captured_at <= config.test_end]
    periods = {
        period: [s for s in ordered if _period(s, config)[0] == period]
        for period in ("training", "validation", "test")
    }
    if any(not group for group in periods.values()):
        raise ValueError("Recorded inputs must contain training, validation and test captures")
    names = None if instrument_names is None else set(instrument_names)
    if names is not None and (not names or not all(isinstance(name, str) for name in names)):
        raise ValueError("Choose nonempty option instrument names")
    pretest = periods["training"] + periods["validation"]
    anchor, cache, protocol = periods["training"][-1], {}, []
    _prepare(periods["training"], config, names, cache, protocol)
    _prepare(periods["validation"], config, names, cache, protocol, anchor=anchor)
    validation_ids = [snapshot.snapshot_id for snapshot in periods["validation"]]
    scores, selections, validation_reports = [], [], []
    for model in STRATEGIES:
        candidates = []
        for threshold in (*config.threshold_grid_cash, None):
            report = _run_policy(pretest, validation_ids, cache, model, threshold, config)
            eligible = (
                report.observations[report.observations.eligible_for_comparison]
                if not report.observations.empty
                else pd.DataFrame()
            )
            trades = report.trades
            outstanding = (
                int(trades.trade_status.eq("outstanding exposure").sum()) if not trades.empty else 0
            )
            funding_gaps = float(trades.funding_gap_seconds.sum()) if not trades.empty else 0
            outcome = (
                float(eligible.opportunity_pnl_native.sum())
                if len(eligible) and eligible.opportunity_pnl_native.notna().all()
                else np.nan
            )
            valid = (
                len(eligible) > 0
                and np.isfinite(outcome)
                and (
                    threshold is None
                    or (
                        len(trades) >= config.minimum_validation_trades
                        and outstanding == 0
                        and funding_gaps == 0
                    )
                )
            )
            score = dict(
                strategy=model,
                threshold_cash=threshold,
                no_trade_policy=threshold is None,
                common_observations=len(eligible),
                hypothetical_trades=len(trades),
                outstanding_trades=outstanding,
                funding_gap_seconds=funding_gaps,
                validation_net_pnl_native=outcome,
                selection_eligible=valid,
                reason="eligible"
                if valid
                else "insufficient common observations/trades, unresolved exits or funding gaps",
            )
            scores.append(score)
            candidates.append(score)
            report.observations["candidate_threshold_cash"] = threshold
            report.trades["candidate_threshold_cash"] = threshold
            report.ledger["candidate_threshold_cash"] = threshold
            validation_reports.append(report)
        valid = [row for row in candidates if row["selection_eligible"]]
        best = (
            max(
                valid,
                key=lambda row: (
                    row["validation_net_pnl_native"],
                    row["no_trade_policy"],
                    row["threshold_cash"] or 0,
                ),
            )
            if valid
            else candidates[-1]
        )
        selections.append(
            dict(
                strategy=model,
                threshold_cash=best["threshold_cash"],
                no_trade_policy=best["no_trade_policy"],
                selection_status="selected" if valid else "insufficient validation; abstain",
                validation_net_pnl_native=best["validation_net_pnl_native"],
            )
        )
    selection = pd.DataFrame(selections)
    selection_record = dict(
        config=config,
        training_inputs=[fingerprint(s.to_dict()) for s in periods["training"]],
        validation_inputs=[fingerprint(s.to_dict()) for s in periods["validation"]],
        validation_scores=scores,
        selections=selections,
        frozen_snapshot=anchor.snapshot_id,
        instrument_names=sorted(names) if names is not None else None,
    )
    selection_hash = fingerprint(selection_record)
    # The first test valuation occurs only after selection has been irreversibly
    # recorded in this run. Test data never enter the score table or selection hash.
    _prepare(periods["test"], config, names, cache, protocol, anchor=anchor)
    reports = []
    for period in ("training", "validation", "test"):
        ids = [snapshot.snapshot_id for snapshot in periods[period]]
        inputs = pretest if period != "test" else periods["test"]
        for model, chosen in zip(STRATEGIES, selections, strict=True):
            # Training is descriptive with a declared fixed threshold; never apply
            # a later validation-selected threshold to earlier entry decisions.
            threshold = (
                config.trading_config.signal_config.minimum_edge_cash
                if period == "training"
                else chosen["threshold_cash"]
            )
            for factor in config.spread_factors if period == "test" else (1.0,):
                reports.append(_run_policy(inputs, ids, cache, model, threshold, config, factor))
    observations, trades, ledger = (
        _concat([getattr(report, key) for report in reports])
        for key in ("observations", "trades", "ledger")
    )
    metrics, curves = _summarize(observations, trades)
    uncertainty, strengths = [], []
    # Three fixed strategy means and two planned paired comparisons form one family.
    family_confidence = 1 - (1 - config.confidence) / 5
    bootstrap = dict(
        bin_seconds=config.bootstrap_bin_seconds,
        block_seconds=config.bootstrap_block_seconds,
        samples=config.bootstrap_samples,
        confidence=family_confidence,
        seed=config.seed,
        minimum_blocks=config.minimum_time_blocks,
    )
    base_test = (
        observations[
            (observations.period == "test")
            & (observations.spread_factor == 1)
            & observations.eligible_for_comparison
        ]
        if not observations.empty
        else pd.DataFrame()
    )
    for model in STRATEGIES:
        group = base_test[base_test.strategy == model] if not base_test.empty else base_test
        interval = moving_block_interval(
            group.signal_time if not group.empty else [],
            group.opportunity_pnl_native if not group.empty else [],
            **bootstrap,
        )
        uncertainty.append(
            dict(
                comparison=model,
                native_currency=config.trading_config.signal_config.settlement,
                **_interval_context(group),
                **interval,
            )
        )
    keys = ["signal_snapshot", "instrument_name"]
    for benchmark in STRATEGIES[1:]:
        left = (
            base_test[base_test.strategy == "relative_value"] if not base_test.empty else base_test
        )
        right = base_test[base_test.strategy == benchmark] if not base_test.empty else base_test
        if left.empty or right.empty:
            interval = moving_block_interval([], [], **bootstrap)
        else:
            joined = left[keys + ["signal_time", "opportunity_pnl_native"]].merge(
                right[keys + ["opportunity_pnl_native"]],
                on=keys,
                how="outer",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            interval = moving_block_interval(
                joined.signal_time,
                joined.opportunity_pnl_native_left - joined.opportunity_pnl_native_right,
                **bootstrap,
            )
        uncertainty.append(
            dict(
                comparison=f"relative_value_minus_{benchmark}",
                native_currency=config.trading_config.signal_config.settlement,
                **_interval_context(_concat([left, right])),
                **interval,
            )
        )
    if not base_test.empty:
        bins = [-np.inf, *config.strength_boundaries_cash, np.inf]
        strength_data = base_test.assign(
            strength_bucket=pd.cut(base_test.signal_strength_cash, bins=bins).astype(str)
        )
        for (model, bucket), group in strength_data.groupby(["strategy", "strength_bucket"]):
            entered = group.trade_status.isin(["closed hypothetically", "outstanding exposure"])
            strengths.append(
                dict(
                    strategy=model,
                    strength_bucket=bucket,
                    common_observations=len(group),
                    hypothetical_trades=int(entered.sum()),
                    net_pnl_native=float(group.opportunity_pnl_native.sum())
                    if np.isfinite(group.opportunity_pnl_native).all()
                    else np.nan,
                    mean_pnl_per_common_observation=float(group.opportunity_pnl_native.mean())
                    if np.isfinite(group.opportunity_pnl_native).all()
                    else np.nan,
                )
            )
    valuations, fits, calibration, pairs = [], [], [], []
    for snapshot_id, prepared in cache.items():
        for model, report in prepared["reports"].items():
            raw = report.signals.copy()
            raw["strategy"], raw["period"] = model, prepared["period"]
            valuations.append(raw)
            for key, destination in (
                ("fits", fits),
                ("calibration", calibration),
                ("parity_pairs", pairs),
            ):
                destination.append(
                    getattr(report, key).assign(
                        strategy=model, period=prepared["period"], signal_snapshot=snapshot_id
                    )
                )
    coverage = (
        observations.groupby(
            ["period", "strategy", "spread_factor", "trade_status", "eligible_for_comparison"],
            dropna=False,
        )
        .size()
        .reset_index(name="observations")
        if not observations.empty
        else pd.DataFrame()
    )
    sensitivity = metrics[metrics.period == "test"].copy() if not metrics.empty else metrics
    return RelativeValueResearchReport(
        pd.DataFrame(protocol),
        selection,
        pd.DataFrame(scores),
        _concat(valuations),
        observations,
        trades,
        ledger,
        metrics,
        coverage,
        pd.DataFrame(uncertainty),
        pd.DataFrame(strengths),
        sensitivity,
        curves,
        _concat(fits),
        _concat(calibration),
        _concat(pairs),
        _concat([report.trades for report in validation_reports]),
        _concat([report.observations for report in validation_reports]),
        _concat([report.ledger for report in validation_reports]),
        dict(
            selection_fingerprint=selection_hash,
            selection_record=selection_record,
            environment=ordered[0].environment,
            sources=sorted({snapshot.source for snapshot in ordered}),
            frozen_training_snapshot=anchor.snapshot_id,
            alpha_claim=False,
            evidence_status="Research accounting only; assess held-out intervals, unresolved exposure, funding coverage and production provenance before any alpha inference.",
            protocol="Chronological boundaries declared in advance; purge entries whose horizon plus tolerance reaches the next boundary. Training uses the declared fixed signal threshold and prior training captures. Validation selects thresholds including abstention; test valuations/outcomes follow the frozen selection. Validation performance is selection-period, not held-out evidence.",
            comparison="Common entry-time valuations and feasible sizes for both directions across all models. Shared prospective option cap; each policy retains its own hedge and depth-budget effects. No exit-success conditioning; PASS/no-position opportunities are zero, exclusions and unknown outcomes are not zero.",
            benchmarks="Flat Black-Scholes uses median peer IV with equal strike weighting. Frozen smile holds excluded-strike training volatility and carry fixed at absolute strike, decaying maturity with the current peer index. Last training cutoff chosen by time alone; no successful-fit replacement.",
            uncertainty="Moving consecutive clock-bin blocks contain entire option cross-sections and overlapping outcomes. Empty elapsed bins stay. Five planned means/differences use Bonferroni confidence allocation. Intervals assume weak dependence/stationarity and enough populated time blocks; no IID option/trade significance or annualized Sharpe.",
            downside="Outcome-time cumulative P&L drawdown, not continuously marked account equity. Downside metrics include PASS zeros on the common opportunity universe.",
            spreads="Execution-only widening around captured midpoints, with frozen valuations, uncertainty, thresholds and baseline common universe. Recompute entry edge, captured-side execution and depth effects. No retuning on stressed test results.",
            selection="Active validation candidates require minimum trades, fully closed exposure and no funding gaps. All rejected candidates and their conservative P&L remain in diagnostics; insufficient validation selects abstention.",
            limitations="Hypothetical executions, approximate funding, scenario marks on unresolved exposure and native-currency P&L. Synthetic/testnet results cannot establish empirical production alpha. No margin, borrowing, liquidation or collateral FX model.",
        ),
    )
