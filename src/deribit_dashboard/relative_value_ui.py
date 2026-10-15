"""Recorded-input BTC relative-value research; no live orders or fill assertions."""

import json
from dataclasses import asdict
from datetime import timedelta

import pandas as pd
import streamlit as st

from deribit_dashboard.experiment_ui import render_save_experiment
from deribit_dashboard.experiments import (
    ExperimentStore,
    fingerprint,
    json_value,
    provenance,
    result_tables,
)
from deribit_dashboard.models import utc_now
from deribit_dashboard.recording_studies import RecordingStudyStore, evaluate_study, study_progress
from deribit_dashboard.relative_value import RelativeValueConfig
from deribit_dashboard.relative_value_execution import TradingEvaluationConfig
from deribit_dashboard.relative_value_research import (
    RelativeValueResearchConfig,
    RelativeValueResearchReport,
    evaluate_research,
)
from deribit_dashboard.storage import SnapshotStore


@st.cache_data(ttl=60, max_entries=3, show_spinner=False)
def load_inputs(database_url, ids):
    store = SnapshotStore(database_url)
    try:
        return sorted((store.load(i) for i in ids), key=lambda snapshot: snapshot.captured_at)
    finally:
        store.close()


@st.cache_data(ttl=600, max_entries=3, show_spinner=False)
def cached_research(snapshots, config, instrument_names, code_fingerprint):
    # Code/dependency provenance participates in the cache key as well as input data.
    return evaluate_research(snapshots, config, instrument_names=instrument_names)


def numbers(text):
    """Parse declared comma-separated numeric grids; configuration validates their bounds."""
    try:
        return tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(
            "Use comma-separated numbers for threshold, spread and strength grids"
        ) from exc


@st.cache_data(ttl=600, max_entries=3, show_spinner=False)
def saved_study_report(database_url, run_id):
    store = SnapshotStore(database_url)
    try:
        payload = ExperimentStore(store.engine).load(run_id)
        return RelativeValueResearchReport(
            **result_tables(payload), diagnostics=payload["diagnostics"]
        )
    finally:
        store.close()


def render_declared_study(store, studies, rows):
    labels = {
        row["digest"]: f"{row['plan']['created_at']} · {row['plan'].get('purpose', 'research')}"
        for row in rows
    }
    digest = st.selectbox(
        "Declared production study", list(labels), format_func=labels.get, key="rv_study"
    )
    row = studies.load(digest)
    plan = row["plan"]
    config = RelativeValueResearchConfig(**plan["research_config"])
    st.subheader("Fixed production protocol")
    st.info(plan["study"])
    st.caption(
        f"{len(plan['candidate_names'])} fixed candidate options · {plan['interval_seconds']}-second captures · "
        f"{config.trading_config.horizon_seconds / 60:g}-minute horizon · plan {digest}"
    )
    st.dataframe(study_progress(store, plan), hide_index=True)
    st.button("Refresh study progress", key="rv_study_refresh")
    with st.expander("Declared candidates and trading controls"):
        st.write(plan["candidate_names"])
        st.json(plan["research_config"])
    st.download_button(
        "Download immutable study plan",
        json.dumps(plan, indent=2),
        "production_study_plan.json",
        "application/json",
        key="rv_study_plan",
    )
    if utc_now() < config.test_end:
        st.info(
            f"Final results are locked until {config.test_end.isoformat()}. This view shows capture counts only; "
            "collection stops at the fixed deadline regardless of trades or profit."
        )
        return
    if row["run_id"] is None:
        st.info(
            "The fixed test period has ended. The recorder saves results automatically after evaluation completes."
        )
        if st.button("Evaluate completed declared study", key="rv_study_evaluate"):
            try:
                with st.spinner("Evaluating the fixed protocol and saving its complete results…"):
                    payload = evaluate_study(store, digest)
                row = studies.load(digest)
                st.success(f"Saved held-out experiment {payload['run_id']}.")
            except (ValueError, KeyError) as exc:
                st.error(f"Cannot evaluate this study: {exc}")
        if row["run_id"] is None:
            return
    st.caption(
        f"Saved experiment: {row['run_id']}. Open Research experiments for export and replay."
    )
    render_report(saved_study_report(str(store.engine.url), row["run_id"]))


def render_relative_value(store, settings):
    st.header("Relative-Value Trading Evaluation")
    st.caption(
        "Research hypothesis: independently valued BTC vanilla-option mispricings predict "
        "positive held-out delta-hedged P&L after costs. All executions are hypothetical."
    )
    st.info(
        "Declare chronological periods and controls before examining test outcomes. "
        "Validation selects thresholds, including abstention; the final test uses frozen choices. "
        "This view makes no alpha claim."
    )
    studies = RecordingStudyStore(store.engine)
    studies.initialize()
    declared = studies.list()
    if declared:
        workflow = st.selectbox(
            "Evaluation workflow",
            ["Declared production study", "Manual recorded-data evaluation"],
            key="rv_workflow",
        )
        if workflow == "Declared production study":
            render_declared_study(store, studies, declared)
            return
        st.caption(
            "Manual period choices produce a separate experiment, not the declared production study result."
        )
    environment = st.selectbox(
        "Trading research snapshot environment",
        [settings.environment],
        key="rv_environment",
    )
    records = store.list(environment, None, limit=10000)
    if len(records) < 3:
        st.info(
            "Select at least three recorded full BTC chains spanning training, validation and test. Record the BTC perpetual alongside options; candles cannot reconstruct option books."
        )
        return
    if len(records) == 10000:
        st.caption(
            "The selector lists the latest 10,000 captures. Use explicit snapshot IDs in the CLI for a different recorded range; no selected capture is silently truncated."
        )
    labels = {
        row[
            "snapshot_id"
        ]: f"{row['captured_at']} · {row['currency']} · {row['ticker_count']} quotes"
        for row in reversed(records)
    }
    selected = st.multiselect(
        "Research input snapshots (3–10,000)",
        list(labels),
        format_func=labels.get,
        key="rv_snapshots",
    )
    if len(selected) < 3:
        return
    snapshots = load_inputs(str(store.engine.url), tuple(sorted(selected)))
    first, last = snapshots[0].captured_at, snapshots[-1].captured_at
    sources = sorted({snapshot.source for snapshot in snapshots})
    st.caption(
        f"{environment.upper()} · {', '.join(sources)} · {len(snapshots)} captures · {first.isoformat()} through {last.isoformat()}"
    )
    if environment != "production" or any("synthetic" in source.lower() for source in sources):
        st.warning("Synthetic/testnet accounting cannot establish empirical production alpha.")
    option_metadata = {
        name: metadata
        for snapshot in snapshots
        for name, metadata in snapshot.instruments.items()
        if metadata.get("kind") == "option" and metadata.get("base_currency") == "BTC"
    }
    settlements = sorted(
        {
            m.get("settlement_currency")
            for m in option_metadata.values()
            if m.get("settlement_currency")
        }
    )
    if not settlements:
        st.info(
            "These captures contain no BTC vanilla-option metadata. Select complete BTC chains."
        )
        return
    settlement = st.selectbox(
        "Trading research native settlement",
        settlements,
        index=settlements.index("BTC") if "BTC" in settlements else 0,
        key="rv_settlement",
    )
    candidates = sorted(
        name for name, m in option_metadata.items() if m.get("settlement_currency") == settlement
    )
    perpetuals = sorted(
        {
            name
            for snapshot in snapshots
            for name, m in snapshot.instruments.items()
            if m.get("kind") == "future"
            and m.get("base_currency") == "BTC"
            and m.get("settlement_period") == "perpetual"
            and m.get("settlement_currency") == settlement
        }
    )
    span = last - first
    with st.form("relative_value_controls"):
        st.subheader("Declared periods and candidate universe")
        st.caption(
            "Suggested boundaries use elapsed clock time only. Final captures may remain outstanding: the suggested test end extends one hour beyond the last capture."
        )
        training_end = st.text_input(
            "Training end (UTC, exclusive)", (first + span * 0.5).isoformat(), key="rv_training_end"
        )
        validation_end = st.text_input(
            "Validation end (UTC, exclusive)",
            (first + span * 0.75).isoformat(),
            key="rv_validation_end",
        )
        test_end = st.text_input(
            "Test end (UTC, inclusive)",
            (last + timedelta(seconds=3631)).isoformat(),
            key="rv_test_end",
        )
        names = st.multiselect(
            "Candidate BTC vanilla options (empty = all)", candidates, key="rv_candidates"
        )
        hedge = st.selectbox(
            "Initial BTC perpetual hedge", perpetuals or ["BTC-PERPETUAL"], key="rv_perpetual"
        )
        horizon = st.number_input(
            "Trading horizon (seconds)", min_value=1.0, value=3600.0, key="rv_horizon"
        )
        tolerance = st.number_input(
            "Exit capture tolerance (seconds)", min_value=0.0, value=30.0, key="rv_tolerance"
        )
        st.subheader("Signals and validation selection")
        thresholds = st.text_input(
            "Validation minimum-edge grid (cash per BTC)", "0,10,25,50,100", key="rv_thresholds"
        )
        minimum_trades = st.number_input(
            "Minimum closed validation trades", min_value=1, value=5, key="rv_min_trades"
        )
        cost_cash = st.number_input(
            "Estimated round-trip costs (cash per BTC)",
            min_value=0.0,
            value=25.0,
            key="rv_cost_cash",
        )
        uncertainty = st.number_input(
            "Model uncertainty multiplier", min_value=0.0, value=1.0, key="rv_uncertainty"
        )
        uncertainty_floor = st.number_input(
            "Model uncertainty floor (cash per BTC)",
            min_value=0.0,
            value=10.0,
            key="rv_uncertainty_floor",
        )
        st.caption(
            "Cash means the option's captured counter/quote denomination. These cost and model-risk buffers are research assumptions; the uncertainty buffer is not a statistical confidence interval. Execution fees and funding are charged separately in the ledger."
        )
        with st.expander("Execution, funding and statistical assumptions"):
            contracts = st.number_input(
                "Requested option contracts", min_value=0.01, value=1.0, key="rv_contracts"
            )
            depth = st.slider("Maximum captured depth fraction", 0.01, 1.0, 0.25, key="rv_depth")
            option_fee = st.number_input(
                "Option execution fee (bps of underlying notional)",
                min_value=0.0,
                value=3.0,
                key="rv_option_fee",
            )
            fee_cap = st.number_input(
                "Option fee cap (fraction of premium)", min_value=0.0, value=0.125, key="rv_fee_cap"
            )
            hedge_fee = st.number_input(
                "Perpetual execution fee (bps)", min_value=0.0, value=5.0, key="rv_hedge_fee"
            )
            funding_age = st.number_input(
                "Captured funding-rate validity (seconds)",
                min_value=0.0,
                value=300.0,
                key="rv_funding_age",
            )
            funding_reserve = st.number_input(
                "Missing funding adverse reserve (bps per eight hours)",
                min_value=0.0,
                value=10.0,
                key="rv_funding_reserve",
            )
            hedge_stress = st.number_input(
                "Outstanding hedge adverse mark (bps)",
                min_value=0.0,
                max_value=9999.0,
                value=100.0,
                key="rv_hedge_stress",
            )
            st.caption(
                "Fee defaults are configurable cost scenarios. Funding uses left-held captured current rates. Missing rates accrue an adverse reserve; unclosed positions retain conservative scenario marks rather than fabricated exits. Native collateral, borrowing, margin and liquidation are not simulated."
            )
            age = st.number_input(
                "Trading quote age limit (seconds)", min_value=0.0, value=60.0, key="rv_quote_age"
            )
            gap = st.number_input(
                "Option/perpetual timestamp gap limit (seconds)",
                min_value=0.0,
                value=5.0,
                key="rv_leg_gap",
            )
            width = st.number_input(
                "Trading maximum spread / midpoint", min_value=0.01, value=1.0, key="rv_width"
            )
            rate = st.number_input("Cash discount rate (annual decimal)", value=0.0, key="rv_rate")
            estimator = st.selectbox(
                "Trading peer-forward estimator",
                ["weighted_median", "huber", "weighted_mean"],
                key="rv_estimator",
            )
            cost_bps = st.number_input(
                "Additional estimated round-trip cost (bps of underlying)",
                min_value=0.0,
                value=0.0,
                key="rv_cost_bps",
            )
            train_edge = st.number_input(
                "Fixed training minimum edge (cash per BTC)",
                min_value=0.0,
                value=0.0,
                key="rv_train_edge",
            )
            spreads = st.text_input("Test spread multipliers", "1,1.5,2", key="rv_spreads")
            strengths = st.text_input(
                "Fixed signal-strength boundaries (cash per BTC)",
                "0,25,100,500",
                key="rv_strengths",
            )
            bin_seconds = st.number_input(
                "Bootstrap clock bin (seconds)", min_value=1.0, value=3600.0, key="rv_bin"
            )
            block_seconds = st.number_input(
                "Bootstrap moving block (seconds)", min_value=1.0, value=86400.0, key="rv_block"
            )
            samples = st.number_input(
                "Bootstrap samples", min_value=100, value=2000, key="rv_samples"
            )
            blocks = st.number_input(
                "Minimum populated time blocks", min_value=2, value=5, key="rv_blocks"
            )
            confidence = st.number_input(
                "Planned interval family confidence",
                min_value=0.01,
                max_value=0.99,
                value=0.95,
                key="rv_confidence",
            )
            seed = st.number_input("Research resampling seed", min_value=0, value=42, key="rv_seed")
            st.caption(
                "Whole clock-bin cross-sections are resampled in consecutive blocks, spanning at least horizon plus tolerance. Five planned intervals share the confidence allocation. Sparse data produces unavailable intervals; block duration and stationarity remain assumptions."
            )
        submitted = st.form_submit_button("Run relative-value trading evaluation")
    try:
        config = RelativeValueResearchConfig(
            training_end,
            validation_end,
            test_end,
            trading_config=TradingEvaluationConfig(
                signal_config=RelativeValueConfig(
                    settlement=settlement,
                    rate=rate,
                    stale_seconds=age,
                    max_relative_width=width,
                    parity_estimator=estimator,
                    estimated_round_trip_cost_cash=cost_cash,
                    estimated_round_trip_cost_bps=cost_bps,
                    minimum_edge_cash=train_edge,
                    uncertainty_multiplier=uncertainty,
                    uncertainty_floor_cash=uncertainty_floor,
                ),
                perpetual_instrument=hedge,
                horizon_seconds=horizon,
                tolerance_seconds=tolerance,
                contracts=contracts,
                max_depth_fraction=depth,
                max_leg_timestamp_gap_seconds=gap,
                option_fee_bps=option_fee,
                option_fee_premium_cap=fee_cap,
                perpetual_fee_bps=hedge_fee,
                funding_max_age_seconds=funding_age,
                missing_funding_rate_8h_bps=funding_reserve,
                residual_hedge_adverse_bps=hedge_stress,
            ),
            threshold_grid_cash=numbers(thresholds),
            minimum_validation_trades=minimum_trades,
            spread_factors=numbers(spreads),
            strength_boundaries_cash=numbers(strengths),
            bootstrap_bin_seconds=bin_seconds,
            bootstrap_block_seconds=block_seconds,
            bootstrap_samples=samples,
            minimum_time_blocks=blocks,
            confidence=confidence,
            seed=seed,
        )
    except (ValueError, TypeError, AttributeError) as exc:
        st.error(f"Invalid research controls: {exc}")
        return
    names = tuple(sorted(names)) if names else None
    code = fingerprint(provenance())
    signature = fingerprint(
        dict(
            inputs=[snapshot.to_dict() for snapshot in snapshots],
            config=config,
            instrument_names=names,
            code=code,
        )
    )
    st.download_button(
        "Download declared research configuration",
        json.dumps(json_value(asdict(config)), indent=2, allow_nan=False),
        "relative_value_config.json",
        "application/json",
        key="rv_config_download",
    )
    if (last - first).total_seconds() < horizon:
        st.warning(
            "The selected capture range is shorter than the trading horizon. It cannot supply a complete captured-book round trip at that horizon; inspect coverage and outstanding exposure."
        )
    if submitted:
        st.session_state.pop("relative_value_result", None)
        try:
            with st.spinner(
                "Selecting on validation, then evaluating frozen policies on test captures…"
            ):
                report = cached_research(snapshots, config, names, code)
            st.session_state["relative_value_result"] = signature, report
        except ValueError as exc:
            st.error(f"Cannot evaluate the declared research periods: {exc}")
    saved = st.session_state.get("relative_value_result")
    if saved is None or saved[0] != signature:
        st.caption(
            "Run the evaluation to apply the current inputs and controls. Results from other selections are not displayed or saved here."
        )
        return
    render_report(saved[1])
    report = saved[1]
    render_save_experiment(
        store,
        kind="relative_value",
        snapshots=snapshots,
        parameters=dict(config=asdict(config), instrument_names=names),
        results={key: value for key, value in vars(report).items() if key != "diagnostics"},
        diagnostics=report.diagnostics,
        key="relative_value",
    )


def render_report(report):
    st.subheader("Frozen validation choices")
    st.dataframe(report.selection, hide_index=True)
    st.caption(
        f"Selection fingerprint: {report.diagnostics['selection_fingerprint']}. Validation outcomes select the policy; only the test period is held out."
    )
    with st.expander("All validation candidates and declared period coverage"):
        st.dataframe(report.validation_scores, hide_index=True)
        st.dataframe(report.protocol, hide_index=True)
    st.subheader("Held-out results after costs")
    test_metrics = (
        report.metrics[(report.metrics.period == "test") & (report.metrics.spread_factor == 1)]
        if not report.metrics.empty
        else report.metrics
    )
    st.dataframe(test_metrics, hide_index=True)
    st.caption(
        "P&L, turnover and costs are in the labelled native settlement. Spread cost is already included in execution cash flows and is not charged twice. Excluded/unknown outcomes remain unavailable; eligible PASS/no-position observations contribute zero."
    )
    if test_metrics.empty or test_metrics.common_observations.eq(0).all():
        st.warning("No common eligible final-test opportunities: performance is unavailable.")
    elif test_metrics.outstanding_trades.gt(0).any():
        st.warning(
            "Final-test P&L includes conservative marks on outstanding exposure. These positions were not closed hypothetically; inspect the trade and ledger audit."
        )
    st.subheader("Dependence-aware uncertainty")
    st.dataframe(report.uncertainty, hide_index=True)
    st.caption(
        "Intervals use moving time blocks containing entire option cross-sections and paired common observations. They are conditional on block/stationarity assumptions and cost scenarios. Unavailable intervals supply no significance evidence; this view does not declare alpha."
    )
    if not report.outcome_curves.empty:
        curve = report.outcome_curves[
            (report.outcome_curves.period == "test") & (report.outcome_curves.spread_factor == 1)
        ]
        if not curve.empty:
            curve = curve.assign(time=pd.to_datetime(curve.time, utc=True))
            st.line_chart(
                curve,
                x="time",
                y="cumulative_pnl_native",
                color="strategy",
                x_label="Outcome evaluation cutoff (UTC)",
                y_label=f"Cumulative outcome P&L ({curve.native_currency.iloc[0]})",
            )
            st.caption(
                "Completed/marked outcome curve, with hypothetical executions. This is not continuously marked account equity and does not measure intrahorizon or margin drawdown."
            )
    st.subheader("Coverage, signal strength and wider spreads")
    st.dataframe(report.coverage, hide_index=True)
    st.dataframe(report.signal_strength, hide_index=True)
    st.dataframe(report.spread_sensitivity, hide_index=True)
    st.caption(
        "Strength buckets were declared before testing. Spread scenarios freeze fair values, uncertainty, thresholds and the baseline common universe, then widen captured execution books without retuning."
    )
    tables = {key: value for key, value in vars(report).items() if key != "diagnostics"}
    with st.expander("Complete research tables, trade ledgers and exclusions"):
        selected = st.selectbox(
            "Research audit table",
            list(tables),
            index=list(tables).index("trades"),
            key="rv_audit_table",
        )
        st.dataframe(tables[selected].head(1000), hide_index=True)
        st.caption(
            "Preview limited to 1,000 rows; the selected-table download and saved experiment include every row."
        )
        st.download_button(
            "Download complete research table",
            tables[selected].to_csv(index=False),
            f"relative_value_{selected}.csv",
            "text/csv",
            key="rv_audit_download",
        )
    with st.expander("Research methodology and limitations"):
        st.json(json_value(report.diagnostics))
