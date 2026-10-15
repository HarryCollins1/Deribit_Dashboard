"""Optional research registry and walk-forward controls; calculations live in services."""

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from deribit_dashboard.experiments import (
    ExperimentStore,
    compare_experiments,
    compare_table,
    fingerprint,
    make_experiment,
    reproduce_experiment,
    result_tables,
)
from deribit_dashboard.walk_forward import WalkForwardConfig, evaluate_walk_forward

cached_walk_forward = st.cache_data(ttl=600, max_entries=6)(evaluate_walk_forward)


def render_save_experiment(
    store, *, kind, snapshots, parameters, results, diagnostics=None, key="run"
):
    if store is None:
        return
    signature = fingerprint(
        dict(inputs=[s.to_dict() for s in snapshots], parameters=parameters, kind=kind)
    )
    with st.expander("Save a reproducible experiment"):
        st.caption(
            "Save the complete input snapshots, controls, results, diagnostics, package version "
            "and source fingerprint in the research registry. Existing snapshots stay unchanged."
        )
        name = st.text_input("Experiment name", value=kind.replace("_", " "), key=f"{key}_name")
        if st.button("Save experiment", key=f"{key}_save"):
            registry = ExperimentStore(store.engine)
            registry.initialize()
            payload = make_experiment(kind, snapshots, parameters, results, diagnostics, name)
            registry.save(payload)
            st.session_state[f"{key}_saved"] = (signature, payload)
            st.success(
                f"Saved experiment {payload['run_id']}. Open Research experiments to compare or reproduce it."
            )
        saved = st.session_state.get(f"{key}_saved")
        if saved and saved[0] == signature:
            st.download_button(
                "Download complete experiment",
                json.dumps(saved[1], indent=2, allow_nan=False),
                f"experiment_{saved[1]['run_id']}.json",
                "application/json",
                key=f"{key}_download",
            )


def render_experiments(store):
    st.header("Research experiments")
    st.caption(
        "Immutable inputs and results • portable JSON • explicit comparisons • reproduce with current code"
    )
    registry = ExperimentStore(store.engine)
    registry.initialize()
    with st.expander("Import an experiment"):
        upload = st.file_uploader("Portable experiment JSON", type=["json"])
        if upload is not None and st.button("Import experiment"):
            try:
                run_id = registry.import_json(upload.getvalue())
                st.success(f"Imported {run_id}. Integrity checks passed.")
            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))
    records = registry.list(limit=500)
    if not records:
        st.info(
            "Run a Quant, Strategy, Barrier, Walk-forward or Relative-Value Trading Evaluation and use Save experiment."
        )
        return
    st.dataframe(pd.DataFrame(records).drop(columns=["digest"]), hide_index=True)
    labels = {r["run_id"]: f"{r['created_at']} · {r['kind']} · {r['name']}" for r in records}
    selected = st.selectbox("Inspect experiment", list(labels), format_func=labels.get)
    payload = registry.load(selected)
    st.json(
        {
            k: payload[k]
            for k in ("run_id", "name", "kind", "parameters", "provenance", "input_hashes")
        }
    )
    st.caption(
        "Every input snapshot is embedded in the download. JSON null means unavailable, not zero."
    )
    tables = result_tables(payload)
    if tables:
        name = st.selectbox("Saved result table", list(tables))
        st.dataframe(tables[name].head(1000), hide_index=True)
        st.caption("Preview limited to 1,000 rows; downloads include all results.")
        st.download_button(
            "Download selected table", tables[name].to_csv(index=False), f"{name}.csv", "text/csv"
        )
    with st.expander("Saved diagnostics"):
        st.json(payload["diagnostics"])
    st.download_button(
        "Export complete experiment",
        registry.export_json(selected),
        f"experiment_{selected}.json",
        "application/json",
    )
    if payload["kind"] in {
        "quant",
        "walk_forward",
        "barrier",
        "strategy",
        "barrier_convergence",
        "barrier_model_hedge",
        "relative_value",
    }:
        st.caption(
            "Reproduction uses the embedded snapshots and current installed code. It creates a new linked run and records any provenance change."
        )
        if st.button("Reproduce with current code"):
            try:
                with st.spinner("Reproducing saved inputs…"):
                    new = reproduce_experiment(payload)
                    registry.save(new)
                st.success(f"Saved reproduced run {new['run_id']}.")
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Cannot reproduce this record: {exc}")
    others = [r for r in records if r["run_id"] != selected and r["kind"] == payload["kind"]]
    if others:
        st.subheader("Compare experiments")
        other_id = st.selectbox(
            "Compare against", [r["run_id"] for r in others], format_func=labels.get
        )
        other = registry.load(other_id)
        differences = compare_experiments(payload, other)
        st.dataframe(differences[differences.changed], hide_index=True)
        st.caption(
            "Changes are right minus left. Check currencies, monitoring, inputs and parameters before interpreting a numeric change."
        )
        common = sorted(set(tables) & set(result_tables(other)))
        if common:
            name = st.selectbox("Result table to compare", common)
            shared = list(tables[name].columns.intersection(result_tables(other)[name].columns))
            candidates = [
                c
                for c in (
                    "source",
                    "model",
                    "barrier",
                    "method",
                    "instrument_name",
                    "expiry",
                    "path",
                    "rule",
                    "status",
                    "signal_snapshot",
                    "strategy",
                    "period",
                    "spread_factor",
                    "native_currency",
                    "comparison",
                    "candidate_threshold_cash",
                )
                if c in shared
            ]
            keys = st.multiselect("Columns identifying a result row", shared, default=candidates)
            if keys:
                try:
                    comparison = compare_table(payload, other, name, keys)
                    st.dataframe(comparison.head(1000), hide_index=True)
                    st.download_button(
                        "Download matched comparison",
                        comparison.to_csv(index=False),
                        "experiment_comparison.csv",
                        "text/csv",
                    )
                except ValueError as exc:
                    st.info(str(exc))


def render_walk_forward(store, settings):
    st.header("Walk-forward evaluation on recorded chains")
    st.caption(
        "Fit at an earlier cutoff; evaluate later recorded quotes. No future fit or future "
        "liquidity enters the signal. Maker entries remain hypothetical; these are not fill records."
    )
    environment = st.selectbox("Walk-forward snapshot environment", [settings.environment])
    records = store.list(environment, None, limit=1000)
    if len(records) < 2:
        st.info(
            "Record at least two full-chain snapshots in this environment. Use the collector; a single-instrument save cannot supply a calibrated chain."
        )
        return
    labels = {
        r["snapshot_id"]: f"{r['captured_at']} · {r['currency']} · {r['ticker_count']} quotes"
        for r in records
    }
    selected = st.multiselect(
        "Chronological input snapshots (2–200)", list(labels), format_func=labels.get
    )
    if len(selected) < 2:
        return
    snapshots = [store.load(i) for i in selected]
    with st.form("walk_forward_controls"):
        base = st.selectbox("Walk-forward underlying", ["BTC", "ETH"])
        settlements = sorted(
            {
                m.get("settlement_currency", "")
                for s in snapshots
                for m in s.instruments.values()
                if m.get("kind") == "option" and m.get("base_currency") == base
            }
        )
        settlement = st.selectbox("Walk-forward settlement", settlements or [base])
        horizon = st.number_input("Forward horizon (seconds)", min_value=1.0, value=60.0)
        tolerance = st.number_input(
            "Capture tolerance after horizon (seconds)", min_value=0.0, value=30.0
        )
        age = st.number_input("Quote age limit (seconds)", min_value=0.0, value=60.0)
        width = st.number_input("Maximum observed spread / midpoint", min_value=0.01, value=1.0)
        rate = st.number_input("Cash discount rate (annual decimal)", value=0.0)
        estimator = st.selectbox(
            "Forward pooling estimator", ["weighted_median", "huber", "weighted_mean"]
        )
        contracts = st.number_input(
            "Intended contracts per hypothetical entry", min_value=0.01, value=1.0
        )
        depth = st.slider(
            "Maximum fraction of displayed size", min_value=0.01, max_value=1.0, value=0.25
        )
        fee = st.number_input("Cash fee per contract per execution", min_value=0.0, value=0.0)
        clicked = st.form_submit_button("Run walk-forward evaluation")
    try:
        config = WalkForwardConfig(
            base=base,
            settlement=settlement,
            rate=rate,
            stale_seconds=age,
            max_relative_width=width,
            horizon_seconds=horizon,
            tolerance_seconds=tolerance,
            contracts=contracts,
            fee_per_contract=fee,
            max_depth_fraction=depth,
            parity_estimator=estimator,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    signature = fingerprint(dict(ids=selected, config=asdict(config)))
    if clicked:
        try:
            with st.spinner("Fitting earlier captures and evaluating later quotes…"):
                report = cached_walk_forward(snapshots, config)
            st.session_state["walk_forward_result"] = (signature, report)
        except ValueError as exc:
            st.error(str(exc))
    saved = st.session_state.get("walk_forward_result")
    if not saved or saved[0] != signature:
        return
    report = saved[1]
    st.subheader("Evaluation coverage and conditional repricing")
    st.dataframe(report.metrics, hide_index=True)
    st.dataframe(report.coverage, hide_index=True)
    st.dataframe(report.folds, hide_index=True)
    st.caption(
        "Conditional repricing uses the later observed spot with the earlier fitted volatility. Its error measures frozen-vol price accuracy, not a forecast of the spot move."
    )
    useful = [
        c
        for c in (
            "signal_time",
            "instrument_name",
            "status",
            "entry_fair",
            "later_mid",
            "conditional_repricing_error",
            "hypothetical_long_close_pnl",
            "hypothetical_short_close_pnl",
        )
        if c in report.observations
    ]
    st.dataframe(report.observations[useful].head(500), hide_index=True)
    with st.expander("Complete quote audit and assumptions"):
        st.dataframe(report.observations.head(1000), hide_index=True)
        st.json(report.diagnostics)
    st.download_button(
        "Download all walk-forward observations",
        report.observations.to_csv(index=False),
        "walk_forward.csv",
        "text/csv",
    )
    render_save_experiment(
        store,
        kind="walk_forward",
        snapshots=snapshots,
        parameters=asdict(config),
        results={
            "observations": report.observations,
            "folds": report.folds,
            "coverage": report.coverage,
            "metrics": report.metrics,
        },
        diagnostics=report.diagnostics,
        key="walk_forward",
    )
