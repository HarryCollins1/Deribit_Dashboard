"""Optional strategy editor and views; calibration/simulation live in the library."""

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from deribit_dashboard.live_ui import stream_source
from deribit_dashboard.pricing import MODELS
from deribit_dashboard.strategies import (
    BARRIER_TYPES,
    SimulationConfig,
    describe_strategy,
    required_cash_currencies,
    simulate_strategy,
    stress_strategy,
)
from deribit_dashboard.strategy_research import (
    MarginScenario,
    PathRegime,
    SurfaceEvolution,
    size_pair,
)
from deribit_dashboard.strategy_workbench import (
    combine_snapshots,
    prepare_strategy,
    scenario_rows,
    simulation_plot_data,
    strategy_catalogue,
    strategy_templates,
)
from deribit_dashboard.workbench import plot_lines

cached_preparation = st.cache_data(ttl=300, max_entries=12)(prepare_strategy)
cached_simulation = st.cache_data(ttl=300, max_entries=6)(simulate_strategy)
cached_stress = st.cache_data(ttl=300, max_entries=12)(stress_strategy)


def render_strategy_page(store, settings):
    st.header("Portfolio strategies and stress tests")
    source = st.selectbox(
        "Strategy market source",
        ["Saved snapshots", "Live BTC + ETH stream"],
    )
    if source == "Saved snapshots":
        environment = st.selectbox("Strategy snapshot environment", [settings.environment])
        records = store.list(environment, None, limit=1000)
        labels = {r["snapshot_id"]: f"{r['captured_at']} · {r['currency']}" for r in records}
        names = st.multiselect("Strategy snapshots", list(labels), format_func=labels.get)
        skew = st.number_input(
            "Maximum difference between capture times (seconds)", min_value=0.0, value=60.0
        )
        if not names:
            st.info("Select a combined BTC/ETH snapshot or contemporaneous single-asset snapshots.")
            return
        try:
            snapshot = combine_snapshots([store.load(n) for n in names], skew)
            snapshot.snapshot_id = "strategy-" + "-".join(sorted(names))
        except ValueError as exc:
            st.warning(str(exc))
            return
    else:
        snapshot = stream_source(settings, "strategy_live")
        if snapshot is None:
            return
    st.caption(
        f"Research input: {snapshot.environment} · {snapshot.source} · {snapshot.captured_at.isoformat()}"
    )
    if st.button("Save strategy market snapshot"):
        # A new capture ID prevents a combined view being mistaken for a stored original.
        from uuid import uuid4

        saved = type(snapshot).from_dict(snapshot.to_dict())
        saved.snapshot_id = str(uuid4())
        store.save(saved)
        st.success(f"Saved market input {saved.snapshot_id}.")
    render_strategy_builder(snapshot, "strategy", store=store)


def render_strategy_builder(snapshot, key="strategy", store=None):
    st.subheader("Build pairs, spreads and barrier strategies")
    st.caption(
        "Choose discovered instruments, enter signed quantities, and test the complete portfolio. "
        "Positive quantities are long; negative quantities are short. Spot uses base-coin amounts; "
        "options and futures use contract counts multiplied by captured contract_size."
    )
    catalogue = strategy_catalogue(snapshot)
    templates = strategy_templates(snapshot)
    if not templates:
        st.info("This market input has no supported BTC/ETH instruments.")
        return
    with st.expander("Available strategy instruments and contract units"):
        st.dataframe(catalogue, hide_index=True)
    names = list(templates)
    default = names.index("BTC / ETH perpetual pair") if "BTC / ETH perpetual pair" in names else 0
    template = st.selectbox("Strategy template", names, index=default, key=f"{key}_template")
    st.caption(
        "Templates are editable examples. A pair's initial equal contract counts do not imply equal notionals or a hedge."
    )
    state_key = f"{key}_leg_rows"
    version_key = f"{key}_editor_version"
    st.session_state.setdefault(version_key, 0)
    if state_key not in st.session_state:
        st.session_state[state_key] = templates[template]
    if st.button("Load strategy template", key=f"{key}_load_template"):
        st.session_state[state_key] = templates[template]
        st.session_state[version_key] += 1
    with st.expander("Restore a saved strategy definition"):
        uploaded = st.file_uploader("Strategy definition JSON", type=["json"], key=f"{key}_upload")
        if uploaded is not None and st.button("Restore strategy legs", key=f"{key}_restore"):
            try:
                definition = json.loads(uploaded.getvalue())
                rows = definition["legs"]
                if (
                    definition.get("schema_version") != 1
                    or not isinstance(rows, list)
                    or not 1 <= len(rows) <= 16
                ):
                    raise ValueError("Expected a schema-v1 strategy with 1–16 legs")
                if any(r.get("instrument_name") not in snapshot.instruments for r in rows):
                    raise ValueError(
                        "Some saved instruments are absent from the current market input"
                    )
                st.session_state[state_key] = rows
                st.session_state[version_key] += 1
                st.success(
                    "Restored the legs. Review the model, FX and simulation assumptions before running."
                )
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Invalid strategy definition: {exc}")
    rows = st.data_editor(
        pd.DataFrame(st.session_state[state_key]),
        num_rows="dynamic",
        hide_index=True,
        key=f"{key}_legs_{st.session_state[version_key]}",
        column_config={
            "instrument_name": st.column_config.SelectboxColumn(
                "Instrument", options=catalogue.instrument_name.tolist(), required=True
            ),
            "quantity": st.column_config.NumberColumn(
                "Signed quantity",
                required=True,
                default=1.0,
                help="Base coin for spot; contract count for derivatives.",
            ),
            "barrier_type": st.column_config.SelectboxColumn(
                "Hypothetical barrier", options=BARRIER_TYPES, default="none", required=True
            ),
            "barrier_level": st.column_config.NumberColumn(
                "Barrier (cash spot units)", min_value=0.0, default=0.0
            ),
            "barrier_hit": st.column_config.CheckboxColumn("Previously touched", default=False),
            "monitoring_days": st.column_config.NumberColumn(
                "Observation interval (days)", min_value=0.01, default=1.0
            ),
        },
    )
    st.caption(
        "Barrier overlays are hypothetical European Black–Scholes contracts with zero rebate. "
        "Up/down barriers knock in or out when touched on the stated observation dates, including "
        "inception and expiry. Between-date touches are not observed. They are not listed Deribit instruments."
    )
    with st.expander("Advanced model comparison"):
        model = st.selectbox("Strategy pricing model", list(MODELS), key=f"{key}_model")
        st.caption(
            "Black–Scholes is the default listed-option research model. Bachelier and Binomial are additional model comparisons; evolving-smile scenarios require Black–Scholes vanilla legs."
        )
    rate = st.number_input(
        "Strategy cash discount rate (annual decimal)", value=0.0, step=0.01, key=f"{key}_rate"
    )
    cash_currency = (
        st.text_input("Reporting cash currency", "USD", key=f"{key}_cash").strip().upper()
    )
    quote_age = st.number_input(
        "Strategy maximum quote age at capture (seconds)",
        min_value=0.0,
        value=60.0,
        key=f"{key}_age",
    )
    conversions = {}
    try:
        currencies = required_cash_currencies(snapshot, rows.to_dict("records"))
    except (KeyError, TypeError, ValueError):
        currencies = []
    for native in currencies:
        if native != cash_currency:
            value = st.number_input(
                f"{cash_currency} per {native} (fixed scenario conversion)",
                min_value=0.000001,
                value=None,
                key=f"{key}_fx_{cash_currency}_{native}",
            )
            if value is not None:
                conversions[native] = value
    if any(c != cash_currency for c in currencies):
        st.caption(
            "Enter explicit conversion assumptions; USD, USDC and other quote currencies are not automatically treated as equal. FX stays fixed during these simulations."
        )

    with st.expander("Size a pair by its risk and captured depth"):
        sizing_method = st.selectbox(
            "Pair neutrality rule", ["dollar", "delta", "beta", "volatility"], key=f"{key}_sizing"
        )
        target_notional = st.number_input(
            "First-leg target notional (reporting cash)",
            min_value=0.01,
            value=10000.0,
            key=f"{key}_target",
        )
        depth_enabled = st.checkbox(
            "Cap both legs at displayed best-quote depth", value=True, key=f"{key}_depth_enabled"
        )
        depth_fraction = st.number_input(
            "Fraction of displayed depth",
            min_value=0.01,
            max_value=1.0,
            value=1.0,
            key=f"{key}_depth_fraction",
        )
        beta_btc = st.number_input("BTC beta assumption", value=1.0, key=f"{key}_beta_btc")
        beta_eth = st.number_input("ETH beta assumption", value=1.2, key=f"{key}_beta_eth")
        sizing_btc_vol = st.number_input(
            "BTC sizing annual volatility", min_value=0.001, value=0.6, key=f"{key}_sizing_btc_vol"
        )
        sizing_eth_vol = st.number_input(
            "ETH sizing annual volatility", min_value=0.001, value=0.7, key=f"{key}_sizing_eth_vol"
        )
        st.caption(
            "Dollar delta measures exposure to a proportional underlying move. Beta/volatility rules weight dollar delta by the supplied assumptions. One common depth scale preserves the ratio before captured quantity increments and minimum sizes are enforced; residual exposure remains in the audit."
        )
        if st.button("Calculate and apply pair quantities", key=f"{key}_apply_size"):
            try:
                preparation = cached_preparation(
                    snapshot,
                    rows,
                    model,
                    rate=rate,
                    cash_currency=cash_currency,
                    conversion_rates=conversions,
                    max_quote_age_seconds=quote_age,
                )
                sized, audit = size_pair(
                    preparation.legs,
                    target_notional,
                    method=sizing_method,
                    betas={"BTC": beta_btc, "ETH": beta_eth},
                    annual_volatility={"BTC": sizing_btc_vol, "ETH": sizing_eth_vol},
                    depth_fraction=depth_fraction if depth_enabled else None,
                )
                resized_rows = rows.to_dict("records")
                for row, leg in zip(resized_rows, sized, strict=True):
                    row["quantity"] = leg.quantity
                st.session_state[state_key] = resized_rows
                st.session_state[f"{key}_sizing_audit"] = audit
                st.session_state[version_key] += 1
                st.rerun()
            except (ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))
        if f"{key}_sizing_audit" in st.session_state:
            st.dataframe(st.session_state[f"{key}_sizing_audit"], hide_index=True)

    st.markdown(
        "**Stress paths** follow the selected shape. Touch-and-retrace passes through the chosen peak before the endpoint; jump-then-hold moves at the event date. Event fraction is a fraction of the horizon. Shocks are fractions: −0.2 means −20%. Barriers are observed only on contract monitoring dates."
    )
    stress_frame = st.data_editor(
        pd.DataFrame(
            [
                dict(
                    path_shape="touch and retrace"
                    if row["name"] == "Touch high then retrace"
                    else "monotone",
                    event_fraction=0.5,
                    BTC_peak=0.3,
                    ETH_peak=0.3,
                    BTC_basis=0.0,
                    ETH_basis=0.0,
                    BTC_funding=None,
                    ETH_funding=None,
                    **row,
                )
                for row in [
                    dict(name="Flat prices", BTC=0.0, ETH=0.0, volatility_shock=0.0, days=1.0),
                    dict(
                        name="Both assets fall", BTC=-0.2, ETH=-0.3, volatility_shock=0.2, days=7.0
                    ),
                    dict(
                        name="BTC outperforms ETH",
                        BTC=0.1,
                        ETH=-0.1,
                        volatility_shock=0.0,
                        days=7.0,
                    ),
                    dict(
                        name="Both assets rise", BTC=0.3, ETH=0.3, volatility_shock=-0.1, days=7.0
                    ),
                    dict(
                        name="Touch high then retrace",
                        BTC=0.1,
                        ETH=0.05,
                        volatility_shock=0.0,
                        days=30.0,
                    ),
                ]
            ]
        ),
        num_rows="dynamic",
        hide_index=True,
        key=f"{key}_stresses",
        column_config={
            "path_shape": st.column_config.SelectboxColumn(
                "Path shape",
                options=["monotone", "touch and retrace", "jump then hold"],
                required=True,
            ),
            "event_fraction": st.column_config.NumberColumn(
                "Event fraction of horizon", min_value=0.01, max_value=0.99, default=0.5
            ),
            "BTC_funding": st.column_config.NumberColumn(
                "BTC annual funding override", help="Blank keeps the simulation funding input."
            ),
            "ETH_funding": st.column_config.NumberColumn(
                "ETH annual funding override", help="Blank keeps the simulation funding input."
            ),
        },
    )
    with st.form(f"{key}_simulation_controls"):
        initial = st.number_input("Strategy starting equity", min_value=0.01, value=100000.0)
        horizon = st.number_input(
            "Simulation horizon (days)", min_value=0.1, max_value=365.0, value=30.0
        )
        steps = st.number_input("Simulation time steps", min_value=2, max_value=365, value=30)
        paths = st.number_input("Simulation paths", min_value=2, max_value=512, value=128)
        seed = st.number_input("Simulation random seed", min_value=0, value=42)
        correlation = st.slider(
            "BTC / ETH return correlation", min_value=-1.0, max_value=1.0, value=0.5, step=0.05
        )
        st.caption(
            "These are user-defined physical-path assumptions; they are separate from the option's surface-calibrated pricing volatility."
        )
        btc_vol = st.number_input("BTC annual path volatility (decimal)", min_value=0.0, value=0.6)
        eth_vol = st.number_input("ETH annual path volatility (decimal)", min_value=0.0, value=0.7)
        btc_drift = st.number_input("BTC annual path drift (decimal)", value=0.0)
        eth_drift = st.number_input("ETH annual path drift (decimal)", value=0.0)
        risk_free = st.number_input("Sharpe reference rate (annual decimal)", value=0.0)
        target = st.number_input("Sortino minimum acceptable return (annual decimal)", value=0.0)
        cost = st.number_input(
            "Entry friction (bps of gross underlying notional)", min_value=0.0, value=0.0
        )
        btc_funding = st.number_input("BTC perpetual funding scenario (annual decimal)", value=0.0)
        eth_funding = st.number_input("ETH perpetual funding scenario (annual decimal)", value=0.0)
        with st.expander("Surface evolution and volatility stress"):
            evolution_mode = st.selectbox(
                "Surface evolution rule",
                ["fixed volatility", "sticky strike", "sticky forward delta"],
            )
            parallel_shock = st.number_input("Parallel IV shock (absolute decimal)", value=0.0)
            skew_shock = st.number_input(
                "Skew shock (decimal IV per unit log-moneyness)", value=0.0
            )
            term_shock = st.number_input("Term shock (decimal IV per remaining year)", value=0.0)
            st.caption(
                "Uses captured initial smile/term data. Sticky delta retains the forward-call-delta smile; maturity interpolation uses total variance. Edge extensions are counted. These are hypothetical mark rules, with no promise of arbitrage preservation. Barrier legs require unchanged fixed-volatility continuation; use Barrier comparison for local/Heston models."
            )
        with st.expander("Jump, correlation, basis and funding regime"):
            regime_enabled = st.checkbox("Apply a simulation regime change", value=False)
            regime_day = st.number_input("Regime change elapsed day", min_value=0.01, value=15.0)
            regime_btc_jump = st.number_input(
                "BTC jump at event (fraction)", min_value=-0.99, value=-0.2
            )
            regime_eth_jump = st.number_input(
                "ETH jump at event (fraction)", min_value=-0.99, value=-0.3
            )
            regime_corr = st.slider(
                "Correlation after event", min_value=-1.0, max_value=1.0, value=0.9
            )
            regime_btc_vol = st.number_input(
                "BTC path-volatility multiplier after event", min_value=0.0, value=1.5
            )
            regime_eth_vol = st.number_input(
                "ETH path-volatility multiplier after event", min_value=0.0, value=1.5
            )
            regime_btc_basis = st.number_input(
                "BTC relative futures-basis level shock", min_value=-0.99, value=0.0
            )
            regime_eth_basis = st.number_input(
                "ETH relative futures-basis level shock", min_value=-0.99, value=0.0
            )
            regime_btc_funding = st.number_input("BTC annual funding after event", value=0.0)
            regime_eth_funding = st.number_input("ETH annual funding after event", value=0.0)
            regime_option_vol = st.number_input(
                "Relative option IV shock after event", min_value=-1.0, value=0.0
            )
        with st.expander("Collateral and margin scenario"):
            margin_enabled = st.checkbox("Simulate collateral and liquidation", value=False)
            initial_margin = st.number_input(
                "Initial margin fraction of gross derivative exposure",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
            )
            maintenance_margin = st.number_input(
                "Maintenance margin fraction", min_value=0.0, max_value=1.0, value=0.1
            )
            coin_fraction = st.number_input(
                "Fraction of initial equity held as coin collateral",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
            )
            collateral_asset = st.selectbox("Collateral coin", ["BTC", "ETH"])
            haircut = st.number_input(
                "Coin collateral haircut", min_value=0.0, max_value=0.99, value=0.0
            )
            liquidation_fee = st.number_input(
                "Liquidation cost (bps of closing gross notional)", min_value=0.0, value=10.0
            )
            st.caption(
                "User-defined research margin, not exchange-certified rules. Coin collateral is part of starting equity. Gross futures and short-option exposure reserves margin without spread offsets. At the first observed maintenance breach all holdings close, costs are paid and remaining cash freezes; negative balances are retained."
            )
        stress_clicked = st.form_submit_button("Run strategy stress tests")
        simulation_clicked = st.form_submit_button("Run strategy simulations")
    try:
        evolution = SurfaceEvolution(evolution_mode, parallel_shock, skew_shock, term_shock)
        margin = MarginScenario(
            margin_enabled,
            initial_margin,
            maintenance_margin,
            coin_fraction,
            collateral_asset,
            haircut,
            liquidation_fee,
        )
        regimes = (
            (
                PathRegime(
                    regime_day,
                    {"BTC": regime_btc_jump, "ETH": regime_eth_jump},
                    {"BTC": regime_btc_vol, "ETH": regime_eth_vol},
                    regime_corr,
                    {"BTC": regime_btc_basis, "ETH": regime_eth_basis},
                    {"BTC": regime_btc_funding, "ETH": regime_eth_funding},
                    regime_option_vol,
                ),
            )
            if regime_enabled
            else ()
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    config = SimulationConfig(
        initial_equity=initial,
        horizon_days=horizon,
        steps=int(steps),
        paths=int(paths),
        seed=int(seed),
        annual_drift={"BTC": btc_drift, "ETH": eth_drift},
        annual_volatility={"BTC": btc_vol, "ETH": eth_vol},
        correlation=correlation,
        risk_free_rate=risk_free,
        minimum_acceptable_return=target,
        entry_cost_bps=cost,
        annual_funding_rates={"BTC": btc_funding, "ETH": eth_funding},
        surface_evolution=evolution,
        margin=margin,
        regimes=regimes,
    )
    signature = (
        snapshot.snapshot_id,
        rows.to_json(),
        model,
        rate,
        cash_currency,
        repr(conversions),
        quote_age,
        repr(config),
        stress_frame.to_json(),
    )
    output_key = f"{key}_results"
    if stress_clicked or simulation_clicked:
        try:
            with st.spinner("Calibrating selected legs and evaluating the strategy…"):
                prepared = cached_preparation(
                    snapshot,
                    rows,
                    model,
                    rate=rate,
                    cash_currency=cash_currency,
                    conversion_rates=conversions,
                    max_quote_age_seconds=quote_age,
                )
                stresses = cached_stress(
                    prepared.legs,
                    scenario_rows(stress_frame),
                    initial_equity=initial,
                    entry_cost_bps=cost,
                    annual_funding_rates=config.annual_funding_rates,
                    surface_evolution=evolution,
                    margin=margin,
                )
                simulation = (
                    cached_simulation(prepared.legs, config) if simulation_clicked else None
                )
                st.session_state[output_key] = (signature, prepared, stresses, simulation)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Strategy cannot be evaluated: {exc}")
    definition = dict(
        schema_version=1,
        snapshot_id=snapshot.snapshot_id,
        legs=rows.to_dict("records"),
        model=model,
        cash_currency=cash_currency,
        conversion_rates=conversions,
        discount_rate=rate,
        simulation=asdict(config),
        stress_scenarios=stress_frame.to_dict("records"),
    )
    st.download_button(
        "Download strategy definition",
        json.dumps(definition, indent=2),
        "strategy.json",
        "application/json",
        key=f"{key}_definition",
    )
    output = st.session_state.get(output_key)
    if output is None:
        return
    if output[0] != signature:
        st.info("The market or strategy inputs changed. Run the tests again to update the results.")
        return
    _, prepared, stresses, simulation = output
    st.markdown("**Resolved portfolio legs and entry conventions**")
    details = describe_strategy(prepared.legs)
    st.dataframe(
        details[
            [
                "instrument_name",
                "quantity",
                "quantity_unit",
                "entry_cash_value",
                "gross_cash_notional",
                "barrier",
            ]
        ],
        hide_index=True,
    )
    with st.expander("Complete leg and calibration audit"):
        st.dataframe(details, hide_index=True)
    if not prepared.calibration.empty:
        st.dataframe(prepared.calibration, hide_index=True)
    st.markdown("**Stress-test results**")
    st.dataframe(
        stresses[
            [
                "scenario",
                "path_shape",
                "pnl",
                "max_drawdown",
                "minimum_equity",
                "barrier_legs_hit",
                "liquidated",
            ]
        ],
        hide_index=True,
    )
    with st.expander("Complete stress audit"):
        st.dataframe(stresses, hide_index=True)
    st.download_button(
        "Download stress results",
        stresses.to_csv(index=False),
        "strategy_stress.csv",
        key=f"{key}_stress_csv",
    )
    if simulation is not None:
        render_simulation(simulation, key)
    if store is not None:
        from deribit_dashboard.experiment_ui import render_save_experiment

        render_save_experiment(
            store,
            kind="strategy",
            snapshots=[snapshot],
            parameters=dict(
                definition, max_quote_age_seconds=quote_age, run_simulation=simulation is not None
            ),
            results={
                "summary": simulation.summary if simulation else {},
                "metrics": simulation.path_metrics if simulation else pd.DataFrame(),
                "equity": simulation.equity if simulation else pd.DataFrame(),
                "stresses": stresses,
                "margin_ledger": simulation.margin_ledger if simulation else pd.DataFrame(),
            },
            diagnostics={
                "assumptions": simulation.assumptions if simulation else [],
                "calibration": prepared.calibration,
            },
            key=f"{key}_run",
        )


def render_simulation(result, key):
    import matplotlib.pyplot as plt

    summary = result.summary
    st.markdown("**Simulated portfolio performance**")
    columns = st.columns(4)
    columns[0].metric(
        f"Mean terminal P&L ({summary['reporting_currency']})",
        f"{summary['mean_terminal_pnl']:,.2f}",
    )
    columns[1].metric("Worst path max drawdown", f"{summary['worst_max_drawdown']:.2%}")
    columns[2].metric("Paths ending at a loss", f"{summary['probability_of_loss']:.2%}")
    columns[3].metric(
        "Scenario liquidation fraction",
        f"{summary['liquidation_fraction']:.2%}" if summary["margin_enabled"] else "Disabled",
    )
    st.caption(
        "Drawdown includes starting equity. Sharpe and Sortino use the time-ordered returns of "
        "each path and the actual time grid; they are undefined when risk is zero or equity becomes "
        "nonpositive. Annualization of a short simulated path can produce unstable ratios. "
        "These metrics describe the supplied simulation assumptions, not historical performance."
    )
    with st.expander("Secondary performance ratios and full assumptions"):
        st.json(summary)
        st.json(result.diagnostics)
        for assumption in result.assumptions:
            st.caption(assumption)
    frame = simulation_plot_data(result)
    fig = plot_lines(
        frame,
        "Elapsed time (days)",
        ["5th percentile", "Median", "95th percentile"],
        "Simulated portfolio equity across paths",
        x_label="Elapsed time (days)",
        y_label=f"Portfolio equity ({summary['reporting_currency']})",
    )
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)
    st.caption(
        "At each date, the lines show cross-path equity percentiles. A percentile line can combine different paths over time; drawdown and risk ratios are calculated on individual paths."
    )
    st.dataframe(
        result.path_metrics[["path", "pnl", "max_drawdown", "minimum_equity", "bankrupt"]],
        hide_index=True,
    )
    with st.expander("Full path metrics including Sharpe and Sortino"):
        st.dataframe(result.path_metrics, hide_index=True)
    if not result.margin_ledger.empty:
        st.markdown("**Collateral and liquidation audit**")
        st.dataframe(
            result.margin_ledger[
                [
                    "path",
                    "liquidated",
                    "liquidation_day",
                    "minimum_margin_buffer",
                    "liquidation_cost",
                    "pnl_reconciliation_error",
                ]
            ],
            hide_index=True,
        )
        with st.expander("Complete collateral and closeout ledger"):
            st.dataframe(result.margin_ledger, hide_index=True)
        st.download_button(
            "Download margin and closeout audit",
            result.margin_ledger.to_csv(index=False),
            "strategy_margin.csv",
            key=f"{key}_margin_csv",
        )
    if not result.barrier_hits.empty:
        st.dataframe(result.barrier_hits, hide_index=True)
    st.download_button(
        "Download path metrics",
        result.path_metrics.to_csv(index=False),
        "strategy_metrics.csv",
        key=f"{key}_metrics_csv",
    )
    st.download_button(
        "Download simulated equity paths",
        result.equity.to_csv(),
        "strategy_equity.csv",
        key=f"{key}_equity_csv",
    )
