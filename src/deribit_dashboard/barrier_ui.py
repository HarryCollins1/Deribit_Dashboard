"""Optional thin controls and rendering for the barrier-comparison service."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st

from deribit_dashboard.barrier_convergence import ConvergenceControls, barrier_convergence
from deribit_dashboard.barrier_history import build_historical_paths
from deribit_dashboard.barrier_model_hedge import ModelHedgeControls, model_hedge_study
from deribit_dashboard.barrier_models import HestonParameters
from deribit_dashboard.barrier_workbench import (
    BarrierExperiment,
    barrier_figure,
    load_barrier_history,
    prepare_barrier_market,
    replay_historical,
    run_barrier_comparison,
)
from deribit_dashboard.experiment_ui import render_save_experiment
from deribit_dashboard.live_ui import stream_source
from deribit_dashboard.models import utc_now
from deribit_dashboard.pricing import DAYS

prepare = st.cache_data(ttl=300, max_entries=4)(prepare_barrier_market)
compare = st.cache_data(ttl=300, max_entries=3)(run_barrier_comparison)
history = st.cache_data(ttl=600, max_entries=3)(load_barrier_history)
replay = st.cache_data(ttl=300, max_entries=3)(replay_historical)
converge = st.cache_data(ttl=600, max_entries=3)(barrier_convergence)
model_hedges = st.cache_data(ttl=600, max_entries=3)(model_hedge_study)


def render_barrier_page(store, settings):
    st.header("Barrier comparison and replication")
    st.caption(
        "Hold the call, expiry, calibration and simulation seed fixed; change only the barrier. Prices and hedge P&L refer to one BTC of option notional."
    )
    source = st.selectbox("Barrier market source", ["Saved snapshot", "Live BTC stream"])
    if source == "Saved snapshot":
        environment = st.selectbox("Barrier snapshot environment", [settings.environment])
        records = store.list(environment, None, limit=1000)
        if not records:
            st.info("Record or import an option-chain snapshot first.")
            return
        labels = {r["snapshot_id"]: f"{r['captured_at']} · {r['currency']}" for r in records}
        selected = st.selectbox("Barrier snapshot", list(labels), format_func=labels.get)
        snapshot = store.load(selected)
    else:
        snapshot = stream_source(replace(settings, currency="BTC"), "barrier_live")
        if snapshot is None:
            return
    st.caption(
        f"Input: {snapshot.source} · {snapshot.environment} · {snapshot.captured_at.isoformat()}"
    )
    calls = {
        n: m
        for n, m in snapshot.instruments.items()
        if m.get("base_currency") == "BTC"
        and m.get("kind") == "option"
        and m.get("option_type") == "call"
    }
    perps = [
        n
        for n, m in snapshot.instruments.items()
        if m.get("base_currency") == "BTC"
        and m.get("kind") == "future"
        and m.get("settlement_period") == "perpetual"
    ]
    if not calls or not perps:
        st.info("The snapshot must contain a BTC option chain and a BTC perpetual with metadata.")
        return
    settlements = sorted({m["settlement_currency"] for m in calls.values()})
    settlement = st.selectbox(
        "Barrier option settlement",
        settlements,
        index=settlements.index("BTC") if "BTC" in settlements else 0,
    )
    calls = {n: m for n, m in calls.items() if m["settlement_currency"] == settlement}
    expiries = sorted({m["expiration_timestamp"] for m in calls.values()})
    target = (snapshot.captured_at + timedelta(days=30)).timestamp() * 1000
    expiry = st.selectbox(
        "Barrier option expiry (UTC)",
        expiries,
        index=min(range(len(expiries)), key=lambda i: abs(expiries[i] - target)),
        format_func=lambda e: datetime.fromtimestamp(e / 1000, UTC).strftime("%d %b %Y, %H:%M UTC"),
    )
    names = sorted(
        [n for n, m in calls.items() if m["expiration_timestamp"] == expiry],
        key=lambda n: calls[n]["strike"],
    )
    selected = st.selectbox(
        "Underlying vanilla call",
        names,
        index=min(range(len(names)), key=lambda i: abs(calls[names[i]]["strike"] - 105000)),
    )
    perp = st.selectbox("Perpetual used to hedge", perps)
    st.caption(
        "The initial call selection is the available strike nearest 105,000. Select the exact listed strike you want; unavailable instruments are never invented. The perpetual must share the option's cash denomination."
    )
    with st.form("barrier_controls"):
        barriers_text = st.text_input(
            "Barrier sweep levels (cash spot units, comma separated)",
            "110000,115000,120000,125000,130000",
        )
        kind = st.selectbox("Barrier contract", ["up-and-out", "up-and-in"])
        monitoring = st.number_input(
            "Barrier monitoring interval (days)", min_value=0.05, max_value=30.0, value=1.0
        )
        rate = st.number_input("Barrier cash discount rate (annual decimal)", value=0.0)
        age = st.number_input("Maximum quote age at capture (seconds)", min_value=0.0, value=60.0)
        paths = st.selectbox(
            "Pricing Monte Carlo paths", [128, 512, 2048, 4096, 8192, 16384], index=3
        )
        steps = st.number_input(
            "Monte Carlo integration steps", min_value=10, max_value=1500, value=120
        )
        hedge_paths = st.selectbox("Paths per hedge evaluation", [16, 64, 256, 512, 1024], index=2)
        seed = st.number_input("Barrier experiment random seed", min_value=0, value=42)
        bump = st.number_input(
            "Gamma spot bump (fraction)", min_value=0.001, max_value=0.04, value=0.01
        )
        level = st.number_input(
            "Fixed finish/touch probability threshold", min_value=1.0, value=120000.0
        )
        fee = st.number_input(
            "Execution fee (bps; option premium or perpetual notional)", min_value=0.0, value=1.0
        )
        execution = st.checkbox("Use captured bid/ask spreads and displayed depth", value=True)
        impact = st.number_input(
            "Beyond-displayed-depth impact multiplier",
            min_value=0.0,
            value=1.0,
            help="Additional unit-price impact = half-spread × max(quantity / side depth − 1, 0) × multiplier. Beyond top-of-book this is a scenario assumption.",
        )
        funding = st.number_input("Perpetual funding assumption (annual decimal)", value=0.0)
        fit_heston = st.checkbox("Calibrate Heston to fitted vanilla smiles", value=True)
        st.caption(
            "For explicit-parameter mode, uncheck calibration. v0 and theta are variances, not volatility percentages. Parameters below are ignored when calibration is enabled."
        )
        v0 = st.number_input("Heston initial variance v0", min_value=0.0001, value=0.30)
        kappa = st.number_input("Heston mean reversion kappa (per year)", min_value=0.05, value=2.0)
        theta = st.number_input("Heston long-run variance theta", min_value=0.0001, value=0.30)
        xi = st.number_input("Heston volatility of variance xi", min_value=0.0, value=0.6)
        rho = st.slider("Heston spot/variance correlation rho", -0.98, 0.98, -0.5, 0.01)
        submitted = st.form_submit_button("Run barrier comparison")
    st.caption(
        "Zero rebate; discrete monitoring includes inception and expiry. Hedge rebalancing occurs at monitoring dates. Captured execution includes vanilla entry bid/ask and stated fees. Later spreads and depth are explicit scenarios around model marks. Vanilla expiry settlement has no trading fee."
    )
    try:
        config = BarrierExperiment(
            barriers=tuple(float(v.strip()) for v in barriers_text.split(",")),
            kind=kind,
            monitoring_days=monitoring,
            paths=paths,
            steps=int(steps),
            seed=int(seed),
            spot_bump_fraction=bump,
            probability_level=level,
            hedge_paths=hedge_paths,
            fee_bps=fee,
            annual_funding=funding,
            fit_heston=fit_heston,
            heston_parameters=None if fit_heston else HestonParameters(v0, kappa, theta, xi, rho),
            market_execution=execution,
            impact_multiplier=impact,
        )
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
        return
    signature = (snapshot.snapshot_id, selected, perp, rate, age, repr(config))
    if submitted:
        try:
            with st.spinner("Fitting dynamics, pricing barriers and replaying hedges…"):
                market = prepare(snapshot, selected, perp, rate, age)
                report = compare(market, config)
                st.session_state["barrier_report"] = (signature, report)
        except (ValueError, KeyError, TypeError, FloatingPointError) as exc:
            st.error(f"Comparison unavailable: {exc}")
    saved = st.session_state.get("barrier_report")
    if saved is None:
        st.info(
            "Run the comparison to generate prices, probability estimates, replication costs and hedge errors."
        )
        return
    if saved[0] != signature:
        st.info("Inputs changed. Run the comparison again before interpreting results.")
        return
    report = saved[1]
    render_report(report)
    registry_parameters = dict(
        config=asdict(config),
        instrument_name=selected,
        perpetual_name=perp,
        rate=rate,
        stale_seconds=age,
    )
    render_save_experiment(
        store,
        kind="barrier",
        snapshots=[snapshot],
        parameters=registry_parameters,
        results={
            name: getattr(report, name)
            for name in (
                "prices",
                "vanilla",
                "probabilities",
                "replication_costs",
                "replication_weights",
                "hedge_metrics",
                "path_results",
            )
        },
        diagnostics=report.diagnostics,
        key="barrier_run",
    )
    render_numerical_studies(report, store, snapshot, registry_parameters)
    render_history(report, settings)


def render_report(report):
    import matplotlib.pyplot as plt

    market = report.market
    st.subheader("Vanilla and barrier prices")
    with st.expander("Contract, units and captured hedge metadata"):
        st.json(
            dict(
                instrument=market.instrument_name,
                spot=market.option.spot,
                strike=market.option.strike,
                days_to_expiry=market.option.years * DAYS,
                pricing_volatility=market.option.volatility,
                cash_currency=market.currency,
                perpetual=asdict(market.perpetual),
            )
        )
    st.dataframe(report.vanilla, hide_index=True)
    st.caption(
        "surface_target is the independently fitted vanilla call price. Model vanilla estimates should be compared with this target before interpreting exotic-price differences. Price and sampling_se are in cash per BTC; SE excludes model and discretization error."
    )
    st.dataframe(
        report.prices,
        hide_index=True,
        column_order=[
            "model",
            "barrier",
            "price",
            "price_se",
            "delta",
            "gamma",
            "gamma_se",
            "bs_grid_crosscheck",
        ],
    )
    with st.expander("All price diagnostics and confidence intervals"):
        st.dataframe(report.prices, hide_index=True)
    st.caption(
        "price is the discounted Monte Carlo barrier value. gamma is the spot second derivative at fixed calibration; gamma_se reports sampling uncertainty using common random numbers. bs_grid_crosscheck is an independent discrete Black–Scholes grid approximation. All barrier levels reuse the same paths."
    )
    st.subheader("Finish and touch at the fixed threshold")
    st.dataframe(
        report.probabilities,
        hide_index=True,
        column_order=[
            "model",
            "threshold",
            "finish_probability",
            "touch_probability",
            "touch_probability_se",
            "total_paths",
        ],
        column_config={
            "finish_probability": st.column_config.NumberColumn(format="percent"),
            "touch_probability": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Finish means the final spot exceeds the threshold. Touch means spot reaches or exceeds it at any scheduled observation, including inception and expiry. The fixed threshold does not change with the barrier sweep. Probabilities are fractions; 0.25 means 25%."
    )
    fig = barrier_figure(report)
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)
    st.caption(
        "Distance is 100 × (barrier / initial spot − 1). Error bars are approximate 95% Monte Carlo sampling intervals. Hedge error is root-mean-square terminal P&L of a short barrier plus its hedge, evaluated on the same Heston paths with the same starting BS premium."
    )
    st.subheader("Static and semi-static replication costs")
    st.dataframe(
        report.replication_costs,
        hide_index=True,
        column_order=[
            "barrier",
            "method",
            "cost",
            "fitted_cost",
            "entry_fee",
            "entry_spread_impact",
            "entry_depth_exceedances",
            "terminal_fit_rmse",
            "boundary_fit_rmse",
        ],
    )
    st.caption(
        "These are finite vanilla-strip approximations. cost includes captured-side entry prices, size-dependent impact and fees when execution is enabled; fitted_cost is the frictionless surface value. Continuation trading uses stated spread/depth assumptions. Low cost alone does not mean successful replication."
    )
    with st.expander("Vanilla strip weights and cash deposits"):
        st.dataframe(report.replication_weights, hide_index=True)
        st.dataframe(report.replication_costs, hide_index=True)
        if market.strip_execution is not None:
            st.dataframe(pd.DataFrame(report.diagnostics["execution_metadata"]), hide_index=True)
    st.subheader("Perpetual hedge and replication P&L")
    st.dataframe(
        report.hedge_metrics,
        hide_index=True,
        column_order=[
            "source",
            "barrier",
            "method",
            "paths",
            "mean_pnl",
            "hedge_rmse",
            "pnl_5pct",
            "loss_fraction",
            "max_reconciliation_error",
        ],
    )
    st.caption(
        "Positive P&L benefits the seller of one barrier with its hedge. Dynamic P&L = received premium + perp gains + cash interest − funding − execution costs − remaining barrier liability. All strategies start with the same BS Monte Carlo premium; static/semi-static P&L includes financing the strip's cost. max_reconciliation_error should be approximately zero."
    )
    with st.expander("Per-path P&L decomposition"):
        st.dataframe(report.hedge_metrics, hide_index=True)
        st.dataframe(report.path_results.head(1000), hide_index=True)
        st.caption("The view shows the first 1,000 rows; the CSV includes every evaluated path.")
    if report.diagnostics["heston_calibration"].get("near_parameter_bound"):
        st.warning(
            "Heston calibration reached a parameter bound. Review vanilla residuals and parameter sensitivity; this is a constrained model fit."
        )
    if report.diagnostics["local_regularized_grid_fraction"] > 0:
        st.warning(
            "Some Dupire local-volatility grid values required regularization. The fraction and runtime counts are in diagnostics."
        )
    with st.expander("Calibration, numerical controls and assumptions"):
        st.json(report.diagnostics)
    for label, frame, file in [
        ("Download barrier price sweep", report.prices, "barrier_prices.csv"),
        ("Download replication costs", report.replication_costs, "replication_costs.csv"),
        (
            "Download finish and touch probabilities",
            report.probabilities,
            "barrier_probabilities.csv",
        ),
        ("Download replication weights", report.replication_weights, "barrier_weights.csv"),
        ("Download all simulated hedge P&L", report.path_results, "barrier_hedge_pnl.csv"),
    ]:
        st.download_button(label, frame.to_csv(index=False), file, "text/csv", key=file)
    st.download_button(
        "Download experiment assumptions",
        json.dumps(
            dict(config=asdict(report.config), diagnostics=report.diagnostics),
            indent=2,
            default=str,
        ),
        "barrier_assumptions.json",
        "application/json",
    )


def render_numerical_studies(report, store, snapshot, registry_parameters):
    st.subheader("Numerical stability and monitoring contract")
    st.caption(
        "Sampling intervals omit numerical bias. Change one control at a time while holding calibration fixed. Changing monitoring frequency changes the contract; continuous bridge pricing is available only for exact lognormal GBM."
    )
    with st.form("barrier_convergence_controls"):
        level = st.selectbox("Barrier for numerical study", sorted(report.config.barriers))
        models = st.multiselect(
            "Models in convergence study",
            ["GBM", "Local vol", "Heston"],
            default=["GBM", "Local vol", "Heston"],
        )
        paths = st.selectbox("Convergence baseline paths", [256, 1024, 4096], index=1)
        steps = st.number_input(
            "Convergence baseline integration steps", min_value=10, max_value=500, value=60
        )
        path_text = st.text_input("Path counts to compare", "256,1024,4096")
        step_text = st.text_input("Integration steps to compare", "30,60,120")
        seed_text = st.text_input("Independent seeds to compare", "11,29,47")
        bump_text = st.text_input("Spot bump fractions to compare", "0.005,0.01,0.02")
        grid_text = st.text_input("Spatial grid sizes to compare", "801,1601,3201")
        monitor_text = st.text_input("Monitoring intervals in days to compare", "0.25,1,2")
        submitted = st.form_submit_button("Run numerical and monitoring study")
    try:
        controls = ConvergenceControls(
            barrier=level,
            models=tuple(models),
            paths=paths,
            steps=int(steps),
            path_counts=tuple(int(v) for v in path_text.split(",")),
            step_counts=tuple(int(v) for v in step_text.split(",")),
            seeds=tuple(int(v) for v in seed_text.split(",")),
            bump_fractions=tuple(float(v) for v in bump_text.split(",")),
            grid_points=tuple(int(v) for v in grid_text.split(",")),
            monitoring_days=tuple(float(v) for v in monitor_text.split(",")),
        )
        signature = (
            report.market.snapshot_id,
            repr(report.config),
            repr(report.market.option),
            repr(controls),
        )
        if submitted:
            with st.spinner("Measuring numerical changes with fixed calibration…"):
                st.session_state["barrier_convergence"] = (signature, converge(report, controls))
        saved = st.session_state.get("barrier_convergence")
        if saved and saved[0] == signature:
            trials, summary = saved[1]
            st.dataframe(summary, hide_index=True)
            st.caption(
                "Price range and empirical standard deviation measure observed sensitivity across trials. They do not certify convergence. Monitoring rows compare different contracts. The bridge-minus-discrete difference uses paired paths."
            )
            st.dataframe(
                trials[trials.dimension == "monitoring contract"],
                hide_index=True,
                column_order=[
                    "model",
                    "contract",
                    "monitoring_days",
                    "price",
                    "price_se",
                    "touch_probability",
                    "paired_price_difference_to_discrete",
                    "paired_difference_se",
                ],
            )
            with st.expander("All numerical trials"):
                st.dataframe(trials, hide_index=True)
            st.download_button(
                "Download numerical and monitoring trials",
                trials.to_csv(index=False),
                "barrier_convergence.csv",
                "text/csv",
            )
            render_save_experiment(
                store,
                kind="barrier_convergence",
                snapshots=[snapshot],
                parameters=dict(registry_parameters, controls=asdict(controls)),
                results=dict(trials=trials, summary=summary),
                diagnostics={
                    "fixed_calibration": report.diagnostics["heston_parameters"],
                    "error_scope": "Sampling intervals exclude numerical bias and model uncertainty; monitoring changes contract",
                },
                key="barrier_convergence_run",
            )
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Numerical study unavailable: {exc}")

    st.subheader("Compare model-consistent hedge rules")
    st.caption(
        "Evaluate Black–Scholes delta against conditional-model delta on identical independent paths. Heston decisions see the current simulated variance state; this is an oracle-state experiment because market variance is latent. Nested Monte Carlo sensitivity noise and quantity caps are reported."
    )
    with st.form("barrier_model_hedge_controls"):
        hedge_level = st.selectbox("Barrier for model hedge study", sorted(report.config.barriers))
        model = st.selectbox("Evaluation path dynamics", ["Heston", "Local vol", "GBM"])
        outer = st.selectbox("Model hedge evaluation paths", [4, 16, 32, 64], index=1)
        inner = st.selectbox("Conditional pricing paths per state", [64, 256, 1024], index=1)
        inner_steps = st.number_input(
            "Conditional integration steps", min_value=10, max_value=250, value=30
        )
        vanilla = st.checkbox(
            "Also hedge Heston variance using the discovered vanilla call", value=True
        )
        cap = st.number_input(
            "Maximum absolute vanilla hedge quantity (BTC)", min_value=0.01, value=10.0
        )
        minimum = st.number_input(
            "Minimum vanilla variance sensitivity (cash per variance unit)",
            min_value=0.01,
            value=1.0,
        )
        variance_bump = st.number_input(
            "Conditional variance bump (variance units)",
            min_value=0.0001,
            value=0.0025,
            format="%.4f",
        )
        nested_spot_bump = st.number_input(
            "Conditional delta spot bump (fraction)", min_value=0.001, max_value=0.04, value=0.01
        )
        nested_seed = st.number_input("Model hedge evaluation seed", min_value=0, value=701)
        run = st.form_submit_button("Run model-consistent hedge study")
    try:
        controls = ModelHedgeControls(
            barrier=hedge_level,
            path_model=model,
            paths=outer,
            inner_paths=inner,
            inner_steps=int(inner_steps),
            spot_bump=nested_spot_bump,
            variance_bump=variance_bump,
            max_vanilla_quantity=cap,
            minimum_variance_sensitivity=minimum,
            add_vanilla_hedge=vanilla,
            seed=int(nested_seed),
        )
        signature = (
            report.market.snapshot_id,
            repr(report.config),
            repr(report.market.option),
            repr(controls),
        )
        if run:
            with st.spinner("Repricing conditional states and reconciling model hedges…"):
                st.session_state["barrier_model_hedges"] = (
                    signature,
                    model_hedges(report, controls),
                )
        saved = st.session_state.get("barrier_model_hedges")
        if saved and saved[0] == signature:
            details, summary, audit = saved[1]
            st.dataframe(summary, hide_index=True)
            st.caption(
                "All methods receive the same independently estimated selected-model premium. The vanilla hedge neutralizes the derivative with respect to current Heston variance, subject to the stated quantity cap and minimum-sensitivity floor. Fees and spreads apply to every trade. A few outer paths illustrate behavior; increase paths and check numerical stability before ranking methods."
            )
            with st.expander("Conditional sensitivities and complete model-hedge P&L"):
                st.dataframe(details, hide_index=True)
                st.dataframe(audit.head(1000), hide_index=True)
            st.download_button(
                "Download model-consistent hedge P&L",
                details.to_csv(index=False),
                "model_hedge_pnl.csv",
                "text/csv",
            )
            st.download_button(
                "Download conditional sensitivity audit",
                audit.to_csv(index=False),
                "model_hedge_sensitivities.csv",
                "text/csv",
            )
            render_save_experiment(
                store,
                kind="barrier_model_hedge",
                snapshots=[snapshot],
                parameters=dict(registry_parameters, controls=asdict(controls)),
                results=dict(path_results=details, summary=summary, sensitivity_audit=audit),
                diagnostics={
                    "variance_information": "Current simulated raw Heston variance is observable to this oracle-state hedge; no market variance estimator",
                    "inner_shocks": "Independent from outer paths; common across finite-difference bumps",
                },
                key="barrier_model_hedge_run",
            )
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Model hedge study unavailable: {exc}")


def render_history(report, settings):
    st.subheader("Replication P&L on historical BTC paths")
    st.caption(
        "This replays observed BTC perpetual returns, rebased to the selected starting spot, under the selected current calibration. It is not a historical option-price or execution backtest. No historical prices are synthesized; incomplete windows are counted. Candle closes can miss intrabar touches, and perpetual prices proxy the settlement index."
    )
    source = st.selectbox(
        "Historical path source", ["Deribit public candles", "Uploaded candle CSV"]
    )
    resolution = st.selectbox("Historical candle interval (minutes)", [60, 360, 1440])
    max_paths = st.number_input("Maximum historical windows", min_value=1, max_value=100, value=12)
    fingerprint = (
        report.market.snapshot_id,
        repr(report.config),
        report.market.instrument_name,
        resolution,
        int(max_paths),
    )
    if source == "Deribit public candles":
        environment = st.selectbox("Historical candle environment", ["production", "testnet"])
        end = st.date_input("Historical end date (UTC, exclusive)", value=utc_now().date())
        start = st.date_input(
            "Historical start date (UTC)", value=utc_now().date() - timedelta(days=365)
        )
        fingerprint += (source, environment, str(start), str(end))
        clicked = st.button("Download BTC paths and replay hedges")

        def load():
            return history(
                replace(settings, environment=environment),
                report.market,
                datetime.combine(start, datetime.min.time(), UTC),
                datetime.combine(end, datetime.min.time(), UTC),
                resolution,
                int(max_paths),
            )
    else:
        uploaded = st.file_uploader("BTC perpetual candle CSV", type=["csv"])
        st.caption(
            "Required columns: timestamp_ms (candle open in UTC milliseconds), open, high, low, close. Optional instrument_name/environment/resolution retain provenance."
        )
        fingerprint += (source, uploaded.getvalue() if uploaded else None)
        clicked = st.button("Replay uploaded BTC paths", disabled=uploaded is None)

        def load():
            return build_historical_paths(
                pd.read_csv(uploaded),
                report.market.option.years * DAYS,
                resolution,
                spot=report.market.option.spot,
                max_paths=int(max_paths),
            )

    if clicked:
        try:
            with st.spinner("Preparing observed historical windows and replaying the hedges…"):
                paths = load()
                st.session_state["barrier_history_data"] = (fingerprint, paths)
                details, metrics = replay(report, paths)
                st.session_state["barrier_history_results"] = (fingerprint, details, metrics)
        except Exception as exc:
            st.error(f"Historical replay unavailable: {exc}")
    data = st.session_state.get("barrier_history_data")
    if data and data[0] == fingerprint:
        st.json(data[1].diagnostics)
        st.dataframe(data[1].labels, hide_index=True)
    output = st.session_state.get("barrier_history_results")
    if output and output[0] == fingerprint:
        st.dataframe(output[2], hide_index=True)
        st.dataframe(output[1], hide_index=True)
        st.download_button(
            "Download historical replication P&L",
            output[1].to_csv(index=False),
            "historical_barrier_pnl.csv",
            "text/csv",
        )
