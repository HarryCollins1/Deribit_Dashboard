"""Thin optional Streamlit view. All calculations live in library services."""

from dataclasses import asdict
from datetime import UTC, datetime, time, timedelta

import pandas as pd
import streamlit as st

from deribit_dashboard.config import Settings
from deribit_dashboard.market import WIDTH_EXPLANATION, simulation_summary
from deribit_dashboard.models import utc_now
from deribit_dashboard.pricing import DAYS, MODELS, aggregate_portfolio
from deribit_dashboard.smile_dynamics import SmileDynamicsConfig, compare_smile_dynamics
from deribit_dashboard.surface import PARITY_ESTIMATORS
from deribit_dashboard.volatility import RV_EXPLANATION
from deribit_dashboard.workbench import (
    analyze,
    cash_currency_label,
    load_live_chain,
    load_volatility,
    mock_replay,
    plot_lines,
    price_selection,
    quote_perturbation_sensitivity,
    quote_selection,
    selections,
)

cached_analysis = st.cache_data(ttl=300, max_entries=12)(analyze)
cached_history = st.cache_data(ttl=300, max_entries=8)(load_volatility)
cached_prices = st.cache_data(ttl=300, max_entries=12)(price_selection)
cached_sensitivity = st.cache_data(ttl=300, max_entries=6)(quote_perturbation_sensitivity)
cached_smile_dynamics = st.cache_data(ttl=300, max_entries=8)(compare_smile_dynamics)


MONEYNESS_LABEL = "Log-moneyness ln(strike / forward) (unitless)"
MONEYNESS_GUIDE = (
    "Read left to right as different strikes: left of zero is below the forward, zero means "
    "strike equals forward, and right of zero is above it. For example, −0.1 is about 9.5% "
    "below the forward and +0.1 is about 10.5% above it."
)


def display_table(frame, columns=None):
    """Presentation-only column selection and human-readable dates/size labels."""
    shown = frame.copy()
    if columns is not None:
        shown = shown[[column for column in columns if column in shown]]
    if "expiry" in shown:
        shown["expiry"] = pd.to_datetime(shown.expiry, unit="ms", utc=True)
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            "expiry": st.column_config.DatetimeColumn("Expiry (UTC)", format="DD MMM YYYY HH:mm"),
            "liquidity": None,
            "minimum_displayed_best_quote_size": st.column_config.NumberColumn(
                "Minimum displayed best quote size",
                help="min(best_bid_amount, best_ask_amount), in the captured option amount units; not volume or full-book liquidity.",
            ),
            "liquidity_segment": "Displayed size group",
        },
    )


def static_chart(frame, x, ys, title, group=None, *, x_label, y_label, explanation, **options):
    import matplotlib.pyplot as plt

    st.markdown(explanation)
    fig = plot_lines(frame, x, ys, title, group, x_label=x_label, y_label=y_label, **options)
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


def render_quant(store, settings, catalogue_loader):
    st.header("Surface, risk and market-making laboratory")
    st.caption("Research simulation • independently computed IV • cash prices per underlying unit")
    source = st.selectbox(
        "Laboratory source",
        ["Saved snapshot", "Live full chain", "Live WebSocket chain"],
    )
    snapshot = None
    if source == "Saved snapshot":
        namespace = st.selectbox("Saved environment", [settings.environment])
        saved = store.list(namespace, None, limit=1000)
        if not saved:
            st.info("Capture or import a full snapshot first.")
            return
        labels = {r["snapshot_id"]: f"{r['captured_at']} · {r['currency']}" for r in saved}
        selected = st.selectbox("Laboratory snapshot", list(labels), format_func=labels.get)
        snapshot = store.load(selected)
    elif source == "Live WebSocket chain":
        from deribit_dashboard.live_ui import stream_source

        snapshot = stream_source(settings, "lab_live")
        if snapshot is None:
            return
    else:
        try:
            catalogue = catalogue_loader(settings.environment)
        except Exception as exc:
            st.error(f"Catalogue failed: {exc}")
            return
        currencies = sorted(
            {m["base_currency"] for m in catalogue.values() if m["kind"] == "option"}
        )
        if not currencies:
            st.info("No active options returned by the exchange.")
            return
        currency = st.selectbox("Live chain currency", currencies)
        key = f"quant_live_{settings.environment}_{currency}"
        st.caption("Full REST capture can take minutes. Each quote retains its exchange timestamp.")
        if st.button("Capture full live chain"):
            try:
                with st.spinner("Collecting the discovered instruments…"):
                    st.session_state[key] = load_live_chain(
                        Settings.from_env(currency=currency, environment=settings.environment)
                    )
            except Exception as exc:
                st.error(f"Capture failed: {exc}")
        snapshot = st.session_state.get(key)
        if snapshot is None:
            return
    st.caption(
        f"Snapshot {snapshot.snapshot_id} · {snapshot.captured_at.isoformat()} · {snapshot.source}"
    )
    if st.button("Save laboratory snapshot"):
        store.save(snapshot)
        st.success("Saved with original raw quotes and metadata.")
    pairs = selections(snapshot)
    if not pairs:
        st.info(
            "No options in this snapshot. Open Portfolio strategies for cash, futures and pairs."
        )
        return
    base, settlement = st.selectbox("Base / settlement currency", pairs)
    cash_unit = cash_currency_label(snapshot, base, settlement)
    options = [
        m
        for m in snapshot.instruments.values()
        if m["kind"] == "option"
        and m["base_currency"] == base
        and m.get("settlement_currency") == settlement
    ]
    expiries = sorted({m["expiration_timestamp"] for m in options})
    selected_expiries = st.multiselect(
        "Fit expiries",
        expiries,
        default=expiries,
        format_func=lambda x: datetime.fromtimestamp(x / 1000, UTC).isoformat(),
    )
    strikes = sorted({float(m["strike"]) for m in options})
    selected_strikes = st.select_slider(
        "Fit strike range", strikes, value=(strikes[0], strikes[-1])
    )
    rate = st.number_input("Cash discount rate (decimal/year)", value=0.0, step=0.01)
    stale = st.number_input("Maximum quote age at capture (seconds)", min_value=0.0, value=60.0)
    width = st.number_input("Maximum spread / midpoint", min_value=0.01, value=1.0)
    estimator = st.selectbox(
        "Parity forward estimator",
        PARITY_ESTIMATORS,
        index=1,
        format_func=lambda value: {
            "weighted_mean": "Spread-weighted mean (legacy)",
            "weighted_median": "Spread-weighted median (robust)",
            "huber": "Huber robust fit",
        }[value],
    )
    huber_delta = st.number_input(
        "Huber cutoff (multiples of robust scale)",
        min_value=0.01,
        value=1.345,
        disabled=estimator != "huber",
    )
    result = cached_analysis(
        snapshot,
        base,
        settlement,
        rate,
        stale,
        width,
        selected_expiries,
        selected_strikes,
        parity_estimator=estimator,
        huber_delta=huber_delta,
    )
    st.subheader("Chain hygiene and parity forwards")
    st.dataframe(result.filters, hide_index=True)
    st.caption(
        "Fractions use the entire selected base/settlement chain. Flagged counts overlap; "
        "removed counts are sequential and sum to unique exclusions. Runtime selections are counted."
    )
    display_table(
        result.forwards,
        [
            "expiry",
            "forward",
            "traded_future",
            "basis_bps",
            "pairs",
            "effective_pair_count",
            "pair_dispersion",
            "spread_uncertainty",
            "downweighted_pairs",
        ],
    )
    with st.expander("Full forward and call/put pair audit"):
        display_table(result.forwards)
        display_table(result.parity_pairs)
        st.download_button(
            "Download forward diagnostics",
            result.forwards.to_csv(index=False),
            "parity_forwards.csv",
        )
        st.download_button(
            "Download parity pair audit",
            result.parity_pairs.to_csv(index=False),
            "parity_pairs.csv",
        )
    static_chart(
        result.forwards,
        "years",
        ["forward", "traded_future"],
        "Parity forward versus traded future",
        x_label="Time remaining to expiry (days)",
        y_label=f"Forward price ({cash_unit} per {base})",
        x_scale=DAYS,
        explanation=(
            "**Do the options and futures agree on the forward?** Each point is an expiry. "
            "The option-implied line comes from matching calls and puts; the futures line comes "
            "from captured bid/ask midpoints. Overlapping lines mean agreement. A gap is the "
            "parity-versus-future basis: inspect the spreads and timestamps before interpreting it. "
            "Moving right means more days remaining until expiry."
        ),
    )
    st.caption(
        "F = K + (call midpoint − put midpoint) / exp(−rT). Precision weights use inverse squared "
        "pair spreads; the selected mean, weighted median or Huber estimator pools the pairs. "
        "The parity interval intersects executable pair bounds; low > high means inconsistency. "
        "Basis = parity forward − same-expiry future midpoint. Asynchronous quotes, bid/ask costs "
        "and discount assumptions can create basis. A missing dated future stays missing."
    )
    st.caption(
        "Pair dispersion is the standard deviation of all pair forwards. Effective pair count "
        "is 1 / sum(normalized weights²); a small value exposes dependence on a few quotes. "
        "Spread uncertainty is sqrt(sum((normalized weight × pair half-spread)²)), a bid/ask "
        "precision proxy, not a confidence interval. Weighted-median diagnostics retain precision "
        "weights; Huber diagnostics include its disclosed downweights. No robust-fit pair is dropped."
    )
    st.subheader("Bottom-up quote investigation")
    name = st.selectbox("Inspect quote", result.chain.instrument_name.tolist())
    inspected = result.chain[result.chain.instrument_name == name]
    display_table(
        inspected,
        [
            "instrument_name",
            "strike",
            "side",
            "bid",
            "ask",
            "own_iv",
            "fitted_iv",
            "minimum_displayed_best_quote_size",
            "fit_status",
            "reasons",
        ],
    )
    st.caption(
        "Minimum displayed best quote size = min(best_bid_amount, best_ask_amount). "
        "The raw ticker below shows both inputs. The download keeps the legacy liquidity alias."
    )
    with st.expander("Full selected-quote audit and captured inputs"):
        display_table(inspected)
        st.json(
            {
                "metadata": snapshot.instruments[name],
                "ticker": snapshot.tickers.get(name),
                "collection_error": snapshot.errors.get(name),
            }
        )
    st.dataframe(result.outcomes(), hide_index=True)
    st.download_button("Download chain audit", result.chain.to_csv(index=False), "chain_audit.csv")
    smile_comparison = None
    if result.surface is None:
        st.warning(f"Surface unavailable: {result.fit_error}")
    else:
        st.subheader("Constrained total variance and diagnostics")
        display_table(
            pd.DataFrame([result.surface.report]),
            [
                "fitted_quotes",
                "fitted_expiries",
                "hygiene_dropped",
                "iv_dropped",
                "domain_or_sparse_dropped",
                "max_constraint_violation",
            ],
        )
        with st.expander("Full surface optimizer diagnostics"):
            st.json(result.surface.report)
        st.caption(
            "Butterfly constraints hold for the convex piecewise-linear call interpolant. "
            "Calendar ordering is in normalized strike K/F; no extrapolation. Greeks below "
            "are model Greeks at fixed calibrated volatility, not derivatives of spline knots."
        )
        points = result.surface.points()
        static_chart(
            points,
            "log_moneyness",
            ["total_variance"],
            "Fitted total variance",
            "expiry",
            x_label=MONEYNESS_LABEL,
            y_label="Total variance (unitless, σ² × years)",
            explanation=(
                "**How much uncertainty is priced over each option's remaining life?** Each "
                "colour is an expiry, shown as a UTC date in the legend. Higher lines mean more "
                "accumulated variance; longer expiries usually sit higher because there is more "
                "time for variation to accumulate. A higher long-dated curve does not by itself "
                "mean higher annualized volatility. Within one expiry, higher points mean higher "
                "fitted volatility at those strikes.\n\n" + MONEYNESS_GUIDE + "\n\n"
                "The height is annualized decimal volatility squared × time in years. For example, "
                "60% volatility over half a year gives 0.60² × 0.5 = 0.18. A height of 0.18 is "
                "neither an 18% probability nor an 18% annual volatility. The curve's slope compares "
                "strikes; it does not forecast the direction of the underlying price."
            ),
        )
        static_chart(
            result.chain,
            "log_moneyness",
            ["own_iv", "fitted_iv"],
            "Independent and fitted IV",
            "expiry",
            x_label=MONEYNESS_LABEL,
            y_label="Annualized implied volatility (%)",
            y_scale=100,
            explanation=(
                "**How closely does the surface reproduce the observed option prices?** Dots show "
                "IV calculated from captured bid/ask midpoints; solid lines show the fitted surface "
                "at the observed strikes. Matching colours identify the same expiry. A gap is a "
                "fitting adjustment: fitted IV above the dots means a higher fitted option price. "
                "Look for persistent gaps across a strike range.\n\n" + MONEYNESS_GUIDE + "\n\n"
                "The vertical axis uses percentages: 53.5% here is 0.535 in the audit table. "
                "Both calls and puts can appear at the same horizontal position. Missing fitted "
                "values leave gaps; check fit_status for excluded observations."
            ),
        )
        static_chart(
            result.chain,
            "log_moneyness",
            ["iv_difference_points"],
            "Own IV minus Deribit mark IV (points)",
            "expiry",
            x_label=MONEYNESS_LABEL,
            y_label="Own IV − exchange mark IV (percentage points)",
            zero_line=True,
            explanation=(
                "**Does our independent IV agree with the exchange's reported mark IV?** Each "
                "dot compares one option. The dashed zero line means agreement; +2 means our IV "
                "is two volatility percentage points higher, and −2 means it is two points lower. "
                "For example, 55% own IV minus 53% exchange IV gives +2 points.\n\n"
                + MONEYNESS_GUIDE
                + "\n\n"
                "This compares observed-midpoint IV with exchange mark IV. The fitted surface "
                "is not part of this subtraction. Different reference prices and forward "
                "assumptions can explain a gap; exchange IV is only a cross-check. "
                "Tiny differences such as 1e−8 points (0.00000001 points) "
                "reflect numerical precision and are effectively zero."
            ),
        )
        st.caption(
            "Deribit mark IV is a comparison only; it is never an inversion or fitting input. "
            "Midpoint-versus-mark prices and forward assumptions also explain cross-check differences."
        )
        display_table(result.segments)
        smile_comparison = render_smile_dynamics(
            result,
            (
                snapshot.snapshot_id,
                base,
                settlement,
                rate,
                stale,
                width,
                tuple(selected_expiries),
                tuple(selected_strikes),
                estimator,
                huber_delta,
            ),
            cash_unit,
            base,
        )
        render_risk(result, snapshot, rate, cash_unit)
    sensitivity = render_sensitivity(
        snapshot,
        base,
        settlement,
        rate,
        stale,
        width,
        selected_expiries,
        selected_strikes,
        estimator,
        huber_delta,
    )
    from deribit_dashboard.experiment_ui import render_save_experiment

    experiment_results = dict(
        chain=result.chain,
        filters=result.filters,
        forwards=result.forwards,
        parity_pairs=result.parity_pairs,
        segments=result.segments,
        surface_points=result.surface.points() if result.surface is not None else pd.DataFrame(),
    )
    if sensitivity is not None:
        experiment_results.update(
            sensitivity_summary=sensitivity.summary,
            sensitivity_forwards=sensitivity.forwards,
            sensitivity_quotes=sensitivity.quotes,
        )
    parameters = dict(
        base=base,
        settlement=settlement,
        rate=rate,
        stale_seconds=stale,
        max_relative_width=width,
        expiries=selected_expiries,
        strike_range=list(selected_strikes),
        parity_estimator=estimator,
        huber_delta=huber_delta,
    )
    diagnostics = (
        dict(result.surface.report)
        if result.surface is not None
        else {"fit_error": result.fit_error}
    )
    if smile_comparison is not None:
        config, comparison = smile_comparison
        parameters["smile_dynamics"] = asdict(config)
        experiment_results.update(
            smile_dynamics_curves=comparison.curves,
            smile_dynamics_summary=comparison.summary,
            smile_dynamics_diagnostics=comparison.diagnostics,
        )
        diagnostics["smile_dynamics"] = comparison.explanation
    render_save_experiment(
        store,
        kind="quant",
        snapshots=[snapshot],
        parameters=parameters,
        results=experiment_results,
        diagnostics=diagnostics,
        key="quant_run",
    )
    st.caption(
        "Use the Portfolio strategies workspace to build pairs, spreads and barrier portfolios."
    )
    render_history(snapshot, settings, base)


def render_smile_dynamics(analysis, calibration_signature, cash_unit, base):
    st.subheader("Smile movement and hedge sensitivity")
    st.caption(
        "The IV charts above show one captured market. They do not assume sticky strike or "
        "sticky delta. This comparison moves that same fitted smile under explicit scenarios."
    )
    if not st.checkbox("Show smile movement comparison"):
        return None
    choices = analysis.chain[analysis.chain.fitted_iv.notna()].sort_values(
        "log_moneyness", key=lambda values: values.abs()
    )
    if choices.empty:
        st.info("This comparison requires a successfully fitted option.")
        return None
    with st.form("quant_smile_movement_controls"):
        instrument = st.selectbox(
            "Smile comparison option", choices.instrument_name.tolist(), key="quant_smile_option"
        )
        shock = st.number_input(
            "Spot and forward move (%)", min_value=-99.0, max_value=500.0, value=5.0, step=1.0
        )
        bump = st.number_input(
            "Greek bump (basis points of spot)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=1.0,
        )
        st.caption(
            "Remaining time, discount rate and forward/spot carry stay fixed. Spot and the "
            "selected expiry's parity forward move by the same percentage. No extra IV shock "
            "is added. The bump controls the numerical hedge-sensitivity calculation."
        )
        submitted = st.form_submit_button("Run smile movement comparison")
    config = SmileDynamicsConfig(instrument, shock / 100, bump / 10000)
    signature = (calibration_signature, config)
    if submitted:
        try:
            with st.spinner("Comparing smile movement and hedge sensitivities…"):
                comparison = cached_smile_dynamics(analysis, config)
            st.session_state["quant_smile_comparison"] = (signature, comparison)
        except ValueError as exc:
            st.session_state.pop("quant_smile_comparison", None)
            st.error(str(exc))
            return None
    saved = st.session_state.get("quant_smile_comparison")
    if saved is None or saved[0] != signature:
        st.info("Run the comparison to apply these controls to the current calibration.")
        return None
    comparison = saved[1]
    with st.expander("Smile movement assumptions and units"):
        st.markdown(comparison.explanation)
    axis = st.selectbox("Smile chart horizontal axis", ["Strike", "Log-moneyness"])
    expiry = pd.to_datetime(comparison.summary.expiry.iloc[0], unit="ms", utc=True)
    static_chart(
        comparison.curves,
        "strike" if axis == "Strike" else "log_moneyness",
        ["iv_percent"],
        f"Smile movement — {expiry.strftime('%d %b %Y, %H:%M UTC')}",
        "scenario",
        x_label=f"Strike ({cash_unit} per {base})" if axis == "Strike" else MONEYNESS_LABEL,
        y_label="Annualized implied volatility (%)",
        highlights=comparison.summary,
        explanation=(
            "**Follow the same option through the move.** Stars mark the selected contract. "
            "Sticky strike keeps each dollar strike's IV unchanged. Sticky forward delta keeps "
            "the forward-call-delta smile unchanged. In strike coordinates the latter smile "
            "moves with the forward; in log-moneyness coordinates it overlaps the initial "
            "smile for this fixed-time experiment. A gap means unavailable support, not zero IV."
        ),
    )
    st.caption(
        f"Values below are per underlying unit (an option on one {base}), before fees and "
        f"contract-size scaling. Spot, forward and strike are in {cash_unit} per {base}; "
        f"price_cash and pnl_cash are in {cash_unit}; price_coin = price_cash / spot is in "
        f"{base}. IV is annualized percent; years is remaining maturity. P&L is hypothetical "
        "revaluation relative to the initial fitted value, with no time elapsed."
    )
    display_table(
        comparison.summary,
        [
            "mode",
            "expiry",
            "spot",
            "forward",
            "strike",
            "years",
            "iv_percent",
            "price_cash",
            "price_coin",
            "pnl_cash",
            "status",
        ],
    )
    st.markdown(
        "**Fixed-IV and smile-movement Greeks answer different questions.** Fixed-IV Greeks "
        "freeze the scenario's volatility. Total Greeks move spot and forward again and "
        "re-evaluate the selected smile rule. Initial has no total Greeks because a single "
        "snapshot specifies no movement rule. Delta is value change per one-unit spot move; "
        "gamma is delta change per one-unit spot move. Cash/coin indicates the value unit."
    )
    display_table(
        comparison.summary,
        [
            "mode",
            "fixed_iv_delta_cash",
            "total_delta_cash",
            "fixed_iv_gamma_cash",
            "total_gamma_cash",
            "fd_status",
        ],
    )
    with st.expander("Coin Greeks, bump sensitivity and full selected-option audit"):
        st.caption(
            "Coin Greeks differentiate P/S: delta_coin = delta_cash/S − P/S²; "
            "gamma_coin = gamma_cash/S − 2 delta_cash/S² + 2P/S³. The full audit compares "
            "the selected bump with half that bump. The fitted call interpolant is piecewise "
            "linear: movement gamma can be near zero between knots and spike across knots. "
            "A bump-dependent value is a numerical scenario sensitivity, not a smooth "
            "analytical gamma. Unsupported bumps remain unavailable."
        )
        display_table(comparison.summary)
    st.markdown("**Scenario coverage and sampled price-shape checks**")
    display_table(comparison.diagnostics)
    st.caption(
        "Checks cover sampled call-price bounds, monotonicity and convexity in this expiry. "
        "Nonzero failure counts identify violations on the sampled grid; the shocked smile "
        "is not repaired or refitted. "
        "They are diagnostics, not a global no-arbitrage proof or a calendar-evolution test. "
        "Unsupported grid points remain in the download with their status."
    )
    st.download_button(
        "Download smile movement curves",
        comparison.curves.to_csv(index=False),
        "smile_movement_curves.csv",
    )
    st.download_button(
        "Download smile movement option audit",
        comparison.summary.to_csv(index=False),
        "smile_movement_option.csv",
    )
    return config, comparison


def render_sensitivity(
    snapshot, base, settlement, rate, stale, width, expiries, strikes, estimator, huber_delta
):
    st.subheader("Calibration sensitivity to bid, midpoint and ask")
    st.caption(
        "Refit all three quote marks to see which forwards and fitted prices depend on the "
        "choice of mark. Every scenario keeps captured bid/ask spreads and records failed quotes."
    )
    signature = (
        snapshot.snapshot_id,
        base,
        settlement,
        rate,
        stale,
        width,
        tuple(expiries),
        tuple(strikes),
        estimator,
        huber_delta,
    )
    if st.button("Run bid / midpoint / ask sensitivity"):
        with st.spinner("Re-estimating forwards and fitting three quote scenarios…"):
            sensitivity = cached_sensitivity(
                snapshot,
                base,
                settlement,
                rate,
                stale,
                width,
                expiries,
                strikes,
                parity_estimator=estimator,
                huber_delta=huber_delta,
            )
        st.session_state["quant_sensitivity"] = (signature, sensitivity)
    saved = st.session_state.get("quant_sensitivity")
    if saved is None or saved[0] != signature:
        return
    sensitivity = saved[1]
    st.caption(sensitivity.explanation)
    display_table(sensitivity.summary)
    display_table(
        sensitivity.forwards,
        [
            "quote_mark",
            "expiry",
            "pairs",
            "forward",
            "forward_change_bps",
            "effective_pair_count",
            "spread_uncertainty",
        ],
    )
    display_table(
        sensitivity.quotes,
        [
            "instrument_name",
            "successful_fit_scenarios",
            "bid_fitted_price",
            "mid_fitted_price",
            "ask_fitted_price",
            "fitted_price_range",
            "fitted_iv_range_points",
        ],
    )
    with st.expander("All perturbation results and exclusions"):
        display_table(sensitivity.quotes)
        for mark, scenario in sensitivity.scenarios.items():
            st.write(f"{mark.capitalize()} quote-mark outcomes")
            display_table(scenario.outcomes())
            st.download_button(
                f"Download {mark} calibration audit",
                scenario.chain.to_csv(index=False),
                f"calibration_{mark}_audit.csv",
            )
    st.download_button(
        "Download calibration sensitivity",
        sensitivity.quotes.to_csv(index=False),
        "calibration_sensitivity.csv",
    )
    st.download_button(
        "Download sensitivity forward changes",
        sensitivity.forwards.to_csv(index=False),
        "calibration_forward_sensitivity.csv",
    )
    return sensitivity


def render_risk(result, snapshot, rate, cash_unit):
    st.subheader("Model pricing and portfolio risk")
    advanced = st.checkbox("Advanced model scenarios", False)
    model, american, parity_carry, dividend = "Black-Scholes", False, True, 0.0
    if advanced:
        model = st.selectbox("Pricing model", list(MODELS))
        american = st.checkbox("American exercise scenario", False, disabled=model != "Binomial")
        if model != "Binomial":
            american = False
        parity_carry = st.checkbox("Use parity-implied carry", True)
        dividend = st.number_input(
            "Continuous dividend yield (scenario)", value=0.0, step=0.01, disabled=parity_carry
        )
    else:
        st.caption(
            "Black–Scholes with parity-implied carry. Enable advanced scenarios for normal-volatility or American/dividend comparisons."
        )
    names = result.chain.loc[result.chain.fit_status == "fitted", "instrument_name"].tolist()
    if not names:
        return
    selected = st.multiselect(
        "Portfolio instruments", names, default=names[len(names) // 2 : len(names) // 2 + 1]
    )
    if not selected:
        return
    try:
        priced = cached_prices(
            result, snapshot, selected, model, rate, dividend, american, parity_carry
        )
    except ValueError as exc:
        st.warning(f"Model cannot price this selection: {exc}")
        return
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "option"} for r in priced])
    display_table(
        table,
        [
            "instrument_name",
            "model_name",
            "cash_value",
            "coin_value",
            "cash_delta",
            "cash_gamma",
            "cash_vega",
            "cash_theta_per_gamma",
        ],
    )
    with st.expander("All cash/coin Greeks and finite-difference checks"):
        display_table(table)
    st.download_button(
        "Download full model risk audit", table.to_csv(index=False), "model_risk_audit.csv"
    )
    quantities = st.data_editor(
        pd.DataFrame({"instrument_name": selected, "contracts": [0.0] * len(selected)}),
        disabled=["instrument_name"],
        key="quant_positions",
        hide_index=True,
    )
    portfolio = aggregate_portfolio(priced, quantities.contracts.to_list())
    display_table(
        pd.DataFrame(
            [
                dict(
                    unit=unit,
                    **{
                        greek: portfolio[f"{unit}_{greek}"]
                        for greek in ("value", "delta", "gamma", "vega", "theta", "theta_per_gamma")
                    },
                )
                for unit in ("cash", "coin")
            ]
        )
    )
    st.caption(
        "Signed quantities are contract counts, multiplied by metadata contract_size. "
        "Cash value V; coin value U=V/S. U_S=V_S/S−V/S²; "
        "U_SS=V_SS/S−2V_S/S²+2V/S³; U_vol=V_vol/S; U_time=V_time/S at fixed S. "
        "Theta is per calendar day; vega per 1.0 native volatility (normal: cash/√year). "
        "Theta/gamma is computed after aggregation. FD columns cross-check analytic models; "
        "tree Greeks are finite differences with a 1% spot bump. American exercise and continuous "
        "dividends are hypothetical scenarios; exchange options are European."
    )
    with st.expander("Illustrative market-making accounting — not a trading backtest."):
        if st.checkbox("Show two-event accounting demonstration", key="quant_show_accounting"):
            render_accounting_demonstration(result, snapshot, priced, selected, model, cash_unit)


def render_accounting_demonstration(result, snapshot, priced, selected, model, cash_unit):
    st.subheader("Two-sided quote and two-event accounting demonstration")
    st.caption(
        "This demonstration schedules two mock orders to explain P&L accounting. "
        "It does not estimate live fills or market-making survival."
    )
    quote_name = st.selectbox("Quote instrument", selected)
    chosen = dict(next(r for r in priced if r["instrument_name"] == quote_name))
    chosen["model_name"] = model
    horizon = st.number_input("Markout horizon (seconds)", min_value=1.0, value=60.0)
    st.caption(WIDTH_EXPLANATION)
    try:
        quoted = quote_selection(result, snapshot, chosen, horizon)
        st.json(quoted)
        spot_move = st.number_input("Mock first spot move (fraction)", min_value=-0.99, value=0.01)
        vol_move = st.number_input("Mock volatility move (fraction)", min_value=-0.99, value=0.0)
        contracts = st.number_input("Mock contracts per order", min_value=0.0, value=1.0)
        fee = st.number_input("Cash fee per contract per fill", min_value=0.0, value=0.0)
        limit = st.number_input("Inventory limit (contracts)", min_value=0.0, value=10.0)
        events = mock_replay(
            chosen,
            quoted,
            snapshot.instruments[quote_name],
            spot_move,
            vol_move,
            horizon,
            contracts,
            fee,
            limit,
        )
        display_table(
            events,
            [
                "fill_contracts",
                "inventory",
                "spread_capture",
                "adverse_selection",
                "inventory_revaluation",
                "hedge_pnl",
                "fees",
                "pnl",
                "cumulative_pnl",
            ],
        )
        if st.checkbox(
            "Show full accounting reconciliation", key="quant_accounting_reconciliation"
        ):
            display_table(events)
            st.download_button(
                "Download accounting demonstration",
                events.to_csv(index=False),
                "accounting_demo.csv",
            )
        capital = st.number_input("Scenario starting cash", min_value=0.01, value=10000.0)
        summary = simulation_summary(events, capital)
        st.json(
            {key: value for key, value in summary.items() if key != "survived_cash_loss_budget"}
        )
        st.caption(
            "cash_budget_breached means a modeled event-end cash balance reached zero or below; exchange margin and liquidation are not included."
        )
        static_chart(
            events.reset_index(),
            "index",
            ["cumulative_pnl"],
            "Two-event accounting demonstration P&L",
            x_label="Completed mock event (count)",
            y_label=f"Cumulative profit / loss ({cash_unit})",
            x_offset=1,
            zero_line=True,
            explanation=(
                "**Has the mock quoting scenario made or lost money so far?** Point 1 is the "
                "result after the first mock order and its markout; point 2 includes the second "
                f"order and markout. Each interval lasts {horizon:g} seconds. Above the dashed zero "
                "line is profit; below it is loss. The last point is final_pnl, including option "
                "trades, inventory revaluation, the simulated hedge and fees.\n\n"
                "The horizontal axis counts completed events, not real trades observed on the "
                "exchange. The replay starts at zero inventory independently of the portfolio "
                "editor. Use the component table to explain each move in P&L."
            ),
        )
        st.caption(
            "Two mock market orders: buyer lifts our ask before the selected spot/vol move; "
            "seller hits our bid before spot returns. Adverse selection is the signed post-fill "
            "fair-value markout. Spread + markout + existing inventory revaluation + hedge − fees "
            "equals cash-flow/end-mark P&L. These scenarios do not establish live survival: "
            "queue priority, real fill probabilities, margin/liquidation and hedge costs are absent."
        )
    except ValueError as exc:
        st.warning(f"Quote/scenario unavailable: {exc}")


def render_history(snapshot, settings, base):
    st.subheader("DVOL and independently computed realized volatility")
    perpetuals = sorted(
        name
        for name, m in snapshot.instruments.items()
        if m["kind"] == "future"
        and m["base_currency"] == base
        and m.get("settlement_period") == "perpetual"
    )
    if not perpetuals:
        st.info("This snapshot has no discovered perpetual instrument for RV history.")
        return
    perpetual = st.selectbox("Perpetual for realized volatility", perpetuals)
    start = st.date_input("Volatility history start", utc_now().date() - timedelta(days=60))
    end = st.date_input("Volatility history end", utc_now().date())
    window = st.number_input("Rolling hourly returns", min_value=2, value=720)
    st.caption(RV_EXPLANATION)
    if st.button("Load volatility history"):
        try:
            frame = cached_history(
                settings,
                base,
                perpetual,
                datetime.combine(start, time.min, UTC),
                datetime.combine(end, time.min, UTC),
                int(window),
            )
            static_chart(
                frame,
                "timestamp",
                ["own_realized_pct", "published_realized_pct", "dvol_pct"],
                "Annualized volatility (%)",
                x_label="Observation date and time (UTC)",
                y_label="Annualized volatility (%)",
                explanation=(
                    "**How does option-implied volatility compare with past price variation?** "
                    "DVOL is the exchange's implied-volatility index. Own realized volatility is "
                    "calculated from perpetual candle returns; exchange realized volatility is "
                    "the published historical series. All three use annualized percentages: "
                    "a value of 50% means 50% annualized volatility.\n\n"
                    "Moving right advances through UTC observation time. The gap between the two "
                    "realized series helps diagnose differences in inputs and calculation methods. "
                    "The DVOL gap compares implied with backward-looking realized volatility; it "
                    "is not a guaranteed trading return. Missing observations remain gaps. These "
                    "histories come from the selected exchange environment."
                ),
            )
            display_table(
                frame,
                [
                    "timestamp",
                    "own_realized_pct",
                    "published_realized_pct",
                    "dvol_pct",
                    "rv_difference_points",
                    "missing_interval",
                ],
            )
            st.download_button(
                "Download volatility reconciliation", frame.to_csv(index=False), "volatility.csv"
            )
        except Exception as exc:
            st.error(f"History unavailable: {exc}")
