"""Streamlit UI, launched via the package command rather than the repository path."""

import asyncio
import json
import os
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from deribit_dashboard.analytics import futures_curve, option_points, snapshot_frame, surface_grid
from deribit_dashboard.client import DeribitClient
from deribit_dashboard.config import Settings
from deribit_dashboard.history import download_candles
from deribit_dashboard.models import Snapshot, utc_now
from deribit_dashboard.storage import SnapshotStore
from deribit_dashboard.universe import in_universe, product_label


def number(value, decimals=4):
    return "—" if pd.isna(value) else f"{value:,.{decimals}f}"


def chart(fig, key):
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=430)
    st.plotly_chart(fig, width="stretch", key=key)


def csv_download(frame, label, filename, key):
    st.download_button(label, frame.to_csv(index=False), filename, "text/csv", key=key)


@st.cache_data(ttl=300, show_spinner=False)
def load_catalogue(environment: str, catalogue_file: str | None = None):
    """Load active BTC/ETH products once per five minutes for UI filtering."""
    if catalogue_file:
        raw = json.loads(Path(catalogue_file).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "instruments" in raw:
            rows = raw["instruments"].values()
        else:
            rows = raw.values() if isinstance(raw, dict) else raw
        return {row["instrument_name"]: row for row in rows if in_universe(row)}

    async def fetch():
        async with DeribitClient(Settings(environment=environment, currency="ANY")) as client:
            return await client.instruments("ANY")

    return asyncio.run(fetch())


def select_instrument(instruments: dict, key_prefix: str, default_name: str | None = None):
    """Three small selectors remain usable with catalogues containing thousands of names."""
    if not instruments:
        raise ValueError("The instrument catalogue is empty")
    rows = list(instruments.values())
    currencies = sorted({row.get("base_currency") or "OTHER" for row in rows})
    default_currency = (
        instruments.get(default_name, {}).get("base_currency") if default_name else None
    )
    currency = st.selectbox(
        "Underlying / base currency",
        currencies,
        index=currencies.index(default_currency) if default_currency in currencies else 0,
        key=f"{key_prefix}_currency",
    )
    kinds = sorted(
        {product_label(row) for row in rows if (row.get("base_currency") or "OTHER") == currency}
    )
    default_kind = product_label(instruments.get(default_name, {})) if default_name else None
    kind = st.selectbox(
        "Instrument type",
        kinds,
        index=kinds.index(default_kind) if default_kind in kinds else 0,
        key=f"{key_prefix}_kind",
    )
    names = sorted(
        name
        for name, row in instruments.items()
        if (row.get("base_currency") or "OTHER") == currency and product_label(row) == kind
    )
    index = names.index(default_name) if default_name in names else 0
    return st.selectbox("Instrument", names, index=index, key=f"{key_prefix}_name")


def render_instrument(snapshot, name, store, settings, show_history=True):
    """Display one selected instrument without assuming it is an option."""
    frame = snapshot_frame(snapshot, settings.stale_seconds).set_index("instrument_name")
    row = frame.loc[name]
    instrument = snapshot.instruments[name]
    ticker = snapshot.tickers.get(name)
    st.subheader(name)
    st.caption(
        f"{instrument['kind']} · {instrument.get('base_currency', '—')} base · "
        f"{instrument.get('settlement_currency', '—')} settlement · "
        f"exchange time {row.exchange_time if ticker else 'missing'}"
    )
    if ticker is None:
        st.warning("This instrument has metadata but no ticker in the selected snapshot.")
        return
    prices = st.columns(4)
    prices[0].metric("Mark", number(row.mark_price, 8))
    prices[1].metric("Last", number(row.last_price, 8))
    prices[2].metric("Best bid", number(row.best_bid_price, 8))
    prices[3].metric("Best ask", number(row.best_ask_price, 8))
    option_like = row.kind in {"option", "option_combo"} and (
        pd.notna(row.mark_iv)
        or any(pd.notna(row[greek]) for greek in ["delta", "gamma", "theta", "vega", "rho"])
    )
    if option_like:
        st.markdown("#### Implied volatility")
        iv = st.columns(3)
        iv[0].metric("Bid IV", number(row.bid_iv, 3) + "%" if pd.notna(row.bid_iv) else "—")
        iv[1].metric("Mark IV", number(row.mark_iv, 3) + "%" if pd.notna(row.mark_iv) else "—")
        iv[2].metric("Ask IV", number(row.ask_iv, 3) + "%" if pd.notna(row.ask_iv) else "—")
        st.markdown("#### Option Greeks")
        greek_columns = st.columns(5)
        for column, greek in zip(greek_columns, ["delta", "gamma", "theta", "vega", "rho"]):
            column.metric(greek.capitalize(), number(row[greek], 8))
        option_fields = [
            "expiry",
            "days_to_expiry",
            "underlying_price",
            "underlying_index",
            "index_price",
            "interest_rate",
            "open_interest",
            "volume_24h",
        ]
        if row.kind == "option":
            option_fields[0:0] = ["option_type", "strike"]
            option_fields.extend(["premium_currency", "mark_usd_indicative"])
        st.dataframe(pd.DataFrame([row[option_fields]]), hide_index=True, width="stretch")
        if show_history:
            history = store.option_history(snapshot.environment, name, snapshot.captured_at)
            if history:
                metric = st.selectbox(
                    "Saved-history metric",
                    ["mark_iv", "delta", "gamma", "theta", "vega", "rho"],
                    key=f"history_{snapshot.snapshot_id}_{name}",
                )
                history_frame = pd.DataFrame(history).sort_values("captured_at")
                chart(
                    px.line(history_frame, x="captured_at", y=metric, markers=True),
                    key=f"instrument_history_{snapshot.snapshot_id}_{name}_{metric}",
                )
    elif row.kind == "future":
        curve = futures_curve(frame.reset_index())
        future = curve[curve.instrument_name == name].iloc[0]
        st.dataframe(
            pd.DataFrame(
                [
                    future[
                        [
                            "expiry",
                            "days_to_expiry",
                            "index_price",
                            "basis_pct",
                            "annualized_basis_pct",
                            "funding_8h",
                            "current_funding",
                            "open_interest",
                            "volume_24h",
                        ]
                    ]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    row[
                        [
                            "kind",
                            "base_currency",
                            "quote_currency",
                            "settlement_currency",
                            "contract_size",
                            "index_price",
                            "open_interest",
                            "volume_24h",
                            "state",
                            "quote_age_seconds",
                        ]
                    ]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Raw Deribit response and instrument metadata"):
        st.json({"instrument": instrument, "ticker": ticker})


def render_snapshot(snapshot, store, settings, include_stale):
    frame = snapshot_frame(snapshot, settings.stale_seconds)
    selected_name = select_instrument(snapshot.instruments, f"snapshot_{snapshot.snapshot_id}")
    render_instrument(snapshot, selected_name, store, settings)
    selected_currency = snapshot.instruments[selected_name].get("base_currency")
    context = frame[frame.base_currency == selected_currency]
    options = context[context.kind == "option"]
    age = (utc_now() - snapshot.captured_at).total_seconds()
    window_label = (
        "Collection session age" if snapshot.source == "websocket" else "Collection window"
    )
    st.caption(
        f"{snapshot.environment.upper()} · {snapshot.currency} · {snapshot.source} · "
        f"Captured {snapshot.captured_at:%Y-%m-%d %H:%M:%S} UTC · "
        f"{window_label} {(snapshot.captured_at - snapshot.collection_started_at).total_seconds():.1f}s"
    )
    if snapshot.environment == "demo":
        st.info("Synthetic demonstration data. Prices, IV and Greeks are illustrative.")
    elif age > settings.stale_seconds:
        st.warning(f"The selected snapshot was captured {age / 60:.1f} minutes ago.")
    columns = st.columns(4)
    index = context.index_price.dropna().median()
    columns[0].metric(f"{selected_currency} index", number(index, 2))
    columns[1].metric("Option instruments", len(options))
    columns[2].metric("Ticker coverage", f"{snapshot.coverage:.1%}")
    columns[3].metric("Stale or missing at capture", int(frame.stale.sum()))
    if snapshot.coverage < 1:
        st.warning("This snapshot is incomplete. Missing instruments remain visible in the chain.")
    if frame.clock_ahead.any():
        st.warning("Some exchange timestamps are ahead of the capture clock; excluded by default.")
    chain_tab, surface_tab, curve_tab, quality_tab = st.tabs(
        ["Option chain & Greeks", "Implied volatility", "Futures curve", "Data & downloads"]
    )
    with chain_tab:
        if options.empty:
            st.info("No option metadata in this snapshot.")
        else:
            expiries = sorted(options.expiry.dropna().unique())
            expiry = st.selectbox(
                "Expiry (UTC)", expiries, format_func=lambda x: x.strftime("%Y-%m-%d")
            )
            side = st.radio("Option type", ["Both", "call", "put"], horizontal=True)
            chain = options[options.expiry == expiry].sort_values(["strike", "option_type"])
            if side != "Both":
                chain = chain[chain.option_type == side]
            columns = [
                "instrument_name",
                "option_type",
                "strike",
                "premium_currency",
                "best_bid_price",
                "best_ask_price",
                "mark_price",
                "mark_usd_indicative",
                "bid_iv",
                "mark_iv",
                "ask_iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
                "open_interest",
                "quote_age_seconds",
                "stale",
                "missing",
            ]
            st.dataframe(chain[columns], width="stretch", hide_index=True)
            st.caption(
                "Premium units follow each instrument's settlement currency; inverse coin-settled "
                "options also show an indicative USD conversion using the saved index price. "
                "IV is shown in percentage points. Greeks are the saved per-option ticker values."
            )
            name = st.selectbox("Inspect an option", chain.instrument_name.tolist())
            if name:
                row = chain[chain.instrument_name == name].iloc[0]
                if row.missing:
                    st.warning("No ticker was captured for this option.")
                greek_columns = st.columns(5)
                for col, greek in zip(greek_columns, ["delta", "gamma", "theta", "vega", "rho"]):
                    col.metric(greek.capitalize(), number(row[greek], 6))
                st.dataframe(
                    pd.DataFrame(
                        [
                            row[
                                [
                                    "underlying_price",
                                    "underlying_index",
                                    "index_price",
                                    "interest_rate",
                                    "contract_size",
                                    "exchange_time",
                                    "received_at",
                                    "state",
                                    "spread",
                                    "crossed_quote",
                                ]
                            ]
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                with st.expander("Saved IV and Greeks over time"):
                    history = store.option_history(snapshot.environment, name, snapshot.captured_at)
                    if history:
                        metric = st.selectbox(
                            "History metric", ["mark_iv", "delta", "gamma", "theta", "vega", "rho"]
                        )
                        table = pd.DataFrame(history).sort_values("captured_at")
                        if metric in table:
                            chart(
                                px.line(table, x="captured_at", y=metric, markers=True),
                                key=f"chain_history_{snapshot.snapshot_id}_{name}_{metric}",
                            )
                        st.caption(
                            "Up to 200 saved snapshots, ending at the selected capture cutoff."
                        )
                    else:
                        st.info("Save more snapshots to build this option's history.")
                with st.expander("Raw metadata and ticker"):
                    st.json(
                        {
                            "instrument": snapshot.instruments[name],
                            "ticker": snapshot.tickers.get(name),
                        }
                    )
            csv_download(chain, "Download selected chain", "option_chain.csv", "chain_csv")
    with surface_tab:
        side = st.selectbox("Volatility slice", ["OTM", "call", "put"])
        points = option_points(frame, side, include_stale)
        st.caption("OTM uses puts below the saved underlying forward and calls at or above it.")
        if points.empty:
            st.info("No eligible positive-IV observations. Check coverage and stale filters.")
        else:
            points = points.copy()
            points["expiry_label"] = points.expiry.dt.strftime("%Y-%m-%d")
            points = points.sort_values(["expiry", "log_moneyness"])
            left, right = st.columns(2)
            with left:
                chart(
                    px.line(
                        points,
                        x="log_moneyness",
                        y="mark_iv",
                        color="expiry_label",
                        markers=True,
                        labels={"mark_iv": "Mark IV (%)", "log_moneyness": "log(K/F)"},
                    ),
                    key=f"iv_smile_{snapshot.snapshot_id}_{side}",
                )
            with right:
                nearest = points.assign(distance=points.log_moneyness.abs()).sort_values("distance")
                nearest = nearest.groupby("expiry", sort=True).first().reset_index()
                chart(
                    px.line(
                        nearest,
                        x="days_to_expiry",
                        y="mark_iv",
                        markers=True,
                        labels={
                            "mark_iv": "Nearest-ATM IV (%)",
                            "days_to_expiry": "Days to expiry",
                        },
                    ),
                    key=f"iv_term_{snapshot.snapshot_id}_{side}",
                )
                st.caption(
                    "Term structure uses the observed strike nearest to ATM for each expiry."
                )
            legacy = st.checkbox("Show legacy descriptive surface views", False)
            st.caption(
                "These charts display captured exchange IV. Use Quant laboratory for independently "
                "computed IV and constrained calibration. Legacy 3D interpolation is optional."
            )
            view = (
                st.radio(
                    "Surface view",
                    ["Observed points", "Interpolated surface", "Heatmap"],
                    horizontal=True,
                )
                if legacy
                else None
            )
            if view == "Observed points":
                chart(
                    px.scatter_3d(
                        points,
                        x="log_moneyness",
                        y="days_to_expiry",
                        z="mark_iv",
                        color="mark_iv",
                        hover_name="instrument_name",
                        labels={"mark_iv": "IV (%)", "log_moneyness": "log(K/F)"},
                    ),
                    key=f"iv_observed_{snapshot.snapshot_id}_{side}",
                )
            elif view is not None:
                x, y, z = surface_grid(points)
                if len(y) < 2:
                    st.info("Need at least two expiries with two eligible strikes for this view.")
                elif view == "Interpolated surface":
                    fig = go.Figure(
                        go.Surface(x=x, y=y, z=z, connectgaps=False, colorbar={"title": "IV (%)"})
                    )
                    fig.update_layout(
                        scene={
                            "xaxis_title": "log(K/F)",
                            "yaxis_title": "Days to expiry",
                            "zaxis_title": "IV (%)",
                        }
                    )
                    chart(fig, key=f"iv_surface_{snapshot.snapshot_id}_{side}")
                else:
                    chart(
                        go.Figure(go.Heatmap(x=x, y=y, z=z, colorbar={"title": "IV (%)"})),
                        key=f"iv_heatmap_{snapshot.snapshot_id}_{side}",
                    )
                st.caption(
                    "Linear interpolation in log(K/F) within each expiry's observed range. "
                    "No strike extrapolation. This display is not an arbitrage-free fitted surface."
                )
            csv_download(
                points, "Download surface observations", "iv_observations.csv", "surface_csv"
            )
    with curve_tab:
        curve = futures_curve(context)
        eligible = curve if include_stale else curve[~curve.stale]
        dated = eligible[~eligible.is_perpetual & (eligible.days_to_expiry > 0)]
        if not dated.empty:
            left, right = st.columns(2)
            with left:
                chart(
                    px.line(
                        dated,
                        x="expiry",
                        y="mark_price",
                        markers=True,
                        hover_name="instrument_name",
                        labels={"mark_price": "Futures mark (USD)"},
                    ),
                    key=f"futures_marks_{snapshot.snapshot_id}",
                )
            with right:
                chart(
                    px.line(
                        dated,
                        x="days_to_expiry",
                        y="annualized_basis_pct",
                        markers=True,
                        labels={"annualized_basis_pct": "Annualized basis (%)"},
                    ),
                    key=f"futures_basis_{snapshot.snapshot_id}",
                )
        else:
            st.info("No eligible dated futures observations.")
        st.dataframe(
            curve[
                [
                    "instrument_name",
                    "expiry",
                    "days_to_expiry",
                    "mark_price",
                    "index_price",
                    "best_bid_price",
                    "best_ask_price",
                    "basis_pct",
                    "annualized_basis_pct",
                    "funding_8h",
                    "current_funding",
                    "quote_age_seconds",
                    "stale",
                    "missing",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Basis = (mark / index − 1) × 100. Annualized basis = basis × 365.25 / days to expiry. "
            "Perpetuals are listed separately by instrument name and have no annualized expiry basis. "
            "Funding fields are raw decimal rates: 0.0001 = 0.01%."
        )
        csv_download(curve, "Download futures curve", "futures_curve.csv", "curve_csv")
    with quality_tab:
        st.dataframe(
            frame[
                [
                    "instrument_name",
                    "exchange_time",
                    "received_at",
                    "quote_age_seconds",
                    "state",
                    "stale",
                    "missing",
                    "clock_ahead",
                    "crossed_quote",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        if snapshot.errors:
            st.json(snapshot.errors)
        st.download_button(
            "Download complete snapshot (JSON)",
            snapshot.to_json(),
            f"snapshot_{snapshot.snapshot_id}.json",
            "application/json",
        )
        csv_download(frame, "Download all observations (CSV)", "observations.csv", "all_csv")
        st.caption(
            f"Stale means missing, older than {settings.stale_seconds:g}s at capture, or clock ahead by >2s. "
            "Capture time is a local cutoff; individual instruments have different exchange timestamps."
        )


def candles_panel(settings):
    st.subheader("Download historical trade candles")
    st.caption(
        "These OHLC candles contain trade prices and volume. They do not supply full past Greeks or IV surfaces."
    )
    catalogue_key = f"candle_catalogue_{settings.environment}"
    if st.button("Load BTC / ETH candle instruments"):
        try:
            st.session_state[catalogue_key] = load_catalogue(
                settings.environment, os.getenv("DERIBIT_CATALOG_FILE")
            )
        except Exception as exc:
            st.error(f"Catalogue unavailable: {exc}")
    catalogue = st.session_state.get(catalogue_key)
    if not catalogue:
        st.info(
            "Load the exchange catalogue to choose a BTC/ETH instrument for historical candles."
        )
        return
    with st.form("candles"):
        instrument = st.selectbox("Candle instrument", sorted(catalogue))
        col1, col2, col3 = st.columns(3)
        start = col1.date_input("From (UTC)", utc_now().date() - timedelta(days=7))
        end = col2.date_input("Through (UTC)", utc_now().date())
        resolution = col3.selectbox("Minutes per candle", ["60", "1", "5", "15", "30", "120", "1D"])
        submitted = st.form_submit_button("Download from Deribit")
    if submitted:

        async def fetch():
            async with DeribitClient(settings) as client:
                await client.instrument(instrument)
                return await download_candles(
                    client,
                    instrument.strip(),
                    datetime.combine(start, time.min, UTC),
                    datetime.combine(end + timedelta(days=1), time.min, UTC),
                    resolution,
                )

        try:
            with st.spinner("Downloading candle history…"):
                st.session_state["candles_result"] = pd.DataFrame(asyncio.run(fetch()))
        except Exception as exc:
            st.error(f"Candle download failed: {exc}")
    result = st.session_state.get("candles_result")
    if result is not None:
        if result.empty:
            st.info("Deribit returned no data for this range.")
        else:
            result = result.copy()
            result["timestamp"] = pd.to_datetime(result.timestamp_ms, unit="ms", utc=True)
            st.caption(f"Result: {result.environment.iloc[0]} · {result.instrument_name.iloc[0]}")
            chart(
                go.Figure(
                    go.Candlestick(
                        x=result.timestamp,
                        open=result.open,
                        high=result.high,
                        low=result.low,
                        close=result.close,
                    )
                ),
                key=(
                    f"candles_{settings.environment}_{result.instrument_name.iloc[0]}_"
                    f"{result.resolution.iloc[0]}"
                ),
            )
            csv_download(result, "Download candles CSV", "candles.csv", "candles_csv")


def live_instrument_snapshot(settings, instrument):
    fixture_file = os.getenv("DERIBIT_LIVE_FIXTURE")
    if fixture_file:
        fixture = Snapshot.from_json(Path(fixture_file).read_text(encoding="utf-8"))
        name = instrument["instrument_name"]
        if name not in fixture.tickers:
            raise ValueError(f"Fixture has no ticker for {name}")
        captured = utc_now()
        return Snapshot(
            currency=instrument.get("base_currency") or settings.currency,
            environment=settings.environment,
            source="fixture",
            captured_at=captured,
            collection_started_at=captured,
            instruments={name: instrument},
            tickers={name: fixture.tickers[name]},
            received_at={name: captured.isoformat()},
        )

    async def fetch():
        async with DeribitClient(settings) as client:
            started = utc_now()
            ticker = await client.ticker(instrument["instrument_name"])
            received = utc_now()
            return Snapshot(
                currency=instrument.get("base_currency") or settings.currency,
                environment=settings.environment,
                source="live-rest",
                captured_at=utc_now(),
                collection_started_at=started,
                instruments={instrument["instrument_name"]: instrument},
                tickers={instrument["instrument_name"]: ticker},
                received_at={instrument["instrument_name"]: received.isoformat()},
            )

    return asyncio.run(fetch())


def main():
    st.set_page_config(page_title="Deribit Market Explorer", page_icon="📈", layout="wide")
    st.title("Deribit market dashboard")
    base_settings = Settings.from_env()
    with st.sidebar:
        st.header("Data selection")
        environment = st.selectbox(
            "Deribit environment",
            ["testnet", "production"],
            index=["testnet", "production"].index(base_settings.environment),
        )
        mode = st.radio(
            "Data mode",
            [
                "Snapshot data",
                "Live data",
                "Quant laboratory",
                "Portfolio strategies",
                "Barrier comparison",
                "Walk-forward evaluation",
                "Relative-Value Trading Evaluation",
                "Research experiments",
            ],
        )
        include_stale = st.checkbox("Include stale observations in charts", False)
        st.caption("All dates and times are UTC.")
    settings = Settings.from_env(currency="ANY", environment=environment)
    store = SnapshotStore(settings.database_url)
    try:
        store.initialize()
    except Exception as exc:
        st.error(f"Cannot connect to snapshot storage: {exc}")
        st.stop()
    try:
        if mode == "Research experiments":
            from deribit_dashboard.experiment_ui import render_experiments

            render_experiments(store)
        elif mode == "Relative-Value Trading Evaluation":
            from deribit_dashboard.relative_value_ui import render_relative_value

            render_relative_value(store, settings)
        elif mode == "Walk-forward evaluation":
            from deribit_dashboard.experiment_ui import render_walk_forward

            render_walk_forward(store, settings)
        elif mode == "Barrier comparison":
            from deribit_dashboard.barrier_ui import render_barrier_page

            render_barrier_page(store, settings)
        elif mode == "Portfolio strategies":
            from deribit_dashboard.strategy_ui import render_strategy_page

            render_strategy_page(store, settings)
        elif mode == "Quant laboratory":
            from deribit_dashboard.quant_ui import render_quant

            render_quant(store, settings, load_catalogue)
        elif mode == "Live data":
            from deribit_dashboard.live_ui import (
                UPDATE_MODES,
                bucket_table,
                feed_status,
                owned_feed,
            )

            st.info(
                "Choose a BTC or ETH option, dated future, perpetual or spot instrument. "
                "Real-time mode consumes public 100 ms ticker updates; 2-second mode groups "
                "those updates into completed UTC quote buckets. Save a market moment for replay."
            )
            try:
                with st.spinner("Loading the active Deribit instrument catalogue…"):
                    catalogue = load_catalogue(environment, os.getenv("DERIBIT_CATALOG_FILE"))
            except Exception as exc:
                st.error(f"Could not load Deribit instruments: {exc}")
                st.stop()
            st.caption(
                f"{len(catalogue):,} active instruments discovered; catalogue refreshes every 5 minutes."
            )
            if not catalogue:
                st.info(
                    "No active BTC/ETH options, futures, perpetuals or spot products were returned."
                )
                return
            selected = select_instrument(catalogue, "live")
            update_label = st.selectbox("Market update mode", list(UPDATE_MODES))
            update_mode = UPDATE_MODES[update_label]
            auto_refresh = st.sidebar.checkbox("Update market display", True)
            enabled = st.sidebar.checkbox("Connect live feed", True)
            feed = (
                None
                if os.getenv("DERIBIT_LIVE_FIXTURE")
                else owned_feed(settings, (selected,), "live")
            )
            if feed is not None:
                feed.start() if enabled else feed.stop()

            @st.fragment(
                run_every=(0.25 if update_mode == "realtime" else 2)
                if auto_refresh and enabled
                else None
            )
            def live_view():
                try:
                    if feed is None:
                        snapshot = live_instrument_snapshot(settings, catalogue[selected])
                        st.caption("Recorded fixture playback; no network feed is running.")
                    else:
                        feed_status(feed)
                        snapshot = feed.snapshot(update_mode)
                except Exception as exc:
                    st.error(f"Live ticker failed: {exc}")
                    return
                if snapshot is None:
                    st.info("Waiting for streaming quotes or the first completed 2-second bucket.")
                    return
                render_instrument(snapshot, selected, store, settings)
                st.caption(
                    f"Display cutoff {snapshot.captured_at:%Y-%m-%d %H:%M:%S.%f} UTC · {snapshot.source}. "
                    "Original exchange timestamps are retained. Real-time rendering is every 250 ms."
                )
                if update_mode == "2s" and feed is not None:
                    bucket_table(feed)
                if st.button("Save this market moment", key=f"save_live_{selected}"):
                    store.save(snapshot)
                    st.success(f"Saved snapshot {snapshot.snapshot_id}.")
                st.download_button(
                    "Download this live response as JSON",
                    snapshot.to_json(),
                    f"{selected}_{snapshot.captured_at:%Y%m%dT%H%M%S}.json",
                    "application/json",
                    key=f"live_json_{selected}",
                )

            live_view()
        else:
            source = st.sidebar.selectbox("Snapshot environment", [environment])
            start = st.sidebar.date_input("Start date", utc_now().date() - timedelta(days=30))
            end = st.sidebar.date_input("End date", utc_now().date())
            if start > end:
                st.error("Start date must precede end date.")
            else:
                rows = store.list(
                    source,
                    None,
                    start=datetime.combine(start, time.min, UTC),
                    end=datetime.combine(end, time.max, UTC),
                    limit=1000,
                )
                if not rows:
                    st.info(
                        "No snapshots in this date range. Capture data or import a snapshot JSON first."
                    )
                else:
                    labels = {
                        row["snapshot_id"]: (
                            f"{row['captured_at']} · {row['currency']} · {row['source']} · "
                            f"{row['ticker_count']}/{row['instrument_count']} tickers"
                        )
                        for row in rows
                    }
                    chosen = st.selectbox(
                        "Select historical snapshot", list(labels), format_func=labels.get
                    )
                    st.caption(
                        "Showing up to 1,000 snapshots. Narrow the date range for older captures."
                    )
                    render_snapshot(store.load(chosen), store, settings, include_stale)
        with st.expander("Historical candle downloads"):
            candles_panel(Settings.from_env(currency="ANY", environment=environment))
        with st.expander("Import a previously exported snapshot"):
            uploaded = st.file_uploader("Snapshot JSON", type=["json"])
            if uploaded is not None and st.button("Import snapshot"):
                try:
                    imported = Snapshot.from_json(uploaded.getvalue().decode("utf-8"))
                    store.save(imported)
                    st.success(
                        f"Imported {imported.currency} snapshot into {imported.environment} history."
                    )
                except Exception as exc:
                    st.error(f"Invalid snapshot or storage error: {exc}")
    finally:
        # Fragment refreshes retain this store; disposing its pool is safe and it reconnects as needed.
        store.close()


if __name__ == "__main__":
    main()
