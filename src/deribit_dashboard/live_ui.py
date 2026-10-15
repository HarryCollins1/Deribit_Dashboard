"""Optional Streamlit controls over the library's public ticker feeds."""

from dataclasses import replace
from uuid import uuid4

import pandas as pd
import streamlit as st

from deribit_dashboard.analytics import snapshot_frame
from deribit_dashboard.feeds import MarketFeed

UPDATE_MODES = {"Real-time public stream (100 ms)": "realtime", "2-second aggregation": "2s"}


@st.cache_resource(max_entries=16, show_spinner=False)
def _cached_feed(settings, instrument_names, session_id):
    settings = replace(settings, ticker_interval="100ms", feed_mode="realtime")
    return MarketFeed(settings, instrument_names=instrument_names, idle_timeout=120)


def owned_feed(settings, instrument_names=None, owner="market"):
    session_id = st.session_state.setdefault("market_session_id", str(uuid4()))
    names = tuple(instrument_names) if instrument_names is not None else None
    feed = _cached_feed(settings, names, session_id)
    previous = st.session_state.get(f"{owner}_feed")
    if previous is not None and previous is not feed:
        previous.stop()
    st.session_state[f"{owner}_feed"] = feed
    return feed


def feed_status(feed):
    status = feed.status()
    st.caption(
        f"Feed: {status['state']} · {status['ticker_count']:,}/{status['instrument_count']:,} tickers "
        f"· {status['updates']:,} streaming updates · {status['reconnects']} reconnects. "
        f"Last received: {status['last_received_at'] or 'waiting'}."
    )
    if status["last_error"]:
        st.warning(f"Last feed diagnostic: {status['last_error']}")
    with st.expander("Feed coverage, late updates and buffer diagnostics"):
        st.json(status)


def bucket_table(feed):
    rows = feed.buckets()
    if rows:
        st.dataframe(pd.DataFrame(rows).tail(100), hide_index=True)
        st.caption(
            "Latest 100 completed UTC-aligned 2-second buckets. OHLC uses the stated quote "
            "reference in native quote currency; update_count counts ticker messages, not trades. "
            "No empty bars are manufactured and 24-hour volume is never summed."
        )
    else:
        st.info("Waiting for a completed 2-second bucket. REST seed quotes do not create bars.")


def stream_source(settings, key):
    """Monitor a feed and explicitly freeze a coherent input for expensive research runs."""
    selected = st.selectbox("Stream update mode", list(UPDATE_MODES), key=f"{key}_update_mode")
    mode = UPDATE_MODES[selected]
    enabled = st.checkbox("Connect BTC / ETH live stream", key=f"{key}_connected")
    feed = owned_feed(settings, owner=key)
    if enabled:
        feed.start()
    else:
        feed.stop()
    st.caption(
        "Public 100 ms exchange ticker updates are consumed in the background. Real-time display "
        "refreshes every 250 ms; the other mode displays completed 2-second quote buckets. "
        "Use the latest market as a fixed input for reproducible calibration and simulations."
    )

    @st.fragment(run_every=(0.25 if mode == "realtime" else 2) if enabled else None)
    def monitor():
        feed_status(feed)
        snapshot = feed.snapshot(mode)
        if snapshot is not None:
            frame = snapshot_frame(snapshot)
            st.dataframe(
                frame[
                    [
                        "instrument_name",
                        "kind",
                        "best_bid_price",
                        "best_ask_price",
                        "mark_price",
                        "quote_currency",
                        "exchange_time",
                    ]
                ].tail(100),
                hide_index=True,
            )
            st.caption(
                "Showing the final 100 instruments in the current feed; coverage counts include all subscribed instruments."
            )
            if st.button("Use latest market for analysis", key=f"{key}_freeze"):
                st.session_state[f"{key}_snapshot"] = snapshot
                st.rerun()
        else:
            st.info("Connect the stream and wait for quotes before selecting an analysis snapshot.")
        if mode == "2s":
            bucket_table(feed)

    monitor()
    return st.session_state.get(f"{key}_snapshot")
