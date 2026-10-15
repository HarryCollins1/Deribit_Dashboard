"""HTTP capture and resilient WebSocket collection. No order or private API methods."""

import asyncio
import copy
import json
import logging
import math
import random
import time

import httpx
import websockets

from deribit_dashboard.client import DeribitClient, DeribitError
from deribit_dashboard.config import Settings
from deribit_dashboard.feeds import FeedState
from deribit_dashboard.models import Snapshot, utc_now
from deribit_dashboard.storage import SnapshotStore
from deribit_dashboard.universe import in_universe, selected_bases

log = logging.getLogger(__name__)


async def capture_snapshot(settings: Settings, instrument_name: str | None = None) -> Snapshot:
    """Fetch full tickers at a bounded rate; capture time is the end of the collection window."""
    started = utc_now()
    async with DeribitClient(settings) as client:
        if instrument_name:
            instrument = await client.instrument(instrument_name)
            if not in_universe(instrument, settings.currency):
                raise ValueError(
                    "Capture requires an active BTC/ETH option, future, perpetual or spot"
                )
            instruments = {instrument_name: instrument}
        else:
            instruments = await client.instruments()
        tickers, received, errors = {}, {}, {}
        queue = asyncio.Queue()
        for name in instruments:
            queue.put_nowait(name)

        async def worker():
            while not queue.empty():
                name = queue.get_nowait()
                try:
                    tickers[name] = await client.ticker(name)
                    received[name] = utc_now().isoformat()
                except (DeribitError, httpx.HTTPError, ValueError) as exc:
                    errors[name] = str(exc)

        await asyncio.gather(*(worker() for _ in range(4)))
    scope = settings.currency
    if instrument_name:
        scope = instruments[instrument_name].get("base_currency") or settings.currency
    return Snapshot(
        currency=scope,
        environment=settings.environment,
        source="rest",
        captured_at=utc_now(),
        collection_started_at=started,
        instruments=instruments,
        tickers=tickers,
        received_at=received,
        errors=errors,
    )


def apply_ticker(tickers: dict, received: dict, data: dict) -> bool:
    """Ignore late older updates, including REST seed responses racing streaming updates."""
    name = data["instrument_name"]
    stamp = data.get("timestamp")
    if (
        isinstance(stamp, bool)
        or not isinstance(stamp, (int, float))
        or not math.isfinite(stamp)
        or stamp < 0
    ):
        raise ValueError("Ticker requires a finite nonnegative exchange timestamp")
    if data["timestamp"] < tickers.get(name, {}).get("timestamp", -1):
        return False
    tickers[name] = copy.deepcopy(data)
    received[name] = utc_now().isoformat()
    return True


class ChainChanged(RuntimeError):
    """Refresh instruments and subscriptions after a lifecycle event."""


async def _stream_session(
    settings: Settings, store: SnapshotStore | None, *, observer=None, instrument_names=None
):
    state = observer if observer is not None else FeedState(settings)
    state.on_status("connecting")
    async with DeribitClient(settings) as client:
        instruments = await client.instruments()
        if instrument_names is not None:
            missing = set(instrument_names) - set(instruments)
            if missing:
                raise ValueError(
                    f"Selected instruments are unavailable in the BTC/ETH catalogue: {sorted(missing)}"
                )
            instruments = {name: instruments[name] for name in instrument_names}
        if not instruments:
            raise RuntimeError("No active instruments returned")
        state.on_catalogue(instruments)
        tickers, received, errors, pending = {}, {}, {}, {}
        chain_changed = asyncio.Event()
        request_id = 0
        async with websockets.connect(
            settings.websocket_url,
            ping_interval=20,
            ping_timeout=20,
            max_size=8 * 1024 * 1024,
            max_queue=1024,
        ) as ws:

            async def rpc(method, params):
                nonlocal request_id
                request_id += 1
                future = asyncio.get_running_loop().create_future()
                pending[request_id] = future
                try:
                    await ws.send(
                        json.dumps(
                            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
                        )
                    )
                    return await asyncio.wait_for(future, settings.request_timeout)
                finally:
                    pending.pop(request_id, None)

            async def reader():
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("id") in pending:
                        future = pending[message["id"]]
                        if not future.done():
                            if "error" in message:
                                error = message["error"]
                                future.set_exception(DeribitError(error["code"], error["message"]))
                            else:
                                future.set_result(message.get("result"))
                    elif message.get("method") == "heartbeat":
                        if message.get("params", {}).get("type") == "test_request":
                            # Acknowledge without blocking the sole receive task on an RPC response.
                            await ws.send(
                                json.dumps({"jsonrpc": "2.0", "method": "public/test", "id": -1})
                            )
                    elif message.get("method") == "subscription":
                        params = message.get("params", {})
                        channel = params.get("channel", "")
                        data = params.get("data", {})
                        if channel.startswith("ticker."):
                            if state.on_ticker(data):
                                apply_ticker(tickers, received, data)
                                errors.pop(data["instrument_name"], None)
                        elif channel.startswith("instrument."):
                            chain_changed.set()
                raise ConnectionError("Deribit WebSocket closed")

            async def seed_missing():
                # Full ticker streams commonly seed immediately; REST fills remaining gaps.
                await asyncio.sleep(5)
                missing = [name for name in instruments if name not in tickers]
                log.info("REST seeding %s instruments missing from WebSocket state", len(missing))
                for name in missing:
                    if name in tickers:
                        continue
                    try:
                        data = await client.ticker(name)
                        if state.on_ticker(data, seed=True):
                            apply_ticker(tickers, received, data)
                    except (DeribitError, httpx.HTTPError, ValueError) as exc:
                        errors[name] = str(exc)
                        state.on_error(name, str(exc))

            async def writer():
                while True:
                    await asyncio.sleep(settings.snapshot_interval)
                    if store is None:
                        # Advance aggregation even when no UI is currently rendering.
                        FeedState.buckets(state)
                        continue
                    snapshot = state.snapshot(settings.feed_mode)
                    if snapshot is None:
                        log.warning("Waiting for first ticker; no empty snapshot saved")
                        continue
                    await asyncio.to_thread(store.save, snapshot)
                    log.info(
                        "Saved %s %s: %.1f%% coverage (%s/%s)",
                        settings.currency,
                        snapshot.snapshot_id,
                        snapshot.coverage * 100,
                        len(tickers),
                        len(instruments),
                    )

            async def lifecycle():
                await chain_changed.wait()
                # Debounce bursts of lifecycle messages before re-discovery.
                await asyncio.sleep(5)
                raise ChainChanged("Instrument lifecycle changed")

            async def catalog_refresh():
                # A periodic full refresh covers ANY scope and instrument kinds without
                # relying on every lifecycle-channel variant being accepted by Deribit.
                while True:
                    await asyncio.sleep(900)
                    refreshed = await client.instruments()
                    if instrument_names is not None:
                        refreshed = {
                            name: row for name, row in refreshed.items() if name in instrument_names
                        }
                    if refreshed != instruments:
                        raise ChainChanged("Instrument catalogue changed")

            tasks = [asyncio.create_task(reader())]
            try:
                await rpc("public/set_heartbeat", {"interval": 30})
                names = list(instruments)
                for offset in range(0, len(names), 100):
                    channels = [
                        f"ticker.{name}.{settings.ticker_interval}"
                        for name in names[offset : offset + 100]
                    ]
                    acknowledged = await rpc("public/subscribe", {"channels": channels})
                    if set(acknowledged or []) != set(channels):
                        raise RuntimeError("Deribit did not acknowledge all ticker subscriptions")
                    await asyncio.sleep(0.3)
                lifecycle_channels = [
                    f"instrument.{event}.{kind}.{base}"
                    for base in selected_bases(settings.currency)
                    for event in ["creation", "state"]
                    for kind in ["option", "future"]
                ]
                await rpc("public/subscribe", {"channels": lifecycle_channels})
                state.on_status("connected")
                log.info("Subscribed to %s tickers on %s", len(instruments), settings.environment)
                tasks.extend(
                    [
                        asyncio.create_task(seed_missing()),
                        asyncio.create_task(writer()),
                        asyncio.create_task(lifecycle()),
                        asyncio.create_task(catalog_refresh()),
                    ]
                )
                # Seeder completion is expected. Any persistent task completion ends the session.
                monitored = set(tasks)
                while monitored:
                    done, _ = await asyncio.wait(monitored, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                        if task is not tasks[1]:
                            return
                        monitored.remove(task)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


async def collect(
    settings: Settings, store: SnapshotStore | None, *, observer=None, instrument_names=None
):
    """Keep reconnecting with capped exponential backoff; re-seed after each connection."""
    failures = 0
    state = observer if observer is not None else FeedState(settings)
    while True:
        connected_at = time.monotonic()
        try:
            await _stream_session(
                settings, store, observer=state, instrument_names=instrument_names
            )
        except ChainChanged:
            failures = 0
            log.info("Refreshing instrument metadata and subscriptions")
            state.on_status("reconnecting", "Refreshing instrument metadata and subscriptions")
        except (
            OSError,
            TimeoutError,
            ConnectionError,
            websockets.WebSocketException,
            httpx.HTTPError,
            DeribitError,
            ValueError,
            RuntimeError,
        ) as exc:
            if time.monotonic() - connected_at > 60:
                failures = 0
            delay = min(2 ** min(failures, 6), 60) + random.random()
            failures += 1
            log.warning("Connection interrupted (%s); reconnecting in %.1fs", exc, delay)
            state.on_status("reconnecting", f"{type(exc).__name__}: {exc}; retry in {delay:.1f}s")
            await asyncio.sleep(delay)
