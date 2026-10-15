"""Bounded public ticker feeds and UTC-aligned two-second quote aggregation.

Every delivered 100ms ticker is consumed. These are quote-reference OHLC bars,
not trade candles: counts are ticker updates, and 24h ticker volume is never summed.
REST seeds populate only the latest view, never manufactured streaming bars.
"""

import asyncio
import copy
import math
import threading
import time
from collections import deque
from datetime import UTC, datetime

from deribit_dashboard.config import Settings
from deribit_dashboard.models import Snapshot, milliseconds, utc_now
from deribit_dashboard.universe import in_universe, product_label


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reference(ticker: dict) -> tuple[float | None, str]:
    mark = ticker.get("mark_price")
    if _finite(mark) and mark >= 0:
        return float(mark), "mark_price"
    bid, ask = ticker.get("best_bid_price"), ticker.get("best_ask_price")
    if _finite(bid) and _finite(ask) and 0 <= bid <= ask and ask > 0:
        return (bid + ask) / 2, "bid_ask_midpoint"
    last = ticker.get("last_price")
    if _finite(last) and last >= 0:
        return float(last), "last_price"
    return None, "unavailable"


class TwoSecondAggregator:
    """Finalize [start, start + 2000ms) buckets in exchange event time.

    Closing uses local UTC minus a bounded lateness allowance. Late updates to
    finalized buckets are counted and rejected; empty buckets are not synthesized.
    Caller serializes access. Timestamps must be nondecreasing per instrument.
    """

    def __init__(self, max_buckets: int = 5000, lateness_ms: int = 250):
        if max_buckets < 1 or not 0 <= lateness_ms <= 2000:
            raise ValueError("Positive bucket limit and 0–2000ms lateness are required")
        self.completed = deque(maxlen=max_buckets)
        self.pending = {}
        self.latest = {}
        self.watermark = -1
        self.lateness_ms = lateness_ms
        self.late_updates = 0
        self.evicted_buckets = 0

    def add(self, ticker: dict, metadata: dict, received_at: str) -> bool:
        stamp = ticker.get("timestamp")
        if not _finite(stamp) or stamp < 0:
            raise ValueError("Ticker requires a finite nonnegative exchange timestamp")
        name = ticker["instrument_name"]
        start = int(stamp) // 2000 * 2000
        if start + 2000 <= self.watermark:
            self.late_updates += 1
            return False
        key = (name, start)
        value, source = _reference(ticker)
        if key not in self.pending:
            self.pending[key] = {
                "instrument_name": name,
                "base_currency": metadata["base_currency"],
                "product": product_label(metadata),
                "quote_currency": metadata.get("quote_currency"),
                "bucket_start_ms": start,
                "bucket_end_ms": start + 2000,
                "bucket_start": datetime.fromtimestamp(start / 1000, UTC).isoformat(),
                "bucket_end": datetime.fromtimestamp((start + 2000) / 1000, UTC).isoformat(),
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "update_count": 0,
                "unpriced_updates": 0,
                "reference_source": source,
            }
        bucket = self.pending[key]
        bucket["update_count"] += 1
        if value is None:
            bucket["unpriced_updates"] += 1
        else:
            if bucket["open"] is None:
                bucket["open"] = bucket["high"] = bucket["low"] = value
                bucket["reference_source"] = source
            elif bucket["reference_source"] != source:
                bucket["reference_source"] = "mixed (mark / midpoint / last)"
            bucket["high"] = max(bucket["high"], value)
            bucket["low"] = min(bucket["low"], value)
            bucket["close"] = value
        for field in ("best_bid_price", "best_ask_price", "best_bid_amount", "best_ask_amount"):
            bucket[field] = ticker.get(field)
        bucket["last_exchange_timestamp_ms"] = stamp
        bucket["last_received_at"] = received_at
        bucket["_ticker"] = copy.deepcopy(ticker)
        return True

    def close(self, now_ms: int) -> None:
        self.watermark = max(self.watermark, now_ms - self.lateness_ms)
        ready = sorted(
            (key for key, row in self.pending.items() if row["bucket_end_ms"] <= self.watermark),
            key=lambda key: (key[1], key[0]),
        )
        for key in ready:
            bucket = self.pending.pop(key)
            if len(self.completed) == self.completed.maxlen:
                self.evicted_buckets += 1
            self.completed.append(bucket)
            self.latest[key[0]] = bucket

    def rows(self) -> list[dict]:
        return [
            copy.deepcopy({k: v for k, v in row.items() if k != "_ticker"})
            for row in self.completed
        ]


class FeedState:
    """Thread-safe observer shared by background feeds and durable CLI collection."""

    def __init__(self, settings: Settings, *, max_events=5000, max_buckets=5000):
        if max_events < 1:
            raise ValueError("Event buffer limit must be positive")
        self.settings = settings
        self._lock = threading.RLock()
        self._instruments = {}
        self._tickers = {}
        self._received = {}
        self._errors = {}
        self._events = deque(maxlen=max_events)
        self._bars = TwoSecondAggregator(max_buckets)
        self._started = utc_now()
        self._status = {
            "state": "stopped",
            "ticker_interval": settings.ticker_interval,
            "provenance": f"Public {settings.ticker_interval} ticker stream (not raw trades)",
            "updates": 0,
            "rest_seeds": 0,
            "out_of_order_updates": 0,
            "invalid_updates": 0,
            "out_of_scope_updates": 0,
            "evicted_events": 0,
            "reconnects": 0,
            "last_error": None,
            "last_received_at": None,
            "instrument_count": 0,
        }

    def on_catalogue(self, instruments: dict) -> None:
        with self._lock:
            self._instruments = copy.deepcopy(
                {
                    name: row
                    for name, row in instruments.items()
                    if in_universe(row, self.settings.currency)
                }
            )
            names = set(self._instruments)
            self._tickers = {n: t for n, t in self._tickers.items() if n in names}
            self._received = {n: t for n, t in self._received.items() if n in names}
            self._errors = {n: t for n, t in self._errors.items() if n in names}
            self._bars.latest = {n: b for n, b in self._bars.latest.items() if n in names}
            self._bars.pending = {k: b for k, b in self._bars.pending.items() if k[0] in names}
            self._status["instrument_count"] = len(names)

    def on_status(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self._status["state"] = state
            if state == "reconnecting":
                self._status["reconnects"] += 1
            if error:
                self._status["last_error"] = error

    def on_error(self, name: str, error: str) -> None:
        with self._lock:
            self._errors[name] = error
            self._status["last_error"] = error

    def on_ticker(self, ticker: dict, *, seed: bool = False, received_at=None) -> bool:
        received_at = received_at or utc_now()
        now_ms = milliseconds(received_at)
        with self._lock:
            name, stamp = ticker.get("instrument_name"), ticker.get("timestamp")
            if name not in self._instruments:
                self._status["out_of_scope_updates"] += 1
                return False
            if not _finite(stamp) or stamp < 0 or stamp > now_ms + 5000:
                self._status["invalid_updates"] += 1
                self.on_error(
                    name, "Invalid exchange timestamp (missing, nonfinite, negative or >5s ahead)"
                )
                return False
            if stamp < self._tickers.get(name, {}).get("timestamp", -1):
                self._status["out_of_order_updates"] += 1
                return False
            receipt = received_at.isoformat()
            self._tickers[name] = copy.deepcopy(ticker)
            self._received[name] = receipt
            self._errors.pop(name, None)
            self._status["last_received_at"] = receipt
            if seed:
                self._status["rest_seeds"] += 1
            else:
                self._status["updates"] += 1
                if len(self._events) == self._events.maxlen:
                    self._status["evicted_events"] += 1
                self._events.append(copy.deepcopy(ticker))
                self._bars.add(ticker, self._instruments[name], receipt)
            self._bars.close(now_ms)
            return True

    def status(self) -> dict:
        with self._lock:
            self._bars.close(milliseconds(utc_now()))
            return {
                **self._status,
                "ticker_count": len(self._tickers),
                "coverage": len(self._tickers) / len(self._instruments) if self._instruments else 0,
                "late_bucket_updates": self._bars.late_updates,
                "evicted_buckets": self._bars.evicted_buckets,
                "completed_buckets": len(self._bars.completed),
                "pending_buckets": len(self._bars.pending),
            }

    def events(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(list(self._events))

    def buckets(self) -> list[dict]:
        with self._lock:
            self._bars.close(milliseconds(utc_now()))
            return self._bars.rows()

    def snapshot(self, mode: str = "realtime") -> Snapshot | None:
        if mode not in {"realtime", "2s"}:
            raise ValueError("Snapshot mode must be realtime or 2s")
        with self._lock:
            now = utc_now()
            self._bars.close(milliseconds(now))
            if mode == "2s":
                tickers, received = {}, {}
                for name, bucket in self._bars.latest.items():
                    tickers[name] = copy.deepcopy(bucket["_ticker"])
                    tickers[name]["aggregation"] = copy.deepcopy(
                        {k: v for k, v in bucket.items() if k != "_ticker"}
                    )
                    received[name] = bucket["last_received_at"]
            else:
                tickers, received = copy.deepcopy(self._tickers), dict(self._received)
            if not self._instruments or not tickers:
                return None
            return Snapshot(
                currency=self.settings.currency,
                environment=self.settings.environment,
                source="websocket-2s" if mode == "2s" else "websocket",
                captured_at=now,
                collection_started_at=self._started,
                instruments=copy.deepcopy(self._instruments),
                tickers=tickers,
                received_at=received,
                errors=dict(self._errors),
            )


class MarketFeed(FeedState):
    """One reusable background WebSocket consumer with explicit stop and idle expiry.

    Access snapshot/status/buckets or call touch() at least once per idle_timeout
    seconds to keep a UI lease alive. All limits are per feed, not per instrument.
    No database writes occur; callers may explicitly save the returned Snapshot.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        instrument_names=None,
        max_events=5000,
        max_buckets=5000,
        idle_timeout=120.0,
    ):
        super().__init__(settings, max_events=max_events, max_buckets=max_buckets)
        if not math.isfinite(idle_timeout) or idle_timeout <= 0:
            raise ValueError("Feed idle timeout must be finite and positive")
        self.instrument_names = tuple(instrument_names) if instrument_names is not None else None
        if self.instrument_names == ():
            raise ValueError("Select at least one instrument, or None for the full BTC/ETH scope")
        self.idle_timeout = idle_timeout
        self._last_access = time.monotonic()
        self._thread = None
        self._loop = None
        self._task = None
        self._stop_requested = threading.Event()

    def touch(self):
        self._last_access = time.monotonic()

    def start(self):
        with self._lock:
            self.touch()
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_requested.clear()
            self.on_status("connecting")
            self._thread = threading.Thread(
                target=self._run_thread, daemon=True, name="deribit-public-market-feed"
            )
            self._thread.start()
        return self

    def _run_thread(self):
        try:
            asyncio.run(self._run())
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.on_status("failed", f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                if self._status["state"] not in {"failed", "idle"}:
                    self.on_status("stopped")
                self._loop = self._task = None

    async def _run(self):
        from deribit_dashboard.collector import collect

        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(
            collect(self.settings, None, observer=self, instrument_names=self.instrument_names)
        )

        async def lease():
            while True:
                await asyncio.sleep(min(1.0, self.idle_timeout))
                if self._stop_requested.is_set():
                    self._task.cancel()
                    return
                if time.monotonic() - self._last_access > self.idle_timeout:
                    self.on_status("idle", "Feed stopped after its UI lease expired")
                    self._task.cancel()
                    return

        watcher = asyncio.create_task(lease())
        try:
            if self._stop_requested.is_set():
                self._task.cancel()
            await self._task
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    def stop(self, timeout: float = 10.0):
        self._stop_requested.set()
        with self._lock:
            loop, task, thread = self._loop, self._task, self._thread
        if loop is not None and task is not None and not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Market feed did not stop within the requested timeout")

    close = stop

    def status(self):
        self.touch()
        return super().status()

    def snapshot(self, mode="realtime"):
        self.touch()
        return super().snapshot(mode)

    def buckets(self):
        self.touch()
        return super().buckets()

    def events(self):
        self.touch()
        return super().events()
