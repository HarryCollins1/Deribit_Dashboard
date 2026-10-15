"""Public, read-only Deribit HTTP client with paced requests and bounded retries."""

import asyncio
import time

import httpx

from deribit_dashboard.config import Settings
from deribit_dashboard.universe import in_universe, selected_bases


class DeribitError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"Deribit error {code}: {message}")


class DeribitClient:
    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self.http = httpx.AsyncClient(
            base_url=settings.http_url + "/", timeout=settings.request_timeout, transport=transport
        )
        self._lock = asyncio.Lock()
        self._next_request = 0.0
        self._metadata = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def call(self, method: str, **params):
        if not method.startswith("public/"):
            raise ValueError("Only public market-data methods are supported")
        if "instrument_name" in params and method != "public/get_instrument":
            await self.instrument(params["instrument_name"])
        if "currency" in params and method != "public/get_instruments":
            selected_bases(params["currency"])
        for attempt in range(4):
            async with self._lock:
                await asyncio.sleep(max(0.0, self._next_request - time.monotonic()))
                # get_instruments has its own 1 request/second rate limit.
                delay = (
                    1.05
                    if method == "public/get_instruments"
                    else (1 / self.settings.requests_per_second)
                )
                self._next_request = time.monotonic() + delay
            try:
                response = await self.http.get(method, params=params)
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    error = payload["error"]
                    raise DeribitError(error.get("code"), error.get("message", "Unknown error"))
                if "result" not in payload:
                    raise ValueError("Malformed Deribit response: no result")
                return payload["result"]
            except (httpx.TransportError, httpx.HTTPStatusError, DeribitError) as exc:
                retry = isinstance(exc, httpx.TransportError)
                retry |= isinstance(exc, httpx.HTTPStatusError) and (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                retry |= isinstance(exc, DeribitError) and exc.code == 10028
                if not retry or attempt == 3:
                    raise
                delay = min(2 ** (attempt + 1), 30)
                if isinstance(exc, httpx.HTTPStatusError):
                    try:
                        delay = min(
                            max(delay, float(exc.response.headers.get("Retry-After", 0))), 60
                        )
                    except ValueError:
                        pass
                await asyncio.sleep(delay)

    async def instruments(self, currency: str | None = None):
        currency = (currency or self.settings.currency).upper()
        selected_bases(currency)
        rows = await self.call(
            "public/get_instruments",
            # Discover across quote/settlement groups, then filter the underlying.
            # This retains BTC/ETH linear instruments and spot pairs as well as inverse ones.
            currency="any",
            expired="false",
        )
        result = {row["instrument_name"]: row for row in rows if in_universe(row, currency)}
        self._metadata.update(result)
        return result

    async def instrument(self, name: str):
        if name not in self._metadata:
            metadata = await self.call("public/get_instrument", instrument_name=name)
            if metadata.get("instrument_name") != name or not in_universe(
                metadata, self.settings.currency, active=False
            ):
                raise ValueError("Instrument is outside the selected BTC/ETH product universe")
            self._metadata[name] = metadata
        metadata = self._metadata[name]
        if not in_universe(metadata, self.settings.currency, active=False):
            raise ValueError("Instrument is outside the selected BTC/ETH product universe")
        return metadata

    async def ticker(self, name: str):
        return await self.call("public/ticker", instrument_name=name)
