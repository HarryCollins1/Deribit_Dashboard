"""Configuration shared by command-line tools and the Streamlit app."""

import math
import os
from dataclasses import dataclass

from deribit_dashboard.universe import selected_bases


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///data/deribit.db"
    environment: str = "testnet"
    currency: str = "BTC"
    snapshot_interval: float = 30.0
    stale_seconds: float = 120.0
    requests_per_second: float = 4.0
    request_timeout: float = 20.0
    ticker_interval: str = "100ms"
    feed_mode: str = "realtime"

    def __post_init__(self):
        object.__setattr__(self, "currency", self.currency.upper())
        if self.environment not in {"testnet", "production"}:
            raise ValueError("Environment must be testnet or production")
        selected_bases(self.currency)
        if not math.isfinite(self.snapshot_interval) or self.snapshot_interval < 2:
            raise ValueError("Snapshot interval must be finite and >= 2s")
        if not math.isfinite(self.stale_seconds) or self.stale_seconds <= 0:
            raise ValueError("Stale threshold must be finite and positive")
        if self.ticker_interval not in {"100ms", "agg2"}:
            raise ValueError(
                "Public ticker interval must be 100ms or agg2; raw requires authentication"
            )
        if self.feed_mode not in {"realtime", "2s"}:
            raise ValueError("Feed mode must be realtime or 2s")
        if self.feed_mode == "2s" and self.ticker_interval != "100ms":
            raise ValueError("Local 2s OHLC aggregation requires the public 100ms ticker stream")
        if not 0 < self.requests_per_second <= 5 or self.request_timeout <= 0:
            raise ValueError("Request rate must be in (0, 5] and timeout must be positive")

    @property
    def host(self):
        return "test.deribit.com" if self.environment == "testnet" else "www.deribit.com"

    @property
    def http_url(self):
        return f"https://{self.host}/api/v2"

    @property
    def websocket_url(self):
        return f"wss://{self.host}/ws/api/v2"

    @classmethod
    def from_env(cls, **overrides):
        values = {
            "database_url": os.getenv("DERIBIT_DATABASE_URL", cls.database_url),
            "environment": os.getenv("DERIBIT_ENVIRONMENT", cls.environment),
            "currency": os.getenv("DERIBIT_CURRENCY", cls.currency),
            "snapshot_interval": float(os.getenv("DERIBIT_SNAPSHOT_INTERVAL", "30")),
            "stale_seconds": float(os.getenv("DERIBIT_STALE_SECONDS", "120")),
            "requests_per_second": float(os.getenv("DERIBIT_REQUESTS_PER_SECOND", "4")),
            "ticker_interval": os.getenv("DERIBIT_TICKER_INTERVAL", cls.ticker_interval),
            "feed_mode": os.getenv("DERIBIT_FEED_MODE", cls.feed_mode),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
