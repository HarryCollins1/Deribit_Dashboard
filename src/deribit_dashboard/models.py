"""Immutable-by-convention snapshots; raw exchange values are never recomputed."""

import copy
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_time(value: str | datetime) -> datetime:
    result = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone, for example 2026-09-12T12:00:00Z")
    return result.astimezone(UTC)


def milliseconds(value: datetime) -> int:
    return int(parse_time(value).timestamp() * 1000)


@dataclass
class Snapshot:
    currency: str
    environment: str
    source: str
    captured_at: datetime
    collection_started_at: datetime
    instruments: dict[str, dict[str, Any]]
    tickers: dict[str, dict[str, Any]]
    received_at: dict[str, str]
    errors: dict[str, str] = field(default_factory=dict)
    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = 1

    def __post_init__(self):
        self.captured_at = parse_time(self.captured_at)
        self.collection_started_at = parse_time(self.collection_started_at)
        if self.schema_version != 1:
            raise ValueError("Unsupported snapshot schema version")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise ValueError("Snapshot requires a non-empty ID")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Snapshot requires a source")
        self.currency = self.currency.upper()
        if not re.fullmatch(r"[A-Z0-9_]{2,20}", self.currency):
            raise ValueError("Unsupported snapshot currency scope")
        if self.environment not in {"testnet", "production", "demo"}:
            raise ValueError("Unsupported snapshot environment")
        if self.collection_started_at > self.captured_at:
            raise ValueError("Collection cannot start after capture time")
        if not self.instruments:
            raise ValueError("A snapshot must contain instrument metadata")
        if set(self.tickers) - set(self.instruments):
            raise ValueError("Ticker metadata is missing")
        if set(self.received_at) != set(self.tickers):
            raise ValueError("Every ticker needs its receive timestamp")
        for name, instrument in self.instruments.items():
            if instrument.get("instrument_name") != name:
                raise ValueError("Instrument key/name mismatch")
            if instrument.get("kind") not in {
                "option",
                "future",
                "spot",
                "future_combo",
                "option_combo",
            }:
                raise ValueError("Unsupported Deribit instrument kind")
            base_currency = instrument.get("base_currency")
            if not isinstance(base_currency, str) or not base_currency:
                raise ValueError("Instrument requires a base currency")
            expiration = instrument.get("expiration_timestamp")
            if expiration is not None and (
                not isinstance(expiration, (int, float))
                or not math.isfinite(expiration)
                or expiration <= 0
            ):
                raise ValueError("Instrument requires expiration_timestamp")
            if instrument["kind"] == "option":
                if expiration is None:
                    raise ValueError("Option requires expiration_timestamp")
                if instrument.get("option_type") not in {"call", "put"}:
                    raise ValueError("Option requires call/put metadata")
                if (
                    not isinstance(instrument.get("strike"), (int, float))
                    or not math.isfinite(instrument["strike"])
                    or instrument["strike"] <= 0
                ):
                    raise ValueError("Option strike must be positive")
        for name, ticker in self.tickers.items():
            if ticker.get("instrument_name") != name:
                raise ValueError("Ticker key/name mismatch")
            timestamp = ticker.get("timestamp")
            if (
                not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                raise ValueError("Ticker requires an exchange timestamp")
            if parse_time(self.received_at[name]) > self.captured_at:
                raise ValueError("Cannot include a ticker received after the snapshot cutoff")

    @property
    def coverage(self) -> float:
        return len(self.tickers) / len(self.instruments)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "currency": self.currency,
            "environment": self.environment,
            "source": self.source,
            "captured_at": self.captured_at.isoformat(),
            "collection_started_at": self.collection_started_at.isoformat(),
            "instruments": copy.deepcopy(self.instruments),
            "tickers": copy.deepcopy(self.tickers),
            "received_at": dict(self.received_at),
            "errors": dict(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, value: dict) -> "Snapshot":
        return cls(**copy.deepcopy(value))

    @classmethod
    def from_json(cls, value: str) -> "Snapshot":
        return cls.from_dict(json.loads(value))
