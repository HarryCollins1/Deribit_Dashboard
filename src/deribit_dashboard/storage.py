"""Atomic snapshot storage on SQLite, with immutable replay payloads."""

from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    event,
    select,
)
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from deribit_dashboard.analytics import snapshot_frame
from deribit_dashboard.models import Snapshot, parse_time

metadata = MetaData()
snapshots = Table(
    "market_snapshots",
    metadata,
    Column("snapshot_id", String, primary_key=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("currency", String, nullable=False),
    Column("environment", String, nullable=False),
    Column("source", String, nullable=False),
    Column("instrument_count", Integer, nullable=False),
    Column("ticker_count", Integer, nullable=False),
    Column("payload", JSON, nullable=False),
    Index("ix_snapshots_selection", "environment", "currency", "captured_at"),
)
observations = Table(
    "market_observations",
    metadata,
    Column("captured_at", DateTime(timezone=True), primary_key=True),
    Column("snapshot_id", String, ForeignKey("market_snapshots.snapshot_id"), primary_key=True),
    Column("instrument_name", String, primary_key=True),
    Column("kind", String, nullable=False),
    Column("exchange_time", DateTime(timezone=True)),
    Column("expiry", DateTime(timezone=True)),
    Column("strike", Float),
    Column("mark_price", Float),
    Column("index_price", Float),
    Column("mark_iv", Float),
    Column("bid_iv", Float),
    Column("ask_iv", Float),
    Column("delta", Float),
    Column("gamma", Float),
    Column("theta", Float),
    Column("vega", Float),
    Column("rho", Float),
    Column("missing", Boolean, nullable=False),
    Index("ix_observations_instrument_time", "instrument_name", "captured_at"),
)


class SnapshotStore:
    def __init__(self, database_url: str):
        url = make_url(database_url)
        if url.get_backend_name() != "sqlite":
            raise ValueError("Use a SQLite database URL, for example sqlite:///data/deribit.db")
        options = {
            "pool_pre_ping": True,
            "connect_args": {"check_same_thread": False, "timeout": 30},
        }
        if url.database in {None, ":memory:"}:
            options["poolclass"] = StaticPool
        else:
            path = Path(url.database).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            url = url.set(database=str(path))
        self.engine = create_engine(url, **options)

        @event.listens_for(self.engine, "connect")
        def configure_sqlite(connection, record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if url.database not in {None, ":memory:"}:
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    def initialize(self):
        metadata.create_all(self.engine)

    def save(self, snapshot: Snapshot) -> str:
        payload = snapshot.to_dict()
        # Revalidate exports before writing; callers may have mutated raw dictionaries.
        Snapshot.from_dict(payload)
        frame = snapshot_frame(snapshot)
        import pandas as pd

        rows = []
        for record in frame.to_dict("records"):
            row = {
                "snapshot_id": snapshot.snapshot_id,
                "captured_at": snapshot.captured_at,
            }
            for column in observations.columns:
                if column.name in record:
                    value = record[column.name]
                    if pd.isna(value):
                        value = None
                    elif isinstance(value, pd.Timestamp):
                        value = value.to_pydatetime()
                    row[column.name] = value
            rows.append(row)
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(snapshots.c.payload).where(snapshots.c.snapshot_id == snapshot.snapshot_id)
            ).scalar_one_or_none()
            if existing is not None:
                if existing != payload:
                    raise ValueError("Snapshot IDs are immutable; use a new ID for different data")
                return snapshot.snapshot_id
            conn.execute(
                snapshots.insert().values(
                    snapshot_id=snapshot.snapshot_id,
                    captured_at=snapshot.captured_at,
                    currency=snapshot.currency,
                    environment=snapshot.environment,
                    source=snapshot.source,
                    instrument_count=len(snapshot.instruments),
                    ticker_count=len(snapshot.tickers),
                    payload=payload,
                )
            )
            conn.execute(observations.insert(), rows)
        return snapshot.snapshot_id

    def list(
        self, environment: str, currency: str | None = None, start=None, end=None, limit: int = 1000
    ):
        if not 1 <= limit <= 10000:
            raise ValueError("Snapshot list limit must be between 1 and 10000")
        columns = [col for col in snapshots.columns if col.name != "payload"]
        query = select(*columns).where(snapshots.c.environment == environment)
        if currency is not None:
            query = query.where(snapshots.c.currency == currency.upper())
        if start is not None:
            query = query.where(snapshots.c.captured_at >= parse_time(start))
        if end is not None:
            query = query.where(snapshots.c.captured_at <= parse_time(end))
        query = query.order_by(snapshots.c.captured_at.desc(), snapshots.c.snapshot_id).limit(limit)
        with self.engine.connect() as conn:
            result = [dict(row) for row in conn.execute(query).mappings()]
        for row in result:
            # SQLite returns naive UTC datetimes for DateTime columns.
            row["captured_at"] = row["captured_at"].isoformat().removesuffix("+00:00") + "Z"
        return result

    def load(self, snapshot_id: str) -> Snapshot:
        with self.engine.connect() as conn:
            payload = conn.execute(
                select(snapshots.c.payload).where(snapshots.c.snapshot_id == snapshot_id)
            ).scalar_one_or_none()
        if payload is None:
            raise KeyError(f"Unknown snapshot: {snapshot_id}")
        return Snapshot.from_dict(payload)

    def at_or_before(self, environment: str, currency: str | None, timestamp) -> Snapshot | None:
        rows = self.list(environment, currency, end=parse_time(timestamp), limit=1)
        return self.load(rows[0]["snapshot_id"]) if rows else None

    def option_history(self, environment: str, instrument: str, end, limit=200):
        """Query normalized values instead of loading hundreds of full-chain JSON payloads."""
        fields = [
            observations.c.captured_at,
            observations.c.mark_iv,
            *[observations.c[name] for name in ["delta", "gamma", "theta", "vega", "rho"]],
        ]
        query = (
            select(*fields)
            .select_from(observations.join(snapshots))
            .where(
                snapshots.c.environment == environment,
                observations.c.instrument_name == instrument,
                observations.c.captured_at <= parse_time(end),
                observations.c.missing.is_(False),
            )
            .order_by(observations.c.captured_at.desc())
            .limit(limit)
        )
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(query).mappings()]

    def close(self):
        self.engine.dispose()
