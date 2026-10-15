
import asyncio
import json
import math
import os
import shutil
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from sqlalchemy import func, select

from deribit_dashboard.collector import collect
from deribit_dashboard.config import Settings
from deribit_dashboard.experiments import fingerprint, json_value, provenance
from deribit_dashboard.feeds import FeedState
from deribit_dashboard.models import milliseconds, parse_time, utc_now
from deribit_dashboard.relative_value import RelativeValueConfig
from deribit_dashboard.relative_value_execution import TradingEvaluationConfig
from deribit_dashboard.relative_value_research import RelativeValueResearchConfig
from deribit_dashboard.storage import SnapshotStore
from deribit_dashboard.storage import snapshots as snapshot_table
from deribit_dashboard.surface import chain_audit


def prepare_plan(
    catalogue,
    index,
    started_at,
    database_url,
    *,
    strikes=9,
    training_days=3,
    validation_days=3,
    test_days=8,
    interval_seconds=241,
    purpose="research",
    candidate_scope="nearest",
    horizon_seconds=3600,
    minimum_validation_trades=5,
):
    """Choose one fixed expiry and paired strikes from metadata and the current index.

    No quoted option prices, IV, performance, or future capture enters selection.
    The closest listed expiry at least seven days beyond test end supplies the cohort.
    """
    started_at = parse_time(started_at)
    if purpose not in {"research", "interview_demonstration"} or candidate_scope not in {
        "nearest",
        "interior",
    }:
        raise ValueError("Choose a supported recording purpose and candidate scope")
    if not isinstance(strikes, int) or strikes < 5 or strikes % 2 != 1:
        raise ValueError("Use an odd cohort size of at least five paired strikes")
    if not math.isfinite(index) or index <= 0:
        raise ValueError("A positive current BTC index is required")
    if not all(
        math.isfinite(value) and value > 0 for value in (training_days, validation_days, test_days)
    ):
        raise ValueError("Declare positive training, validation and test durations")
    if interval_seconds not in {60, 67, 120, 240, 241}:
        raise ValueError(
            "Use 60, 67, 120, 240 or 241 seconds; the buffered cadences keep exit captures inside horizon tolerance"
        )
    end = started_at + timedelta(days=training_days + validation_days + test_days)
    expected = math.ceil((end - started_at).total_seconds() / interval_seconds)
    if expected > 10000:
        raise ValueError("The declared cohort exceeds the registry's 10,000-capture limit")
    if test_days < 6 and purpose == "research":
        raise ValueError(
            "Use at least six test days for the initial five-block coverage requirement"
        )
    instruments = (
        catalogue if isinstance(catalogue, dict) else {m["instrument_name"]: m for m in catalogue}
    )
    hedge = instruments.get("BTC-PERPETUAL", {})
    if not (
        hedge.get("base_currency") == "BTC"
        and hedge.get("kind") == "future"
        and hedge.get("settlement_currency") == "BTC"
        and hedge.get("settlement_period") == "perpetual"
    ):
        raise ValueError("Production BTC-PERPETUAL metadata is required")
    expiries = {}
    for name, metadata in instruments.items():
        if (
            metadata.get("kind") == "option"
            and metadata.get("base_currency") == "BTC"
            and metadata.get("settlement_currency") == metadata.get("quote_currency") == "BTC"
            and metadata.get("option_type") in {"call", "put"}
            and metadata.get("is_active", True)
            and metadata["expiration_timestamp"] >= milliseconds(end + timedelta(days=7))
        ):
            expiries.setdefault(metadata["expiration_timestamp"], {}).setdefault(
                metadata["strike"], {}
            )[metadata["option_type"]] = name
    chosen = None
    for expiry, by_strike in sorted(expiries.items()):
        paired = [strike for strike, rights in by_strike.items() if set(rights) == {"call", "put"}]
        if len(paired) >= strikes:
            ordered = sorted(paired)
            center = min(
                range(len(ordered)), key=lambda i: (abs(math.log(ordered[i] / index)), ordered[i])
            )
            # Consecutive strikes, centered on the nearest strike without silently
            # selecting an endpoint candidate that would need extrapolation.
            offset = max(0, min(center - strikes // 2, len(ordered) - strikes))
            cohort = ordered[offset : offset + strikes]
            candidate_strike = ordered[center]
            if candidate_strike in cohort[1:-1]:
                chosen = expiry, by_strike, cohort, candidate_strike
                break
    if chosen is None:
        raise ValueError(
            "No expiry has enough paired strikes and an interior index-nearest candidate"
        )
    expiry, by_strike, cohort, candidate_strike = chosen
    candidate_strikes = [candidate_strike] if candidate_scope == "nearest" else cohort[1:-1]
    candidates = [by_strike[k][side] for k in candidate_strikes for side in ("call", "put")]
    names = sorted(
        [by_strike[k][side] for k in cohort for side in ("call", "put")] + ["BTC-PERPETUAL"]
    )
    trading = TradingEvaluationConfig(
        horizon_seconds=horizon_seconds,
        signal_config=RelativeValueConfig(
            estimated_round_trip_cost_cash=25,
            estimated_round_trip_cost_bps=12,
            uncertainty_floor_cash=10,
            minimum_edge_cash=25,
        ),
    )
    research = RelativeValueResearchConfig(
        started_at + timedelta(days=training_days),
        started_at + timedelta(days=training_days + validation_days),
        end,
        trading_config=trading,
        minimum_validation_trades=minimum_validation_trades,
    )
    plan = json_value(
        dict(
            schema_version=1,
            environment="production",
            purpose=purpose,
            created_at=started_at,
            database_url=database_url,
            research_config=asdict(research),
            interval_seconds=interval_seconds,
            expected_captures=expected,
            instrument_names=names,
            candidate_names=candidates,
            expiry=expiry,
            strikes=cohort,
            initial_index=index,
            initial_metadata={name: instruments[name] for name in names},
            provenance=provenance(),
            alpha_claim=False,
            selection="Fixed paired strikes chosen by metadata and initial index only; no quote-price, IV or outcome selection. No automatic rolling or successful-fit replacement.",
            candidate_scope=candidate_scope,
            study=(
                "Short prospective production interview demonstration. Fixed clock stop, training/validation/test rules and costs; no stopping on profit. Short samples cannot establish durable profitability or supply the default multi-day uncertainty blocks."
                if purpose == "interview_demonstration"
                else "Initial 14-day default: 3 training, 3 validation, 8 test days. Coverage and block assumptions do not guarantee statistical power or alpha; extend using a newly declared future holdout if inconclusive."
            ),
        )
    )
    plan["digest"] = fingerprint(plan)
    return plan


def load_plan(path):
    return validate_plan(json.loads(Path(path).read_text()))


def validate_plan(plan):
    """Validate the same immutable protocol whether read from a file or SQLite."""
    if plan.get("schema_version") != 1 or plan.get("environment") != "production":
        raise ValueError("Expected a production recording plan")
    if plan.get("digest") != fingerprint({k: v for k, v in plan.items() if k != "digest"}):
        raise ValueError(
            "Recording plan integrity mismatch; create a new plan to change the design"
        )
    RelativeValueResearchConfig(**plan["research_config"])
    return plan


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_value(value), indent=2, allow_nan=False)
    if exclusive:
        with path.open("x") as handle:
            handle.write(encoded)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(path)


def recorded_ids(store, plan):
    """Query the entire declared range, ordered chronologically, with no list truncation."""
    config = RelativeValueResearchConfig(**plan["research_config"])
    query = (
        select(snapshot_table.c.snapshot_id)
        .where(
            snapshot_table.c.environment == "production",
            snapshot_table.c.captured_at >= parse_time(plan["created_at"]),
            snapshot_table.c.captured_at <= config.test_end,
        )
        .order_by(snapshot_table.c.captured_at, snapshot_table.c.snapshot_id)
    )
    with store.engine.connect() as connection:
        ids = list(connection.scalars(query))
    if len(ids) > 10000:
        raise ValueError(
            "More than 10,000 captures: declare a new recording design, never silently truncate"
        )
    return ids


def recording_status(store, plan):
    with store.engine.connect() as connection:
        count = connection.scalar(select(func.count()).select_from(snapshot_table))
    rows = store.list("production", None, limit=1)
    result = dict(
        captures=count,
        last_capture=rows[0]["captured_at"] if rows else None,
        plan_digest=plan["digest"],
        alpha_claim=False,
        environment="production",
    )
    if rows:
        latest = store.load(rows[0]["snapshot_id"])
        audit, _ = chain_audit(latest, "BTC", "BTC", stale_seconds=60, max_relative_width=1)
        stamp = milliseconds(latest.captured_at)
        result.update(
            instruments=len(latest.instruments),
            tickers=len(latest.tickers),
            coverage=latest.coverage,
            audited_options=len(audit),
            accepted_options=int(audit.accepted.sum()) if not audit.empty else 0,
            after_cutoff_events=sum(
                ticker["timestamp"] > stamp for ticker in latest.tickers.values()
            ),
            available_disk_bytes=shutil.disk_usage(Path(store.engine.url.database).parent).free,
            approximate_snapshot_bytes=len(latest.to_json().encode()),
            seconds_since_last_capture=max(0, (utc_now() - latest.captured_at).total_seconds()),
        )
        hedge = latest.tickers.get("BTC-PERPETUAL", {})
        rate = hedge.get("current_funding")
        age = (stamp - hedge.get("timestamp", stamp)) / 1000
        result["fresh_current_funding"] = (
            isinstance(rate, (int, float)) and math.isfinite(rate) and 0 <= age <= 300
        )
    return result


async def record(
    plan, status_path, *, seconds, interval_seconds=None, max_database_bytes=2_000_000_000
):
    """Bound elapsed runtime/disk usage and persist operational status across reconnects."""
    config = RelativeValueResearchConfig(**plan["research_config"])
    remaining = (config.test_end - utc_now()).total_seconds()
    if not math.isfinite(seconds) or seconds <= 0 or remaining <= 0:
        raise ValueError("Recording needs a positive duration before the declared test end")
    interval = interval_seconds or plan["interval_seconds"]
    if interval not in {60, 67, 120, 240, 241}:
        raise ValueError("Use a supported 60, 67, 120, 240 or 241-second capture interval")
    if not isinstance(max_database_bytes, int) or max_database_bytes <= 0:
        raise ValueError("A positive database disk budget is required")
    duration = min(seconds, remaining)
    expected_new = math.ceil(duration / interval)
    if expected_new > 10000:
        raise ValueError("Duration/cadence exceeds the 10,000-capture registry limit")
    settings = Settings(
        database_url=plan["database_url"],
        environment="production",
        currency="BTC",
        snapshot_interval=interval,
        feed_mode="realtime",
        ticker_interval="100ms",
    )
    store = SnapshotStore(settings.database_url)
    store.initialize()
    if len(recorded_ids(store, plan)) + expected_new > 10000:
        store.close()
        raise ValueError(
            "Existing plus requested captures exceed the 10,000-capture registry limit"
        )
    state = FeedState(settings)
    loop = asyncio.get_running_loop()
    stop = loop.time() + duration
    task = asyncio.create_task(
        collect(settings, store, observer=state, instrument_names=plan["instrument_names"])
    )
    reason = "duration elapsed"
    started = utc_now()
    try:
        while loop.time() < stop:
            status = recording_status(store, plan)
            database = Path(store.engine.url.database)
            used = sum(
                path.stat().st_size
                for path in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm"))
                if path.exists()
            )
            status.update(
                state="recording",
                pid=os.getpid(),
                started_at=started,
                scheduled_stop=started + timedelta(seconds=duration),
                interval_seconds=interval,
                feed=state.status(),
                database_bytes=used,
            )
            write_json(status_path, status)
            if used >= max_database_bytes:
                reason = "database disk budget reached"
                break
            if task.done():
                task.result()
                reason = "collector ended"
                break
            await asyncio.sleep(min(15, max(0, stop - loop.time())))
    except BaseException:
        reason = "interrupted or failed; inspect collector log"
        raise
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        state.on_status("stopped")
        status = recording_status(store, plan)
        status.update(
            state="stopped",
            reason=reason,
            pid=os.getpid(),
            stopped_at=utc_now(),
            feed=state.status(),
            interval_seconds=interval,
        )
        write_json(status_path, status)
        store.close()


def check_test_deadline(plan, now=None):
    if (
        parse_time(now or utc_now())
        < RelativeValueResearchConfig(**plan["research_config"]).test_end
    ):
        raise ValueError(
            "Final test is still recording. Use status or accounting-only pilot-report; do not inspect final-test P&L before the declared end."
        )
