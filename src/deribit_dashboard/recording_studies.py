

from pathlib import Path

import pandas as pd
from sqlalchemy import JSON, Column, DateTime, MetaData, String, Table, func, select, update
from sqlalchemy.engine import make_url

from deribit_dashboard.experiments import ExperimentStore, make_experiment, provenance
from deribit_dashboard.models import parse_time, utc_now
from deribit_dashboard.recording import check_test_deadline, recorded_ids, validate_plan
from deribit_dashboard.relative_value_research import RelativeValueResearchConfig, evaluate_research
from deribit_dashboard.storage import snapshots

metadata = MetaData()
studies = Table(
    "relative_value_recording_studies",
    metadata,
    Column("digest", String, primary_key=True),
    Column("registered_at", DateTime(timezone=True), nullable=False),
    Column("plan", JSON, nullable=False),
    Column("run_id", String),
)


class RecordingStudyStore:


    def __init__(self, engine):
        self.engine = engine

    def initialize(self):
        metadata.create_all(self.engine)

    def register(self, plan):
        plan = validate_plan(plan)
        declared = make_url(plan["database_url"])
        if (
            declared.get_backend_name() != "sqlite"
            or Path(declared.database).resolve() != Path(self.engine.url.database).resolve()
        ):
            raise ValueError("Register the study in its declared SQLite database")
        with self.engine.begin() as connection:
            existing = connection.execute(select(studies)).mappings().first()
            if existing:
                if existing["digest"] != plan["digest"]:
                    raise ValueError("This database already has a different immutable study")
                validate_plan(existing["plan"])
                return existing["digest"]
            if connection.scalar(select(func.count()).select_from(snapshots)):
                raise ValueError("Register a prospective study before recording any snapshots")
            if utc_now() >= parse_time(plan["research_config"]["training_end"]):
                raise ValueError("Register a prospective study before training ends")
            connection.execute(
                studies.insert().values(
                    digest=plan["digest"], registered_at=utc_now(), plan=plan, run_id=None
                )
            )
        return plan["digest"]

    def list(self):
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(studies)).mappings()]

    def load(self, digest):
        with self.engine.connect() as connection:
            row = (
                connection.execute(select(studies).where(studies.c.digest == digest))
                .mappings()
                .first()
            )
        if row is None:
            raise KeyError("Unknown declared recording study")
        result = dict(row)
        validate_plan(result["plan"])
        return result

    def attach_result(self, digest, run_id):
        with self.engine.begin() as connection:
            connection.execute(
                update(studies)
                .where(studies.c.digest == digest, studies.c.run_id.is_(None))
                .values(run_id=run_id)
            )
        return self.load(digest)["run_id"]


def study_progress(store, plan, *, now=None):

    plan = validate_plan(plan)
    config = RelativeValueResearchConfig(**plan["research_config"])
    now = parse_time(now or utc_now())
    boundaries = [
        parse_time(plan["created_at"]),
        config.training_end,
        config.validation_end,
        config.test_end,
    ]
    rows = []
    with store.engine.connect() as connection:
        for index, period in enumerate(("training", "validation", "test")):
            start, end = boundaries[index : index + 2]
            upper = (
                snapshots.c.captured_at <= end
                if period == "test"
                else snapshots.c.captured_at < end
            )
            count, latest = connection.execute(
                select(func.count(), func.max(snapshots.c.captured_at)).where(
                    snapshots.c.environment == "production",
                    snapshots.c.captured_at >= start,
                    snapshots.c.captured_at <= now,
                    upper,
                )
            ).one()
            rows.append(
                dict(
                    period=period,
                    start_utc=start,
                    end_utc=end,
                    status="complete" if now >= end else "recording" if now >= start else "pending",
                    captures=count,
                    latest_capture_utc=latest,
                )
            )
    return pd.DataFrame(rows)


def evaluate_study(store, digest, *, now=None):
    """Save the first complete evaluation without retuning or inspecting an early test."""
    declared = RecordingStudyStore(store.engine)
    row = declared.load(digest)
    plan = row["plan"]
    check_test_deadline(plan, now)
    registry = ExperimentStore(store.engine)
    registry.initialize()
    if row["run_id"]:
        return registry.load(row["run_id"])
    current = provenance()
    if any(current[key] != plan["provenance"][key] for key in ("source_sha256", "dependencies")):
        raise ValueError(
            "Research code or dependencies changed after declaration. Evaluate with the frozen source/environment; do not replace the declared design."
        )
    inputs = [store.load(snapshot_id) for snapshot_id in recorded_ids(store, plan)]
    report = evaluate_research(
        inputs,
        RelativeValueResearchConfig(**plan["research_config"]),
        instrument_names=plan["candidate_names"],
    )
    payload = make_experiment(
        "relative_value",
        inputs,
        dict(config=plan["research_config"], instrument_names=plan["candidate_names"]),
        {key: value for key, value in vars(report).items() if key != "diagnostics"},
        dict(report.diagnostics, recording_plan=plan, study_digest=digest),
        name=f"Declared BTC OOS study · {plan['created_at']}",
    )
    registry.save(payload)
    linked = declared.attach_result(digest, payload["run_id"])
    return payload if linked == payload["run_id"] else registry.load(linked)
