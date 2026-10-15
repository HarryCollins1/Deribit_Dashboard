import hashlib
import json
import math
import platform
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import JSON, Column, DateTime, Index, MetaData, String, Table, select

from deribit_dashboard.models import Snapshot, parse_time, utc_now

metadata = MetaData()
RELATIVE_VALUE_INPUT_LIMIT = 10_000
RELATIVE_VALUE_IMPORT_BYTES = 1_000_000_000
LEGACY_IMPORT_BYTES = 100_000_000
runs = Table(
    "research_experiments",
    metadata,
    Column("run_id", String, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("kind", String, nullable=False),
    Column("name", String, nullable=False),
    Column("digest", String, nullable=False),
    Column("payload", JSON, nullable=False),
    Index("ix_research_kind_time", "kind", "created_at"),
)


def json_value(value):
    """Encode research data without non-standard NaN JSON; retain table boundaries."""
    if isinstance(value, pd.DataFrame):
        columns = list(map(str, value.columns))
        if len(set(columns)) != len(columns):
            raise ValueError("Experiment tables require unique column names")
        return {
            "__table__": True,
            "columns": columns,
            "records": json_value(value.to_dict("records")),
            "index": json_value(value.index.tolist()),
            "index_name": value.index.name,
            "index_kind": "multi" if isinstance(value.index, pd.MultiIndex) else "single",
            "index_names": list(value.index.names),
        }
    if isinstance(value, pd.Series):
        return json_value(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported experiment value: {type(value).__name__}")


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def provenance():
    from deribit_dashboard import __version__

    dependencies = {}
    for package in ("numpy", "scipy", "pandas", "SQLAlchemy"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "unavailable"
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return dict(
        package_version=__version__,
        source_sha256=source.hexdigest(),
        python=platform.python_version(),
        dependencies=dependencies,
    )


def make_experiment(
    kind, snapshots, parameters, results, diagnostics=None, name="", parent_run_id=None
):
    if kind not in {
        "quant",
        "strategy",
        "barrier",
        "walk_forward",
        "barrier_convergence",
        "barrier_model_hedge",
        "relative_value",
    }:
        raise ValueError("Unknown experiment kind")
    limit = RELATIVE_VALUE_INPUT_LIMIT if kind == "relative_value" else 200
    if not 1 <= len(snapshots) <= limit:
        raise ValueError(f"An experiment needs 1–{limit:,} input snapshots")
    inputs = [Snapshot.from_dict(s.to_dict()).to_dict() for s in snapshots]
    if len({s["snapshot_id"] for s in inputs}) != len(inputs):
        raise ValueError("Input snapshot IDs must be unique")
    payload = json_value(
        dict(
            schema_version=1,
            run_id=str(uuid4()),
            created_at=utc_now(),
            kind=kind,
            name=name.strip() or kind.replace("_", " "),
            parent_run_id=parent_run_id,
            provenance=provenance(),
            inputs=inputs,
            input_hashes=[fingerprint(s) for s in inputs],
            parameters=parameters,
            results=results,
            diagnostics=diagnostics or {},
            missing_numeric_convention="Nonfinite or unavailable outputs are JSON null, never zero.",
        )
    )
    payload["digest"] = fingerprint(payload)
    return payload


def validate_experiment(payload):
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Expected research experiment schema version 1")
    required = {
        "run_id",
        "created_at",
        "kind",
        "name",
        "provenance",
        "inputs",
        "input_hashes",
        "parameters",
        "results",
        "diagnostics",
        "digest",
    }
    if not required <= payload.keys() or not isinstance(payload["parameters"], dict):
        raise ValueError("Experiment is missing required fields")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise ValueError("Experiment requires a run ID")
    parse_time(payload["created_at"])
    limit = RELATIVE_VALUE_INPUT_LIMIT if payload["kind"] == "relative_value" else 200
    if not 1 <= len(payload["inputs"]) <= limit:
        raise ValueError("Invalid input snapshot count")
    if payload["digest"] != fingerprint({k: v for k, v in payload.items() if k != "digest"}):
        raise ValueError("Experiment integrity digest does not match")
    for raw in payload["inputs"]:
        Snapshot.from_dict(raw)
    if payload["input_hashes"] != [fingerprint(s) for s in payload["inputs"]]:
        raise ValueError("Input snapshot integrity mismatch")
    return payload


class ExperimentStore:
    """Uses an existing engine; creates an additive table and never rewrites snapshots."""

    def __init__(self, engine):
        self.engine = engine

    def initialize(self):
        metadata.create_all(self.engine)

    def save(self, payload):
        payload = validate_experiment(json_value(payload))
        with self.engine.begin() as conn:
            existing = conn.scalar(select(runs.c.digest).where(runs.c.run_id == payload["run_id"]))
            if existing is not None:
                if existing != payload["digest"]:
                    raise ValueError("Experiment IDs are immutable; save a new run")
                return payload["run_id"]
            conn.execute(
                runs.insert().values(
                    run_id=payload["run_id"],
                    created_at=parse_time(payload["created_at"]),
                    kind=payload["kind"],
                    name=payload["name"],
                    digest=payload["digest"],
                    payload=payload,
                )
            )
        return payload["run_id"]

    def load(self, run_id):
        with self.engine.connect() as conn:
            payload = conn.scalar(select(runs.c.payload).where(runs.c.run_id == run_id))
        if payload is None:
            raise KeyError(f"Unknown experiment: {run_id}")
        return validate_experiment(payload)

    def list(self, kind=None, limit=100):
        if not 1 <= limit <= 1000:
            raise ValueError("List 1–1,000 experiments")
        query = select(runs.c.run_id, runs.c.created_at, runs.c.kind, runs.c.name, runs.c.digest)
        if kind:
            query = query.where(runs.c.kind == kind)
        query = query.order_by(runs.c.created_at.desc(), runs.c.run_id).limit(limit)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(query).mappings()]

    def import_json(self, data):
        if len(data) > RELATIVE_VALUE_IMPORT_BYTES:
            raise ValueError("Relative-value experiment import is limited to 1 GB")
        payload = json.loads(data)
        if len(data) > LEGACY_IMPORT_BYTES and (
            not isinstance(payload, dict) or payload.get("kind") != "relative_value"
        ):
            raise ValueError("Experiment import is limited to 100 MB")
        return self.save(payload)

    def export_json(self, run_id):
        return json.dumps(self.load(run_id), indent=2, allow_nan=False)


def result_tables(payload):
    """Extract named tables from a portable run, retaining nulls and UTC strings."""
    found = {}

    def visit(value, prefix):
        if isinstance(value, dict) and value.get("__table__") is True:
            frame = pd.DataFrame(value["records"], columns=value["columns"])
            if "index" in value:
                if value.get("index_kind") == "multi":
                    names = value.get("index_names")
                    frame.index = (
                        pd.MultiIndex.from_tuples(
                            [tuple(row) for row in value["index"]], names=names
                        )
                        if value["index"]
                        else pd.MultiIndex.from_arrays([[] for _ in names], names=names)
                    )
                else:
                    frame.index = pd.Index(value["index"], name=value.get("index_name"))
            found[prefix] = frame
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{prefix}.{key}" if prefix else key)

    visit(payload["results"], "")
    return found


def compare_experiments(left, right):

    validate_experiment(left)
    validate_experiment(right)
    rows = []

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            return {
                p: x
                for k, v in value.items()
                for p, x in flatten(v, f"{prefix}.{k}" if prefix else k).items()
            }
        return {prefix: value}

    a = flatten({k: left[k] for k in ("kind", "parameters", "provenance", "input_hashes")})
    b = flatten({k: right[k] for k in ("kind", "parameters", "provenance", "input_hashes")})
    for key in sorted(a.keys() | b.keys()):
        rows.append(
            dict(
                field=key,
                left=canonical(a.get(key)),
                right=canonical(b.get(key)),
                changed=a.get(key) != b.get(key),
            )
        )
    return pd.DataFrame(rows)


def compare_table(left, right, name, keys):
    """Outer join explicit user-selected economic keys; differences only for numeric cells."""
    a, b = result_tables(left)[name], result_tables(right)[name]
    if left["kind"] != right["kind"]:
        raise ValueError("Compare tables from the same experiment kind")

    def currency(record):
        if record["kind"] == "relative_value":
            signal = record["parameters"]["config"]["trading_config"]["signal_config"]
            from deribit_dashboard.workbench import cash_currency_label

            return signal["settlement"], cash_currency_label(
                Snapshot.from_dict(record["inputs"][0]), "BTC", signal["settlement"]
            )
        explicit = record["parameters"].get("cash_currency") or record["diagnostics"].get(
            "cash_currency"
        )
        if explicit:
            return explicit
        parameters = record["parameters"]
        raw = record["inputs"][0]
        if record["kind"].startswith("barrier"):
            instrument = raw["instruments"][parameters["instrument_name"]]
            base, settlement = instrument["base_currency"], instrument["settlement_currency"]
        else:
            base, settlement = parameters.get("base"), parameters.get("settlement")
        if base and settlement:
            from deribit_dashboard.workbench import cash_currency_label

            return cash_currency_label(Snapshot.from_dict(raw), base, settlement)
        return None

    left_currency, right_currency = currency(left), currency(right)
    if left_currency is not None and right_currency is not None and left_currency != right_currency:
        raise ValueError("Cannot subtract results in different cash currencies")
    for field in ("base", "settlement", "cash_currency"):
        a_unit, b_unit = left["parameters"].get(field), right["parameters"].get(field)
        if a_unit != b_unit:
            raise ValueError(f"Cannot subtract results with different {field} conventions")
    if not keys or not set(keys) <= set(a.columns) & set(b.columns):
        raise ValueError("Choose shared row identity columns")
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Selected keys do not uniquely identify rows; include further columns")
    merged = a.merge(
        b, on=keys, how="outer", suffixes=("_left", "_right"), indicator="row_presence"
    )
    for column in set(a.columns) & set(b.columns) - set(keys):
        if (
            pd.api.types.is_numeric_dtype(a[column])
            and pd.api.types.is_numeric_dtype(b[column])
            and not pd.api.types.is_bool_dtype(a[column])
            and not pd.api.types.is_bool_dtype(b[column])
        ):
            merged[f"{column}_change"] = merged[f"{column}_right"] - merged[f"{column}_left"]
    return merged


def reproduce_experiment(payload):

    validate_experiment(payload)
    snapshots = [Snapshot.from_dict(s) for s in payload["inputs"]]
    params = payload["parameters"]
    kind = payload["kind"]
    if kind == "quant":
        from deribit_dashboard.workbench import analyze

        calibration_params = dict(params)
        smile_params = calibration_params.pop("smile_dynamics", None)
        analysis = analyze(snapshots[0], **calibration_params)
        results = dict(
            chain=analysis.chain,
            filters=analysis.filters,
            forwards=analysis.forwards,
            segments=analysis.segments,
        )
        if hasattr(analysis, "parity_pairs"):
            results["parity_pairs"] = analysis.parity_pairs
        if analysis.surface is not None:
            results["surface_points"] = analysis.surface.points()
        if "sensitivity_summary" in payload["results"]:
            from deribit_dashboard.workbench import quote_perturbation_sensitivity

            sensitivity = quote_perturbation_sensitivity(
                snapshots[0], **{k: v for k, v in calibration_params.items() if k != "quote_mark"}
            )
            results.update(
                sensitivity_summary=sensitivity.summary,
                sensitivity_forwards=sensitivity.forwards,
                sensitivity_quotes=sensitivity.quotes,
            )
        diagnostics = (
            dict(analysis.surface.report) if analysis.surface else {"fit_error": analysis.fit_error}
        )
        if smile_params is not None:
            from deribit_dashboard.smile_dynamics import (
                SmileDynamicsConfig,
                compare_smile_dynamics,
            )

            comparison = compare_smile_dynamics(analysis, SmileDynamicsConfig(**smile_params))
            results.update(
                smile_dynamics_curves=comparison.curves,
                smile_dynamics_summary=comparison.summary,
                smile_dynamics_diagnostics=comparison.diagnostics,
            )
            diagnostics["smile_dynamics"] = comparison.explanation
    elif kind == "walk_forward":
        from deribit_dashboard.walk_forward import WalkForwardConfig, evaluate_walk_forward

        report = evaluate_walk_forward(snapshots, WalkForwardConfig(**params))
        results = dict(
            observations=report.observations,
            folds=report.folds,
            coverage=report.coverage,
            metrics=report.metrics,
        )
        diagnostics = report.diagnostics
    elif kind == "relative_value":
        from deribit_dashboard.relative_value_research import (
            RelativeValueResearchConfig,
            evaluate_research,
        )

        report = evaluate_research(
            snapshots,
            RelativeValueResearchConfig(**params["config"]),
            instrument_names=params.get("instrument_names"),
        )
        results = {key: value for key, value in vars(report).items() if key != "diagnostics"}
        diagnostics = report.diagnostics
    elif kind in {"barrier", "barrier_convergence", "barrier_model_hedge"}:
        from deribit_dashboard.barrier_models import HestonParameters
        from deribit_dashboard.barrier_workbench import (
            BarrierExperiment,
            prepare_barrier_market,
            run_barrier_comparison,
        )

        values = dict(params["config"])
        if values.get("heston_parameters"):
            values["heston_parameters"] = HestonParameters(**values["heston_parameters"])
        market = prepare_barrier_market(
            snapshots[0],
            params["instrument_name"],
            params["perpetual_name"],
            params.get("rate", 0),
            params.get("stale_seconds", 60),
        )
        report = run_barrier_comparison(market, BarrierExperiment(**values))
        results = {
            n: getattr(report, n)
            for n in (
                "prices",
                "vanilla",
                "probabilities",
                "replication_costs",
                "replication_weights",
                "hedge_metrics",
                "path_results",
            )
        }
        diagnostics = report.diagnostics
        if kind == "barrier_convergence":
            from deribit_dashboard.barrier_convergence import (
                ConvergenceControls,
                barrier_convergence,
            )

            trials, summary = barrier_convergence(report, ConvergenceControls(**params["controls"]))
            results = dict(trials=trials, summary=summary)
        elif kind == "barrier_model_hedge":
            from deribit_dashboard.barrier_model_hedge import ModelHedgeControls, model_hedge_study

            details, summary, audit = model_hedge_study(
                report, ModelHedgeControls(**params["controls"])
            )
            results = dict(path_results=details, summary=summary, sensitivity_audit=audit)
    elif kind == "strategy":
        from deribit_dashboard.strategies import simulate_strategy, stress_strategy
        from deribit_dashboard.strategy_workbench import (
            prepare_strategy,
            restore_simulation_config,
            scenario_rows,
        )

        config = restore_simulation_config(params["simulation"])
        prepared = prepare_strategy(
            snapshots[0],
            params["legs"],
            params["model"],
            rate=params["discount_rate"],
            cash_currency=params["cash_currency"],
            conversion_rates=params.get("conversion_rates", {}),
            max_quote_age_seconds=params.get("max_quote_age_seconds", 60),
        )
        stresses = stress_strategy(
            prepared.legs,
            scenario_rows(pd.DataFrame(params["stress_scenarios"])),
            initial_equity=config.initial_equity,
            entry_cost_bps=config.entry_cost_bps,
            annual_funding_rates=config.annual_funding_rates,
            surface_evolution=config.surface_evolution,
            margin=config.margin,
        )
        simulation = (
            simulate_strategy(prepared.legs, config) if params.get("run_simulation", True) else None
        )
        results = dict(
            summary=simulation.summary if simulation else {},
            metrics=simulation.path_metrics if simulation else pd.DataFrame(),
            equity=simulation.equity if simulation else pd.DataFrame(),
            stresses=stresses,
            margin_ledger=simulation.margin_ledger if simulation else pd.DataFrame(),
        )
        diagnostics = dict(
            assumptions=simulation.assumptions if simulation else [],
            calibration=prepared.calibration,
        )
    else:
        raise ValueError(
            "This experiment stores complete inputs and results; automated replay for this kind is unavailable"
        )
    diagnostics = dict(
        diagnostics,
        reproduced_from=payload["run_id"],
        original_provenance=payload["provenance"],
        provenance_changed=payload["provenance"] != provenance(),
    )
    return make_experiment(
        kind,
        snapshots,
        params,
        results,
        diagnostics,
        f"Replay: {payload['name']}",
        payload["run_id"],
    )
