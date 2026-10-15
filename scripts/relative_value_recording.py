"""Prepare, run and inspect a fixed BTC public-data cohort without placing orders."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from deribit_dashboard.client import DeribitClient
from deribit_dashboard.config import Settings
from deribit_dashboard.experiments import ExperimentStore, make_experiment
from deribit_dashboard.models import utc_now
from deribit_dashboard.recording import (
    check_test_deadline,
    load_plan,
    prepare_plan,
    record,
    recorded_ids,
    recording_status,
    write_json,
)
from deribit_dashboard.recording_studies import RecordingStudyStore, evaluate_study
from deribit_dashboard.relative_value_execution import TradingEvaluationConfig
from deribit_dashboard.relative_value_research import RelativeValueResearchConfig, evaluate_research
from deribit_dashboard.storage import SnapshotStore
from deribit_dashboard.walk_forward import evaluate_trading


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Discover and predeclare a fixed 14-day study")
    prepare.add_argument("--directory", type=Path, required=True)
    prepare.add_argument("--catalogue", type=Path)
    prepare.add_argument("--strikes", type=int, default=9)
    interview = commands.add_parser(
        "prepare-interview", help="Declare a fixed 90-minute production demonstration"
    )
    interview.add_argument("--directory", type=Path, required=True)
    interview.add_argument("--catalogue", type=Path)
    interview.add_argument("--strikes", type=int, default=9)
    run = commands.add_parser("record", help="Record public books for a bounded duration")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--hours", type=float, default=2)
    run.add_argument("--interval", type=int, choices=[60, 67, 120, 240, 241])
    run.add_argument("--max-database-gb", type=float, default=2)
    run.add_argument(
        "--evaluate-on-completion",
        action="store_true",
        help="Save registered study results after the fixed test deadline, never early",
    )
    status = commands.add_parser("status", help="Inspect capture/funding health without test P&L")
    status.add_argument("--plan", type=Path, required=True)
    pilot = commands.add_parser(
        "pilot-report", help="Accounting-only diagnostic, never OOS evidence"
    )
    pilot.add_argument("--plan", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate the predeclared study after test end")
    evaluate.add_argument("--plan", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command in {"prepare", "prepare-interview"}:
        plan_path = args.directory / "plan.json"
        if plan_path.exists():
            raise ValueError("Plan already exists; use a new directory for a different design")

        async def discover():
            async with DeribitClient(Settings(environment="production", currency="BTC")) as client:
                catalogue = (
                    json.loads(args.catalogue.read_text())
                    if args.catalogue
                    else await client.instruments()
                )
                ticker = await client.ticker("BTC-PERPETUAL")
                return catalogue, ticker["index_price"]

        catalogue, index = asyncio.run(discover())
        directory = args.directory.resolve()
        plan = prepare_plan(
            catalogue,
            index,
            utc_now(),
            f"sqlite:///{directory / 'production.db'}",
            strikes=args.strikes,
            **(
                dict(
                    training_days=30 / 1440,
                    validation_days=30 / 1440,
                    test_days=30 / 1440,
                    interval_seconds=67,
                    purpose="interview_demonstration",
                    candidate_scope="interior",
                    horizon_seconds=600,
                    minimum_validation_trades=1,
                )
                if args.command == "prepare-interview"
                else {}
            ),
        )
        write_json(plan_path, plan, exclusive=True)
        write_json(args.directory / "research-config.json", plan["research_config"], exclusive=True)
        if args.command == "prepare-interview":
            store = SnapshotStore(plan["database_url"])
            try:
                store.initialize()
                studies = RecordingStudyStore(store.engine)
                studies.initialize()
                studies.register(plan)
            finally:
                store.close()
        print(
            f"Prepared {plan_path}; {len(plan['instrument_names'])} instruments, candidates {plan['candidate_names']}"
        )
        print(
            f"Fixed test end: {plan['research_config']['test_end']}; expected study captures: {plan['expected_captures']}"
        )
        return
    plan = load_plan(args.plan)
    if args.command == "record":
        # An OS-held lock is released on normal exit and crashes. The operational
        # launcher targets macOS/Linux; the existing core collector stays portable.
        import fcntl

        if args.evaluate_on_completion:
            store = SnapshotStore(plan["database_url"])
            try:
                studies = RecordingStudyStore(store.engine)
                studies.initialize()
                studies.load(plan["digest"])
            finally:
                store.close()
        with args.plan.with_name("collector.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("This plan already has an active collector") from exc
            asyncio.run(
                record(
                    plan,
                    args.plan.with_name("status.json"),
                    seconds=args.hours * 3600,
                    interval_seconds=args.interval,
                    max_database_bytes=int(args.max_database_gb * 1e9),
                )
            )
            if args.evaluate_on_completion:
                try:
                    check_test_deadline(plan)
                except ValueError:
                    logging.warning(
                        "Stopped before the fixed test deadline; no early test evaluation"
                    )
                else:
                    store = SnapshotStore(plan["database_url"])
                    status_path = args.plan.with_name("status.json")
                    state = json.loads(status_path.read_text())
                    try:
                        state["evaluation_state"] = "evaluating"
                        write_json(status_path, state)
                        payload = evaluate_study(store, plan["digest"])
                        write_json(args.plan.with_name("experiment.json"), payload)
                        state.update(evaluation_state="complete", run_id=payload["run_id"])
                        write_json(status_path, state)
                        print(f"Saved held-out experiment {payload['run_id']}; alpha_claim=False")
                    except BaseException as exc:
                        state.update(evaluation_state="failed", evaluation_error=str(exc))
                        write_json(status_path, state)
                        raise
                    finally:
                        store.close()
        return
    if args.command == "evaluate":
        check_test_deadline(plan)
    store = SnapshotStore(plan["database_url"])
    store.initialize()
    try:
        if args.command == "status":
            print(json.dumps(recording_status(store, plan), indent=2))
            return
        ids = recorded_ids(store, plan)
        if args.command == "pilot-report":
            studies = RecordingStudyStore(store.engine)
            studies.initialize()
            if any(row["digest"] == plan["digest"] for row in studies.list()):
                check_test_deadline(plan)
        inputs = [store.load(snapshot_id) for snapshot_id in ids]
        if args.command == "pilot-report":
            config = TradingEvaluationConfig(**plan["research_config"]["trading_config"])
            report = evaluate_trading(inputs, config, instrument_names=plan["candidate_names"])
            write_json(
                args.output,
                dict(
                    label="Accounting-only pilot; hypothetical executions, not out-of-sample evidence",
                    alpha_claim=False,
                    plan_digest=plan["digest"],
                    report=vars(report),
                ),
            )
            print(
                f"Saved accounting-only diagnostic {args.output}; {len(report.trades)} hypothetical trades"
            )
        else:
            studies = RecordingStudyStore(store.engine)
            studies.initialize()
            if any(row["digest"] == plan["digest"] for row in studies.list()):
                payload = evaluate_study(store, plan["digest"])
                write_json(args.output, payload)
                print(f"Saved experiment {payload['run_id']} to {args.output}; alpha_claim=False")
                return
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
                dict(report.diagnostics, recording_plan=plan),
            )
            registry = ExperimentStore(store.engine)
            registry.initialize()
            registry.save(payload)
            write_json(args.output, payload)
            print(f"Saved experiment {payload['run_id']} to {args.output}; alpha_claim=False")
    finally:
        store.close()


if __name__ == "__main__":
    main()
