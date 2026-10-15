"""Command-line entry points, including installed-package Streamlit launch."""

import argparse
import asyncio
import csv
import json
import logging
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from deribit_dashboard.client import DeribitClient
from deribit_dashboard.collector import capture_snapshot, collect
from deribit_dashboard.config import Settings
from deribit_dashboard.demo import demo_snapshot
from deribit_dashboard.history import download_candles
from deribit_dashboard.models import Snapshot, parse_time, utc_now
from deribit_dashboard.storage import SnapshotStore


def parser():
    result = argparse.ArgumentParser(
        description="BTC/ETH options, futures, perpetual and spot toolkit"
    )
    result.add_argument("--database-url", help="SQLite URL (or DERIBIT_DATABASE_URL)")
    result.add_argument("--environment", choices=["testnet", "production"])
    result.add_argument(
        "--currency",
        choices=["BTC", "ETH", "ANY"],
        help="Underlying scope; ANY means BTC and ETH only",
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create the package's SQLite tables")
    capture = commands.add_parser("capture", help="Capture BTC/ETH products in a REST snapshot")
    capture.add_argument("--instrument", help="Capture only this exact instrument")
    capture.add_argument("--output", type=Path, help="Also export a snapshot JSON")
    stream = commands.add_parser("collect", help="Stream tickers and save periodic snapshots")
    stream.add_argument(
        "--interval", type=float, help="Database snapshot interval in seconds, minimum 2"
    )
    stream.add_argument(
        "--mode",
        choices=["realtime", "2s"],
        help="Save latest tickers or completed local two-second OHLC buckets",
    )
    stream.add_argument(
        "--ticker-interval",
        choices=["100ms", "agg2"],
        help="Public exchange subscription interval (default 100ms); 2s mode requires 100ms",
    )
    stream.add_argument(
        "--instrument",
        action="append",
        help="Stream only this discovered instrument; repeat for multiple legs",
    )
    demo = commands.add_parser("demo", help="Save synthetic snapshots in a separate demo namespace")
    demo.add_argument("--count", type=int, default=6)
    listing = commands.add_parser("snapshots", help="List saved snapshots")
    listing.add_argument("--demo", action="store_true")
    listing.add_argument("--start", type=parse_time)
    listing.add_argument("--end", type=parse_time)
    export = commands.add_parser("export", help="Export a snapshot JSON")
    export.add_argument("snapshot_id")
    export.add_argument("output", type=Path)
    imp = commands.add_parser("import", help="Import the package snapshot JSON format")
    imp.add_argument("input", type=Path)
    candles = commands.add_parser("candles", help="Download historical OHLCV, not Greeks/surfaces")
    candles.add_argument("--instrument", required=True)
    candles.add_argument("--start", type=parse_time, required=True)
    candles.add_argument("--end", type=parse_time, required=True)
    candles.add_argument("--resolution", default="60")
    candles.add_argument("--output", type=Path, required=True)
    dashboard = commands.add_parser("dashboard", help="Launch Streamlit from the installed package")
    dashboard.add_argument("--port", type=int, default=8501)
    catalogue = commands.add_parser(
        "instruments", help="Discover active BTC/ETH products across settlement currencies"
    )
    catalogue.add_argument("--output", type=Path, default=Path("data/instruments.json"))
    experiment_list = commands.add_parser("experiments", help="List immutable research experiments")
    experiment_list.add_argument("--kind")
    experiment_export = commands.add_parser(
        "experiment-export", help="Export a complete portable experiment"
    )
    experiment_export.add_argument("run_id")
    experiment_export.add_argument("output", type=Path)
    experiment_import = commands.add_parser(
        "experiment-import", help="Validate and import portable experiment JSON"
    )
    experiment_import.add_argument("input", type=Path)
    experiment_replay = commands.add_parser(
        "experiment-replay", help="Reproduce embedded inputs with current code"
    )
    experiment_replay.add_argument("run_id")
    experiment_replay.add_argument("--output", type=Path)
    walk = commands.add_parser(
        "walk-forward", help="Evaluate later quotes from recorded option-chain snapshots"
    )
    walk.add_argument("snapshot_ids", nargs="+")
    walk.add_argument("--base", choices=["BTC", "ETH"], default="BTC")
    walk.add_argument("--settlement", default="BTC")
    walk.add_argument("--horizon", type=float, default=60.0)
    walk.add_argument("--tolerance", type=float, default=30.0)
    walk.add_argument(
        "--config", type=Path, help="JSON WalkForwardConfig; overrides individual defaults"
    )
    walk.add_argument("--output", type=Path)
    relative = commands.add_parser(
        "relative-value", help="Chronological held-out BTC delta-hedged trading research"
    )
    relative.add_argument("snapshot_ids", nargs="+")
    relative.add_argument(
        "--config", type=Path, required=True, help="JSON RelativeValueResearchConfig"
    )
    relative.add_argument(
        "--instrument",
        action="append",
        help="Candidate BTC vanilla option; repeat for multiple options, default all",
    )
    relative.add_argument("--output", type=Path, help="Complete portable experiment JSON")
    return result


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.from_env(
        database_url=args.database_url,
        environment=args.environment,
        currency=args.currency,
        snapshot_interval=getattr(args, "interval", None),
        ticker_interval=getattr(args, "ticker_interval", None),
        feed_mode=getattr(args, "mode", None),
    )
    if args.command == "dashboard":
        import os

        env = dict(os.environ)
        env.update(
            DERIBIT_DATABASE_URL=settings.database_url,
            DERIBIT_ENVIRONMENT=settings.environment,
            DERIBIT_CURRENCY=settings.currency,
        )
        app = Path(__file__).with_name("app.py")
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(app),
                    "--server.port",
                    str(args.port),
                ],
                env=env,
            )
        )
    store = SnapshotStore(settings.database_url)
    try:
        store.initialize()
        match args.command:
            case (
                "experiments"
                | "experiment-export"
                | "experiment-import"
                | "experiment-replay"
                | "walk-forward"
                | "relative-value"
            ):
                from deribit_dashboard.experiments import (
                    ExperimentStore,
                    make_experiment,
                    reproduce_experiment,
                )

                registry = ExperimentStore(store.engine)
                registry.initialize()
                if args.command == "experiments":
                    for row in registry.list(kind=args.kind):
                        print(f"{row['run_id']}  {row['created_at']}  {row['kind']}  {row['name']}")
                elif args.command == "experiment-export":
                    write_text(args.output, registry.export_json(args.run_id))
                    print(f"Exported {args.output}")
                elif args.command == "experiment-import":
                    print(
                        f"Imported {registry.import_json(args.input.read_text(encoding='utf-8'))}"
                    )
                else:
                    if args.command == "experiment-replay":
                        payload = reproduce_experiment(registry.load(args.run_id))
                    elif args.command == "relative-value":
                        from dataclasses import asdict

                        from deribit_dashboard.relative_value_research import (
                            RelativeValueResearchConfig,
                            evaluate_research,
                        )

                        inputs = [store.load(i) for i in args.snapshot_ids]
                        config = RelativeValueResearchConfig(
                            **json.loads(args.config.read_text(encoding="utf-8"))
                        )
                        result = evaluate_research(inputs, config, instrument_names=args.instrument)
                        payload = make_experiment(
                            "relative_value",
                            inputs,
                            dict(
                                config=asdict(config),
                                instrument_names=sorted(set(args.instrument))
                                if args.instrument
                                else None,
                            ),
                            {
                                key: value
                                for key, value in vars(result).items()
                                if key != "diagnostics"
                            },
                            result.diagnostics,
                        )
                    else:
                        from dataclasses import asdict

                        from deribit_dashboard.walk_forward import (
                            WalkForwardConfig,
                            evaluate_walk_forward,
                        )

                        inputs = [store.load(i) for i in args.snapshot_ids]
                        values = (
                            json.loads(args.config.read_text(encoding="utf-8"))
                            if args.config
                            else dict(
                                base=args.base,
                                settlement=args.settlement,
                                horizon_seconds=args.horizon,
                                tolerance_seconds=args.tolerance,
                            )
                        )
                        config = WalkForwardConfig(**values)
                        result = evaluate_walk_forward(inputs, config)
                        payload = make_experiment(
                            "walk_forward",
                            inputs,
                            asdict(config),
                            dict(
                                observations=result.observations,
                                folds=result.folds,
                                coverage=result.coverage,
                                metrics=result.metrics,
                            ),
                            result.diagnostics,
                        )
                    run_id = registry.save(payload)
                    if args.output:
                        write_text(args.output, registry.export_json(run_id))
                    print(f"Saved experiment {run_id}")
            case "init-db":
                print("Snapshot tables ready.")
            case "capture":
                snapshot = asyncio.run(capture_snapshot(settings, args.instrument))
                store.save(snapshot)
                if args.output:
                    write_text(args.output, snapshot.to_json())
                print(f"Saved {snapshot.snapshot_id}: {snapshot.coverage:.1%} ticker coverage")
            case "collect":
                asyncio.run(collect(settings, store, instrument_names=args.instrument))
            case "demo":
                if settings.currency not in {"BTC", "ETH"}:
                    raise ValueError("Synthetic demo data is available for BTC or ETH")
                if not 1 <= args.count <= 1000:
                    raise ValueError("Demo count must be between 1 and 1000")
                now = utc_now()
                for i in range(args.count):
                    snapshot = demo_snapshot(
                        settings.currency, now - timedelta(minutes=30 * i), shift=0.002 * i
                    )
                    store.save(snapshot)
                print(f"Saved {args.count} synthetic snapshots under environment=demo.")
            case "snapshots":
                rows = store.list(
                    "demo" if args.demo else settings.environment,
                    None if settings.currency == "ANY" else settings.currency,
                    start=args.start,
                    end=args.end,
                )
                for row in rows:
                    print(
                        f"{row['snapshot_id']}  {row['captured_at']}  {row['source']}  "
                        f"{row['ticker_count']}/{row['instrument_count']}"
                    )
                if not rows:
                    print("No snapshots found.")
            case "export":
                write_text(args.output, store.load(args.snapshot_id).to_json())
                print(f"Exported {args.output}")
            case "import":
                snapshot = Snapshot.from_json(args.input.read_text(encoding="utf-8"))
                print(f"Saved {store.save(snapshot)} under {snapshot.environment}.")
            case "candles":

                async def fetch():
                    async with DeribitClient(settings) as client:
                        return await download_candles(
                            client, args.instrument, args.start, args.end, args.resolution
                        )

                rows = asyncio.run(fetch())
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w", newline="", encoding="utf-8") as handle:
                    fields = [
                        "timestamp_ms",
                        "instrument_name",
                        "environment",
                        "resolution",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Downloaded {len(rows)} candles to {args.output}")
            case "instruments":

                async def fetch_catalogue():
                    async with DeribitClient(settings) as client:
                        return await client.instruments()

                catalogue = asyncio.run(fetch_catalogue())
                write_text(args.output, json.dumps(list(catalogue.values()), indent=2))
                print(f"Downloaded {len(catalogue)} active instruments to {args.output}")
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        store.close()
