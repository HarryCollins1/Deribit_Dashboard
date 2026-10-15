"""Snapshot preparation and strategy composition, independent of the dashboard."""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from deribit_dashboard.models import Snapshot
from deribit_dashboard.workbench import analyze, price_selection


def product_type(metadata):
    if metadata.get("kind") == "future" and metadata.get("settlement_period") == "perpetual":
        return "perpetual"
    return metadata.get("kind", "unknown")


def combine_snapshots(snapshots, max_skew_seconds=60.0):
    """Combine explicitly chosen same-environment captures without rewriting their data."""
    if not snapshots or max_skew_seconds < 0:
        raise ValueError("Choose snapshots and a nonnegative maximum capture-time difference")
    if len({s.environment for s in snapshots}) != 1:
        raise ValueError("Strategies cannot mix demo, testnet and production snapshots")
    captured = max(s.captured_at for s in snapshots)
    if (captured - min(s.captured_at for s in snapshots)).total_seconds() > max_skew_seconds:
        raise ValueError("Selected snapshot capture times exceed the allowed difference")
    instruments, tickers, received, errors = {}, {}, {}, {}
    for snapshot in sorted(snapshots, key=lambda s: s.captured_at):
        for name, metadata in snapshot.instruments.items():
            if metadata.get("base_currency") not in {"BTC", "ETH"} or metadata.get("kind") not in {
                "option",
                "future",
                "spot",
            }:
                continue
            incoming = snapshot.tickers.get(name)
            existing = tickers.get(name)
            if existing and (not incoming or incoming["timestamp"] < existing["timestamp"]):
                continue
            instruments[name] = copy.deepcopy(metadata)
            if incoming:
                tickers[name] = copy.deepcopy(incoming)
                received[name] = snapshot.received_at[name]
                errors.pop(name, None)
            elif name in snapshot.errors:
                errors[name] = snapshot.errors[name]
    if not instruments:
        raise ValueError(
            "These snapshots contain no BTC/ETH option, future, perpetual or spot data"
        )
    return Snapshot(
        currency="ANY",
        environment=snapshots[0].environment,
        source="strategy-combined",
        captured_at=captured,
        collection_started_at=min(s.collection_started_at for s in snapshots),
        instruments=instruments,
        tickers=tickers,
        received_at=received,
        errors=errors,
    )


def strategy_catalogue(snapshot):
    rows = []
    for name, metadata in snapshot.instruments.items():
        if metadata.get("base_currency") not in {"BTC", "ETH"} or metadata.get("kind") not in {
            "option",
            "future",
            "spot",
        }:
            continue
        ticker = snapshot.tickers.get(name, {})
        expiry = metadata.get("expiration_timestamp")
        rows.append(
            dict(
                instrument_name=name,
                base=metadata["base_currency"],
                product=product_type(metadata),
                quote_currency=metadata.get("quote_currency"),
                settlement_currency=metadata.get("settlement_currency"),
                contract_size=metadata.get("contract_size"),
                quantity_unit="base coin" if metadata["kind"] == "spot" else "contracts",
                expiry=pd.to_datetime(expiry, unit="ms", utc=True)
                if expiry and product_type(metadata) != "perpetual"
                else pd.NaT,
                strike=metadata.get("strike"),
                side=metadata.get("option_type"),
                ticker_available=bool(ticker),
                best_bid_amount=ticker.get("best_bid_amount"),
                best_ask_amount=ticker.get("best_ask_amount"),
            )
        )
    return pd.DataFrame(rows)


def strategy_row(name, quantity=1.0, **overrides):
    row = dict(
        instrument_name=name,
        quantity=float(quantity),
        barrier_type="none",
        barrier_level=0.0,
        barrier_hit=False,
        monitoring_days=1.0,
    )
    row.update(overrides)
    return row


def strategy_templates(snapshot):
    """Discover illustrative templates; leg identifiers always come from metadata."""
    catalogue = strategy_catalogue(snapshot)
    if catalogue.empty:
        return {}
    templates = {"Custom legs": [strategy_row(catalogue.instrument_name.iloc[0])]}
    perpetuals = catalogue[catalogue["product"] == "perpetual"]
    btc, eth = perpetuals[perpetuals.base == "BTC"], perpetuals[perpetuals.base == "ETH"]
    if not btc.empty and not eth.empty:
        templates["BTC / ETH perpetual pair"] = [
            strategy_row(btc.instrument_name.iloc[0], 1),
            strategy_row(eth.instrument_name.iloc[0], -1),
        ]
    for base in sorted(catalogue.base.unique()):
        options = catalogue[(catalogue.base == base) & (catalogue["product"] == "option")]
        if options.empty:
            continue
        # A template uses one settlement convention, with all choices still editable.
        settlement = (
            base
            if base in options.settlement_currency.values
            else options.settlement_currency.iloc[0]
        )
        options = options[options.settlement_currency == settlement]
        expiries = sorted(options.expiry.dropna().unique())
        if not expiries:
            continue
        expiry = min(
            expiries, key=lambda e: abs((e - snapshot.captured_at).total_seconds() / 86400 - 30)
        )
        same_expiry = options[options.expiry == expiry]
        spots = [snapshot.tickers.get(n, {}).get("index_price") for n in options.instrument_name]
        valid = [s for s in spots if s is not None and np.isfinite(s) and s > 0]
        if not valid:
            continue
        spot = float(np.median(valid))
        calls = same_expiry[same_expiry.side == "call"].sort_values("strike")
        if calls.empty:
            continue
        near = calls.iloc[int(np.argmin(abs(calls.strike.to_numpy() - spot)))]
        above = calls[calls.strike > near.strike]
        if not above.empty:
            templates[f"{base} call vertical"] = [
                strategy_row(near.instrument_name),
                strategy_row(above.instrument_name.iloc[0], -1),
            ]
        puts = same_expiry[(same_expiry.side == "put") & (same_expiry.strike == near.strike)]
        if not puts.empty:
            templates[f"{base} long straddle"] = [
                strategy_row(near.instrument_name),
                strategy_row(puts.instrument_name.iloc[0]),
            ]
        later = options[
            (options.expiry > expiry) & (options.side == "call") & (options.strike == near.strike)
        ].sort_values("expiry")
        if not later.empty:
            templates[f"{base} call calendar"] = [
                strategy_row(near.instrument_name, -1),
                strategy_row(later.instrument_name.iloc[0]),
            ]
        row = strategy_row(near.instrument_name)
        row.update(barrier_type="up-and-out", barrier_level=spot * 1.2)
        templates[f"{base} barrier call (hypothetical)"] = [row]
        spots = catalogue[(catalogue.base == base) & (catalogue["product"] == "spot")]
        if not spots.empty:
            multiplier = float(snapshot.instruments[near.instrument_name]["contract_size"])
            templates[f"{base} covered call"] = [
                strategy_row(spots.instrument_name.iloc[0], multiplier),
                strategy_row(near.instrument_name, -1),
            ]
    return templates


@dataclass
class StrategyPreparation:
    legs: list
    calibration: pd.DataFrame


def prepare_strategy(
    snapshot,
    rows,
    model_name="Black-Scholes",
    *,
    rate=0.0,
    cash_currency="USD",
    conversion_rates=None,
    max_quote_age_seconds=60.0,
):
    from deribit_dashboard.strategies import build_strategy_legs

    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    if not rows or len(rows) > 16:
        raise ValueError("Choose between 1 and 16 strategy legs")
    names = [r.get("instrument_name") for r in rows]
    if any(name not in snapshot.instruments for name in names):
        raise ValueError("Every leg must select an instrument from this snapshot")
    option_names = {n for n in names if snapshot.instruments[n]["kind"] == "option"}
    pairs = {
        (snapshot.instruments[n]["base_currency"], snapshot.instruments[n]["settlement_currency"])
        for n in option_names
    }
    calibrated, audit = {}, []
    for base, settlement in sorted(pairs):
        result = analyze(snapshot, base, settlement, rate, max_quote_age_seconds)
        selected = sorted(
            n
            for n in option_names
            if snapshot.instruments[n]["base_currency"] == base
            and snapshot.instruments[n]["settlement_currency"] == settlement
        )
        if result.surface is None:
            raise ValueError(f"{base}/{settlement} surface unavailable: {result.fit_error}")
        chain = result.chain.set_index("instrument_name")
        for name in selected:
            row = chain.loc[name]
            if row.fit_status != "fitted":
                raise ValueError(
                    f"{name}: {row.fit_status}; select a fitted option or refresh the chain"
                )
            audit.append(
                dict(
                    instrument_name=name,
                    own_iv=row.own_iv,
                    fitted_iv=row.fitted_iv,
                    fitted_price=row.fitted_price,
                    fit_status=row.fit_status,
                )
            )
        priced = price_selection(result, snapshot, selected, model_name, rate)
        fitted = result.chain[result.chain.fit_status == "fitted"]
        reference = tuple(
            (float(part.years.iloc[0]), tuple(part.strike), tuple(part.fitted_iv))
            for _, part in (
                (years, group.drop_duplicates("strike").sort_values("strike"))
                for years, group in fitted.groupby("years", sort=True)
            )
        )
        for row in priced:
            row["surface_reference"] = reference
        calibrated.update({r["instrument_name"]: r for r in priced})
    legs = build_strategy_legs(
        snapshot,
        rows,
        calibrated_options=calibrated,
        cash_currency=cash_currency,
        conversion_rates=conversion_rates,
        max_quote_age_seconds=max_quote_age_seconds,
    )
    return StrategyPreparation(legs, pd.DataFrame(audit))


def scenario_rows(frame):
    """Convert explicitly labelled UI scenario inputs into the public simulation schema."""
    rows = frame.to_dict("records") if isinstance(frame, pd.DataFrame) else frame
    return [
        dict(
            name=str(row["name"]),
            spot_shocks={"BTC": float(row["BTC"]), "ETH": float(row["ETH"])},
            volatility_shock=float(row["volatility_shock"]),
            days=float(row["days"]),
            path_shape=str(row.get("path_shape", "monotone")),
            event_day=float(row["days"]) * float(row.get("event_fraction", 0.5)),
            peak_shocks={
                "BTC": float(row.get("BTC_peak", 0.0)),
                "ETH": float(row.get("ETH_peak", 0.0)),
            },
            basis_shocks={
                "BTC": float(row.get("BTC_basis", 0.0)),
                "ETH": float(row.get("ETH_basis", 0.0)),
            },
            annual_funding_rates={
                asset: float(row[f"{asset}_funding"])
                for asset in ("BTC", "ETH")
                if f"{asset}_funding" in row and pd.notna(row[f"{asset}_funding"])
            },
        )
        for row in rows
    ]


def simulation_plot_data(result):
    """Summaries for static plots; percentiles are cross-path, metrics remain per-path."""
    equity = result.equity
    return pd.DataFrame(
        {
            "Elapsed time (days)": equity.index,
            "5th percentile": equity.quantile(0.05, axis=1).to_numpy(),
            "Median": equity.median(axis=1).to_numpy(),
            "95th percentile": equity.quantile(0.95, axis=1).to_numpy(),
        }
    )


def restore_simulation_config(values):
    """Reconstruct a saved experiment's explicit nested scenario inputs."""
    from deribit_dashboard.strategies import SimulationConfig
    from deribit_dashboard.strategy_research import MarginScenario, PathRegime, SurfaceEvolution

    values = dict(values)
    values["surface_evolution"] = SurfaceEvolution(**values.get("surface_evolution", {}))
    values["margin"] = MarginScenario(**values.get("margin", {}))
    values["regimes"] = tuple(PathRegime(**row) for row in values.get("regimes", ()))
    return SimulationConfig(**values)
