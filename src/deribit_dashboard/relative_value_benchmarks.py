"""Simple peer-flat Black--Scholes and a training-anchored frozen-strike smile.

All strategies share the original candidate-strike exclusion. Benchmarks reuse
audited peer IV/forward inputs; they never fit the held-out option's market price.
"""

from dataclasses import replace
from math import log, sqrt

import numpy as np
import pandas as pd

from deribit_dashboard.models import milliseconds
from deribit_dashboard.pricing import YEAR_SECONDS, BlackScholes, Option, implied_vol, risk
from deribit_dashboard.relative_value import (
    RelativeValueReport,
    decide_signal,
    evaluate_relative_value,
)
from deribit_dashboard.relative_value_execution import IDENTITY_FIELDS
from deribit_dashboard.surface import Contract

STRATEGIES = ("relative_value", "flat_black_scholes", "frozen_smile")


def classify_report(report, config, threshold):
    """Apply a fixed cash threshold (None explicitly selects the no-trade policy)."""
    rows = []
    for source in report.signals.to_dict("records"):
        row = dict(source)
        if row["status"] == "valued":
            decision = decide_signal(
                row["fair_value_cash"],
                row["market_bid_cash"],
                row["market_ask_cash"],
                estimated_cost_cash=row["estimated_round_trip_cost_cash"],
                uncertainty_cash=row["uncertainty_buffer_cash"],
                minimum_edge_cash=threshold if threshold is not None else 0.0,
            )
            row.update(
                signal=decision.signal,
                minimum_edge_cash=threshold,
                gross_buy_edge_cash=decision.gross_buy_edge_cash,
                gross_sell_edge_cash=decision.gross_sell_edge_cash,
                buy_net_edge_cash=decision.buy_net_edge_cash,
                sell_net_edge_cash=decision.sell_net_edge_cash,
            )
            row["signal_strength_cash"] = max(
                decision.buy_net_edge_cash, decision.sell_net_edge_cash
            )
            if threshold is None:
                row.update(signal="PASS", reason="selected no-trade policy")
            elif (
                decision.signal != "PASS"
                and row[
                    "ask_depth_contracts" if decision.signal == "BUY" else "bid_depth_contracts"
                ]
                <= 0
            ):
                row.update(signal="PASS", reason=f"no displayed entry depth for {decision.signal}")
            else:
                row["reason"] = "edge below required threshold" if decision.signal == "PASS" else ""
        rows.append(row)
    return replace(report, signals=pd.DataFrame(rows) if rows else report.signals.copy())


def _priced_row(source, metadata, config, options, residual):
    model = BlackScholes()
    values = {mark: model.price(option) for mark, option in options.items()}
    option, fair = options["mid"], values["mid"]
    contract = Contract.from_metadata(metadata)
    greeks = risk(model, option, contract.multiplier)
    factor = contract.cash_factor(option.spot)
    sensitivity = max(abs(values[mark] - fair) for mark in ("bid", "ask"))
    uncertainty = max(config.uncertainty_floor_cash, residual, sensitivity)
    return dict(
        source,
        status="valued",
        reason="",
        signal="PASS",
        fair_value_cash=fair,
        fair_value_quote=fair / factor,
        peer_reference_spot=option.spot,
        peer_forward=option.forward,
        years=option.years,
        volatility=option.volatility,
        carry_yield=option.dividend,
        rate=option.rate,
        cash_delta_per_contract=greeks["cash_delta"],
        coin_delta_per_contract=greeks["coin_delta"],
        market_bid_cash=source["market_bid_quote"] * factor,
        market_ask_cash=source["market_ask_quote"] * factor,
        peer_residual_rmse=residual,
        peer_quote_sensitivity_cash=sensitivity,
        bid_refit_fair_value_cash=values["bid"],
        ask_refit_fair_value_cash=values["ask"],
        uncertainty_cash=uncertainty,
        uncertainty_buffer_cash=config.uncertainty_multiplier * uncertainty,
        estimated_round_trip_cost_cash=config.estimated_round_trip_cost_cash
        + option.spot * config.estimated_round_trip_cost_bps / 10000,
    )


def _flat_options(source, report, config):
    options, residual = {}, np.nan
    if "fold_id" not in source or report.calibration.empty:
        raise ValueError(source["reason"] or "no eligible peer calibration")
    for mark in ("mid", "bid", "ask"):
        fits = report.fits[
            (report.fits.fold_id == source["fold_id"]) & (report.fits.quote_mark == mark)
        ]
        if fits.empty:
            raise ValueError(f"{mark} peer calibration unavailable for flat benchmark")
        fit = fits.iloc[0]
        peers = report.calibration[
            (report.calibration.fold_id == source["fold_id"])
            & (report.calibration.quote_mark == mark)
        ]
        if "iv_status" not in peers:
            raise ValueError("peer IV unavailable for flat benchmark")
        usable = peers[peers.iv_status == "included"]
        if (
            usable.strike.nunique() < 3
            or not np.isfinite(fit.peer_forward)
            or fit.peer_forward <= 0
        ):
            raise ValueError(
                "flat benchmark needs a valid peer forward and three invertible strikes"
            )
        # Equal strike weighting avoids double-weighting a strike with both rights.
        volatility = float(usable.groupby("strike").own_iv.median().median())
        carry = config.rate - log(fit.peer_forward / fit.peer_reference_spot) / usable.years.iloc[0]
        options[mark] = Option(
            fit.peer_reference_spot,
            source["strike"],
            usable.years.iloc[0],
            volatility,
            config.rate,
            carry,
            source["side"],
        )
        if mark == "mid":
            errors = [
                BlackScholes().price(replace(options[mark], strike=row.strike, side=row.side))
                - row.calibration_price
                for row in usable.itertuples()
            ]
            residual = sqrt(float(np.mean(np.square(errors))))
    return options, residual


def _frozen_options(snapshot, source, current, anchor, anchor_report, config):
    if anchor is None or anchor_report is None:
        raise ValueError("no earlier training anchor")
    if anchor.captured_at > snapshot.captured_at:
        raise ValueError("training anchor is after the signal cutoff")
    name = source["instrument_name"]
    expected, actual = anchor.instruments.get(name), snapshot.instruments[name]
    if expected is None or any(expected.get(key) != actual.get(key) for key in IDENTITY_FIELDS):
        raise ValueError("frozen anchor contract identity unavailable or changed")
    matches = anchor_report.signals[anchor_report.signals.instrument_name == name]
    if matches.empty or matches.iloc[0].status != "valued":
        reason = matches.iloc[0].reason if not matches.empty else "not captured at anchor"
        raise ValueError(f"frozen anchor valuation unavailable: {reason}")
    original = matches.iloc[0]
    if current.fits.empty:
        raise ValueError("no signal-time peer index for frozen benchmark")
    fits = current.fits[
        (current.fits.fold_id == source.get("fold_id")) & (current.fits.quote_mark == "mid")
    ]
    if fits.empty or not np.isfinite(fits.iloc[0].peer_reference_spot):
        raise ValueError("no signal-time peer index for frozen benchmark")
    spot = float(fits.iloc[0].peer_reference_spot)
    options = {}
    for mark in ("mid", "bid", "ask"):
        fit = anchor_report.fits[
            (anchor_report.fits.fold_id == original.fold_id)
            & (anchor_report.fits.quote_mark == mark)
        ].iloc[0]
        carry = config.rate - log(fit.peer_forward / original.peer_reference_spot) / original.years
        template = Option(
            original.peer_reference_spot,
            original.strike,
            original.years,
            0.5,
            config.rate,
            carry,
            original.side,
        )
        price = (
            original.fair_value_cash if mark == "mid" else original[f"{mark}_refit_fair_value_cash"]
        )
        volatility = implied_vol(price, template)
        options[mark] = replace(
            template,
            spot=spot,
            years=source["years"]
            if "years" in source and np.isfinite(source["years"])
            else (actual["expiration_timestamp"] - milliseconds(snapshot.captured_at))
            / 1000
            / YEAR_SECONDS,
            volatility=volatility,
        )
    residual = original.peer_residual_rmse * spot / original.peer_reference_spot
    return options, residual


def evaluate_benchmarks(
    snapshot,
    config,
    *,
    relative_report=None,
    frozen_snapshot=None,
    frozen_report=None,
    instrument_names=None,
):
    """Return three auditable signal reports; frozen parameters never use later data."""
    current = (
        relative_report
        if relative_report is not None
        else evaluate_relative_value(snapshot, config, instrument_names=instrument_names)
    )
    for capture, report in ((snapshot, current), (frozen_snapshot, frozen_report)):
        if report is not None and not report.signals.empty:
            if capture is None or not (
                report.signals.signal_snapshot.eq(capture.snapshot_id).all()
                and report.signals.signal_time.eq(capture.captured_at).all()
            ):
                raise ValueError("Benchmark reports must match their exact capture cutoff")
    output = {"relative_value": classify_report(current, config, config.minimum_edge_cash)}
    for strategy in STRATEGIES[1:]:
        rows, diagnostics = [], []
        for source in current.signals.to_dict("records"):
            row = dict(source)
            # Quote exclusions are shared; calibration/domain failures remain model-specific.
            if source["status"] != "quote excluded":
                try:
                    options, residual = (
                        _flat_options(source, current, config)
                        if strategy == "flat_black_scholes"
                        else _frozen_options(
                            snapshot, source, current, frozen_snapshot, frozen_report, config
                        )
                    )
                    row = _priced_row(
                        source,
                        snapshot.instruments[source["instrument_name"]],
                        config,
                        options,
                        residual,
                    )
                except ValueError as exc:
                    row.update(
                        status="benchmark unavailable",
                        reason=str(exc),
                        signal="PASS",
                        fair_value_cash=np.nan,
                        fair_value_quote=np.nan,
                    )
            row["strategy"] = strategy
            if strategy == "frozen_smile":
                row["frozen_snapshot"] = frozen_snapshot.snapshot_id if frozen_snapshot else None
            rows.append(row)
            diagnostics.append(
                dict(
                    signal_snapshot=snapshot.snapshot_id,
                    instrument_name=row["instrument_name"],
                    strategy=strategy,
                    status=row["status"],
                    reason=row["reason"],
                )
            )
        result = RelativeValueReport(
            pd.DataFrame(rows) if rows else current.signals.copy(),
            pd.DataFrame(diagnostics),
            pd.DataFrame(),
            pd.DataFrame(),
            dict(strategy=strategy, hypothetical=True),
        )
        output[strategy] = classify_report(result, config, config.minimum_edge_cash)
    return output
