"""Independent BTC vanilla smile values and cost-buffered research signals.

Each expiry/strike fold removes both option rights before parity, IV inversion,
smile calibration and sensitivity calculations. Only one captured snapshot enters
this service. A signal is research intent, never a fill or evidence of trading alpha.
"""

from dataclasses import dataclass, field, replace
from math import log, sqrt
from typing import Literal

import numpy as np
import pandas as pd

from deribit_dashboard.models import Snapshot, milliseconds
from deribit_dashboard.pricing import BlackScholes, Option, risk
from deribit_dashboard.surface import (
    PARITY_ESTIMATORS,
    Contract,
    Surface,
    chain_audit,
    fit_surface,
    independently_invert,
    parity_forwards,
)


@dataclass(frozen=True)
class RelativeValueConfig:
    """Prices/costs/edges are cash per underlying BTC, never per contract.

    Estimated round-trip costs cover option and hedge fees, expected exit spread,
    slippage and funding. Entry spread is already included by comparing with the
    captured ask/bid. The bps component applies to peer reference index notional.
    These estimates must be supplied for the intended study; zero is a frictionless
    diagnostic assumption. Actual execution accounting is a separate service.
    """

    settlement: str = "BTC"
    rate: float = 0.0
    stale_seconds: float = 60.0
    max_relative_width: float = 1.0
    parity_estimator: str = "weighted_median"
    estimated_round_trip_cost_cash: float = 0.0
    estimated_round_trip_cost_bps: float = 0.0
    minimum_edge_cash: float = 0.0
    uncertainty_multiplier: float = 1.0
    uncertainty_floor_cash: float = 0.0

    def __post_init__(self):
        if not isinstance(self.settlement, str) or not self.settlement.strip():
            raise ValueError("A settlement currency is required")
        if self.parity_estimator not in PARITY_ESTIMATORS:
            raise ValueError("Unknown parity estimator")
        nonnegative = (
            self.stale_seconds,
            self.estimated_round_trip_cost_cash,
            self.estimated_round_trip_cost_bps,
            self.minimum_edge_cash,
            self.uncertainty_multiplier,
            self.uncertainty_floor_cash,
        )
        if (
            not np.isfinite([self.rate, self.max_relative_width, *nonnegative]).all()
            or min(nonnegative) < 0
            or self.max_relative_width <= 0
        ):
            raise ValueError("Use finite controls, nonnegative costs/buffers and positive width")


@dataclass(frozen=True)
class SignalDecision:
    signal: Literal["BUY", "SELL", "PASS"]
    gross_buy_edge_cash: float
    gross_sell_edge_cash: float
    buy_net_edge_cash: float
    sell_net_edge_cash: float


def decide_signal(
    fair_value,
    bid,
    ask,
    *,
    estimated_cost_cash=0.0,
    uncertainty_cash=0.0,
    minimum_edge_cash=0.0,
):
    """Require strictly positive edge beyond all costs, buffers and minimum edge.

    Cash values share one denomination and are per underlying BTC. Depth and
    quote eligibility are checked by ``evaluate_relative_value``. Equality passes.
    """
    values = [fair_value, bid, ask, estimated_cost_cash, uncertainty_cash, minimum_edge_cash]
    if not np.isfinite(values).all() or min(values) < 0 or bid > ask:
        raise ValueError("Signal inputs require finite nonnegative values and uncrossed quotes")
    buy, sell = fair_value - ask, bid - fair_value
    deduction = estimated_cost_cash + uncertainty_cash
    net_buy, net_sell = buy - deduction, sell - deduction
    signal = (
        "BUY" if net_buy > minimum_edge_cash else "SELL" if net_sell > minimum_edge_cash else "PASS"
    )
    return SignalDecision(signal, buy, sell, net_buy, net_sell)


@dataclass
class RelativeValueReport:
    signals: pd.DataFrame
    fits: pd.DataFrame
    calibration: pd.DataFrame
    parity_pairs: pd.DataFrame
    diagnostics: dict


@dataclass
class _SmileFold:
    surface: Surface | None = None
    forward: float = np.nan
    spot: float = np.nan
    residual_rmse: float = np.nan
    error: str | None = None
    audit: dict = field(default_factory=dict)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)


def _cash_currency(metadata):
    quote, settlement = metadata.get("quote_currency"), metadata.get("settlement_currency")
    if quote == settlement == "BTC":
        counter = metadata.get("counter_currency")
        return counter if counter and counter != "BTC" else None
    return quote if quote == settlement and quote != "BTC" else None


def _audit(snapshot, config):
    chain, _ = chain_audit(
        snapshot, "BTC", config.settlement, config.stale_seconds, config.max_relative_width
    )
    cutoff = milliseconds(snapshot.captured_at)
    for index, row in chain.iterrows():
        ticker = snapshot.tickers.get(row.instrument_name, {})
        reasons = list(row.reasons)
        if ticker.get("timestamp", 0) > cutoff:
            reasons.append("after_signal_cutoff")
        for side in ("bid", "ask"):
            amount = ticker.get(f"best_{side}_amount")
            valid = isinstance(amount, (int, float)) and np.isfinite(amount) and amount >= 0
            chain.loc[index, f"{side}_depth_underlying"] = float(amount) if valid else np.nan
            if not valid:
                reasons.append(f"invalid_{side}_depth")
        currency = _cash_currency(snapshot.instruments[row.instrument_name])
        if currency is None:
            reasons.append("ambiguous_cash_currency")
        chain.loc[index, "cash_currency"] = currency
        chain.at[index, "reasons"] = tuple(reasons)
        chain.loc[index, "accepted"] = not reasons
    return chain


def _training_snapshot(snapshot, strike):
    """Physically remove held-out prices and all after-cutoff tickers from the fit input."""
    instruments = {
        name: metadata
        for name, metadata in snapshot.instruments.items()
        if not (
            metadata.get("kind") == "option"
            and metadata.get("base_currency") == "BTC"
            and metadata.get("strike") == strike
        )
    }
    cutoff = milliseconds(snapshot.captured_at)
    tickers = {
        name: ticker
        for name, ticker in snapshot.tickers.items()
        if name in instruments and ticker["timestamp"] <= cutoff
    }
    # The universe is guaranteed nonempty when a fold has eligible peers.
    return replace(
        snapshot,
        instruments=instruments,
        tickers=tickers,
        received_at={name: snapshot.received_at[name] for name in tickers},
        errors={name: error for name, error in snapshot.errors.items() if name in instruments},
    )


def _fit_fold(snapshot, peers, config, quote_mark):
    result = _SmileFold(calibration=peers.copy())
    # A calibration midpoint needs displayed liquidity on both sides. Candidate
    # eligibility remains directional: a BUY needs ask depth and a SELL bid depth.
    for index, row in result.calibration.iterrows():
        if row.accepted and min(row.bid_depth_underlying, row.ask_depth_underlying) <= 0:
            result.calibration.loc[index, "accepted"] = False
            result.calibration.at[index, "reasons"] += ("no_two_sided_peer_depth",)
    result.audit = dict(
        peer_quotes=len(peers),
        hygiene_exclusions=int((~result.calibration.accepted).sum()),
        included_quotes=0,
        invertible_strikes=0,
        parity_pair_count=0,
    )
    try:
        if not result.calibration.accepted.any():
            raise ValueError("No eligible peer quotes remain after strike exclusion")
        result.spot = float(result.calibration.loc[result.calibration.accepted, "spot"].median())
        marked = result.calibration.assign(calibration_price=result.calibration[quote_mark])
        forwards, result.pairs = parity_forwards(
            snapshot,
            marked,
            config.rate,
            config.stale_seconds,
            estimator=config.parity_estimator,
            return_pairs=True,
        )
        forward_row = forwards.iloc[0]
        result.forward = float(forward_row.forward)
        result.audit["parity_pair_count"] = int(forward_row.pairs)
        result.calibration = independently_invert(marked, forwards, config.rate)
        usable = result.calibration[result.calibration.iv_status == "included"]
        result.audit.update(
            included_quotes=len(usable),
            invertible_strikes=int(usable.strike.nunique()),
            iv_exclusions=int((marked.accepted & ~marked.index.isin(usable.index)).sum()),
        )
        if not np.isfinite(result.forward) or result.forward <= 0:
            raise ValueError("No valid parity forward after strike exclusion")
        if not forward_row.estimator_converged:
            raise ValueError("Peer parity estimator did not converge")
        result.surface = fit_surface(result.calibration, forwards, config.rate)
        fitted = []
        result.calibration["peer_fitted_price"] = np.nan
        for index, row in usable.iterrows():
            value = result.surface.price(row.strike, row.years, row.side)
            result.calibration.loc[index, "peer_fitted_price"] = value
            fitted.append(value - row.calibration_price)
        result.residual_rmse = sqrt(float(np.mean(np.square(fitted))))
        result.audit.update(result.surface.report)
    except ValueError as exc:
        result.error = str(exc)
    return result


def evaluate_relative_value(snapshot, config=RelativeValueConfig(), *, instrument_names=None):
    """Value selected BTC vanilla options using only peers at this signal cutoff.

    Fits are per expiry, avoiding cross-expiry domain intersections. Both rights at
    the candidate strike are excluded; other expiries never enter its fit. Calls
    and puts at the same strike share three fits (bid/mid/ask), cached within this
    invocation. No extrapolation, exchange IV/Greeks or future captures are used.

    All selected candidates and every attempted fit survive in the returned audit.
    Missing sensitivities force PASS even if the midpoint valuation succeeded.
    """
    # Validate receive timestamps and detach the caller's mutable snapshot dictionaries.
    snapshot = Snapshot.from_dict(snapshot.to_dict())
    candidates = {
        name
        for name, metadata in snapshot.instruments.items()
        if metadata.get("kind") == "option"
        and metadata.get("base_currency") == "BTC"
        and metadata.get("settlement_currency") == config.settlement
    }
    selected = candidates if instrument_names is None else set(instrument_names)
    if selected - candidates:
        raise ValueError(
            "Selected instruments must be captured BTC vanilla options in this settlement"
        )
    if not selected:
        raise ValueError("Select at least one captured BTC vanilla option")
    chain = _audit(snapshot, config)
    cache, rows, fits, calibration, pairs = {}, [], [], [], []
    for entry in (
        chain[chain.instrument_name.isin(selected)].sort_values("instrument_name").itertuples()
    ):
        name = entry.instrument_name
        metadata, ticker = snapshot.instruments[name], snapshot.tickers.get(name, {})
        row = dict(
            signal_snapshot=snapshot.snapshot_id,
            signal_time=snapshot.captured_at,
            quote_timestamp=ticker.get("timestamp"),
            quote_age_seconds=entry.quote_age_seconds,
            instrument_name=name,
            expiry=entry.expiry,
            strike=entry.strike,
            side=entry.side,
            settlement_currency=config.settlement,
            cash_currency=entry.cash_currency,
            market_bid_quote=ticker.get("best_bid_price"),
            market_ask_quote=ticker.get("best_ask_price"),
            signal="PASS",
            status="quote excluded",
            reason=", ".join(entry.reasons),
            fair_value_cash=np.nan,
            fair_value_quote=np.nan,
            uncertainty_cash=np.nan,
            uncertainty_buffer_cash=np.nan,
        )
        rows.append(row)
        if not entry.accepted:
            continue
        contract = Contract.from_metadata(metadata)
        row.update(
            multiplier=contract.multiplier,
            bid_depth_contracts=entry.bid_depth_underlying / contract.multiplier,
            ask_depth_contracts=entry.ask_depth_underlying / contract.multiplier,
        )
        key = (entry.expiry, entry.strike, entry.cash_currency)
        fold_id = f"{snapshot.snapshot_id}:{entry.expiry}:{entry.strike:g}:{entry.cash_currency}"
        row["fold_id"] = fold_id
        if key not in cache:
            peers = (
                chain[
                    (chain.expiry == entry.expiry)
                    & (chain.strike != entry.strike)
                    & (chain.cash_currency == entry.cash_currency)
                ]
                .sort_values("instrument_name")
                .copy()
            )
            excluded_names = tuple(
                sorted(
                    name
                    for name, m in snapshot.instruments.items()
                    if m.get("kind") == "option"
                    and m.get("base_currency") == "BTC"
                    and m.get("strike") == entry.strike
                )
            )
            training = (
                _training_snapshot(snapshot, entry.strike) if peers.accepted.any() else snapshot
            )
            scenarios = {
                mark: _fit_fold(training, peers, config, mark) for mark in ("mid", "bid", "ask")
            }
            cache[key] = scenarios
            for mark, fitted in scenarios.items():
                fits.append(
                    dict(
                        fold_id=fold_id,
                        expiry=entry.expiry,
                        excluded_strike=entry.strike,
                        excluded_instruments=excluded_names,
                        cash_currency=entry.cash_currency,
                        quote_mark=mark,
                        status="failed" if fitted.error else "fitted",
                        reason=fitted.error or "",
                        peer_forward=fitted.forward,
                        peer_reference_spot=fitted.spot,
                        peer_residual_rmse=fitted.residual_rmse,
                        **fitted.audit,
                    )
                )
                calibration.append(fitted.calibration.assign(fold_id=fold_id, quote_mark=mark))
                if not fitted.pairs.empty:
                    pairs.append(fitted.pairs.assign(fold_id=fold_id, quote_mark=mark))
        scenarios = cache[key]
        mid = scenarios["mid"]
        if mid.error:
            row.update(status="valuation unavailable", reason=mid.error)
            continue
        try:
            fair = mid.surface.price(entry.strike, entry.years, entry.side)
            volatility = mid.surface.volatility(entry.strike, entry.years)
            carry = config.rate - log(mid.forward / mid.spot) / entry.years
            option = Option(
                mid.spot, entry.strike, entry.years, volatility, config.rate, carry, entry.side
            )
            greeks = risk(BlackScholes(), option, contract.multiplier)
            factor = contract.cash_factor(mid.spot)
            row.update(
                fair_value_cash=fair,
                fair_value_quote=fair / factor,
                peer_reference_spot=mid.spot,
                peer_forward=mid.forward,
                years=entry.years,
                volatility=volatility,
                carry_yield=carry,
                rate=config.rate,
                cash_delta_per_contract=greeks["cash_delta"],
                coin_delta_per_contract=greeks["coin_delta"],
                market_bid_cash=float(ticker["best_bid_price"]) * factor,
                market_ask_cash=float(ticker["best_ask_price"]) * factor,
                peer_residual_rmse=mid.residual_rmse,
            )
        except ValueError as exc:
            row.update(status="valuation unavailable", reason=str(exc))
            continue
        try:
            scenario_values = {}
            for mark in ("bid", "ask"):
                scenario = scenarios[mark]
                if scenario.error:
                    raise ValueError(f"{mark} peer refit: {scenario.error}")
                scenario_values[mark] = scenario.surface.price(
                    entry.strike, entry.years, entry.side
                )
            sensitivity = max(abs(value - fair) for value in scenario_values.values())
            uncertainty = max(config.uncertainty_floor_cash, mid.residual_rmse, sensitivity)
            buffer = config.uncertainty_multiplier * uncertainty
            cost = (
                config.estimated_round_trip_cost_cash
                + mid.spot * config.estimated_round_trip_cost_bps / 10000
            )
            decision = decide_signal(
                fair,
                row["market_bid_cash"],
                row["market_ask_cash"],
                estimated_cost_cash=cost,
                uncertainty_cash=buffer,
                minimum_edge_cash=config.minimum_edge_cash,
            )
            row.update(
                status="valued",
                signal=decision.signal,
                reason="edge below required threshold" if decision.signal == "PASS" else "",
                bid_refit_fair_value_cash=scenario_values["bid"],
                ask_refit_fair_value_cash=scenario_values["ask"],
                peer_quote_sensitivity_cash=sensitivity,
                uncertainty_cash=uncertainty,
                uncertainty_buffer_cash=buffer,
                estimated_round_trip_cost_cash=cost,
                minimum_edge_cash=config.minimum_edge_cash,
                gross_buy_edge_cash=decision.gross_buy_edge_cash,
                gross_sell_edge_cash=decision.gross_sell_edge_cash,
                buy_net_edge_cash=decision.buy_net_edge_cash,
                sell_net_edge_cash=decision.sell_net_edge_cash,
            )
            if decision.signal != "PASS":
                depth = row[
                    "ask_depth_contracts" if decision.signal == "BUY" else "bid_depth_contracts"
                ]
                if depth <= 0:
                    row.update(
                        signal="PASS", reason=f"no displayed entry depth for {decision.signal}"
                    )
        except ValueError as exc:
            row.update(status="uncertainty unavailable", reason=str(exc))
    signals = pd.DataFrame(rows)
    return RelativeValueReport(
        signals=signals,
        fits=pd.DataFrame(fits),
        calibration=pd.concat(calibration, ignore_index=True) if calibration else pd.DataFrame(),
        parity_pairs=pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(),
        diagnostics=dict(
            environment=snapshot.environment,
            signal_snapshot=snapshot.snapshot_id,
            signal_time=snapshot.captured_at.isoformat(),
            selected_observations=len(signals),
            valued_observations=int(signals.status.eq("valued").sum()),
            signals=signals.signal.value_counts().to_dict(),
            outcomes=signals.status.value_counts().to_dict(),
            unique_exclusion_folds=len(cache),
            exclusion="Both call and put prices at the candidate strike are removed from all expiries before fitting; each smile uses only its own expiry and cash denomination.",
            information="Captured and received by the signal cutoff; strictly no after-cutoff exchange timestamps, future snapshots, exchange IV or exchange Greeks.",
            units="Prices, edges and costs: cash per underlying BTC. Coin-quoted markets are normalized using the excluded fold's peer median index. Greeks: per metadata contract, holding model volatility and carry fixed.",
            uncertainty="max(peer midpoint-fit residual RMSE, absolute bid/ask peer-refit price deviations, configured floor), multiplied by the configured buffer. A model-risk heuristic, not a statistical confidence interval.",
            costs="Configured round-trip cash plus bps of peer index notional, covering fees, hedge friction, expected exit spread, slippage and funding. Entry spread is already in bid/ask. Zero costs mean a frictionless diagnostic.",
            execution="BUY/SELL are hypothetical research intent. This phase does not simulate fills, size positions, hedge, or measure out-of-sample P&L.",
        ),
    )
