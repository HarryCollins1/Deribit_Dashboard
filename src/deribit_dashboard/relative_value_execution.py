"""Captured-book, native-currency accounting for hypothetical BTC relative-value trades.

No orders are sent. Entry intent is frozen before exit selection. Missing liquidity
leaves outstanding positions with explicit conservative marks and cost reserves.
"""

from dataclasses import dataclass, field
from math import copysign, exp, floor

import numpy as np
import pandas as pd

from deribit_dashboard.barrier_hedging import PerpetualHedge
from deribit_dashboard.execution_accounting import (
    fee_from_notional,
    funding_amount,
    perpetual_gain,
    perpetual_notional,
)
from deribit_dashboard.models import milliseconds
from deribit_dashboard.pricing import YEAR_SECONDS
from deribit_dashboard.relative_value import RelativeValueConfig
from deribit_dashboard.surface import Contract

IDENTITY_FIELDS = (
    "instrument_name",
    "kind",
    "base_currency",
    "quote_currency",
    "counter_currency",
    "settlement_currency",
    "contract_size",
    "strike",
    "expiration_timestamp",
    "option_type",
    "settlement_period",
)


@dataclass(frozen=True)
class TradingEvaluationConfig:
    signal_config: RelativeValueConfig = field(default_factory=RelativeValueConfig)
    perpetual_instrument: str = "BTC-PERPETUAL"
    horizon_seconds: float = 3600.0
    tolerance_seconds: float = 30.0
    contracts: float = 1.0
    max_depth_fraction: float = 0.25
    max_leg_timestamp_gap_seconds: float = 5.0
    option_fee_bps: float = 3.0
    option_fee_premium_cap: float = 0.125
    perpetual_fee_bps: float = 5.0
    funding_max_age_seconds: float = 300.0
    missing_funding_rate_8h_bps: float = 10.0
    residual_hedge_adverse_bps: float = 100.0

    def __post_init__(self):
        # Supports the existing dataclass-to-JSON-to-dataclass reproduction convention.
        if isinstance(self.signal_config, dict):
            object.__setattr__(self, "signal_config", RelativeValueConfig(**self.signal_config))
        if not isinstance(self.signal_config, RelativeValueConfig):
            raise ValueError("A RelativeValueConfig is required")
        if not isinstance(self.perpetual_instrument, str) or not self.perpetual_instrument:
            raise ValueError("Select a captured BTC perpetual")
        positive = (self.horizon_seconds, self.contracts, self.max_depth_fraction)
        nonnegative = (
            self.tolerance_seconds,
            self.max_leg_timestamp_gap_seconds,
            self.option_fee_bps,
            self.option_fee_premium_cap,
            self.perpetual_fee_bps,
            self.funding_max_age_seconds,
            self.missing_funding_rate_8h_bps,
            self.residual_hedge_adverse_bps,
        )
        if (
            not np.isfinite([*positive, *nonnegative]).all()
            or min(positive) <= 0
            or min(nonnegative) < 0
            or self.max_depth_fraction > 1
            or self.option_fee_premium_cap > 1
            or self.residual_hedge_adverse_bps >= 10000
        ):
            raise ValueError("Invalid horizon, quantity, depth, fee, funding or marking controls")


@dataclass(frozen=True)
class _Book:
    bid: float
    ask: float
    bid_depth: float
    ask_depth: float
    index: float
    timestamp: float

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    def price(self, signed_quantity):
        return self.ask if signed_quantity > 0 else self.bid


class DepthBudget:
    """Consume each instrument/event/side's captured depth once across all trades.

    Repeated snapshots of the same event cannot replenish liquidity. A new event
    timestamp provides a new hypothetical capacity, not proof of actual availability.
    """

    def __init__(self, fraction):
        self.fraction = fraction
        self.limits = {}
        self.used = {}

    def _key(self, name, book, signed_quantity, multiplier):
        return (
            name,
            book.timestamp,
            "ask" if signed_quantity > 0 else "bid",
            book.price(signed_quantity),
            multiplier,
        )

    def remaining(self, name, book, signed_quantity, multiplier):
        key = self._key(name, book, signed_quantity, multiplier)
        depth = book.ask_depth if signed_quantity > 0 else book.bid_depth
        capacity = depth / multiplier * self.fraction
        self.limits[key] = min(self.limits.get(key, capacity), capacity)
        return max(0.0, self.limits[key] - self.used.get(key, 0.0))

    def consume(self, name, book, signed_quantity, multiplier):
        key = self._key(name, book, signed_quantity, multiplier)
        available = self.remaining(name, book, signed_quantity, multiplier)
        if abs(signed_quantity) > available + 1e-10:
            raise ValueError("Hypothetical execution exceeds remaining captured depth")
        self.used[key] = self.used.get(key, 0.0) + abs(signed_quantity)


def _identity_matches(expected, actual):
    return actual is not None and all(
        expected.get(key) == actual.get(key) for key in IDENTITY_FIELDS
    )


def _book(snapshot, name, config, *, expected=None, target_ms=None):
    metadata, ticker = snapshot.instruments.get(name), snapshot.tickers.get(name)
    if metadata is None or ticker is None:
        raise ValueError("missing captured instrument or quote")
    if expected is not None and not _identity_matches(expected, metadata):
        raise ValueError("contract identity or units changed")
    cutoff, stamp = milliseconds(snapshot.captured_at), ticker["timestamp"]
    if stamp > cutoff or (target_ms is not None and stamp < target_ms):
        raise ValueError("quote timestamp outside execution cutoff/horizon")
    if (cutoff - stamp) / 1000 > config.signal_config.stale_seconds:
        raise ValueError("stale captured quote")
    if ticker.get("state", "open") != "open":
        raise ValueError("closed captured market")
    values = [
        ticker.get(key)
        for key in (
            "best_bid_price",
            "best_ask_price",
            "best_bid_amount",
            "best_ask_amount",
            "index_price",
        )
    ]
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in values):
        raise ValueError("missing or nonfinite price, depth or index")
    bid, ask, bid_depth, ask_depth, index = map(float, values)
    if min(bid, ask, index) <= 0 or bid > ask or min(bid_depth, ask_depth) < 0:
        raise ValueError("invalid bid/ask, displayed depth or index")
    if (ask - bid) / ((ask + bid) / 2) > config.signal_config.max_relative_width:
        raise ValueError("captured spread exceeds configured limit")
    _quantity_rules(metadata)
    if metadata.get("kind") == "option":
        try:
            Contract.from_metadata(metadata)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid captured option tick/contract metadata") from exc
        if metadata["expiration_timestamp"] <= cutoff:
            raise ValueError("expiry settlement unavailable")
    elif (
        not isinstance(metadata.get("tick_size"), (int, float))
        or not np.isfinite(metadata["tick_size"])
        or metadata["tick_size"] <= 0
    ):
        raise ValueError("invalid captured perpetual tick metadata")
    return _Book(bid, ask, bid_depth, ask_depth, index, stamp)


def _quantity_rules(metadata):
    size = metadata.get("contract_size")
    if not isinstance(size, (int, float)) or not np.isfinite(size) or size <= 0:
        raise ValueError("invalid captured contract multiplier")
    size = float(size)
    minimum = metadata.get("min_trade_amount")
    step = metadata.get("amount_step")
    for value in (minimum, step):
        if value is not None and (
            not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
        ):
            raise ValueError("invalid captured quantity minimum/step")
    if step is not None:
        return float(step) / size, float(minimum or step) / size, "captured amount_step"
    if minimum is not None:
        return float(minimum) / size, float(minimum) / size, "min_trade_amount assumed as increment"
    return None, 0.0, "fractional research contracts; no captured minimum/step"


def _round_quantity(quantity, metadata):
    step, minimum, _ = _quantity_rules(metadata)
    magnitude = abs(quantity)
    if step is not None:
        magnitude = floor(magnitude / step + 1e-12) * step
    if magnitude + 1e-12 < minimum:
        magnitude = 0.0
    return copysign(magnitude, quantity)


def _option_fee(units, price, index, inverse, config):
    underlying = 1.0 if inverse else index
    return min(
        float(fee_from_notional(units * underlying, config.option_fee_bps)),
        abs(units) * price * config.option_fee_premium_cap,
    )


def _hedge_fee(quantity, price, hedge, config):
    notional = perpetual_notional(quantity, hedge.contract_size, hedge.inverse, price)
    # Inverse face amounts are converted at the execution price for native BTC fees.
    return float(
        fee_from_notional(notional / price if hedge.inverse else notional, config.perpetual_fee_bps)
    )


def _hedge_spread(quantity, price, midpoint, hedge):
    return abs(float(perpetual_gain(quantity, hedge.contract_size, hedge.inverse, midpoint, price)))


@dataclass
class HypotheticalTrade:
    row: dict
    option_metadata: dict
    hedge_metadata: dict
    option_book: _Book
    hedge_book: _Book
    hedge: PerpetualHedge
    option_quantity: float
    hedge_quantity: float
    ledger: list = field(default_factory=list)


def _event(
    trade,
    event,
    time,
    *,
    cash=0.0,
    mark=0.0,
    fee=0.0,
    spread=0.0,
    quantity=0.0,
    position=0.0,
    turnover=0.0,
    quote_timestamp=None,
    **details,
):
    trade.ledger.append(
        dict(
            trade_id=trade.row["trade_id"],
            event=event,
            time=time,
            execution="hypothetical",
            native_currency=trade.row["native_currency"],
            cash_flow_native=float(cash),
            mark_value_native=float(mark),
            fee_native=float(fee),
            spread_cost_native=float(spread),
            signed_contracts=float(quantity),
            marked_position_contracts=float(position),
            turnover_native=float(turnover),
            quote_timestamp=quote_timestamp,
            **details,
        )
    )


def prepare_trade(snapshot, signal, config, budget):
    """Preflight both legs, freeze sizes, then consume entry depth and book cash flows.

    This function receives no exit snapshot, later price or later liquidity.
    """
    if signal.get("signal") not in {"BUY", "SELL"} or signal.get("status") != "valued":
        raise ValueError("Entry requires a valued BUY or SELL research signal")
    name, hedge_name = signal["instrument_name"], config.perpetual_instrument
    option_metadata = snapshot.instruments[name]
    hedge_metadata = snapshot.instruments.get(hedge_name)
    if hedge_metadata is None:
        raise ValueError("selected BTC perpetual metadata unavailable")
    target_ms = milliseconds(snapshot.captured_at) + config.horizon_seconds * 1000
    if target_ms >= option_metadata["expiration_timestamp"]:
        raise ValueError("configured horizon reaches option expiry; no fabricated settlement")
    option_book = _book(snapshot, name, config)
    hedge_book = _book(snapshot, hedge_name, config)
    if (
        abs(option_book.timestamp - hedge_book.timestamp) / 1000
        > config.max_leg_timestamp_gap_seconds
    ):
        raise ValueError("entry option/perpetual timestamps exceed leg gap limit")
    hedge = PerpetualHedge.from_metadata(
        hedge_metadata, hedge_book.mid, signal["peer_reference_spot"], signal["cash_currency"]
    )
    if hedge_metadata.get("settlement_currency") != option_metadata.get("settlement_currency"):
        raise ValueError("option and hedge require the same native settlement; no implicit FX")
    contract = Contract.from_metadata(option_metadata)
    sign = 1.0 if signal["signal"] == "BUY" else -1.0
    delta = (
        signal["coin_delta_per_contract"] if hedge.inverse else signal["cash_delta_per_contract"]
    )
    coefficient = (
        -delta * hedge_book.mid * signal["peer_reference_spot"] / hedge.contract_size
        if hedge.inverse
        else -delta / (hedge.contract_size * hedge.initial_basis_ratio)
    )
    option_capacity = budget.remaining(name, option_book, sign, contract.multiplier)
    comparison_cap = signal.get("comparison_contract_cap", config.contracts)
    if (
        not isinstance(comparison_cap, (int, float))
        or not np.isfinite(comparison_cap)
        or comparison_cap <= 0
    ):
        raise ValueError("Comparison quantity cap must be finite and positive")
    magnitude = min(config.contracts, comparison_cap, option_capacity)
    if abs(coefficient) > 1e-14:
        hedge_capacity = budget.remaining(
            hedge_name, hedge_book, sign * coefficient, hedge.contract_size
        )
        magnitude = min(magnitude, hedge_capacity / abs(coefficient))
    quantity = _round_quantity(sign * magnitude, option_metadata)
    hedge_quantity = _round_quantity(quantity * coefficient, hedge_metadata)
    if abs(quantity) <= 1e-14:
        raise ValueError("entry depth/quantity minimum cannot support both legs")
    if abs(quantity * coefficient) > 1e-14 and abs(hedge_quantity) <= 1e-14:
        raise ValueError("captured hedge quantity minimum prevents an initial delta hedge")
    row = dict(signal)
    option_price, hedge_price = option_book.price(quantity), hedge_book.price(hedge_quantity)
    row.update(
        trade_id=f"{snapshot.snapshot_id}:{name}",
        entry_time=snapshot.captured_at,
        target_time=pd.to_datetime(target_ms, unit="ms", utc=True),
        target_ms=target_ms,
        requested_contracts=config.contracts,
        hypothetical_option_contracts=quantity,
        hypothetical_hedge_contracts=hedge_quantity,
        intended_hedge_contracts=quantity * coefficient,
        option_entry_price_quote=option_price,
        hedge_entry_price=hedge_price,
        option_entry_quote_timestamp=option_book.timestamp,
        hedge_entry_quote_timestamp=hedge_book.timestamp,
        hedge_instrument=hedge_name,
        hedge_inverse=hedge.inverse,
        hedge_contract_size=hedge.contract_size,
        native_currency=option_metadata["settlement_currency"],
        execution="hypothetical",
        option_quantity_rule=_quantity_rules(option_metadata)[2],
        hedge_quantity_rule=_quantity_rules(hedge_metadata)[2],
        hedge_delta_numeraire="BTC" if hedge.inverse else signal["cash_currency"],
        residual_initial_delta=(
            quantity * delta
            + hedge_quantity
            * hedge.contract_size
            / (hedge_book.mid * signal["peer_reference_spot"])
            if hedge.inverse
            else quantity * delta + hedge_quantity * hedge.contract_size * hedge.initial_basis_ratio
        ),
    )
    trade = HypotheticalTrade(
        row,
        option_metadata,
        hedge_metadata,
        option_book,
        hedge_book,
        hedge,
        quantity,
        hedge_quantity,
    )
    budget.consume(name, option_book, quantity, contract.multiplier)
    budget.consume(hedge_name, hedge_book, hedge_quantity, hedge.contract_size)
    units = quantity * contract.multiplier
    fee = _option_fee(units, option_price, option_book.index, contract.inverse, config)
    _event(
        trade,
        "option_entry",
        snapshot.captured_at,
        cash=-units * option_price - fee,
        fee=fee,
        spread=abs(units) * abs(option_price - option_book.mid),
        quantity=quantity,
        turnover=abs(units * option_price),
        quote_timestamp=option_book.timestamp,
        reference_index=option_book.index,
        price=option_price,
    )
    fee = _hedge_fee(hedge_quantity, hedge_price, hedge, config)
    _event(
        trade,
        "hedge_entry",
        snapshot.captured_at,
        cash=-fee,
        fee=fee,
        spread=_hedge_spread(hedge_quantity, hedge_price, hedge_book.mid, hedge),
        quantity=hedge_quantity,
        turnover=abs(
            float(
                perpetual_notional(hedge_quantity, hedge.contract_size, hedge.inverse, hedge_price)
            )
        )
        / (hedge_price if hedge.inverse else 1.0),
        quote_timestamp=hedge_book.timestamp,
        reference_index=hedge_book.index,
        price=hedge_price,
    )
    return trade


def _funding(trade, snapshots, end, config):
    """Left-held captured current rates; expired/missing rates get an adverse reserve."""
    start_ms, end_ms = milliseconds(trade.row["entry_time"]), milliseconds(end)
    frames = [s for s in snapshots if start_ms <= milliseconds(s.captured_at) < end_ms]
    paid = reserved = gaps = reconstructed = 0.0
    last_index = trade.hedge_book.index
    for i, snapshot in enumerate(frames):
        left = milliseconds(snapshot.captured_at)
        right = milliseconds(frames[i + 1].captured_at) if i + 1 < len(frames) else end_ms
        ticker = snapshot.tickers.get(config.perpetual_instrument, {})
        stamp = ticker.get("timestamp", -np.inf)
        rate, index = ticker.get("current_funding"), ticker.get("index_price")
        identity = _identity_matches(
            trade.hedge_metadata, snapshot.instruments.get(config.perpetual_instrument)
        )
        valid_index = (
            identity
            and stamp <= left
            and ticker.get("state", "open") == "open"
            and isinstance(index, (int, float))
            and np.isfinite(index)
            and index > 0
        )
        if valid_index:
            last_index = float(index)
        valid_rate = valid_index and isinstance(rate, (int, float)) and np.isfinite(rate)
        fresh_until = (
            min(right, stamp + config.funding_max_age_seconds * 1000) if valid_rate else left
        )
        observed_seconds = max(0.0, (fresh_until - left) / 1000)
        gap_seconds = (right - left) / 1000 - observed_seconds
        notional = float(
            perpetual_notional(
                trade.hedge_quantity, trade.hedge.contract_size, trade.hedge.inverse, last_index
            )
        )
        native_notional = notional / last_index if trade.hedge.inverse else notional
        if observed_seconds:
            amount = float(funding_amount(native_notional, rate, observed_seconds / 28800))
            paid += amount
            reconstructed += observed_seconds
            _event(
                trade,
                "captured_rate_funding",
                pd.to_datetime(fresh_until, unit="ms", utc=True),
                cash=-amount,
                funding_paid_native=amount,
                interval_seconds=observed_seconds,
                period_rate=rate,
                quote_timestamp=stamp,
                reference_index=last_index,
            )
        if gap_seconds:
            amount = float(
                funding_amount(
                    abs(native_notional),
                    config.missing_funding_rate_8h_bps / 10000,
                    gap_seconds / 28800,
                )
            )
            reserved += amount
            gaps += gap_seconds
            _event(
                trade,
                "funding_gap_reserve",
                pd.to_datetime(right, unit="ms", utc=True),
                mark=-amount,
                reserve_native=amount,
                interval_seconds=gap_seconds,
                reference_index=last_index,
            )
    return paid, reserved, gaps, reconstructed


def finish_trade(trade, snapshots, exit_snapshot, config, budget):
    """Close available quantities once; mark remaining exposure without fabricating fills."""
    row = trade.row
    end = (
        exit_snapshot.captured_at
        if exit_snapshot is not None
        else row["target_time"].to_pydatetime()
    )
    errors, books = [], {}
    for leg, name, expected in (
        ("option", row["instrument_name"], trade.option_metadata),
        ("hedge", config.perpetual_instrument, trade.hedge_metadata),
    ):
        try:
            if exit_snapshot is None:
                raise ValueError("missing horizon capture")
            books[leg] = _book(
                exit_snapshot, name, config, expected=expected, target_ms=row["target_ms"]
            )
        except ValueError as exc:
            errors.append(f"{leg}: {exc}")
    if (
        len(books) == 2
        and abs(books["option"].timestamp - books["hedge"].timestamp) / 1000
        > config.max_leg_timestamp_gap_seconds
    ):
        errors.append("exit option/perpetual timestamps exceed leg gap limit")
        books = {}
    option_closed = hedge_closed = option_proceeds = hedge_realized = 0.0
    for leg, quantity, metadata, multiplier in (
        ("option", trade.option_quantity, trade.option_metadata, row["multiplier"]),
        ("hedge", trade.hedge_quantity, trade.hedge_metadata, trade.hedge.contract_size),
    ):
        if leg not in books or abs(quantity) <= 1e-14:
            continue
        book = books[leg]
        name = row["instrument_name"] if leg == "option" else config.perpetual_instrument
        available = budget.remaining(name, book, -quantity, multiplier)
        # Current captured trading rules can affect the close but cannot resize entry.
        close = _round_quantity(
            copysign(min(abs(quantity), available), quantity), exit_snapshot.instruments[name]
        )
        if abs(close) + 1e-12 < abs(quantity):
            errors.append(f"{leg}: insufficient exit depth/quantity increment")
        if abs(close) <= 1e-14:
            continue
        price = book.price(-close)
        budget.consume(name, book, -close, multiplier)
        if leg == "option":
            option_closed = close
            option_proceeds = close * multiplier * price
            fee = _option_fee(
                close * multiplier,
                price,
                book.index,
                Contract.from_metadata(metadata).inverse,
                config,
            )
            spread = abs(close * multiplier) * abs(price - book.mid)
            cash = option_proceeds - fee
        else:
            hedge_closed = close
            hedge_realized = float(
                perpetual_gain(
                    close, multiplier, trade.hedge.inverse, row["hedge_entry_price"], price
                )
            )
            fee = _hedge_fee(close, price, trade.hedge, config)
            spread = _hedge_spread(close, price, book.mid, trade.hedge)
            cash = hedge_realized - fee
        row[f"{leg}_exit_price"] = price
        row[f"{leg}_exit_quote_timestamp"] = book.timestamp
        _event(
            trade,
            f"{leg}_close",
            end,
            cash=cash,
            fee=fee,
            spread=spread,
            quantity=-close,
            turnover=abs(close * multiplier * price)
            if leg == "option"
            else abs(float(perpetual_notional(close, multiplier, trade.hedge.inverse, price)))
            / (price if trade.hedge.inverse else 1.0),
            quote_timestamp=book.timestamp,
            reference_index=book.index,
            price=price,
        )
    option_remaining = trade.option_quantity - option_closed
    hedge_remaining = trade.hedge_quantity - hedge_closed
    # The latest valid as-of book is a marking reference only, never a substituted exit.
    reference, reference_snapshot = trade.hedge_book, row["signal_snapshot"]
    for snapshot in snapshots:
        if snapshot.captured_at < row["entry_time"] or snapshot.captured_at > end:
            continue
        try:
            reference = _book(
                snapshot, config.perpetual_instrument, config, expected=trade.hedge_metadata
            )
            reference_snapshot = snapshot.snapshot_id
        except ValueError:
            continue
    remaining_years = max(
        0.0,
        (trade.option_metadata["expiration_timestamp"] - milliseconds(end)) / 1000 / YEAR_SECONDS,
    )
    inverse_option = Contract.from_metadata(trade.option_metadata).inverse
    factor = reference.index if inverse_option else 1.0
    # Conservative cash-model upper bounds, floored at the native BTC call payoff cap.
    if trade.option_metadata["option_type"] == "call":
        upper = max(1.0, exp(-row["carry_yield"] * remaining_years)) * reference.index / factor
    else:
        upper = (
            trade.option_metadata["strike"] * exp(max(0.0, -row["rate"] * remaining_years)) / factor
        )
    option_mark = 0.0 if option_remaining >= 0 else upper
    option_residual = option_remaining * row["multiplier"] * option_mark
    hedge_mark = reference.price(-hedge_remaining) * (
        1 - copysign(config.residual_hedge_adverse_bps / 10000, hedge_remaining)
    )
    hedge_residual = float(
        perpetual_gain(
            hedge_remaining,
            trade.hedge.contract_size,
            trade.hedge.inverse,
            row["hedge_entry_price"],
            hedge_mark,
        )
    )
    close_reserve = 0.0
    if abs(option_remaining) > 1e-14:
        close_reserve += _option_fee(
            option_remaining * row["multiplier"],
            option_mark,
            reference.index,
            inverse_option,
            config,
        )
        _event(
            trade,
            "option_residual_mark",
            end,
            mark=option_residual,
            position=option_remaining,
            marking="long zero / short upper bound",
            price=option_mark,
        )
    if abs(hedge_remaining) > 1e-14:
        close_reserve += _hedge_fee(hedge_remaining, hedge_mark, trade.hedge, config)
        _event(
            trade,
            "hedge_residual_mark",
            end,
            mark=hedge_residual,
            position=hedge_remaining,
            marking="adverse as-of price scenario; not a guaranteed loss bound",
            price=hedge_mark,
            quote_timestamp=reference.timestamp,
        )
    if close_reserve:
        _event(
            trade,
            "residual_close_fee_reserve",
            end,
            mark=-close_reserve,
            reserve_native=close_reserve,
        )
    paid, funding_reserve, gaps, reconstructed = _funding(trade, snapshots, end, config)
    option_fees = sum(
        event["fee_native"] for event in trade.ledger if event["event"].startswith("option_")
    )
    hedge_fees = sum(
        event["fee_native"] for event in trade.ledger if event["event"].startswith("hedge_")
    )
    option_gross = (
        option_proceeds
        + option_residual
        - trade.option_quantity * row["multiplier"] * row["option_entry_price_quote"]
    )
    hedge_gross = hedge_realized + hedge_residual
    net = (
        option_gross
        + hedge_gross
        - option_fees
        - hedge_fees
        - paid
        - funding_reserve
        - close_reserve
    )
    direct = sum(event["cash_flow_native"] + event["mark_value_native"] for event in trade.ledger)
    outstanding = abs(option_remaining) > 1e-12 or abs(hedge_remaining) > 1e-12
    row.update(
        trade_status="outstanding exposure" if outstanding else "closed hypothetically",
        exit_reason="; ".join(errors),
        evaluation_snapshot=exit_snapshot.snapshot_id if exit_snapshot else None,
        evaluation_time=end,
        elapsed_seconds=(end - row["entry_time"]).total_seconds(),
        closed_option_contracts=option_closed,
        closed_hedge_contracts=hedge_closed,
        outstanding_option_contracts=option_remaining,
        outstanding_hedge_contracts=hedge_remaining,
        option_gross_pnl_native=option_gross,
        hedge_gross_pnl_native=hedge_gross,
        option_fees_native=option_fees,
        hedge_fees_native=hedge_fees,
        captured_rate_funding_paid_native=paid,
        funding_gap_reserve_native=funding_reserve,
        funding_gap_seconds=gaps,
        funding_reconstructed_seconds=reconstructed,
        residual_close_fee_reserve_native=close_reserve,
        net_pnl_native=net,
        direct_ledger_pnl_native=direct,
        reconciliation_error_native=net - direct,
        spread_cost_native=sum(event["spread_cost_native"] for event in trade.ledger),
        option_premium_turnover_native=sum(
            event["turnover_native"]
            for event in trade.ledger
            if event["event"].startswith("option_")
        ),
        hedge_notional_turnover_native=sum(
            event["turnover_native"]
            for event in trade.ledger
            if event["event"].startswith("hedge_")
        ),
        indicative_net_pnl_cash=net * reference.index if inverse_option else net,
        marking_reference_snapshot=reference_snapshot,
        marking_reference_index=reference.index,
        marking_quote_age_seconds=(milliseconds(end) - reference.timestamp) / 1000,
        residual_option_mark_quote=option_mark,
        residual_hedge_mark_price=hedge_mark,
    )
    return row
