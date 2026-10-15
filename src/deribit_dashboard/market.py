"""Derived two-sided quotes and explicit event replay accounting (no order execution)."""

from math import ceil, floor, sqrt

import numpy as np
import pandas as pd

from deribit_dashboard.pricing import YEAR_SECONDS
from deribit_dashboard.surface import Contract

WIDTH_EXPLANATION = (
    "Cash half-width = sqrt(residual_RMSE² + (vega × IV_RMSE × sqrt(min(horizon/T, 1)))² "
    "+ (live_half_spread / sqrt(1 + displayed_size/contract_size))²). "
    "Vega is per one absolute model-volatility unit. Normal IV uncertainty uses the local "
    "conversion S × lognormal IV RMSE. Width is rounded outwards to the "
    "metadata tick ladder. Inventory limits, fees and a mock next mark are runtime inputs."
)


def outward_round(price, contract, up):
    candidate = max(0.0, price)
    for _ in range(len(contract.steps) + 3):
        tick = contract.tick_at(candidate)
        new = (ceil if up else floor)(price / tick) * tick
        new = max(0.0, new)
        if contract.tick_at(new) == tick:
            return new
        candidate = new
    raise ValueError("Tick ladder does not converge")


def quote_market(
    theoretical,
    vega,
    residual_rmse,
    iv_rmse,
    years,
    horizon_seconds,
    live_bid,
    live_ask,
    liquidity,
    metadata,
    spot,
):
    if not np.isfinite(
        [
            theoretical,
            vega,
            residual_rmse,
            iv_rmse,
            years,
            horizon_seconds,
            live_bid,
            live_ask,
            liquidity,
            spot,
        ]
    ).all():
        raise ValueError("Finite quote inputs required")
    if min(theoretical, residual_rmse, iv_rmse, horizon_seconds, liquidity) < 0 or years <= 0:
        raise ValueError("Invalid quote uncertainty inputs")
    if not 0 <= live_bid <= live_ask:
        raise ValueError("Cannot quote against a crossed or invalid live market")
    contract = Contract.from_metadata(metadata)
    live_half = (live_ask - live_bid) / 2
    time_term = abs(vega) * iv_rmse * sqrt(min(horizon_seconds / YEAR_SECONDS / years, 1))
    liquidity_term = live_half / sqrt(1 + liquidity / contract.multiplier)
    half = sqrt(residual_rmse**2 + time_term**2 + liquidity_term**2)
    factor = contract.cash_factor(spot)
    bid = outward_round(max(0, theoretical - half) / factor, contract, False) * factor
    ask = outward_round((theoretical + half) / factor, contract, True) * factor
    if ask <= bid:
        ask = outward_round(ask / factor + contract.tick_at(ask / factor), contract, True) * factor

    def position(price, side):
        if (side == "bid" and price >= live_ask) or (side == "ask" and price <= live_bid):
            return "crossed"
        if (side == "bid" and price > live_bid) or (side == "ask" and price < live_ask):
            return "inside"
        return "outside" if price != (live_bid if side == "bid" else live_ask) else "at market"

    return dict(
        bid=bid,
        ask=ask,
        theoretical=theoretical,
        live_bid=live_bid,
        live_ask=live_ask,
        half_width=half,
        residual_component=residual_rmse,
        time_vega_component=time_term,
        liquidity_component=liquidity_term,
        bid_position=position(bid, "bid"),
        ask_position=position(ask, "ask"),
        quote_bid=bid / factor,
        quote_ask=ask / factor,
    )


def replay(events, multiplier, fee_per_contract=0.0, inventory_limit=np.inf):
    """Measured markouts from supplied chronological events, cash settlement accounting.

    Each event: bid/ask, fair_before/fair_after, next_spot/spot, hedge_delta,
    taker_side (buy/sell/none), trade_price, contracts. A touch is a mock fill;
    queue priority/partial fills are not inferred from snapshots.
    """
    if multiplier <= 0 or fee_per_contract < 0 or inventory_limit < 0:
        raise ValueError("Invalid simulation contract/fee/limit")
    inventory, total, rows = 0.0, 0.0, []
    for event in events:
        e = dict(event)
        side = e["taker_side"]
        if side not in ("buy", "sell", "none") or e["contracts"] < 0:
            raise ValueError("Invalid event side or quantity")
        filled = (side == "buy" and e["trade_price"] >= e["ask"]) or (
            side == "sell" and e["trade_price"] <= e["bid"]
        )
        q = (-1 if side == "buy" else 1) * e["contracts"] if filled else 0
        rejected = abs(inventory + q) > inventory_limit
        if rejected:
            q = 0
        execution = e["ask"] if side == "buy" else e["bid"]
        spread = q * multiplier * (e["fair_before"] - execution)
        adverse = q * multiplier * (e["fair_after"] - e["fair_before"])
        carry = inventory * multiplier * (e["fair_after"] - e["fair_before"])
        hedge = -inventory * multiplier * e["hedge_delta"] * (e["next_spot"] - e["spot"])
        fees = -abs(q) * fee_per_contract
        pnl = spread + adverse + carry + hedge + fees
        # Independent cash-flow plus end-mark identity.
        direct = (inventory + q) * multiplier * e["fair_after"] - inventory * multiplier * e[
            "fair_before"
        ]
        direct += -q * multiplier * execution + hedge + fees
        inventory += q
        total += pnl
        rows.append(
            dict(
                fill_contracts=q,
                inventory=inventory,
                spread_capture=spread,
                adverse_selection=adverse,
                inventory_revaluation=carry,
                hedge_pnl=hedge,
                fees=fees,
                pnl=pnl,
                direct_pnl=direct,
                reconciliation=pnl - direct,
                cumulative_pnl=total,
                inventory_rejected=rejected,
            )
        )
    return pd.DataFrame(rows)


def simulation_summary(events, starting_cash):
    """Accounting demonstration cash-budget check; no trading-survival inference.

    ``survived_cash_loss_budget`` remains a compatibility alias. New callers should
    report ``cash_budget_breached`` and the cash balance, not market survival.
    """
    if not np.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("Positive starting cash required")
    equity = starting_cash + events.cumulative_pnl
    breached = bool((equity <= 0).any())
    return dict(
        final_pnl=float(events.pnl.sum()),
        minimum_cash_equity=float(min(starting_cash, equity.min()))
        if len(equity)
        else starting_cash,
        cash_budget_breached=breached,
        survived_cash_loss_budget=not breached,
        fills=int((events.fill_contracts != 0).sum()),
        rejected_orders=int(events.inventory_rejected.sum()),
        total_signed_markout=float(events.adverse_selection.sum()),
        maximum_reconciliation_error=float(events.reconciliation.abs().max()),
    )
