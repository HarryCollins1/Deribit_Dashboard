"""Fresh-acquisition scope, separate from permissive historical snapshot decoding."""

BASE_CURRENCIES = ("BTC", "ETH")
PRODUCT_KINDS = ("option", "future", "spot")


def selected_bases(currency: str = "ANY") -> tuple[str, ...]:
    currency = currency.upper()
    if currency == "ANY":
        return BASE_CURRENCIES
    if currency not in BASE_CURRENCIES:
        raise ValueError("Fresh market data supports BTC, ETH, or ANY (BTC and ETH only)")
    return (currency,)


def in_universe(metadata: dict, currency: str = "ANY", *, active: bool = True) -> bool:
    """Use exchange metadata, not instrument-name conventions or settlement currency."""
    return (
        metadata.get("base_currency") in selected_bases(currency)
        and metadata.get("kind") in PRODUCT_KINDS
        and (not active or metadata.get("is_active", True))
        and bool(metadata.get("instrument_name"))
    )


def product_label(metadata: dict) -> str:
    if metadata.get("kind") == "future" and metadata.get("settlement_period") == "perpetual":
        return "perpetual"
    return metadata.get("kind", "unknown")
