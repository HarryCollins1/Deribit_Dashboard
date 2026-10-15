"""Historical trade candles; these are not historical option-Greek snapshots."""

from deribit_dashboard.client import DeribitClient
from deribit_dashboard.models import milliseconds, parse_time

RESOLUTIONS = {"1", "3", "5", "10", "15", "30", "60", "120", "180", "360", "720", "1D"}


async def download_candles(client: DeribitClient, instrument: str, start, end, resolution="60"):
    start, end = parse_time(start), parse_time(end)
    if start >= end:
        raise ValueError("Start must precede end")
    if resolution not in RESOLUTIONS:
        raise ValueError("Unsupported candle resolution")
    interval_ms = (1440 if resolution == "1D" else int(resolution)) * 60000
    cursor, stop, rows = milliseconds(start), milliseconds(end), {}
    # Bounded windows avoid relying on a single arbitrarily large API response.
    while cursor < stop:
        window_end = min(cursor + 500 * interval_ms, stop)
        data = await client.call(
            "public/get_tradingview_chart_data",
            instrument_name=instrument,
            start_timestamp=cursor,
            end_timestamp=window_end,
            resolution=resolution,
        )
        if data.get("status") == "ok":
            ticks = data.get("ticks", [])
            for field in ["open", "high", "low", "close", "volume"]:
                if len(data.get(field, [])) != len(ticks):
                    raise ValueError("Candle arrays have inconsistent lengths")
            for i, timestamp in enumerate(ticks):
                if milliseconds(start) <= timestamp <= stop:
                    rows[timestamp] = {
                        "timestamp_ms": timestamp,
                        "instrument_name": instrument,
                        "environment": client.settings.environment,
                        "resolution": resolution,
                        **{
                            field: data[field][i]
                            for field in ["open", "high", "low", "close", "volume"]
                        },
                    }
        elif data.get("status") != "no_data":
            raise ValueError("Unknown historical candle response status")
        cursor = window_end
    return [rows[key] for key in sorted(rows)]
