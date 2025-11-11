"""
Batch fetch 5-minute cryptocurrency candles for a fixed 14-day period of the current year.

This script uses the hourly-to-5m expansion logic from `fetch_alternative_timeframe`
so it can retrieve data for dates beyond the 7-day minute-data window of free APIs.

Example:
    python -m data_collection.fetch_random_period_batch
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from data_collection.fetch_alternative_timeframe import main as fetch_alt_main

# Symbols inferred from files in `data/historical/*_5m.csv`
SYMBOLS: Iterable[str] = (
    "AAVE",
    "APT",
    "AVAX",
    "AVNT",
    "BIO",
    "BMT",
    "BNB",
    "BTC",
    "CAKE",
    "CRV",
    "DOGE",
    "DOT",
    "EIGEN",
    "ENA",
    "FLOKI",
    "HBAR",
    "HEMI",
    "ICP",
    "LINEA",
    "LINK",
    "LTC",
    "OMNI",
    "ONDO",
    "PAXG",
    "PENDLE",
    "PEPE",
    "PLUME",
    "POL",
    "PUMP",
    "SEI",
    "SHIB",
    "SOMI",
    "TAO",
    "TRUMP",
    "UNI",
    "WLD",
    "XLM",
    "XPL",
    "XRP",
    "ZEC",
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 5-minute data for a fixed 14-day period for multiple cryptocurrencies."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-03-03T00:00:00+00:00",
        help="Inclusive start timestamp (ISO-8601). Defaults to 2025-03-03.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-03-17T00:00:00+00:00",
        help="Exclusive end timestamp (ISO-8601). Defaults to 2025-03-17 (14 days).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "random_period",
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between requests to avoid rate limiting.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Subset of symbols to fetch. Defaults to the full inferred list.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        start_dt = datetime.fromisoformat(args.start).astimezone(timezone.utc)
        end_dt = datetime.fromisoformat(args.end).astimezone(timezone.utc)
    except ValueError as exc:
        print(f"Invalid start/end timestamp: {exc}", file=sys.stderr)
        return 1

    if end_dt <= start_dt:
        print("--end must be after --start", file=sys.stderr)
        return 1

    symbols = [s.strip().upper() for s in (args.symbols or SYMBOLS)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching 5-minute candles for {len(symbols)} symbols")
    print(f"Period: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    succeeded: list[str] = []
    failed: list[str] = []

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] {symbol}")
        result = fetch_alt_main(
            [
                "--symbol",
                symbol,
                "--start",
                start_dt.isoformat(),
                "--end",
                end_dt.isoformat(),
                "--output-dir",
                str(args.output_dir),
                "--request-pause",
                "0.2",
            ]
        )
        if result == 0:
            succeeded.append(symbol)
        else:
            failed.append(symbol)

        time.sleep(args.delay)

    print("\n" + "=" * 60)
    print("Random period batch summary")
    print("=" * 60)
    print(f"Successful: {len(succeeded)}")
    if succeeded:
        print(f"  {', '.join(succeeded)}")

    if failed:
        print(f"Failed: {len(failed)}")
        print(f"  {', '.join(failed)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

