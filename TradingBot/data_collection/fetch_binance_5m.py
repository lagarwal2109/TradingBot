"""
Utility script to backfill historical 5-minute cryptocurrency candles from Binance.

Example:
    python fetch_binance_5m.py --symbol BTCUSDT --output-dir data/historical
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_EPOCH_START = datetime(2017, 8, 17, tzinfo=timezone.utc)


@dataclass
class Candle:
    open_time: datetime
    open_price: str
    high_price: str
    low_price: str
    close_price: str
    volume: str
    close_time: datetime
    quote_asset_volume: str
    number_of_trades: int
    taker_buy_base_volume: str
    taker_buy_quote_volume: str

    @classmethod
    def from_raw(cls, raw: List) -> "Candle":
        """Convert the Binance REST API payload into a Candle instance."""
        return cls(
            open_time=_ms_to_datetime(raw[0]),
            open_price=raw[1],
            high_price=raw[2],
            low_price=raw[3],
            close_price=raw[4],
            volume=raw[5],
            close_time=_ms_to_datetime(raw[6]),
            quote_asset_volume=raw[7],
            number_of_trades=int(raw[8]),
            taker_buy_base_volume=raw[9],
            taker_buy_quote_volume=raw[10],
        )

    def to_csv_row(self) -> List[str]:
        return [
            self.open_time.isoformat(),
            self.open_price,
            self.high_price,
            self.low_price,
            self.close_price,
            self.volume,
            self.close_time.isoformat(),
            self.quote_asset_volume,
            str(self.number_of_trades),
            self.taker_buy_base_volume,
            self.taker_buy_quote_volume,
        ]


def _ms_to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def fetch_klines(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: Optional[int],
    limit: int,
    session: requests.Session,
) -> List:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_time_ms,
        "limit": limit,
    }
    # Only add endTime if explicitly needed (some regions may block requests with endTime)
    # if end_time_ms is not None:
    #     params["endTime"] = end_time_ms

    response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
    if response.status_code == 451:
        raise requests.exceptions.HTTPError(
            f"451 Client Error: Binance API may be restricted in your region. "
            f"Response: {response.text}"
        )
    response.raise_for_status()
    return response.json()


def iter_candles(
    symbol: str,
    interval: str,
    start: datetime,
    end: Optional[datetime],
    batch_limit: int,
    request_pause: float,
) -> Iterable[Candle]:
    session = requests.Session()
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000) if end else None

    while True:
        raw_candles = fetch_klines(symbol, interval, start_ms, end_ms, batch_limit, session)
        if not raw_candles:
            break

        for raw in raw_candles:
            candle = Candle.from_raw(raw)
            if end and candle.open_time >= end:
                return
            yield candle

        last_close_ms = raw_candles[-1][6]
        next_open_ms = last_close_ms + 1
        if next_open_ms == start_ms:
            # Safeguard: Binance returned a single candle repeatedly. Avoid tight loop.
            break
        start_ms = next_open_ms
        time.sleep(request_pause)


def write_csv(candles: Iterable[Candle], dest: Path, symbol: str) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "symbol",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    count = 0

    with dest.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for candle in candles:
            writer.writerow([symbol.upper(), *candle.to_csv_row()])
            count += 1

    return count


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Binance historical 5-minute candles into a CSV file."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g., BTCUSDT.")
    parser.add_argument(
        "--interval",
        default="5m",
        help="Candle interval as accepted by Binance (default: 5m).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Inclusive UTC start timestamp (ISO-8601). Defaults to Binance inception.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Exclusive UTC end timestamp (ISO-8601). Defaults to now.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "historical",
        help="Directory where the CSV file will be created.",
    )
    parser.add_argument(
        "--request-pause",
        type=float,
        default=0.25,
        help="Seconds to pause between API requests to respect rate limits.",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=1000,
        help="Number of candles to request per API call (max 1000).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.interval != "5m":
        print("Warning: Interval differs from requested 5-minute default.", file=sys.stderr)

    try:
        start = (
            datetime.fromisoformat(args.start).astimezone(timezone.utc)
            if args.start
            else BINANCE_EPOCH_START
        )
    except ValueError as exc:
        print(f"Invalid --start value: {exc}", file=sys.stderr)
        return 1

    try:
        end = (
            datetime.fromisoformat(args.end).astimezone(timezone.utc)
            if args.end
            else datetime.now(timezone.utc)
        )
    except ValueError as exc:
        print(f"Invalid --end value: {exc}", file=sys.stderr)
        return 1

    if end <= start:
        print("--end must be greater than --start", file=sys.stderr)
        return 1

    output_file = args.output_dir / f"{args.symbol.upper()}_{args.interval}.csv"

    print(f"Fetching {args.interval} candles for {args.symbol.upper()}...")
    candle_iter = iter_candles(
        symbol=args.symbol,
        interval=args.interval,
        start=start,
        end=end,
        batch_limit=args.batch_limit,
        request_pause=args.request_pause,
    )
    total = write_csv(candle_iter, output_file, args.symbol)
    print(f"Wrote {total} rows to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

