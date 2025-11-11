"""
Utility script to backfill historical 5-minute cryptocurrency candles using yfinance.

Example:
    python fetch_yfinance_5m.py --symbol BTC-USD --output-dir data/historical
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance is not installed. Install it with: pip install yfinance", file=sys.stderr)
    sys.exit(1)


def fetch_historical_data(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "5m",
) -> list:
    """Fetch historical data using yfinance."""
    print(f"Fetching {interval} candles for {symbol}...")
    print(f"  Date range: {start.isoformat()} to {end.isoformat()}")
    
    # yfinance uses ticker format
    ticker = yf.Ticker(symbol)
    
    # yfinance interval options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    # Note: For intervals < 1d, yfinance only provides data for the last 7-60 days depending on interval
    # For 5m, it's typically last 60 days
    
    try:
        data = ticker.history(start=start, end=end, interval=interval)
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return []
    
    if data.empty:
        print("No data returned from yfinance.", file=sys.stderr)
        return []
    
    # Convert to list of dictionaries
    candles = []
    for idx, row in data.iterrows():
        # Convert timezone-aware index to UTC
        if idx.tzinfo is None:
            open_time = idx.replace(tzinfo=timezone.utc)
        else:
            open_time = idx.astimezone(timezone.utc)
        
        # Calculate close time (5 minutes later for 5m interval)
        close_time = open_time + timedelta(minutes=5)
        
        candles.append({
            "open_time": open_time,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "close_time": close_time,
        })
    
    return candles


def write_csv(candles: list, dest: Path, symbol: str) -> int:
    """Write candles to CSV file."""
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
    ]
    
    with dest.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for candle in candles:
            writer.writerow([
                symbol.upper(),
                candle["open_time"].isoformat(),
                str(candle["open"]),
                str(candle["high"]),
                str(candle["low"]),
                str(candle["close"]),
                str(candle["volume"]),
                candle["close_time"].isoformat(),
            ])
    
    return len(candles)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill yfinance historical 5-minute candles into a CSV file."
    )
    parser.add_argument(
        "--symbol",
        default="BTC-USD",
        help="Cryptocurrency symbol in yfinance format, e.g., BTC-USD, ETH-USD (default: BTC-USD).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Inclusive UTC start timestamp (ISO-8601). Defaults to 60 days ago (yfinance 5m limit).",
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
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    
    try:
        if args.start:
            start = datetime.fromisoformat(args.start).astimezone(timezone.utc)
        else:
            # yfinance 5m interval typically allows last 60 days
            # Use last 7 days to be safe
            start = datetime.now(timezone.utc) - timedelta(days=7)
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
    
    # Check if date range is too large for yfinance 5m data
    days_diff = (end - start).days
    if days_diff > 60:
        print(f"Warning: yfinance 5m interval typically only supports last 60 days.", file=sys.stderr)
        print(f"  Your range is {days_diff} days. Data may be incomplete.", file=sys.stderr)
        # Adjust start to 60 days ago
        start = end - timedelta(days=60)
        print(f"  Adjusted start to: {start.isoformat()}", file=sys.stderr)
    
    output_file = args.output_dir / f"{args.symbol.replace('-', '')}_5m.csv"
    
    candles = fetch_historical_data(
        symbol=args.symbol,
        start=start,
        end=end,
        interval="5m",
    )
    
    if not candles:
        print("No data fetched.", file=sys.stderr)
        return 1
    
    total = write_csv(candles, output_file, args.symbol)
    print(f"\nWrote {total} rows to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

