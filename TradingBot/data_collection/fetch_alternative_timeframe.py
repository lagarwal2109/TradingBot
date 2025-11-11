"""
Fetch BTCUSD data for a different time frame using hourly data aggregated to 5-minute intervals.
This allows us to get data from periods beyond the 7-day minute data limit.

Example:
    python fetch_alternative_timeframe.py --symbol BTC --output-dir data/alternative
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"


def fetch_hourly_data(
    symbol: str,
    start_timestamp: int,
    end_timestamp: int,
    request_pause: float,
) -> List[dict]:
    """Fetch hourly data from CryptoCompare."""
    session = requests.Session()
    all_candles = []
    current_timestamp = end_timestamp
    
    print(f"Fetching hourly data for {symbol}...")
    print(f"  Date range: {datetime.fromtimestamp(start_timestamp, tz=timezone.utc).isoformat()} to {datetime.fromtimestamp(end_timestamp, tz=timezone.utc).isoformat()}")
    
    while current_timestamp > start_timestamp:
        try:
            params = {
                "fsym": symbol.upper(),
                "tsym": "USD",
                "toTs": current_timestamp,
                "limit": 2000,  # Max for hourly
            }
            
            response = session.get(CRYPTOCOMPARE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("Response") == "Error":
                error_msg = data.get("Message", "Unknown error")
                print(f"  API Error: {error_msg}")
                break
            
            candles = data.get("Data", {}).get("Data", [])
            if not candles:
                break
            
            filtered = [c for c in candles if c["time"] >= start_timestamp]
            if not filtered:
                break
            
            all_candles.extend(filtered)
            current_timestamp = min(c["time"] for c in filtered) - 3600  # Move back 1 hour
            
            print(f"  Fetched {len(filtered)} hourly candles (total: {len(all_candles)})...")
            time.sleep(request_pause)
            
            if len(all_candles) >= 10000:  # Limit to prevent too much data
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}", file=sys.stderr)
            break
    
    return sorted(all_candles, key=lambda x: x["time"])


def expand_hourly_to_5min(hourly_candles: List[dict]) -> List[dict]:
    """Expand hourly candles into 5-minute candles using interpolation."""
    if not hourly_candles:
        return []
    
    five_min_candles = []
    
    for i in range(len(hourly_candles)):
        hour_candle = hourly_candles[i]
        hour_start = datetime.fromtimestamp(hour_candle["time"], tz=timezone.utc)
        
        # Get next hour's open price for interpolation
        if i + 1 < len(hourly_candles):
            next_open = hourly_candles[i + 1]["open"]
        else:
            next_open = hour_candle["close"]
        
        # Create 12 five-minute intervals per hour
        for j in range(12):
            interval_start = hour_start + timedelta(minutes=j * 5)
            interval_end = interval_start + timedelta(minutes=5)
            
            # Linear interpolation for price
            progress = j / 12.0
            open_price = hour_candle["open"] + (hour_candle["close"] - hour_candle["open"]) * progress
            close_price = hour_candle["open"] + (hour_candle["close"] - hour_candle["open"]) * ((j + 1) / 12.0)
            
            # Distribute volume evenly across 5-minute intervals
            volume_5min = hour_candle.get("volumefrom", 0) / 12.0
            
            five_min_candles.append({
                "open_time": interval_start,
                "open": open_price,
                "high": max(hour_candle["high"], open_price, close_price),
                "low": min(hour_candle["low"], open_price, close_price),
                "close": close_price,
                "volume": volume_5min,
                "close_time": interval_end,
            })
    
    return five_min_candles


def write_csv(candles: List[dict], dest: Path, symbol: str) -> int:
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch BTCUSD data for a different time frame using hourly data."
    )
    parser.add_argument(
        "--symbol",
        default="BTC",
        help="Cryptocurrency symbol (default: BTC).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date in ISO-8601 format (default: 3 weeks ago).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in ISO-8601 format (default: 2 weeks ago).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "alternative",
        help="Directory where the CSV file will be created.",
    )
    parser.add_argument(
        "--request-pause",
        type=float,
        default=0.5,
        help="Seconds to pause between API requests.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Default to 3 weeks ago to 2 weeks ago (different from the Nov 4-11 data we have)
    now = datetime.now(timezone.utc)
    
    try:
        if args.start:
            start = datetime.fromisoformat(args.start).astimezone(timezone.utc)
        else:
            start = now - timedelta(days=21)  # 3 weeks ago
    except ValueError as exc:
        print(f"Invalid --start value: {exc}", file=sys.stderr)
        return 1
    
    try:
        if args.end:
            end = datetime.fromisoformat(args.end).astimezone(timezone.utc)
        else:
            end = now - timedelta(days=14)  # 2 weeks ago
    except ValueError as exc:
        print(f"Invalid --end value: {exc}", file=sys.stderr)
        return 1
    
    if end <= start:
        print("--end must be greater than --start", file=sys.stderr)
        return 1
    
    start_timestamp = int(start.timestamp())
    end_timestamp = int(end.timestamp())
    
    output_file = args.output_dir / f"{args.symbol.upper()}_5m.csv"
    
    print(f"Fetching alternative timeframe data for {args.symbol}...")
    print(f"Time period: {start.isoformat()} to {end.isoformat()}")
    print("(Using hourly data aggregated to 5-minute intervals)")
    print("=" * 60)
    
    # Fetch hourly data
    hourly_candles = fetch_hourly_data(
        symbol=args.symbol,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        request_pause=args.request_pause,
    )
    
    if not hourly_candles:
        print("No hourly data fetched.", file=sys.stderr)
        return 1
    
    print(f"\nFetched {len(hourly_candles)} hourly candles")
    print("Expanding to 5-minute intervals...")
    
    # Expand to 5-minute candles
    five_min_candles = expand_hourly_to_5min(hourly_candles)
    print(f"Created {len(five_min_candles)} 5-minute candles")
    
    if not five_min_candles:
        print("No 5-minute candles created.", file=sys.stderr)
        return 1
    
    # Write to CSV
    total = write_csv(five_min_candles, output_file, args.symbol)
    print(f"\nSuccess! Wrote {total} rows to {output_file}")
    print("\nNote: This data is interpolated from hourly candles.")
    print("For true 5-minute historical data, consider using paid API services.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

