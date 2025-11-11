"""
Utility script to backfill historical cryptocurrency prices using CoinGecko API.
Note: CoinGecko doesn't provide 5-minute intervals directly, so we use hourly data
and interpolate/aggregate. For true 5-minute data, consider using a paid API.

Example:
    python fetch_coingecko_5m.py --symbol bitcoin --output-dir data/historical
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

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{}/market_chart"


def fetch_historical_data(
    coin_id: str,
    days: int,
    session: requests.Session,
) -> dict:
    """Fetch historical data from CoinGecko."""
    url = COINGECKO_URL.format(coin_id)
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "hourly" if days > 90 else "daily",
    }
    
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def aggregate_to_5min(prices: List[List], volumes: List[List]) -> List[dict]:
    """Aggregate hourly/daily data into approximate 5-minute intervals."""
    # CoinGecko returns data in format: [[timestamp_ms, value], ...]
    # We'll create 5-minute intervals by interpolating between data points
    
    if not prices:
        return []
    
    candles = []
    
    # Sort by timestamp
    price_data = sorted(prices, key=lambda x: x[0])
    volume_data = {int(v[0]): v[1] for v in volumes} if volumes else {}
    
    for i in range(len(price_data) - 1):
        start_ts = price_data[i][0] / 1000  # Convert ms to seconds
        end_ts = price_data[i + 1][0] / 1000
        start_price = price_data[i][1]
        end_price = price_data[i + 1][1]
        
        # Create 5-minute intervals between these two points
        current_ts = start_ts
        interval_count = 0
        
        while current_ts < end_ts:
            # Calculate interpolated price (linear interpolation)
            progress = (current_ts - start_ts) / (end_ts - start_ts) if end_ts > start_ts else 0
            interpolated_price = start_price + (end_price - start_price) * progress
            
            # Use the volume from the nearest hourly point
            nearest_hour_ts = int(start_ts // 3600) * 3600
            volume = volume_data.get(nearest_hour_ts * 1000, 0.0) / 12  # Divide by 12 (5-min intervals per hour)
            
            open_time = datetime.fromtimestamp(current_ts, tz=timezone.utc)
            close_time = open_time + timedelta(minutes=5)
            
            candles.append({
                "open_time": open_time,
                "open": interpolated_price,
                "high": max(start_price, end_price),
                "low": min(start_price, end_price),
                "close": interpolated_price,
                "volume": volume,
                "close_time": close_time,
            })
            
            current_ts += 300  # 5 minutes in seconds
            interval_count += 1
            
            # Limit to prevent too many interpolated points
            if interval_count >= 12:  # Max 12 intervals per hour
                break
    
    return candles


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill CoinGecko historical cryptocurrency data (approximated to 5-minute intervals)."
    )
    parser.add_argument(
        "--symbol",
        default="bitcoin",
        help="CoinGecko coin ID, e.g., bitcoin, ethereum, solana (default: bitcoin).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch (default: 365, max recommended: 365).",
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
        default=1.0,
        help="Seconds to pause between API requests (CoinGecko free tier: 10-50 calls/minute).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    output_file = args.output_dir / f"{args.symbol}_5m.csv"
    
    session = requests.Session()
    
    print(f"Fetching historical data for {args.symbol}...")
    print(f"  Requesting {args.days} days of data...")
    
    try:
        data = fetch_historical_data(args.symbol, args.days, session)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("Error: Rate limit exceeded. Please wait and try again.", file=sys.stderr)
        else:
            print(f"Error fetching data: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    
    if not prices:
        print("No price data returned.", file=sys.stderr)
        return 1
    
    print(f"  Received {len(prices)} data points")
    print("  Aggregating into 5-minute intervals...")
    
    candles = aggregate_to_5min(prices, volumes)
    
    if not candles:
        print("No candles created.", file=sys.stderr)
        return 1
    
    total = write_csv(candles, output_file, args.symbol)
    print(f"\nWrote {total} rows to {output_file}")
    print("\nNote: This data is interpolated from hourly/daily points.")
    print("For true 5-minute historical data, consider using a paid API service.")
    
    return 0


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


if __name__ == "__main__":
    sys.exit(main())


