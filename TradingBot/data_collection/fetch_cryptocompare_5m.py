"""
Utility script to backfill historical 5-minute cryptocurrency candles from CryptoCompare.

Example:
    python fetch_cryptocompare_5m.py --symbol BTC --output-dir data/historical
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histominute"


def fetch_klines(
    symbol: str,
    to_timestamp: int,
    limit: int,
    session: requests.Session,
) -> dict:
    """Fetch historical minute data from CryptoCompare."""
    params = {
        "fsym": symbol.upper(),
        "tsym": "USD",
        "toTs": to_timestamp,
        "limit": min(limit, 2000),  # CryptoCompare max is 2000
    }
    
    response = session.get(CRYPTOCOMPARE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    if data.get("Response") == "Error":
        raise ValueError(f"CryptoCompare API error: {data.get('Message', 'Unknown error')}")
    
    return data


def aggregate_to_5min(candles: List[dict]) -> List[dict]:
    """Aggregate 1-minute candles into 5-minute candles."""
    if not candles:
        return []
    
    aggregated = []
    current_group = []
    
    for candle in sorted(candles, key=lambda x: x["time"]):
        dt = datetime.fromtimestamp(candle["time"], tz=timezone.utc)
        # Round down to nearest 5-minute mark
        rounded_minute = (dt.minute // 5) * 5
        group_time = dt.replace(minute=rounded_minute, second=0, microsecond=0)
        group_timestamp = int(group_time.timestamp())
        
        if not current_group or current_group[0]["group_time"] != group_timestamp:
            if current_group:
                # Aggregate the previous group
                agg = {
                    "time": current_group[0]["group_time"],
                    "open": current_group[0]["open"],
                    "high": max(c["high"] for c in current_group),
                    "low": min(c["low"] for c in current_group),
                    "close": current_group[-1]["close"],
                    "volumefrom": sum(c["volumefrom"] for c in current_group),
                    "volumeto": sum(c["volumeto"] for c in current_group),
                }
                aggregated.append(agg)
            current_group = []
        
        candle["group_time"] = group_timestamp
        current_group.append(candle)
    
    # Don't forget the last group
    if current_group:
        agg = {
            "time": current_group[0]["group_time"],
            "open": current_group[0]["open"],
            "high": max(c["high"] for c in current_group),
            "low": min(c["low"] for c in current_group),
            "close": current_group[-1]["close"],
            "volumefrom": sum(c["volumefrom"] for c in current_group),
            "volumeto": sum(c["volumeto"] for c in current_group),
        }
        aggregated.append(agg)
    
    return aggregated


def fetch_all_historical(
    symbol: str,
    start_timestamp: int,
    end_timestamp: int,
    request_pause: float,
) -> List[dict]:
    """Fetch all historical 5-minute candles between start and end timestamps."""
    session = requests.Session()
    all_minute_candles = []
    current_timestamp = end_timestamp
    
    print(f"Fetching historical data for {symbol}...")
    print(f"  Date range: {datetime.fromtimestamp(start_timestamp, tz=timezone.utc).isoformat()} to {datetime.fromtimestamp(end_timestamp, tz=timezone.utc).isoformat()}")
    
    while current_timestamp > start_timestamp:
        try:
            data = fetch_klines(symbol, current_timestamp, 2000, session)
            candles = data.get("Data", {}).get("Data", [])
            
            if not candles:
                break
            
            # Filter candles within our time range
            filtered = [c for c in candles if c["time"] >= start_timestamp]
            
            if not filtered:
                # We've gone past our start time
                break
            
            all_minute_candles.extend(filtered)
            
            # Move to the earliest timestamp we got minus 1 minute
            current_timestamp = min(c["time"] for c in filtered) - 60
            
            print(f"  Fetched {len(filtered)} minute candles (total: {len(all_minute_candles)})...")
            time.sleep(request_pause)
            
        except Exception as e:
            print(f"Error fetching data: {e}", file=sys.stderr)
            break
    
    # Aggregate minute candles into 5-minute candles
    print(f"\nAggregating {len(all_minute_candles)} minute candles into 5-minute candles...")
    aggregated = aggregate_to_5min(all_minute_candles)
    print(f"Created {len(aggregated)} 5-minute candles")
    
    return aggregated


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
        "quote_asset_volume",
    ]
    
    with dest.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for candle in candles:
            open_time = datetime.fromtimestamp(candle["time"], tz=timezone.utc)
            close_time = datetime.fromtimestamp(candle["time"] + 300, tz=timezone.utc)  # 5 minutes later
            
            writer.writerow([
                symbol.upper(),
                open_time.isoformat(),
                str(candle["open"]),
                str(candle["high"]),
                str(candle["low"]),
                str(candle["close"]),
                str(candle["volumefrom"]),  # Base volume
                close_time.isoformat(),
                str(candle["volumeto"]),  # Quote volume (USD)
            ])
    
    return len(candles)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill CryptoCompare historical 5-minute candles into a CSV file."
    )
    parser.add_argument(
        "--symbol",
        default="BTC",
        help="Cryptocurrency symbol, e.g., BTC, ETH, SOL (default: BTC).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Inclusive UTC start timestamp (ISO-8601). Defaults to 1 year ago.",
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
        default=0.5,
        help="Seconds to pause between API requests to respect rate limits.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    try:
        if args.start:
            start = datetime.fromisoformat(args.start).astimezone(timezone.utc)
        else:
            # Default to 2 years ago (CryptoCompare free tier allows up to ~2000 data points per request)
            # We'll fetch as much as possible
            start = datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 2)
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
    
    start_timestamp = int(start.timestamp())
    end_timestamp = int(end.timestamp())
    
    output_file = args.output_dir / f"{args.symbol.upper()}_5m.csv"
    
    candles = fetch_all_historical(
        symbol=args.symbol,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        request_pause=args.request_pause,
    )
    
    if not candles:
        print("No data fetched. Check your symbol and date range.", file=sys.stderr)
        return 1
    
    total = write_csv(candles, output_file, args.symbol)
    print(f"\nWrote {total} rows to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

