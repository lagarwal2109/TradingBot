"""
Master script to fetch maximum historical 5-minute cryptocurrency data using multiple APIs.
Combines data from yfinance (last 60 days) and CryptoCompare (last 7 days) to get maximum coverage.

Example:
    python fetch_all_5m.py --symbol BTC --output-dir data/historical
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

try:
    import yfinance as yf
except ImportError:
    yf = None

import requests

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histominute"


def fetch_yfinance_data(symbol: str, days: int = 60) -> List[dict]:
    """Fetch data from yfinance (supports last 60 days for 5m interval)."""
    if yf is None:
        return []
    
    # yfinance uses format like BTC-USD
    yf_symbol = f"{symbol}-USD" if not symbol.endswith("-USD") else symbol
    
    print(f"Fetching yfinance data for {yf_symbol} (last {days} days)...")
    
    try:
        ticker = yf.Ticker(yf_symbol)
        end = datetime.now(timezone.utc)
        # Use shorter period to ensure data is available
        start = end - timedelta(days=min(days, 7))
        
        data = ticker.history(start=start, end=end, interval="5m")
        
        if data.empty:
            return []
        
        candles = []
        for idx, row in data.iterrows():
            if idx.tzinfo is None:
                open_time = idx.replace(tzinfo=timezone.utc)
            else:
                open_time = idx.astimezone(timezone.utc)
            
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
        
        print(f"  Fetched {len(candles)} candles from yfinance")
        return candles
    except Exception as e:
        print(f"  Error fetching yfinance data: {e}", file=sys.stderr)
        return []


def fetch_cryptocompare_data(symbol: str, days: int = 7) -> List[dict]:
    """Fetch data from CryptoCompare (supports last 7 days for minute data)."""
    print(f"Fetching CryptoCompare data for {symbol} (last {days} days)...")
    
    session = requests.Session()
    all_minute_candles = []
    current_timestamp = int(datetime.now(timezone.utc).timestamp())
    start_timestamp = current_timestamp - (days * 24 * 60 * 60)
    
    try:
        while current_timestamp > start_timestamp:
            params = {
                "fsym": symbol.upper(),
                "tsym": "USD",
                "toTs": current_timestamp,
                "limit": 2000,
            }
            
            response = session.get(CRYPTOCOMPARE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("Response") == "Error":
                break
            
            candles = data.get("Data", {}).get("Data", [])
            if not candles:
                break
            
            filtered = [c for c in candles if c["time"] >= start_timestamp]
            if not filtered:
                break
            
            all_minute_candles.extend(filtered)
            current_timestamp = min(c["time"] for c in filtered) - 60
            
            if len(all_minute_candles) >= 10000:  # Limit to prevent too much data
                break
        
        # Aggregate to 5-minute candles
        if all_minute_candles:
            aggregated = aggregate_to_5min(all_minute_candles)
            print(f"  Fetched {len(aggregated)} 5-minute candles from CryptoCompare")
            return aggregated
    except Exception as e:
        print(f"  Error fetching CryptoCompare data: {e}", file=sys.stderr)
    
    return []


def aggregate_to_5min(candles: List[dict]) -> List[dict]:
    """Aggregate 1-minute candles into 5-minute candles."""
    if not candles:
        return []
    
    aggregated = []
    current_group = []
    
    for candle in sorted(candles, key=lambda x: x["time"]):
        dt = datetime.fromtimestamp(candle["time"], tz=timezone.utc)
        rounded_minute = (dt.minute // 5) * 5
        group_time = dt.replace(minute=rounded_minute, second=0, microsecond=0)
        group_timestamp = int(group_time.timestamp())
        
        if not current_group or current_group[0]["group_time"] != group_timestamp:
            if current_group:
                agg = {
                    "open_time": datetime.fromtimestamp(current_group[0]["group_time"], tz=timezone.utc),
                    "open": current_group[0]["open"],
                    "high": max(c["high"] for c in current_group),
                    "low": min(c["low"] for c in current_group),
                    "close": current_group[-1]["close"],
                    "volume": sum(c.get("volumefrom", 0) for c in current_group),
                    "close_time": datetime.fromtimestamp(current_group[0]["group_time"] + 300, tz=timezone.utc),
                }
                aggregated.append(agg)
            current_group = []
        
        candle["group_time"] = group_timestamp
        current_group.append(candle)
    
    if current_group:
        agg = {
            "open_time": datetime.fromtimestamp(current_group[0]["group_time"], tz=timezone.utc),
            "open": current_group[0]["open"],
            "high": max(c["high"] for c in current_group),
            "low": min(c["low"] for c in current_group),
            "close": current_group[-1]["close"],
            "volume": sum(c.get("volumefrom", 0) for c in current_group),
            "close_time": datetime.fromtimestamp(current_group[0]["group_time"] + 300, tz=timezone.utc),
        }
        aggregated.append(agg)
    
    return aggregated


def merge_candles(candle_lists: List[List[dict]]) -> List[dict]:
    """Merge and deduplicate candles from multiple sources."""
    all_candles = []
    seen_times = set()
    
    for candle_list in candle_lists:
        for candle in candle_list:
            time_key = candle["open_time"].isoformat()
            if time_key not in seen_times:
                seen_times.add(time_key)
                all_candles.append(candle)
    
    # Sort by time
    all_candles.sort(key=lambda x: x["open_time"])
    return all_candles


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
        description="Fetch maximum historical 5-minute cryptocurrency data using multiple APIs."
    )
    parser.add_argument(
        "--symbol",
        default="BTC",
        help="Cryptocurrency symbol, e.g., BTC, ETH, SOL (default: BTC).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "historical",
        help="Directory where the CSV file will be created.",
    )
    parser.add_argument(
        "--yfinance-days",
        type=int,
        default=60,
        help="Days of yfinance data to fetch (default: 60, max for 5m interval).",
    )
    parser.add_argument(
        "--cryptocompare-days",
        type=int,
        default=7,
        help="Days of CryptoCompare data to fetch (default: 7, max for minute data).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    output_file = args.output_dir / f"{args.symbol.upper()}_5m.csv"
    
    print(f"Fetching maximum historical 5-minute data for {args.symbol}...")
    print("=" * 60)
    
    # Fetch from multiple sources
    all_candle_lists = []
    
    # Try yfinance (best for recent 60 days)
    yf_candles = fetch_yfinance_data(args.symbol, args.yfinance_days)
    if yf_candles:
        all_candle_lists.append(yf_candles)
    
    # Try CryptoCompare (best for last 7 days, can supplement)
    cc_candles = fetch_cryptocompare_data(args.symbol, args.cryptocompare_days)
    if cc_candles:
        all_candle_lists.append(cc_candles)
    
    if not all_candle_lists:
        print("\nError: No data fetched from any source.", file=sys.stderr)
        return 1
    
    # Merge and deduplicate
    print("\nMerging data from all sources...")
    merged = merge_candles(all_candle_lists)
    print(f"Total unique candles: {len(merged)}")
    
    if merged:
        date_range = f"{merged[0]['open_time'].isoformat()} to {merged[-1]['open_time'].isoformat()}"
        print(f"Date range: {date_range}")
    
    # Write to CSV
    total = write_csv(merged, output_file, args.symbol)
    print(f"\nSuccess! Wrote {total} rows to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

