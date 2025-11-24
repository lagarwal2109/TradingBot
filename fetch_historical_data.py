#!/usr/bin/env python3
"""Fetch historical data from Binance for all Roostoo trading pairs."""

import time
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

from bot.config import get_config
from bot.roostoo_v3 import RoostooV3Client
from bot.binance import BinancePublicClient
from bot.utils import from_roostoo_pair


def convert_roostoo_to_binance_pair(roostoo_pair: str) -> Optional[str]:
    """Convert Roostoo pair format to Binance format.
    
    Args:
        roostoo_pair: Roostoo pair (e.g., "BTC/USD" or "BTCUSD")
        
    Returns:
        Binance symbol (e.g., "BTCUSDT") or None if conversion not possible
    """
    # Remove slash if present
    pair = roostoo_pair.replace("/", "").replace("_", "")
    
    # Roostoo uses USD, Binance uses USDT
    if pair.endswith("USD"):
        binance_symbol = pair.replace("USD", "USDT")
        return binance_symbol
    
    return None


def fetch_pair_data(
    binance_client: BinancePublicClient,
    roostoo_pair: str,
    days: int = 30,
    interval: str = "1m"
) -> List[Dict]:
    """Fetch historical data for a single pair.
    
    Args:
        binance_client: Binance client
        roostoo_pair: Roostoo pair format
        days: Number of days to fetch
        interval: Data interval (1m, 5m, 1h, etc.)
        
    Returns:
        List of data points with timestamp, price, volume
    """
    binance_symbol = convert_roostoo_to_binance_pair(roostoo_pair)
    if not binance_symbol:
        return []
    
    try:
        # Fetch klines from Binance
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_klines = []
        current_start = start_time
        
        # Binance limits to 1000 klines per request
        while current_start < end_time:
            klines = binance_client.get_klines(
                symbol=binance_symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000
            )
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Update start time for next batch
            if klines:
                current_start = klines[-1].close_time + 1
            else:
                break
            
            # Rate limiting
            time.sleep(0.1)
        
        # Convert to our format
        data_points = []
        for kline in all_klines:
            data_points.append({
                "timestamp": kline.timestamp,
                "price": kline.close,  # Use close price
                "volume": kline.volume
            })
        
        return data_points
        
    except Exception as e:
        print(f"  Error fetching {binance_symbol}: {e}")
        return []


def save_data_to_csv(
    data_dir: Path,
    pair: str,
    data: List[Dict],
    merge_existing: bool = True
) -> None:
    """Save data to CSV file, optionally merging with existing data.
    
    Args:
        data_dir: Data directory
        pair: Trading pair (e.g., "BTCUSD")
        data: List of data points
        merge_existing: If True, merge with existing CSV data
    """
    if not data:
        return
    
    # Clean pair name for filename
    safe_pair = pair.replace("/", "_").replace("\\", "_")
    csv_path = data_dir / f"{safe_pair}.csv"
    
    # Merge with existing data if file exists
    if merge_existing and csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            existing_data = existing_df.to_dict("records")
            # Combine and remove duplicates
            all_data = existing_data + data
            # Remove duplicates by timestamp, keeping the newer one
            seen = {}
            for point in all_data:
                ts = point["timestamp"]
                if ts not in seen or point.get("price", 0) != 0:  # Prefer non-zero prices
                    seen[ts] = point
            data = list(seen.values())
            print(f"  Merged with existing data, total: {len(data)} records", end=" ")
        except Exception as e:
            print(f"  Warning: Could not merge with existing data: {e}", end=" ")
    
    # Sort by timestamp
    data = sorted(data, key=lambda x: x["timestamp"])
    
    # Write to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price", "volume"])
        
        for point in data:
            writer.writerow([
                point["timestamp"],
                point["price"],
                point["volume"]
            ])
    
    print(f"Saved {len(data)} records to {csv_path.name}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch historical data from Binance")
    parser.add_argument("--days", type=int, default=60, help="Number of days to fetch (default: 60, need at least 4 days for indicators)")
    parser.add_argument("--interval", type=str, default="1m", help="Data interval: 1m, 5m, 1h, etc. (default: 1m)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--pairs", type=str, nargs="+", help="Specific pairs to fetch (optional, otherwise fetches all)")
    
    args = parser.parse_args()
    
    config = get_config()
    data_dir = args.data_dir or config.data_dir
    data_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Fetching Historical Data from Binance")
    print(f"{'='*60}")
    print(f"Days: {args.days}")
    print(f"Interval: {args.interval}")
    print(f"Data directory: {data_dir}\n")
    
    # Initialize clients
    binance_client = BinancePublicClient()
    roostoo_client = RoostooV3Client(
        api_key=config.api_key,
        api_secret=config.api_secret,
        base_url=config.base_url
    )
    
    # Get list of trading pairs
    if args.pairs:
        # Use specified pairs
        pairs_to_fetch = args.pairs
        print(f"Fetching data for {len(pairs_to_fetch)} specified pairs...\n")
    else:
        # Get all pairs from Roostoo
        try:
            print("Fetching trading pairs from Roostoo...")
            exchange_info = roostoo_client.exchange_info()
            pairs_to_fetch = [info.pair for info in exchange_info if info.can_trade]
            print(f"Found {len(pairs_to_fetch)} tradable pairs\n")
        except Exception as e:
            print(f"Error fetching pairs from Roostoo: {e}")
            print("Using pairs from existing data files...")
            # Fallback: use pairs from existing CSV files
            pairs_to_fetch = []
            for csv_file in data_dir.glob("*.csv"):
                if csv_file.stem != "state":
                    pairs_to_fetch.append(csv_file.stem.replace("_", ""))
            print(f"Found {len(pairs_to_fetch)} pairs from existing files\n")
    
    # Fetch data for each pair
    successful = 0
    failed = 0
    
    for i, roostoo_pair in enumerate(pairs_to_fetch, 1):
        print(f"[{i}/{len(pairs_to_fetch)}] Fetching {roostoo_pair}...", end=" ", flush=True)
        
        # Convert to internal format (remove slash)
        internal_pair = from_roostoo_pair(roostoo_pair) if "/" in roostoo_pair else roostoo_pair
        
        # Fetch data
        data = fetch_pair_data(
            binance_client,
            internal_pair,
            days=args.days,
            interval=args.interval
        )
        
        if data:
            # Save to CSV
            save_data_to_csv(data_dir, internal_pair, data)
            successful += 1
        else:
            print("  No data available")
            failed += 1
        
        # Rate limiting between pairs
        time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"Data Fetch Complete")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(pairs_to_fetch)}")
    print(f"\nData saved to: {data_dir}")


if __name__ == "__main__":
    main()

