#!/usr/bin/env python3
"""Backfill historical data using Binance API."""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from bot.config import get_config
from bot.binance import BinancePublicClient
from bot.datastore import DataStore


def backfill_from_binance(
    binance_client: BinancePublicClient,
    datastore: DataStore,
    assets: list,
    days: int = 7,
    interval: str = "1h"
):
    """Backfill historical data from Binance API.
    
    Args:
        binance_client: Binance API client
        datastore: DataStore instance
        assets: List of asset symbols
        days: Number of days to backfill
        interval: Data interval (15m, 1h, or 1d)
    """
    print(f"Backfilling {days} days of {interval} data for {len(assets)} assets from Binance...")
    print("-" * 70)
    
    for asset in assets:
        try:
            print(f"\nFetching {asset}...", end=" ", flush=True)
            
            # Binance uses USDT pairs, not USD
            binance_symbol = f"{asset}USDT"
            
            # Get historical data from Binance
            klines = binance_client.get_historical_data(
                symbol=binance_symbol,
                days=days,
                interval=interval
            )
            
            if not klines:
                print("No data returned")
                continue
                
            print(f"Got {len(klines)} candles")
            
            # Store to CSV
            pair = f"{asset}USD"
            for kline in klines:
                # Store each candle as a minute bar
                # Use close price and volume
                datastore.append_minute_bar(
                    pair=pair,
                    timestamp=kline.timestamp,
                    price=kline.close,
                    volume=kline.volume
                )
            
            print(f"  Stored {len(klines)} bars for {pair}")
            
            # For minute-level data, we need to interpolate if using hourly/daily intervals
            # This is handled by the aggregation functions in DataStore
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("Backfill complete!")


def main():
    """Main entry point for backfill script."""
    parser = argparse.ArgumentParser(description="Backfill historical market data")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "LINK", "DOT", "UNI", "AVAX", "MATIC", "SHIB", "LTC", "TRX"],
        help="List of asset symbols to backfill"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backfill (default: 30, minimum recommended: 30 for ML training)"
    )
    parser.add_argument(
        "--interval",
        choices=["15m", "1h", "1d"],
        default="1h",
        help="Data interval"
    )
    
    args = parser.parse_args()
    
    # Initialize clients
    config = get_config()
    datastore = DataStore()
    binance_client = BinancePublicClient()
    
    print("=" * 70)
    print("Historical Data Backfill - Binance API")
    print("=" * 70)
    print(f"Assets: {', '.join(args.assets)}")
    print(f"Days: {args.days}")
    print(f"Interval: {args.interval}")
    print()
    
    # Backfill from Binance
    backfill_from_binance(
        binance_client,
        datastore,
        args.assets,
        args.days,
        args.interval
    )


if __name__ == "__main__":
    main()
