"""
Batch script to fetch 5-minute historical data for multiple cryptocurrencies.

Example:
    python fetch_all_cryptos_5m.py --output-dir data/historical
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Import the main fetch function
from data_collection.fetch_all_5m import main as fetch_single

# List of all cryptocurrencies to fetch
ALL_CRYPTOS = [
    "BTC", "PAXG", "APT", "TAO", "ONDO", "XLM", "AAVE", "PEPE", "BNB", "ICP",
    "XPL", "HBAR", "ENA", "ZEC", "PUMP", "LINEA", "AVNT", "POL", "SOMI", "DOT",
    "BMT", "UNI", "HEMI", "PLUME", "XRP", "TRUMP", "CRV", "CAKE", "AVAX", "SHIB",
    "SEI", "LINK", "DOGE", "FLOKI", "PENDLE", "WLD", "EIGEN", "BIO", "LTC", "OMNI",
    "ARB", "SOL", "CFX", "STO", "MIRA", "SUI", "ADA", "FIL", "FORM", "PENGU",
    "WIF", "OPEN", "NEAR", "EDEN", "LISTA", "VIRTUAL", "TRX", "TON", "WLFI", "TUT",
    "ZEN", "ETH", "FET", "SUSD", "BONK", "ASTER", "1000CHEEMS"
]


def fetch_crypto(symbol: str, output_dir: Path, delay: float = 1.0) -> bool:
    """Fetch data for a single cryptocurrency."""
    print(f"\n{'='*60}")
    print(f"Fetching data for {symbol}...")
    print(f"{'='*60}")
    
    try:
        # Import and call the main function from fetch_all_5m
        from data_collection.fetch_all_5m import main as fetch_main
        result = fetch_main(["--symbol", symbol, "--output-dir", str(output_dir)])
        
        if result == 0:
            print(f"Successfully fetched {symbol}")
            time.sleep(delay)  # Rate limiting
            return True
        else:
            print(f"Failed to fetch {symbol}")
            time.sleep(delay)
            return False
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        time.sleep(delay)
        return False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 5-minute historical data for multiple cryptocurrencies."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "historical",
        help="Directory where CSV files will be created.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Specific symbols to fetch (default: all cryptocurrencies).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between each cryptocurrency fetch (default: 2.0).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cryptocurrencies that already have CSV files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    symbols = args.symbols if args.symbols else ALL_CRYPTOS
    
    print(f"Starting batch fetch for {len(symbols)} cryptocurrencies...")
    print(f"Output directory: {args.output_dir}")
    print(f"Delay between requests: {args.delay} seconds")
    
    if args.skip_existing:
        print("Skipping existing files enabled")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    failed = []
    skipped = []
    
    for i, symbol in enumerate(symbols, 1):
        output_file = args.output_dir / f"{symbol.upper()}_5m.csv"
        
        if args.skip_existing and output_file.exists():
            print(f"\n[{i}/{len(symbols)}] Skipping {symbol} (file already exists)")
            skipped.append(symbol)
            continue
        
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        
        if fetch_crypto(symbol, args.output_dir, args.delay):
            successful.append(symbol)
        else:
            failed.append(symbol)
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH FETCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total cryptocurrencies: {len(symbols)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")
    
    if successful:
        print(f"\nSuccessful: {', '.join(successful)}")
    
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    
    if skipped:
        print(f"\nSkipped: {', '.join(skipped)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

