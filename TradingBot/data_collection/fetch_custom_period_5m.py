"""
Fetch 5-minute historical data for all trading cryptocurrencies for a custom 15-day period.

This script fetches data for all symbols defined in the trading bot's COEFFICIENTS,
excluding EXCLUDED_SYMBOLS, for a user-specified time period.

Example:
    python fetch_custom_period_5m.py --start-date 2025-01-01 --days 15 --output-dir data/historical
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

# Import symbols from the trading bot
sys.path.insert(0, str(Path(__file__).parent.parent))
from multi_asset_linear_momentum_trading import COEFFICIENTS, EXCLUDED_SYMBOLS

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Symbol mapping: Our symbols to Binance symbols
SYMBOL_MAPPING = {
    "SHIB": "SHIBUSDT",
    "PEPE": "PEPEUSDT",
    "FLOKI": "FLOKIUSDT",
    "PUMP": "PUMPUSDT",
    "LINEA": "LINEAUSDT",
    "BMT": "BMTUSDT",
    "HEMI": "HEMIUSDT",
    "PLUME": "PLUMEUSDT",
    "BIO": "BIOUSDT",
    "DOGE": "DOGEUSDT",
    "HBAR": "HBARUSDT",
    "SEI": "SEIUSDT",
    "POL": "POLUSDT",
    "XLM": "XLMUSDT",
    "ENA": "ENAUSDT",
    "CRV": "CRVUSDT",
    "XPL": "XPLUSDT",
    "SOMI": "SOMIUSDT",
    "ONDO": "ONDOUSDT",
    "AVNT": "AVNTUSDT",
    "WLD": "WLDUSDT",
    "EIGEN": "EIGENUSDT",
    "XRP": "XRPUSDT",
    "CAKE": "CAKEUSDT",
    "PENDLE": "PENDLEUSDT",
    "DOT": "DOTUSDT",
    "APT": "APTUSDT",
    "LINK": "LINKUSDT",
    "AVAX": "AVAXUSDT",
    "ICP": "ICPUSDT",
    "UNI": "UNIUSDT",
    "OMNI": "OMNIUSDT",
    "LTC": "LTCUSDT",
    "AAVE": "AAVEUSDT",
    "BNB": "BNBUSDT",
}


def fetch_binance_klines(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    session: requests.Session,
) -> List[dict]:
    """Fetch 5-minute klines from Binance for a specific time period."""
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    all_klines = []
    current_start = start_ms
    limit = 1000  # Binance max per request
    
    print(f"  Fetching {symbol} from {start_time.isoformat()} to {end_time.isoformat()}...")
    
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": current_start,
            "limit": limit,
        }
        
        try:
            response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
            
            if response.status_code == 451:
                print(f"    [WARN] Binance API restricted in your region for {symbol}")
                return []
            
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
            
            # Convert to our format
            for k in klines:
                open_time_ms = k[0]
                close_time_ms = k[6]
                
                if open_time_ms >= end_ms:
                    break
                
                if open_time_ms < start_ms:
                    continue
                
                candle = {
                    "open_time": datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc),
                }
                all_klines.append(candle)
            
            # Update for next iteration
            if klines:
                last_time = klines[-1][0]
                if last_time >= current_start:
                    current_start = last_time + (5 * 60 * 1000)  # Next 5-minute interval
                else:
                    break
            else:
                break
            
            # Rate limiting
            time.sleep(0.2)
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                # Symbol might not exist on Binance
                print(f"    [WARN] Symbol {symbol} not available on Binance: {e}")
                return []
            raise
        except Exception as e:
            print(f"    [ERROR] Failed to fetch {symbol}: {e}")
            return []
    
    print(f"    Fetched {len(all_klines)} candles")
    return all_klines


def write_csv(candles: List[dict], dest: Path, symbol: str) -> int:
    """Write candles to CSV file in the same format as existing data."""
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
        description="Fetch 5-minute historical data for all trading cryptocurrencies for a custom time period."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format (e.g., 2025-01-01)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=15,
        help="Number of days to fetch (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("TradingBot") / "data" / "historical",
        help="Directory where CSV files will be created (default: TradingBot/data/historical)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Fetch data for a single symbol only (for testing)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Parse start date
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"Error: Invalid date format. Use YYYY-MM-DD (e.g., 2025-01-01)", file=sys.stderr)
        return 1
    
    end_date = start_date + timedelta(days=args.days)
    
    print("=" * 60)
    print("FETCHING CUSTOM PERIOD DATA")
    print("=" * 60)
    print(f"Start Date: {start_date.date()}")
    print(f"End Date: {end_date.date()}")
    print(f"Duration: {args.days} days")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)
    
    # Get symbols to fetch
    if args.symbol:
        symbols_to_fetch = [args.symbol.upper()]
    else:
        symbols_to_fetch = sorted(
            symbol for symbol in COEFFICIENTS.keys()
            if symbol not in EXCLUDED_SYMBOLS
        )
    
    print(f"\nFetching data for {len(symbols_to_fetch)} symbols:")
    print(", ".join(symbols_to_fetch))
    print()
    
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "trading-bot-data-fetcher/1.0"
    })
    
    successful = 0
    failed = 0
    
    for symbol in symbols_to_fetch:
        binance_symbol = SYMBOL_MAPPING.get(symbol)
        if not binance_symbol:
            print(f"[SKIP] {symbol}: No Binance mapping available")
            failed += 1
            continue
        
        try:
            candles = fetch_binance_klines(binance_symbol, start_date, end_date, session)
            
            if candles:
                # Filter to exact date range
                filtered = [
                    c for c in candles
                    if start_date <= c["open_time"] < end_date
                ]
                
                if filtered:
                    output_file = args.output_dir / f"{symbol}_5m.csv"
                    count = write_csv(filtered, output_file, symbol)
                    print(f"  [OK] {symbol}: Wrote {count} candles to {output_file}")
                    successful += 1
                else:
                    print(f"  [WARN] {symbol}: No data in specified date range")
                    failed += 1
            else:
                print(f"  [WARN] {symbol}: No data fetched")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            failed += 1
        
        # Rate limiting between symbols
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(symbols_to_fetch)}")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

