# Cryptocurrency Historical Data Collection

This directory contains utilities to download historical 5-minute cryptocurrency candles (including price and volume) from various APIs and store them as CSV files.

## Available Scripts

### 1. `fetch_all_5m.py` (Recommended)
**Best for maximum historical data coverage**

Combines data from multiple APIs (yfinance + CryptoCompare) to get the maximum amount of historical 5-minute data available.

```bash
cd TradingBot
python -m data_collection.fetch_all_5m --symbol BTC --output-dir data/historical
```

**Features:**
- Fetches from yfinance (last 7 days of 5-minute data)
- Fetches from CryptoCompare (last 7 days, aggregated to 5-minute)
- Automatically merges and deduplicates data
- Gets maximum coverage from free APIs

**Arguments:**
- `--symbol`: Cryptocurrency symbol, e.g., BTC, ETH, SOL (default: BTC)
- `--output-dir`: Directory for CSV output (default: `data/historical`)
- `--yfinance-days`: Days of yfinance data (default: 60, but uses 7 for safety)
- `--cryptocompare-days`: Days of CryptoCompare data (default: 7)

### 2. `fetch_yfinance_5m.py`
**Best for recent data (last 7-60 days)**

Uses yfinance to fetch real 5-minute candles. Limited to last 60 days for 5-minute intervals.

```bash
python -m data_collection.fetch_yfinance_5m --symbol BTC-USD --output-dir data/historical
```

**Arguments:**
- `--symbol`: Symbol in yfinance format, e.g., BTC-USD, ETH-USD (default: BTC-USD)
- `--start`: Start timestamp in ISO-8601 format (default: 7 days ago)
- `--end`: End timestamp in ISO-8601 format (default: now)
- `--output-dir`: Directory for CSV output (default: `data/historical`)

**Limitations:**
- yfinance 5-minute data is typically limited to last 60 days
- Requires `pip install yfinance`

### 3. `fetch_cryptocompare_5m.py`
**Best for last 7 days with minute-level precision**

Uses CryptoCompare API to fetch minute data and aggregates to 5-minute candles.

```bash
python -m data_collection.fetch_cryptocompare_5m --symbol BTC --output-dir data/historical
```

**Arguments:**
- `--symbol`: Cryptocurrency symbol, e.g., BTC, ETH (default: BTC)
- `--start`: Start timestamp in ISO-8601 format (default: 2 years ago, but API limits apply)
- `--end`: End timestamp in ISO-8601 format (default: now)
- `--output-dir`: Directory for CSV output (default: `data/historical`)
- `--request-pause`: Seconds between API requests (default: 0.5)

**Limitations:**
- CryptoCompare free tier only provides minute data for last 7 days
- For older data, uses hourly/daily data which is interpolated

### 4. `fetch_binance_5m.py`
**Best for regions where Binance API is accessible**

Downloads historical 5-minute candles from Binance spot API. **Note:** May be restricted in some regions (HTTP 451 error).

```bash
python -m data_collection.fetch_binance_5m --symbol BTCUSDT --output-dir data/historical
```

**Arguments:**
- `--symbol`: Trading pair symbol, e.g., BTCUSDT (default: BTCUSDT)
- `--interval`: Candle interval (default: 5m)
- `--start`: Start timestamp in ISO-8601 format (default: Binance inception)
- `--end`: End timestamp in ISO-8601 format (default: now)
- `--output-dir`: Directory for CSV output (default: `data/historical`)
- `--request-pause`: Seconds between API requests (default: 0.25)
- `--batch-limit`: Candles per API request (default: 1000, max)

**Limitations:**
- May be blocked in some regions (geographic restrictions)
- Requires stable internet connection

### 5. `fetch_coingecko_5m.py`
**For very long-term historical data (interpolated)**

Uses CoinGecko API for historical data. Note: CoinGecko doesn't provide true 5-minute data, so this interpolates from hourly/daily data.

```bash
python -m data_collection.fetch_coingecko_5m --symbol bitcoin --output-dir data/historical
```

**Arguments:**
- `--symbol`: CoinGecko coin ID, e.g., bitcoin, ethereum (default: bitcoin)
- `--days`: Number of days of historical data (default: 365)
- `--output-dir`: Directory for CSV output (default: `data/historical`)
- `--request-pause`: Seconds between API requests (default: 1.0)

**Limitations:**
- Data is interpolated, not true 5-minute candles
- Free tier has rate limits (10-50 calls/minute)

## CSV Output Format

All scripts output CSV files with the following columns:
- `symbol`: Cryptocurrency symbol
- `open_time`: Opening timestamp (ISO-8601 UTC)
- `open`: Opening price
- `high`: Highest price in the interval
- `low`: Lowest price in the interval
- `close`: Closing price
- `volume`: Trading volume (base currency)
- `close_time`: Closing timestamp (ISO-8601 UTC)
- Additional columns may vary by script (e.g., `quote_asset_volume`, `number_of_trades`)

## Recommendations

1. **For maximum recent data (last 7 days):** Use `fetch_all_5m.py` - it combines multiple sources
2. **For regions with Binance access:** Use `fetch_binance_5m.py` for the most historical data
3. **For very long-term data:** Consider using paid APIs or data providers for true 5-minute historical data

## Installation

Required packages:
```bash
pip install requests yfinance
```

## Notes

- Free APIs have limitations on historical 5-minute data availability
- True 5-minute historical data going back years typically requires paid API services
- All timestamps are in UTC (ISO-8601 format)
- Scripts automatically create output directories if they don't exist
