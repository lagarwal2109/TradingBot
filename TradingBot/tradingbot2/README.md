# MACD Rules-Based Trading System

A rules-based trading system using MACD (Moving Average Convergence Divergence) indicators with trend filtering and strict entry/exit rules.

## Strategy Overview

### MACD Configuration
- **Fast EMA**: 12 days (12 * 1440 minutes)
- **Slow EMA**: 26 days (26 * 1440 minutes)
- **Signal Line**: 9-day EMA of MACD line
- **Histogram**: MACD - Signal (indicates momentum)

### Entry Rules

#### Long Positions
- MACD crosses **above** Signal line
- Both MACD and Signal are **below zero**
- Price is **above** 200-day Moving Average (uptrend)

#### Short Positions
- MACD crosses **below** Signal line
- MACD is **above zero**
- Price is **below** 200-day Moving Average (downtrend)

### Exit Rules
- **Stop Loss**: Price reaches 200-day Moving Average
- **Take Profit**: 1.5x entry price (e.g., entry $100 → exit $150)

### Trading Constraints
- Maximum 5 trades per minute
- Transaction fee: 0.1% per trade
- Maximum position size: 40% of equity
- Maximum simultaneous positions: 3

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure parent TradingBot has `.env` file with API credentials:
```
ROOSTOO_API_KEY=your_key
ROOSTOO_API_SECRET=your_secret
ROOSTOO_BASE_URL=https://mock-api.roostoo.com
```

3. Ensure historical data is available in `../data/data/` directory (from parent TradingBot)

## Usage

### Run Backtest

```bash
python scripts/run_backtest.py --days 15 --pairs BTCUSD ETHUSD
```

Options:
- `--days`: Number of days to backtest (default: 15)
- `--pairs`: Trading pairs to test (default: all available with sufficient data)
- `--output-dir`: Output directory for results (default: figures)
- `--capital`: Initial capital (default: 10000)

### Output Files

After running a backtest, the following files are generated in the output directory:

- `macd_backtest_trades.csv`: Detailed trade log with entry/exit prices, P&L, fees
- `macd_backtest_equity.csv`: Equity curve over time
- `macd_backtest_equity_curve.png`: Visualization of equity curve
- `macd_backtest_summary.json`: Summary metrics (return, Sharpe, drawdown, etc.)

## Project Structure

```
tradingbot2/
├── bot/
│   ├── config.py          # Configuration management
│   ├── macd_strategy.py   # MACD signal generation
│   ├── backtester.py      # Backtesting engine
│   └── utils.py           # Utility functions
├── scripts/
│   └── run_backtest.py    # CLI for backtesting
├── data/                  # Market data (symlinked to parent)
├── figures/               # Backtest results
└── requirements.txt
```

## Strategy Logic

### MACD Calculation
1. Calculate 12-day EMA (fast) and 26-day EMA (slow) of prices
2. MACD Line = Fast EMA - Slow EMA
3. Signal Line = 9-day EMA of MACD Line
4. Histogram = MACD - Signal

### Signal Generation
- **Green Histogram** (MACD > Signal): Bullish momentum
- **Red Histogram** (MACD < Signal): Bearish momentum
- **Growing Histogram**: Increasing momentum
- **Decreasing Histogram**: Weakening momentum

### Trend Filter
- 200-day Moving Average determines market trend
- Only trade longs in uptrends (price > 200-day MA)
- Only trade shorts in downtrends (price < 200-day MA)

## Performance Metrics

The backtest calculates:
- Total return (%)
- Sharpe ratio (annualized)
- Maximum drawdown (%)
- Win rate (%)
- Average win/loss
- Total fees paid
- Number of trades

## Notes

- The system uses minute-bar data from the parent TradingBot data directory
- All calculations are done in minutes (e.g., 12 days = 12 * 1440 minutes)
- The system respects rate limits (5 trades/minute) and transaction costs (0.1%)
- Positions are automatically closed at stop loss or take profit levels


