# ORB + Scalp Trading System

A complete trading system implementing Opening Range Breakout (ORB) and EMA(5/8) scalping strategies on close-only 1-minute data.

## Features

- **ORB Strategy**: Breakout from opening range with volume confirmation
- **Scalp Strategy**: EMA(5/8) crossovers with RSI exhaustion filters
- **Risk Management**: Position sizing, trailing stops, daily kill-switch
- **Realistic Execution**: Fees, slippage, and latency modeling
- **Comprehensive Metrics**: Sharpe, Sortino, Calmar, Profit Factor, R-multiples
- **Walk-Forward Testing**: Built-in support for parameter optimization

## Installation

```bash
cd TradingBot/orb_scalp
pip install -r requirements.txt
```

## Usage

### Basic Backtest

```bash
python run_backtest.py --days 30 --symbols BTCUSD ETHUSD
```

### With Custom Config

```bash
python run_backtest.py --config config.yaml --data-dir ../data2/data --capital 10000
```

### Options

- `--config`: Path to config YAML file (default: config.yaml)
- `--data-dir`: Directory with CSV data files (default: ../data2/data)
- `--output-dir`: Output directory for results (default: results)
- `--capital`: Initial capital (default: 10000)
- `--days`: Number of days to backtest (default: use all data)
- `--symbols`: Symbols to trade (default: from config)

## Data Format

Input CSV files should have:
- `timestamp`: Milliseconds since epoch (UTC)
- `price`: Close price
- `volume`: Volume (optional, defaults to 0)

Example:
```csv
timestamp,price,volume
1760213340000,230.67,228.362
1760213400000,230.44,111.886
```

## Configuration

Edit `config.yaml` to adjust:
- Strategy parameters (RSI thresholds, ATR multipliers, etc.)
- Risk settings (position sizing, daily limits)
- Fee and slippage models
- Trading hours and filters

## Output Files

After running a backtest:
- `trades.csv`: Detailed trade log
- `equity_curve.csv`: Equity over time
- `metrics.json`: Performance metrics
- `equity_curve.png`: Visualization

## Strategy Details

### ORB (Opening Range Breakout)

- Entry: Price breaks above/below opening range (first 15 minutes)
- Filters: Volume spike (1.2× median), RSI confirmation
- Stop: 0.8× ATR (min 0.5%)
- Target: 1.5R
- Timeout: 6 hours

### Scalp (EMA 5/8)

- Entry: EMA(5) > EMA(8) for longs, opposite for shorts
- Filters: RSI not exhausted, realized volatility check
- Stop: 0.6× ATR (min 0.5%)
- Target: 1.2%
- Trailing: 0.4% after breakeven

## Risk Management

- Position sizing: 0.75% risk for ORB, 0.5% for scalps
- Daily kill-switch: Stop trading if down 3% from day start
- Max positions: 4 concurrent, 2 per symbol
- Notional exposure: 120% of equity max

## Performance Targets

- Return: 12-20% over 15 days
- Max Drawdown: < 6%
- Sharpe: ≥ 1.3
- Profit Factor: ≥ 1.2



