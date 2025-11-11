# Quick Start Guide

## Installation

```bash
cd TradingBot/orb_scalp
pip install -r requirements.txt
```

## Run Your First Backtest

```bash
# Basic backtest with default config
python run_backtest.py --days 15 --symbols BTCUSD ETHUSD

# With custom capital
python run_backtest.py --days 30 --symbols BTCUSD ETHUSD SOLUSD --capital 50000

# Use all available data
python run_backtest.py --symbols BTCUSD ETHUSD
```

## Understanding the Output

After running, check the `results/` directory:

- **trades.csv**: Every trade with entry/exit, P&L, R-multiples
- **equity_curve.csv**: Equity over time
- **metrics.json**: All performance metrics
- **equity_curve.png**: Visual chart

## Key Metrics to Watch

- **Sharpe Ratio**: Should be ≥ 1.3 for good risk-adjusted returns
- **Max Drawdown**: Should be < 6% for acceptable risk
- **Profit Factor**: Should be ≥ 1.2 (profits > losses)
- **Win Rate**: 40-60% is typical for trend-following
- **Average R**: Should be positive (average trade makes money)

## Customizing Strategy

Edit `config.yaml` to adjust:

1. **Entry Filters**: RSI thresholds, volume multipliers
2. **Risk Settings**: Position sizing, daily limits
3. **Stop/Target**: ATR multipliers, percentage stops
4. **Trading Hours**: UTC hours to trade

## Troubleshooting

### No Trades Generated

- Check if data has enough bars (need 200+ for indicators)
- Verify trading hours match your data timezone
- Lower RSI thresholds if too strict
- Check volume spike multiplier (may be too high)

### Poor Performance

- Increase stop distance (ATR multiplier)
- Tighten entry filters (higher RSI thresholds)
- Reduce position sizing (lower risk per trade)
- Check if fees/slippage are too high

### Errors

- Ensure data files exist in `../data2/data/`
- Check CSV format: `timestamp,price,volume`
- Verify timestamps are in milliseconds (UTC)

## Next Steps

1. Run backtest on your data
2. Review metrics and trades
3. Adjust config parameters
4. Re-run and compare results
5. Implement walk-forward testing for robustness



