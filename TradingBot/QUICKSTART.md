# Quick Start Guide - Enhanced Breakout Strategy

## 1. Setup (Windows Development)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Create .env file from template
copy env.template .env

# Then edit .env and add your Roostoo trading credentials
# Horus API key is already included in the template
```

## 2. Collect Data First

```powershell
# Run data collector for at least 4 hours (240 minutes)
python run.py --mode collect
```

## 3. Backtest Strategy

```powershell
# Basic backtest with enhanced strategy (RECOMMENDED)
python backtest.py --mode enhanced

# Legacy Sharpe backtest
python backtest.py --mode sharpe --window 240 --momentum 20 --max-position 0.4
```

## 4. Run Live Trading

```powershell
# Test with one cycle using enhanced strategy
python run.py --mode enhanced --once

# Run continuously with enhanced strategy (RECOMMENDED)
python run.py --mode enhanced

# Run with legacy Sharpe strategy
python run.py --mode sharpe
```

## 5. Ubuntu/Linux Production Deployment

```bash
# SSH to your Ubuntu server and clone the repo
git clone <your-repo-url>
cd roostoo-sharpe-bot

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh

# Create .env from template
cp env.template .env

# Edit .env with your Roostoo API credentials (Horus key already included)
nano .env

# Start data collector
./scripts/start-collector.sh

# After collecting data, start trading bot
./scripts/start-bot.sh

# Monitor status
./scripts/status.sh
./scripts/logs.sh -f
```

## Dual API Setup

The bot uses **two APIs** working together:
- **Horus API**: Market data (prices, volumes) - Key already in template
- **Roostoo API**: Trading execution (orders, balances) - You need to add yours

This provides better data quality and faster execution.

## Key Files to Edit

1. **.env** - Add your Roostoo API credentials (Horus key already included)
2. **bot/config.py** - Adjust trading parameters if needed
3. **run.py** - Main entry point with command-line options
4. **backtest.py** - Test strategies on historical data

See `DUAL_API_INTEGRATION.md` for detailed API documentation.

## Enhanced Strategy Overview

The bot now uses a sophisticated **multi-timeframe breakout strategy** based on extensive research:

1. **Trend Analysis**: 
   - Daily (24h) for overall direction
   - 4-hour for trend confirmation  
   - 1-hour for entry timing

2. **Breakout Detection**:
   - Identifies support/resistance from 7 days of data
   - Requires 2% move beyond level
   - Must have 2x average volume

3. **Risk Management**:
   - 2% stop loss
   - 5% take profit
   - 2% trailing stop
   - Position sizing based on signal quality

## Legacy Parameters (Sharpe Mode)

- **Window Size**: 240 minutes (4 hours) of historical data
- **Momentum Lookback**: 20 minutes for trend detection
- **Max Position**: 40% of total equity
- **Min Sharpe**: 0.0 (only positive Sharpe ratio)
- **Trade Frequency**: 1 trade per minute maximum

## Important Notes

1. **Collect sufficient data first** - Enhanced strategy needs 7+ days of historical data
2. **Start with backtesting** - Test your strategy before going live
3. **Monitor logs** - Check logs/bot.log for detailed information
4. **Use --once flag** - Test single cycles during development
5. **API Rate Limits** - The bot respects the 1 trade/minute limit
6. **Enhanced mode is recommended** - Uses proven breakout strategy with volume confirmation
