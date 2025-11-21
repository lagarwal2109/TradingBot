# Roostoo Trading Bot - Enhanced Breakout Strategy

A sophisticated Python 3.10 trading bot that uses **Horus API for market data** and **Roostoo API for trading execution**. Implements a multi-timeframe trend-following breakout strategy with volume confirmation, optimized for the HK University Web3 Quant Hackathon.

## Trading Strategy

The bot uses an **Enhanced Trend-Following Breakout Strategy** that combines:
- **Multi-timeframe Analysis**: Uses 24-hour (daily) trends for direction, 4-hour for confirmation, and 1-hour for entry timing
- **Volume Confirmation**: All breakouts must be confirmed by 2x average volume
- **Support/Resistance Levels**: Identifies key levels from 7 days of historical data
- **Momentum Filters**: Only trades in the direction of prevailing momentum
- **Advanced Risk Management**: Trailing stops, position sizing based on signal quality

### Strategy Modes

1. **Enhanced Mode** (Default - Recommended): Multi-timeframe breakout strategy with volume
2. **Sharpe Mode**: Legacy rolling Sharpe ratio strategy (4-hour window)
3. **Tangency Mode**: Markowitz portfolio optimization approach

## Dual API Architecture

The bot uses two APIs working together:
- **🔵 Horus API**: Market data (prices, volumes, order book, historical data)
- **🟢 Roostoo API**: Trading execution (orders, balances, account management)

This separation provides better data quality, faster execution, and improved reliability.

## Key Features

- **Dual API Integration**: Horus for data + Roostoo for trading
- **Multi-Timeframe Trend Analysis**: Combines daily, 4-hour, and hourly timeframes
- **Volume-Based Breakout Detection**: Confirms breakouts with volume surge (2x average)
- **Support/Resistance Recognition**: Automatically identifies key price levels
- **Dynamic Position Sizing**: Scales position size based on signal quality and volatility
- **Trailing Stop Loss**: Locks in profits as trades move favorably
- **Risk Management**: Stop loss, take profit, and position limits
- **Data Collection**: Continuous minute bar data with volume tracking from Horus
- **Backtesting**: Comprehensive backtesting with performance metrics
- **Kill Switch**: Automatic shutdown after consecutive errors
- **One Trade Per Minute**: Strictly enforces exchange limits 

## Performance Metrics

The bot calculates and optimizes for the following metrics:
- **Sharpe Ratio**: Risk-adjusted returns (no annualization)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annualized return / Maximum Drawdown
- **Competition Score**: 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar

## Installation

### Prerequisites

- Ubuntu/Debian Linux (for deployment scripts)
- Python 3.10 or higher
- Roostoo API credentials

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/roostoo-sharpe-bot.git
cd roostoo-sharpe-bot
```

2. Run the installation script:
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

3. Configure API credentials:
```bash
# Edit .env file
nano .env
```

Add your credentials:
```
ROOSTOO_API_KEY=your_api_key_here
ROOSTOO_API_SECRET=your_api_secret_here
ROOSTOO_BASE_URL=https://api.roostoo.com
```

### Manual Setup

1. Create virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data logs figures
```

4. Create `.env` file with your API credentials (see above)

## Usage

### Data Collection

Before trading, collect historical minute bar data:

```bash
# Start the data collector
python run.py --mode collect

# Or use the service script
./scripts/start-collector.sh
```

The collector runs continuously and saves minute bar data to `data/<PAIR>.csv` files.

### Backtesting

Test your strategy on historical data:

```bash
# Basic backtest
python backtest.py

# With custom parameters
python backtest.py --mode sharpe --window 240 --momentum 20 --max-position 0.4

# Backtest specific date range
python backtest.py --start-date 2024-01-01 --end-date 2024-12-31

# Test tangency portfolio mode
python backtest.py --mode tangency --trade-frequency 60
```

Backtest outputs:
- Performance metrics in terminal
- `figures/equity_curve.png` - Portfolio value over time
- `figures/returns_analysis.png` - Returns distribution
- `figures/backtest_results.csv` - Detailed results
- `figures/backtest_metrics.json` - Performance metrics

### Live Trading

Run the trading bot:

```bash
# Run with enhanced breakout strategy (RECOMMENDED)
python run.py --mode enhanced

# Run with legacy Sharpe strategy
python run.py --mode sharpe

# Run with tangency portfolio mode
python run.py --mode tangency

# Run once (for testing)
python run.py --mode enhanced --once

# Custom parameters
python run.py --max-position 0.3
```

### Deployment (Ubuntu/Systemd)

Use the provided scripts for production deployment:

```bash
# Start trading bot (runs every minute via systemd timer)
./scripts/start-bot.sh

# Stop trading bot
./scripts/stop-bot.sh

# Check status
./scripts/status.sh

# View logs
./scripts/logs.sh
./scripts/logs.sh -f  # Follow mode
```

## Configuration

### Environment Variables (.env)

**Roostoo API (Trading):**
- `ROOSTOO_API_KEY`: Your Roostoo API key for trading
- `ROOSTOO_API_SECRET`: Your Roostoo API secret
- `ROOSTOO_BASE_URL`: Roostoo API base URL (default: https://api.roostoo.com)

**Horus API (Market Data):**
- `HORUS_API_KEY`: Your Horus API key for market data (already included in template)
- `HORUS_BASE_URL`: Horus API base URL (default: https://api.horus.com)
- `USE_HORUS_DATA`: Enable Horus data collection (default: true)

### Trading Parameters

The enhanced strategy uses optimized parameters based on research:

**Multi-Timeframe Windows:**
- Long-term trend: 1440 minutes (24 hours)
- Short-term trend: 240 minutes (4 hours)  
- Entry timing: 60 minutes (1 hour)
- Volume analysis: 480 minutes (8 hours)

**Breakout Detection:**
- Support/Resistance lookback: 7 days
- Breakout threshold: 2% beyond level
- Volume confirmation: 2x average volume

**Risk Management:**
- Maximum position: 40% of equity (scales with signal quality)
- Stop loss: 2% (adjustable per trade)
- Take profit: 5% (adjustable per trade)
- Trailing stop: 2% from peak

Command-line arguments:
- `--mode`: Strategy mode (enhanced/sharpe/tangency)
- `--max-position`: Maximum position size as fraction of equity
- `--once`: Run single cycle for testing

### Risk Management

- **Position Sizing**: Inversely proportional to volatility (size ∝ 1/σ)
- **Maximum Position**: Capped at 40% of total equity by default
- **Minimum Order**: Enforces exchange minimum order requirements
- **Precision Handling**: Rounds amounts and prices to exchange-specified precision

## Architecture

### Core Components

- **bot/config.py**: Configuration management and environment variables
- **bot/roostoo.py**: API client with HMAC-SHA256 authentication
- **bot/datastore.py**: Minute bar persistence and state management
- **bot/signals.py**: Feature computation (Sharpe, momentum, volatility)
- **bot/risk.py**: Position sizing and risk constraints
- **bot/engine.py**: Trading logic and rebalancing engine
- **run.py**: Main execution loop with safe startup
- **backtest.py**: Historical strategy testing

### Data Storage

- **data/<PAIR>.csv**: Minute bar data (timestamp, price)
- **data/state.json**: Current position and trading state
- **logs/bot.log**: JSON-formatted log entries

### Trading Logic

1. **Signal Generation**:
   - Compute 240-minute rolling Sharpe ratio for each pair
   - Calculate 20-minute momentum
   - Apply liquidity filtering

2. **Position Selection**:
   - Sharpe Mode: Select highest Sharpe ratio asset where Sharpe > 0 and momentum > 0
   - Tangency Mode: Compute optimal long-only weights using Markowitz optimization

3. **Execution**:
   - Check one-minute trade cooldown
   - Close current position if changing assets
   - Open new position with volatility-adjusted sizing
   - Record trade and update state

## Safety Features

- **Kill Switch**: Automatic shutdown after 5 consecutive errors
- **Order Timeout**: Cancels orders older than 5 minutes
- **Safe Startup**: Reconciles state with actual balances on start
- **Trade Throttling**: Enforces one trade per minute limit
- **Error Recovery**: Automatic retry with exponential backoff

## Monitoring

### Log Files

- **logs/bot.log**: Main application logs (JSON format)
- **logs/systemd.log**: Systemd service logs
- **logs/collector.log**: Data collector logs

### Health Checks

```bash
# Check if bot is running
./scripts/status.sh

# Monitor real-time logs
tail -f logs/bot.log

# Check systemd service
sudo systemctl status roostoo-bot.timer
```

## Development

### Running Tests

```bash
# Run unit tests (if implemented)
python -m pytest tests/

# Run a single backtest iteration
python backtest.py --trade-frequency 1 --end-date 2024-01-02
```

### Adding New Strategies

1. Extend `SignalGenerator` class in `bot/signals.py`
2. Implement feature computation methods
3. Add strategy selection logic in `bot/engine.py`
4. Update command-line arguments in `run.py`

## Troubleshooting

### Common Issues

1. **API Authentication Errors**:
   - Check API credentials in `.env`
   - Ensure API secret is correct
   - Verify timestamp synchronization

2. **Insufficient Data**:
   - Run collector for at least 240 minutes before trading
   - Check `data/` directory for CSV files

3. **Order Failures**:
   - Check minimum order requirements
   - Verify sufficient balance
   - Review precision settings

4. **Performance Issues**:
   - Reduce window size for faster computation
   - Increase trade frequency for less frequent rebalancing
   - Check system resources

### Debug Mode

```bash
# Run with debug logging
python run.py --once  # Single cycle for debugging
```

## Competition Notes

The bot is optimized for the HK University Web3 Quant Hackathon scoring:
- Composite Score = 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar
- Focuses on risk-adjusted returns over absolute returns
- Implements strict risk management to minimize drawdowns

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `logs/` directory
3. Ensure all dependencies are correctly installed
4. Verify API connectivity and credentials