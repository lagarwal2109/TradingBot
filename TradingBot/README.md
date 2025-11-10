# Regime-Adaptive Ensemble Trading Bot

A sophisticated Python 3.10+ trading bot that uses **machine learning ensemble models** with **regime detection** to predict cryptocurrency price movements. Optimized for the HK University Web3 Quant Hackathon, focusing on maximizing Sharpe, Sortino, and Calmar ratios.

## Trading Strategy

The bot implements a **Regime-Adaptive Ensemble Strategy** that combines:

### Core Components

1. **Regime Detection**:
   - **GMM (Gaussian Mixture Model)**: Detects microstructure regimes (calm vs volatile)
   - **HMM (Hidden Markov Model)**: Detects trend regimes (bearish, neutral, bullish)
   - **Regime Fusion**: Combines both regime signals for adaptive trading

2. **Ensemble Machine Learning**:
   - **Stacked Ensemble**: Combines XGBoost, LightGBM, and RandomForest models
   - **Feature Engineering**: 100+ technical indicators, volume features, volatility metrics, multi-timeframe returns
   - **Regime-Aware Features**: Incorporates regime probabilities into predictions

3. **Portfolio Optimization**:
   - **Minimum Variance Frontier**: Optimizes portfolio weights to maximize Sharpe, Sortino, and Calmar ratios
   - **Covariance Matrix**: Full correlation analysis across all trading pairs
   - **Downside Risk**: Separate downside covariance for Sortino optimization

4. **Risk Management**:
   - **Adaptive Position Sizing**: Based on confidence, regime, and volatility
   - **Stop Loss & Take Profit**: Regime-adaptive thresholds
   - **Portfolio Limits**: Maximum simultaneous positions and allocation limits
   - **Long & Short Trading**: Can profit from both upward and downward movements

## Key Features

- **Machine Learning Ensemble**: Stacked XGBoost, LightGBM, RandomForest models
- **Regime Detection**: GMM for microstructure, HMM for trend detection
- **Portfolio Optimization**: Minimum variance frontier for optimal capital allocation
- **Feature Engineering**: 100+ technical and statistical features
- **Multi-Timeframe Analysis**: Combines multiple timeframes for robust signals
- **Backtesting**: Comprehensive backtesting with performance metrics
- **Parameter Optimization**: Optuna-based hyperparameter tuning
- **Competition Scoring**: Optimized for 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar

## Performance Metrics

The bot calculates and optimizes for the following metrics:
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Sortino Ratio**: Downside risk-adjusted returns (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annualized return / Maximum Drawdown
- **Competition Score**: 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar

## Installation

### Prerequisites

- Python 3.10 or higher
- Windows, Linux, or macOS
- Historical price data in CSV format (minute bars)

### Quick Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd TradingBot
```

2. Create virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data logs figures models
```

5. Prepare your data:
   - Place CSV files in `data/` directory
   - Format: `{SYMBOL}USD.csv` (e.g., `BTCUSD.csv`, `ETHUSD.csv`)
   - Required columns: `timestamp` (index), `price`, `volume` (optional)
   - Example:
     ```csv
     timestamp,price,volume
     2024-10-26 08:00:00,50000.0,100.5
     2024-10-26 08:01:00,50010.0,95.2
     ```

## Usage

### Step 1: Train Regime Detection Models

First, train GMM and HMM models for regime detection:

```bash
# Train GMM models (microstructure: calm/volatile)
python scripts/train_regime_models.py --mode gmm --days 30

# Train HMM models (trend: bearish/neutral/bullish)
python scripts/train_regime_models.py --mode hmm --days 30

# Train both
python scripts/train_regime_models.py --mode both --days 30
```

**Output**: Models saved to `models/` directory (e.g., `gmm_regime_BTCUSD_v1.0.0.pkl`)

### Step 2: Train Ensemble Model

Train the stacked ensemble model for price prediction:

```bash
# Train ensemble on historical data
python scripts/train_ensemble.py --days 30 --forward-window 180

# Options:
# --days: Number of days of historical data to use (default: 30)
# --forward-window: Prediction window in minutes (default: 180)
# --pairs: Comma-separated list of pairs (default: all pairs with data)
```

**Output**: Ensemble model saved to `models/ensemble_stacked_v1.0.0.pkl`

### Step 3: Backtest Strategy

Test the strategy on historical data:

```bash
# Basic backtest (uses optimized parameters if available)
python backtest_regime_ensemble.py --days 15

# Backtest with custom parameters
python backtest_regime_ensemble.py --days 15 --no-optimized-params

# Backtest with specific parameter overrides
python backtest_regime_ensemble.py --days 15 --min-confidence 0.3 --trade-interval 5
```

**Output**:
- Performance metrics in terminal
- `figures/backtest_regime_ensemble.json` - Detailed results
- Console output with Sharpe, Sortino, Calmar, and Competition Score

### Step 4: Optimize Parameters (Optional)

Use Optuna to find optimal strategy parameters:

```bash
# Run optimization (may take 1-2 hours)
python scripts/optimize_competition_strategy.py --trials 50 --days 15

# Options:
# --trials: Number of optimization trials (default: 50)
# --days: Days of data for backtesting (default: 15)
```

**Output**: `figures/optimized_strategy_params.json` - Best parameters for competition score

### Step 5: Live Trading

Run the bot for live trading:

```bash
# Run with regime ensemble strategy (RECOMMENDED)
python run.py --mode regime_ensemble

# Run once for testing
python run.py --mode regime_ensemble --once

# Run with custom parameters
python run.py --mode regime_ensemble --max-position 0.3
```

## Project Structure

```
TradingBot/
├── bot/
│   ├── models/
│   │   ├── ensemble_model.py          # Stacked ensemble (XGBoost, LightGBM, RF)
│   │   ├── regime_detection.py        # GMM and HMM regime detectors
│   │   ├── feature_engineering.py      # 100+ technical indicators
│   │   ├── portfolio_optimizer.py      # Minimum variance frontier optimization
│   │   └── model_storage.py            # Model persistence
│   ├── portfolio_manager.py           # Adaptive position sizing
│   ├── engine_regime_ensemble.py       # Main trading engine
│   ├── datastore.py                    # Data loading and management
│   └── config.py                       # Configuration management
├── scripts/
│   ├── train_regime_models.py          # Train GMM/HMM models
│   ├── train_ensemble.py               # Train ensemble model
│   ├── optimize_competition_strategy.py # Optuna parameter optimization
│   └── validate_models.py             # Model validation
├── backtest_regime_ensemble.py         # Main backtesting script
├── run.py                              # Live trading entry point
├── data/                               # CSV price data files
├── models/                             # Trained model files (.pkl)
├── figures/                            # Backtest results and plots
└── requirements.txt                    # Python dependencies
```

## Configuration

### Data Format

CSV files in `data/` directory must have:
- **Index**: `timestamp` (datetime)
- **Columns**: `price` (required), `volume` (optional)
- **Format**: `{SYMBOL}USD.csv` (e.g., `BTCUSD.csv`)

### Trading Parameters

Key parameters (can be optimized):

- **Signal Generation**:
  - `min_confidence`: Minimum confidence threshold (default: 0.20)
  - `high_confidence_threshold`: High confidence threshold (default: 0.45)
  - `momentum_threshold`: Momentum filter threshold (default: 0.0001)

- **Position Sizing**:
  - `base_position_pct`: Base position size (default: 0.25)
  - `calm_multiplier`: Position multiplier in calm regime (default: 1.1)
  - `volatile_multiplier`: Position multiplier in volatile regime (default: 0.9)
  - `bullish_multiplier`: Position multiplier in bullish trend (default: 1.3)
  - `bearish_multiplier`: Position multiplier in bearish trend (default: 0.3)

- **Risk Management**:
  - `base_stop_loss_pct`: Base stop loss percentage (default: 0.02)
  - `base_take_profit_pct`: Base take profit percentage (default: 0.03)
  - `max_simultaneous_positions`: Maximum concurrent positions (default: 5)
  - `max_portfolio_allocation`: Maximum capital allocation (default: 0.9)

- **Trading Frequency**:
  - `trade_interval_minutes`: Minutes between trade evaluations (default: 5)

### Environment Variables (.env)

Optional configuration (for API integration):

```env
# Roostoo API (if using live trading)
ROOSTOO_API_KEY=your_api_key_here
ROOSTOO_API_SECRET=your_api_secret_here
ROOSTOO_BASE_URL=https://api.roostoo.com

# Horus API (if using live data collection)
HORUS_API_KEY=your_api_key_here
HORUS_BASE_URL=https://api.horus.com
```

## Workflow for New Users

### Complete Setup Workflow

1. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\Activate on Windows
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Place CSV files in `data/` directory
   - Ensure files have `timestamp` (index), `price`, and optionally `volume` columns

3. **Train Models** (in order):
   ```bash
   # Step 1: Train regime detectors
   python scripts/train_regime_models.py --mode both --days 30
   
   # Step 2: Train ensemble model
   python scripts/train_ensemble.py --days 30 --forward-window 180
   ```

4. **Backtest Strategy**:
   ```bash
   python backtest_regime_ensemble.py --days 15
   ```

5. **Optimize Parameters** (optional, but recommended):
   ```bash
   python scripts/optimize_competition_strategy.py --trials 50 --days 15
   ```

6. **Run Final Backtest with Optimized Parameters**:
   ```bash
   python backtest_regime_ensemble.py --days 15
   # Will automatically use optimized parameters from figures/optimized_strategy_params.json
   ```

7. **Live Trading** (if configured):
   ```bash
   python run.py --mode regime_ensemble
   ```

## Understanding the Strategy

### How It Works

1. **Regime Detection**: 
   - Analyzes price volatility patterns (GMM) to identify calm vs volatile periods
   - Analyzes price trends (HMM) to identify bearish, neutral, or bullish markets
   - Adapts trading behavior based on detected regime

2. **Signal Generation**:
   - Computes 100+ features from price/volume data
   - Feeds features to ensemble model to predict price direction
   - Generates long/short signals with confidence scores

3. **Portfolio Optimization**:
   - Converts signals to expected returns
   - Calculates covariance matrix from historical returns
   - Optimizes portfolio weights to maximize Sharpe/Sortino/Calmar

4. **Execution**:
   - Opens positions based on optimized weights
   - Manages risk with adaptive stop-loss and take-profit
   - Closes positions when targets are hit or regime changes

### Key Concepts

- **Regime**: Market state (calm/volatile, bearish/neutral/bullish)
- **Ensemble**: Combination of multiple ML models for robust predictions
- **Portfolio Optimization**: Mathematical optimization to allocate capital optimally
- **Competition Score**: Weighted combination of Sharpe, Sortino, and Calmar ratios

## Troubleshooting

### Common Issues

1. **"Could not load regime detectors"**:
   - Solution: Train regime models first with `python scripts/train_regime_models.py --mode both`

2. **"No training data prepared"**:
   - Solution: Ensure CSV files exist in `data/` directory with sufficient historical data (at least 24 hours)

3. **"ValueError: cannot convert float NaN to integer"**:
   - Solution: Check data quality, ensure no missing values in price column

4. **Low number of trades in backtest**:
   - Solution: Lower `min_confidence` threshold or reduce `trade_interval_minutes`

5. **Poor backtest performance**:
   - Solution: Run parameter optimization with `scripts/optimize_competition_strategy.py`
   - Check that models are trained on recent data
   - Verify data quality and completeness

### Debug Mode

```bash
# Run backtest with verbose output
python backtest_regime_ensemble.py --days 15 --no-optimized-params

# Check model files
ls models/*.pkl

# Validate models
python scripts/validate_models.py
```

## Competition Notes

The bot is optimized for the HK University Web3 Quant Hackathon scoring:
- **Competition Score** = 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar
- Focuses on risk-adjusted returns over absolute returns
- Implements strict risk management to minimize drawdowns
- Uses portfolio optimization to maximize all three ratios simultaneously

## Performance Tips

1. **More Data = Better Models**: Train on at least 30 days of historical data
2. **Regular Retraining**: Retrain models weekly or when market conditions change
3. **Parameter Optimization**: Run optimization before competition to find best parameters
4. **Diversification**: Trade across multiple pairs to reduce risk
5. **Regime Awareness**: Strategy adapts to market conditions automatically

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs in `logs/` directory
3. Ensure all dependencies are correctly installed
4. Verify data format matches requirements
5. Check that models are trained before backtesting
