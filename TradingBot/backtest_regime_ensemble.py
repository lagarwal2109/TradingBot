#!/usr/bin/env python3
"""Backtesting for regime-adaptive ensemble strategy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json

from bot.datastore import DataStore
from bot.models.feature_engineering import FeatureEngineer
from bot.models.regime_detection import GMMRegimeDetector, HMMTrendDetector, RegimeFusion
from bot.models.ensemble_model import StackedEnsemble
from bot.models.model_storage import ModelStorage
from bot.models.portfolio_optimizer import PortfolioOptimizer
from bot.portfolio_manager import RegimeAdaptivePortfolioManager
from bot.config import get_config, Config


class RegimeEnsembleBacktest:
    """Backtest regime-adaptive ensemble strategy."""
    
    def __init__(self, config: Optional[Config] = None, data_dir: Optional[Path] = None, 
                 initial_capital: float = 50000.0, param_overrides: Optional[Dict[str, Any]] = None,
                 use_optimized_params: bool = True):
        """Initialize backtest.
        
        Args:
            config: Config object (default: get_config())
            data_dir: Data directory (default: Path("data"))
            initial_capital: Starting capital
            param_overrides: Dictionary of parameter overrides for optimization
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.config = config or get_config()
        self.data_dir = data_dir or Path("data")
        
        # Load optimized parameters if requested
        if use_optimized_params and param_overrides is None:
            optimized_file = Path("figures") / "optimized_strategy_params.json"
            if optimized_file.exists():
                try:
                    import json
                    with open(optimized_file, 'r') as f:
                        optimized = json.load(f)
                        self.param_overrides = optimized.get("best_params", {})
                        # Override trade_interval_minutes to use 5 minutes for more opportunities
                        self.param_overrides["trade_interval_minutes"] = 5
                        print(f"Loaded optimized parameters from {optimized_file}")
                        print(f"  Overriding trade_interval_minutes to 5 minutes for more opportunities")
                except Exception as e:
                    print(f"Warning: Could not load optimized parameters: {e}")
                    self.param_overrides = {}
            else:
                self.param_overrides = {}
        else:
            self.param_overrides = param_overrides or {}
        
        # Ensure trade_interval_minutes is set to 5 if not specified
        if "trade_interval_minutes" not in self.param_overrides:
            self.param_overrides["trade_interval_minutes"] = 5
        
        # Track positions
        self.positions: Dict[str, Dict] = {}
        
        # Portfolio optimizer for maximum Sharpe/Sortino/Calmar
        self.portfolio_optimizer = PortfolioOptimizer(risk_free_rate=0.0)
        
        # Track historical returns for covariance calculation
        self.historical_returns: Dict[str, List[float]] = {}
        
        # Track performance
        self.equity_curve = []
        self.trades = []
        self.regime_history = []
        
        # Components
        self.model_storage = ModelStorage()
        self.feature_engineer = FeatureEngineer()
        self.portfolio_manager = RegimeAdaptivePortfolioManager()
        
        # Load models (if available)
        self.gmm_detector = None
        self.hmm_detector = None
        self.regime_fusion = RegimeFusion()
        self.ensemble = None
        
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        try:
            pairs = DataStore().get_all_pairs_with_data()
            if pairs:
                pair_name = pairs[0].replace("USD", "")
                
                try:
                    self.gmm_detector = GMMRegimeDetector(model_storage=self.model_storage)
                    self.gmm_detector.load(name=f"gmm_regime_{pair_name}")
                except:
                    pass
                
                try:
                    self.hmm_detector = HMMTrendDetector(model_storage=self.model_storage)
                    self.hmm_detector.load(name=f"hmm_trend_{pair_name}")
                except:
                    pass
                
                try:
                    self.ensemble = StackedEnsemble([], model_storage=self.model_storage)
                    self.ensemble.load(name="stacked_ensemble")
                except:
                    pass
        except:
            pass
    
    def detect_regime(self, pair: str, df_minute: pd.DataFrame, df_4h: pd.DataFrame) -> Tuple[Dict, str]:
        """Detect regime for a pair."""
        try:
            if self.gmm_detector and not df_minute.empty:
                gmm_proba = self.gmm_detector.predict_proba(df_minute)
            else:
                gmm_proba = {"calm": 0.5, "volatile": 0.5}
            
            if self.hmm_detector and not df_4h.empty:
                hmm_proba = self.hmm_detector.predict_proba(df_4h)
            else:
                hmm_proba = {"bearish": 0.33, "neutral": 0.33, "bullish": 0.34}
            
            combined = self.regime_fusion.fuse_regimes(gmm_proba, hmm_proba)
            gmm_regime = self.gmm_detector.get_dominant_regime(gmm_proba) if self.gmm_detector else "calm"
            hmm_regime = self.hmm_detector.get_dominant_trend(hmm_proba) if self.hmm_detector else "neutral"
            dominant = self.regime_fusion.get_combined_regime(gmm_regime, hmm_regime)
            
            return combined, dominant
        except:
            return {"calm_bullish": 0.5}, "calm_bullish"
    
    def run_backtest(self, datastore: DataStore, days: int = 7) -> Dict:
        """Run backtest on historical data.
        
        Args:
            datastore: DataStore instance
            days: Number of days to backtest
        
        Returns:
            Backtest results dictionary
        """
        print(f"Running regime ensemble backtest on {days} days of data...")
        
        # Load data
        pairs = datastore.get_all_pairs_with_data()
        all_data = {}
        
        for pair in pairs:
            try:
                df = datastore.read_minute_bars(pair, limit=None)
                if df.empty:
                    continue
                
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp")
                    else:
                        continue
                
                # Check if we have enough data by date range, not row count
                if len(df) < 100:  # Minimum 100 data points
                    continue
                
                end_date = df.index.max()
                start_date = end_date - timedelta(days=days)
                
                # Filter to requested date range
                df_filtered = df[df.index >= start_date].copy()
                
                # Check if we have at least some data in the range
                if len(df_filtered) >= 100:  # Need at least 100 data points
                    all_data[pair] = df_filtered
            except Exception as e:
                print(f"Warning: Error loading data for {pair}: {e}")
                continue
        
        if not all_data:
            raise ValueError(f"No data available for backtesting. Found {len(pairs)} pairs but none had sufficient data for {days} days.")
        
        # Get all timestamps
        all_timestamps = set()
        for df in all_data.values():
            all_timestamps.update(df.index)
        timestamps = sorted(all_timestamps)
        
        print(f"Backtesting on {len(timestamps)} timestamps across {len(all_data)} pairs")
        
        # Determine trade interval (minutes) from overrides
        # Default: 5 minutes for crypto (more frequent = more opportunities)
        # Tested intervals: 5min (optimal for high-frequency), 10min (good), 15min (balanced), 30min (too sparse)
        trade_interval_minutes = int(self.param_overrides.get("trade_interval_minutes", 5))
        trade_interval_minutes = max(1, trade_interval_minutes)  # safety
        print(f"Using trade interval: {trade_interval_minutes} minutes")
        
        # Simulate trading
        for i, timestamp in enumerate(timestamps[::trade_interval_minutes]):  # Use configured interval
            if i % 24 == 0:
                print(f"Progress: {i}/{len(timestamps[::trade_interval_minutes])} steps")
            
            # Update positions and check stops
            current_prices = {}
            for pair, df in all_data.items():
                if timestamp in df.index:
                    price_val = df.loc[timestamp, "price"]
                    # Ensure scalar value
                    if isinstance(price_val, pd.Series):
                        price_val = float(price_val.iloc[-1] if len(price_val) > 0 else 0)
                    else:
                        price_val = float(price_val)
                    current_prices[pair] = price_val
            
            # Check existing positions
            positions_to_close = []
            for symbol, pos in list(self.positions.items()):
                pair = f"{symbol}USD"
                if pair in current_prices:
                    price = current_prices[pair]
                    # Ensure price is scalar
                    if isinstance(price, pd.Series):
                        price = float(price.iloc[-1] if len(price) > 0 else 0)
                    else:
                        price = float(price)
                    
                    # Check stops (handle both long and short)
                    side = pos.get("side", "long")
                    if price > 0:
                        if side == "long":
                            # Long: stop if price <= stop_loss or price >= take_profit
                            if price <= pos["stop_loss"] or price >= pos["take_profit"]:
                                positions_to_close.append(symbol)
                        else:  # short
                            # Short: stop if price >= stop_loss (goes up) or price <= take_profit (goes down)
                            if price >= pos["stop_loss"] or price <= pos["take_profit"]:
                                positions_to_close.append(symbol)
            
            # Close positions
            for symbol in positions_to_close:
                pair = f"{symbol}USD"
                if pair in current_prices:
                    self.close_position(symbol, current_prices[pair], timestamp)
            
            # Generate signals if ensemble available
            if self.ensemble is not None:
                signals = self.generate_signals(all_data, timestamp, current_prices)
                
                # Debug: log signal generation stats periodically
                if i % 50 == 0 or len(signals) > 0:  # Log when signals are generated
                    print(f"  Step {i}, Timestamp {timestamp}: Generated {len(signals)} signals from {len(all_data)} pairs")
                    if len(signals) > 0:
                        for pair, sig in list(signals.items())[:3]:  # Show first 3 signals
                            print(f"    {pair}: {sig.get('side', 'unknown')} - conf={sig.get('confidence', 0):.3f}, prob={sig.get('probability', 0):.3f}")
                
                # Use portfolio optimization if we have multiple signals
                if len(signals) >= 2:
                    try:
                        # Calculate expected returns from signals
                        expected_returns = self.portfolio_optimizer.calculate_expected_returns_from_signals(
                            signals,
                            base_return_estimate=0.02  # 2% base return
                        )
                        
                        # Get historical returns for covariance
                        historical_returns_dict = {}
                        lookback_periods = 100  # Use last 100 periods
                        
                        for pair in signals.keys():
                            if pair in all_data:
                                df_hist = all_data[pair]
                                # Get returns up to current timestamp
                                df_up_to_now = df_hist[df_hist.index <= timestamp].tail(lookback_periods)
                                if len(df_up_to_now) > 1:
                                    prices = df_up_to_now["price"].values
                                    returns = np.diff(prices) / prices[:-1]
                                    historical_returns_dict[pair] = pd.Series(returns, index=df_up_to_now.index[1:])
                        
                        # Only optimize if we have enough historical data
                        if len(historical_returns_dict) >= 2:
                            # Calculate covariance matrices
                            covariance = self.portfolio_optimizer.calculate_covariance_matrix(historical_returns_dict)
                            downside_covariance = self.portfolio_optimizer.calculate_downside_covariance(historical_returns_dict)
                            
                            # Optimize portfolio for maximum competition score
                            max_weight = self.param_overrides.get("max_position_pct", 0.3)  # Max 30% per asset
                            optimal_weights, metrics = self.portfolio_optimizer.optimize_combined_portfolio(
                                expected_returns=expected_returns,
                                covariance=covariance,
                                downside_covariance=downside_covariance,
                                historical_returns=historical_returns_dict,
                                long_only=True,
                                max_weight=max_weight,
                                sharpe_weight=0.3,
                                sortino_weight=0.4,
                                calmar_weight=0.3
                            )
                            
                            # Execute trades based on optimal portfolio weights
                            total_equity = self.calculate_equity(current_prices)
                            max_allocation = self.param_overrides.get("max_portfolio_allocation", 0.90)  # Use 90% of capital
                            available_capital = total_equity * max_allocation
                            
                            # Allocate capital according to optimal weights
                            for pair, weight in optimal_weights.items():
                                if weight > 0.01:  # Only if weight > 1%
                                    symbol = pair.replace("USD", "")
                                    if symbol not in self.positions:
                                        price = current_prices.get(pair, 0)
                                        if price > 0:
                                            # Calculate position size from portfolio weight
                                            position_value = available_capital * weight
                                            amount = position_value / price
                                            
                                            if amount > 0:
                                                signal = signals[pair]
                                                # Open position with portfolio-optimized size
                                                self.open_position_with_size(pair, signal, price, amount, timestamp)
                        
                        else:
                            # Fallback: use individual signal sizing if not enough data
                            max_positions = self.param_overrides.get("max_simultaneous_positions", self.config.max_simultaneous_positions)
                            for pair, signal in signals.items():
                                if len(self.positions) < max_positions:
                                    symbol = pair.replace("USD", "")
                                    if symbol not in self.positions:
                                        price = current_prices.get(pair, 0)
                                        if price > 0:
                                            self.open_position(pair, signal, price, timestamp)
                    
                    except Exception as e:
                        # Fallback on error
                        import logging
                        logging.getLogger(__name__).debug(f"Portfolio optimization failed: {e}, using individual signals")
                        max_positions = self.param_overrides.get("max_simultaneous_positions", self.config.max_simultaneous_positions)
                        for pair, signal in signals.items():
                            if len(self.positions) < max_positions:
                                symbol = pair.replace("USD", "")
                                if symbol not in self.positions:
                                    price = current_prices.get(pair, 0)
                                    if price > 0:
                                        self.open_position(pair, signal, price, timestamp)
                else:
                    # Single signal or no signals - use individual sizing
                    max_positions = self.param_overrides.get("max_simultaneous_positions", self.config.max_simultaneous_positions)
                    for pair, signal in signals.items():
                        if len(self.positions) < max_positions:
                            symbol = pair.replace("USD", "")
                            if symbol not in self.positions:
                                price = current_prices.get(pair, 0)
                                if price > 0:
                                    self.open_position(pair, signal, price, timestamp)
            
            # Record equity
            equity = self.calculate_equity(current_prices)
            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": equity,
                "positions": len(self.positions)
            })
        
        # Calculate metrics
        results = self.calculate_metrics()
        return results
    
    def calculate_momentum(self, df: pd.DataFrame, periods: List[int] = [20, 50]) -> Dict[str, float]:
        """Calculate momentum indicators.
        
        Args:
            df: DataFrame with price column
            periods: List of periods for momentum calculation
            
        Returns:
            Dictionary with momentum values
        """
        if len(df) < max(periods):
            return {f"momentum_{p}": 0.0 for p in periods}
        
        prices = df["price"].values
        momentum_dict = {}
        
        for period in periods:
            if len(prices) >= period:
                # Calculate percentage change over period
                momentum = (prices[-1] / prices[-period] - 1) if prices[-period] > 0 else 0.0
                momentum_dict[f"momentum_{period}"] = float(momentum)
            else:
                momentum_dict[f"momentum_{period}"] = 0.0
        
        return momentum_dict
    
    def generate_signals(self, all_data: Dict, timestamp: pd.Timestamp, prices: Dict[str, float]) -> Dict:
        """Generate trading signals with long and short support."""
        signals = {}
        
        # Get thresholds from overrides or config
        # Use very relaxed defaults to generate more signals
        min_confidence = self.param_overrides.get("min_confidence", 0.15)  # Very low - just need some confidence
        # Accept either specific long/short thresholds or a generic 'high_confidence_threshold'
        generic_high_conf = self.param_overrides.get("high_confidence_threshold", 0.40)  # Lowered further
        long_confidence_threshold = self.param_overrides.get(
            "long_confidence_threshold",
            generic_high_conf
        )
        short_confidence_threshold = self.param_overrides.get(
            "short_confidence_threshold",
            generic_high_conf
        )
        momentum_threshold = self.param_overrides.get("momentum_threshold", 0.0001)  # Very low threshold (0.01%)
        use_momentum_filter = self.param_overrides.get("use_momentum_filter", False)  # Disabled by default for more signals
        
        for pair, df in all_data.items():
            if timestamp not in df.index:
                continue
            
            try:
                # Get historical data up to timestamp
                df_hist = df[df.index <= timestamp].tail(500)
                if len(df_hist) < 100:
                    continue
                
                # Calculate momentum
                momentum = self.calculate_momentum(df_hist)
                momentum_20 = momentum.get("momentum_20", 0.0)
                momentum_50 = momentum.get("momentum_50", 0.0)
                
                # Detect regime
                from bot.datastore import DataStore
                datastore = DataStore()
                df_4h = datastore.read_aggregated_bars(pair, interval="4h", limit=100)
                regime_proba, regime = self.detect_regime(pair, df_hist, df_4h)
                
                # Create features
                feature_matrix, _ = self.feature_engineer.create_feature_matrix(
                    {pair: df_hist},
                    regime_detector=None
                )
                
                if feature_matrix.empty:
                    continue
                
                # Get prediction
                feature_cols = [col for col in feature_matrix.columns if col != "pair"]
                X = feature_matrix[feature_cols].values
                X = np.nan_to_num(X, nan=0.0)
                
                proba = self.ensemble.predict_proba(X)
                confidence = self.ensemble.get_confidence_score(proba)[0]
                prob_up = proba[0, 1] if proba.shape[1] == 2 else proba[0, 0]
                
                # Generate LONG signal
                # Extremely relaxed: prob_up > 0.45 OR (high confidence AND prob_up > 0.40)
                # Momentum is optional (can be disabled)
                long_signal = False
                if confidence >= min_confidence:
                    # Extremely relaxed probability thresholds - accept any edge
                    prob_condition = prob_up > 0.45 or (confidence >= long_confidence_threshold and prob_up > 0.40)
                    momentum_condition = not use_momentum_filter or momentum_20 > momentum_threshold
                    if prob_condition and momentum_condition:
                        long_signal = True
                
                # Generate SHORT signal
                # Extremely relaxed: prob_up < 0.55 OR (high confidence AND prob_up < 0.60)
                # Momentum is optional (can be disabled)
                short_signal = False
                if confidence >= min_confidence:
                    # Extremely relaxed probability thresholds - accept any edge
                    prob_condition = prob_up < 0.55 or (confidence >= short_confidence_threshold and prob_up < 0.60)
                    momentum_condition = not use_momentum_filter or momentum_20 < -momentum_threshold
                    if prob_condition and momentum_condition:
                        short_signal = True
                
                # Create signal if either long or short
                if long_signal:
                    signals[pair] = {
                        "side": "long",
                        "confidence": float(confidence),
                        "probability": float(prob_up),
                        "regime": regime,
                        "momentum_20": momentum_20,
                        "momentum_50": momentum_50
                    }
                elif short_signal:
                    signals[pair] = {
                        "side": "short",
                        "confidence": float(confidence),
                        "probability": float(prob_up),
                        "regime": regime,
                        "momentum_20": momentum_20,
                        "momentum_50": momentum_50
                    }
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Error generating signal for {pair}: {e}")
                continue
        
        return signals
    
    def open_position_with_size(self, pair: str, signal: Dict, price: float, amount: float, timestamp: pd.Timestamp):
        """Open a position with a specific amount (for portfolio optimization)."""
        if amount <= 0 or price <= 0:
            return
        
        side = signal.get("side", "long")
        confidence = signal.get("confidence", 0.5)
        regime = signal.get("regime", "calm_bullish")
        
        # Calculate stops
        stop_loss, take_profit = self.portfolio_manager.calculate_adaptive_stops(
            entry_price=price,
            regime=regime,
            confidence=confidence,
            volatility=0.02,  # Default volatility
            param_overrides=self.param_overrides
        )
        
        # For short positions, invert stop loss and take profit
        if side == "short":
            original_stop_loss = stop_loss
            original_take_profit = take_profit
            stop_loss = 2 * price - original_take_profit
            take_profit = 2 * price - original_stop_loss
        
        # Calculate cost
        cost = amount * price * (1 + 0.001)  # 0.1% fee
        
        if self.capital >= cost:
            self.capital -= cost
            
            symbol = pair.replace("USD", "")
            self.positions[symbol] = {
                "pair": pair,
                "side": side,
                "entry_price": price,
                "amount": amount,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "regime": regime,
                "confidence": confidence,
                "entry_time": timestamp
            }
            
            self.trades.append({
                "timestamp": timestamp,
                "action": "open",
                "pair": pair,
                "side": side,
                "price": price,
                "amount": amount,
                "value": cost,
                "regime": regime
            })
    
    def open_position(self, pair: str, signal: Dict, price: float, timestamp: pd.Timestamp):
        """Open a position."""
        # Ensure price is a scalar float
        if isinstance(price, pd.Series):
            price = float(price.iloc[-1] if len(price) > 0 else 0)
        elif hasattr(price, '__iter__') and not isinstance(price, str):
            price = float(price[0] if len(price) > 0 else 0)
        else:
            price = float(price)
        
        if price <= 0 or pd.isna(price):
            return
        
        symbol = pair.replace("USD", "")
        regime = signal.get("regime", "calm_bullish")
        confidence = signal.get("confidence", 0.5)
        side = signal.get("side", "long")  # "long" or "short"
        
        # Calculate position size with maximum capital deployment
        total_equity = self.calculate_equity({pair: price})
        max_allocation = self.param_overrides.get("max_portfolio_allocation", 0.90)  # 90% deployment
        
        # Calculate how much capital is already allocated
        allocated_value = 0.0
        for symbol, pos in self.positions.items():
            pos_pair = f"{symbol}USD"
            if pos_pair in {pair: price}:
                allocated_value += pos["amount"] * price
            else:
                allocated_value += pos["amount"] * pos["entry_price"]
        
        available_capital = total_equity * max_allocation - allocated_value
        
        # Base position size - more aggressive to use available capital
        volatility = 0.02  # Simplified
        base_size = self.param_overrides.get("base_position_pct", 0.25)  # 25% base
        
        position_value = self.portfolio_manager.calculate_position_size(
            base_size=base_size,
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            total_equity=total_equity,
            param_overrides=self.param_overrides
        )
        
        # Ensure we don't exceed available capital
        position_value = min(position_value, available_capital)
        
        if position_value <= 0:
            return
        
        amount = position_value / price
        
        # Calculate stops (for shorts, stop_loss > entry, take_profit < entry)
        stop_loss, take_profit = self.portfolio_manager.calculate_adaptive_stops(
            entry_price=price,
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            param_overrides=self.param_overrides
        )
        
        # For short positions, invert stop loss and take profit
        if side == "short":
            # Short: profit when price goes down, loss when price goes up
            # Stop loss should be above entry (price goes up = loss)
            # Take profit should be below entry (price goes down = profit)
            original_stop_loss = stop_loss
            original_take_profit = take_profit
            # Invert: stop_loss becomes the upper bound, take_profit becomes lower bound
            stop_loss = 2 * price - original_take_profit  # Mirror take_profit above entry
            take_profit = 2 * price - original_stop_loss  # Mirror stop_loss below entry
        
        # Deduct capital (with fees) - same for long and short
        cost = position_value * (1 + 0.001)  # 0.1% fee
        if cost > self.capital:
            return
        
        self.capital -= cost
        
        self.positions[symbol] = {
            "pair": pair,
            "side": side,
            "entry_price": price,
            "amount": amount,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "regime": regime,
            "confidence": confidence,
            "entry_time": timestamp
        }
        
        self.trades.append({
            "timestamp": timestamp,
            "action": "open",
            "pair": pair,
            "side": side,
            "price": price,
            "amount": amount,
            "regime": regime
        })
    
    def close_position(self, symbol: str, price: float, timestamp: pd.Timestamp):
        """Close a position."""
        if symbol not in self.positions:
            return
        
        # Ensure price is scalar
        if isinstance(price, pd.Series):
            price = float(price.iloc[-1] if len(price) > 0 else 0)
        else:
            price = float(price)
        
        if price <= 0:
            return
        
        pos = self.positions[symbol]
        side = pos.get("side", "long")
        entry_price = pos["entry_price"]
        amount = pos["amount"]
        
        # Calculate PnL based on side
        if side == "long":
            # Long: profit when exit > entry
            proceeds = amount * price * (1 - 0.001)  # 0.1% fee
            cost = amount * entry_price * (1 + 0.001)  # Entry fee
            pnl = proceeds - cost
        else:  # short
            # Short: profit when exit < entry
            # We sold at entry_price, now buying back at price
            proceeds = amount * entry_price * (1 - 0.001)  # Entry fee (selling)
            cost = amount * price * (1 + 0.001)  # Exit fee (buying back)
            pnl = proceeds - cost
        
        self.capital += proceeds  # Add proceeds from closing
        
        self.trades.append({
            "timestamp": timestamp,
            "action": "close",
            "pair": pos["pair"],
            "side": side,
            "price": price,
            "pnl": pnl,
            "regime": pos["regime"]
        })
        
        del self.positions[symbol]
    
    def calculate_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (handles both long and short positions)."""
        equity = self.capital
        for symbol, pos in self.positions.items():
            pair = f"{symbol}USD"
            if pair in prices:
                current_price = prices[pair]
                side = pos.get("side", "long")
                entry_price = pos["entry_price"]
                amount = pos["amount"]
                
                if side == "long":
                    # Long: value is current price * amount
                    equity += amount * current_price
                else:  # short
                    # Short: value is entry_price * amount (what we sold for)
                    # minus the current cost to buy back
                    # Simplified: entry_value - (current_price - entry_price) * amount
                    # = entry_value + (entry_price - current_price) * amount
                    equity += amount * entry_price + (entry_price - current_price) * amount
                    # Which simplifies to: 2 * entry_price * amount - current_price * amount
        return equity
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.equity_curve:
            return {}
        
        # Create equity series with timestamps as index
        equity_df = pd.DataFrame(self.equity_curve)
        if "timestamp" in equity_df.columns:
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
            equity_df = equity_df.set_index("timestamp")
        equity_series = equity_df["equity"]
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        
        # Calculate intervals per day based on data frequency
        # If we have minute data, there are 1440 minutes per day
        # If we're sampling every N minutes, intervals_per_day = 1440 / N
        if len(returns) > 1:
            time_diff = (equity_series.index[-1] - equity_series.index[0]).total_seconds() / 60  # minutes
            intervals_per_day = (1440 * len(returns)) / time_diff if time_diff > 0 else 1440
        else:
            intervals_per_day = 1440  # Default: assume minute data
        
        # Annualization factor
        annualization_factor = np.sqrt(252 * intervals_per_day)
        
        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * annualization_factor
        else:
            sharpe = 0
        
        # Sortino ratio (downside deviation, annualized)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * annualization_factor
        else:
            # If no downside returns, Sortino = Sharpe
            sortino = sharpe
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Calmar ratio (annualized return / max drawdown)
        if max_drawdown > 0:
            # Annualized return
            periods_per_year = 252 * intervals_per_day
            annualized_return = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (periods_per_year / len(returns)) - 1) * 100
            calmar = annualized_return / max_drawdown
        else:
            calmar = 0
        
        # Competition score
        competition_score = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
        
        # Trade statistics
        closed_trades = [t for t in self.trades if t["action"] == "close"]
        winning_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        results = {
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_drawdown,
            "calmar_ratio": calmar,
            "competition_score": competition_score,
            "total_trades": len(closed_trades),
            "win_rate": win_rate,
            "final_equity": equity_series.iloc[-1]
        }
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest regime ensemble strategy")
    parser.add_argument("--days", type=int, default=7, help="Days to backtest")
    parser.add_argument("--capital", type=float, default=50000.0, help="Initial capital")
    
    args = parser.parse_args()
    
    datastore = DataStore()
    backtest = RegimeEnsembleBacktest(initial_capital=args.capital)
    
    results = backtest.run_backtest(datastore, days=args.days)
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("="*70)
    
    # Save results
    results_file = Path("figures") / "backtest_regime_ensemble.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

