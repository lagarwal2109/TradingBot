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
            # Use the data_dir from initialization
            datastore = DataStore(data_dir=self.data_dir)
            pairs = datastore.get_all_pairs_with_data()
            if pairs:
                pair_name = pairs[0].replace("USD", "")
                
                try:
                    self.gmm_detector = GMMRegimeDetector(model_storage=self.model_storage)
                    self.gmm_detector.load(name=f"gmm_regime_{pair_name}")
                    print(f"Loaded GMM detector for {pair_name}")
                except Exception as e:
                    print(f"Could not load GMM detector: {e}")
                    pass
                
                try:
                    self.hmm_detector = HMMTrendDetector(model_storage=self.model_storage)
                    self.hmm_detector.load(name=f"hmm_trend_{pair_name}")
                    print(f"Loaded HMM detector for {pair_name}")
                except Exception as e:
                    print(f"Could not load HMM detector: {e}")
                    pass
                
                try:
                    self.ensemble = StackedEnsemble([], model_storage=self.model_storage)
                    self.ensemble.load(name="stacked_ensemble")
                    print("Loaded ensemble model")
                except Exception as e:
                    print(f"Could not load ensemble model: {e}")
                    print("Will use fallback momentum/trend strategy")
                    self.ensemble = None
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Will use fallback momentum/trend strategy")
            self.ensemble = None
    
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
                
                # If we don't have enough data for the requested days, use what we have
                # but only if we have at least 1 day of data
                actual_days = (df.index.max() - df.index.min()).days
                if len(df_filtered) < 100 and actual_days >= 1:
                    # Use all available data if we have at least 1 day
                    df_filtered = df.copy()
                    print(f"Warning: {pair} only has {actual_days} days of data, using all available data")
                
                # Check if we have at least some data in the range
                if len(df_filtered) >= 100:  # Need at least 100 data points
                    all_data[pair] = df_filtered
            except Exception as e:
                print(f"Warning: Error loading data for {pair}: {e}")
                continue
        
        if not all_data:
            # Provide more helpful error message
            if len(pairs) == 0:
                error_msg = f"No data files found in {datastore.data_dir}. Please ensure CSV files exist."
            else:
                # Check actual data range
                max_days = 0
                for pair in pairs:
                    try:
                        df = datastore.read_minute_bars(pair, limit=None)
                        if not df.empty and len(df) >= 100:
                            actual_days = (df.index.max() - df.index.min()).days
                            max_days = max(max_days, actual_days)
                    except:
                        pass
                
                error_msg = (
                    f"No data available for backtesting. Found {len(pairs)} pairs but none had sufficient data.\n"
                    f"  Requested: {days} days\n"
                    f"  Maximum available: {max_days} days\n"
                    f"  Data directory: {datastore.data_dir}\n"
                    f"  Try: --days {min(max_days, 7)} (or less)"
                )
            raise ValueError(error_msg)
        
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
                    
                    # Only handle long positions (no shorts)
                    if price > 0:
                        # Long: stop if price <= stop_loss or price >= take_profit
                        if price <= pos["stop_loss"] or price >= pos["take_profit"]:
                            positions_to_close.append(symbol)
            
            # Close positions
            for symbol in positions_to_close:
                pair = f"{symbol}USD"
                if pair in current_prices:
                    self.close_position(symbol, current_prices[pair], timestamp)
            
            # Generate signals - use ensemble if available, otherwise use fallback
            signals = {}
            if self.ensemble is not None:
                try:
                    signals = self.generate_signals(all_data, timestamp, current_prices)
                except Exception as e:
                    print(f"Warning: Ensemble signal generation failed: {e}, using fallback")
                    signals = self.generate_fallback_signals(all_data, timestamp, current_prices)
            else:
                # No ensemble available - use simple momentum/trend fallback
                signals = self.generate_fallback_signals(all_data, timestamp, current_prices)
            
            # Check for high-confidence signals (confidence > 0.6) for rebalancing
            # Only trade when confidence > 0.6 to avoid equity drops
            high_confidence_signals = {
                pair: sig for pair, sig in signals.items() 
                if sig.get("confidence", 0) > 0.6 and sig.get("probability", 0) > 0.5  # Must also have prob > 0.5 for longs
            }
            
            # Initialize expected returns dict (will be populated if high-confidence signals exist)
            signal_expected_returns = {}
            
            # If we have high-confidence signals, trigger rebalancing
            if len(high_confidence_signals) > 0:
                # Calculate expected returns for all high-confidence signals
                # Expected return = probability * confidence * base_return_estimate (as per plan)
                base_return_estimate = 0.02  # 2% base return estimate
                for pair, sig in high_confidence_signals.items():
                    confidence = sig.get("confidence", 0.5)
                    prob_up = sig.get("probability", 0.5)
                    # Expected return = probability * confidence * base_return_estimate
                    # This gives higher return for higher probability AND higher confidence
                    expected_return = prob_up * confidence * base_return_estimate
                    signal_expected_returns[pair] = expected_return
                
                # Sort signals by expected return (highest first)
                sorted_signals = sorted(
                    high_confidence_signals.items(),
                    key=lambda x: signal_expected_returns.get(x[0], 0),
                    reverse=True
                )
                
                # Get top N signals (where N = max positions)
                max_positions = self.param_overrides.get("max_simultaneous_positions", self.config.max_simultaneous_positions)
                top_signals = dict(sorted_signals[:max_positions])
                top_pairs = set(top_signals.keys())
                
                # Close positions that are NOT in top high-confidence signals
                positions_to_rebalance = []
                for symbol, pos in list(self.positions.items()):
                    pair = pos["pair"]
                    if pair not in top_pairs:
                        positions_to_rebalance.append(symbol)
                
                # Close positions that should be rebalanced
                for symbol in positions_to_rebalance:
                    pair = f"{symbol}USD"
                    if pair in current_prices:
                        print(f"  [REBALANCE] Closing {pair} (not in top {len(top_signals)} high-confidence signals)")
                        self.close_position(symbol, current_prices[pair], timestamp)
                
                # Now use only top high-confidence signals for allocation
                signals = top_signals
                print(f"  [REBALANCE] Using {len(signals)} high-confidence signals (conf > 0.6) sorted by expected return")
            
            # Calculate current equity and positions
            current_equity = self.calculate_equity(current_prices)
            allocated_value = sum(
                pos["amount"] * current_prices.get(f"{symbol}USD", pos["entry_price"])
                for symbol, pos in self.positions.items()
            )
            cash = self.capital
            realized_pnl = sum(t.get("pnl", 0) for t in self.trades if t.get("action") == "close")
            unrealized_pnl = current_equity - self.initial_capital - realized_pnl
            
            # Debug: log signal generation stats periodically
            if i % 50 == 0 or len(signals) > 0 or len(positions_to_close) > 0:  # Log when signals are generated or positions closed
                print(f"\n  === Step {i}, {timestamp} ===")
                print(f"  Equity: ${current_equity:.2f} | Cash: ${cash:.2f} | Allocated: ${allocated_value:.2f}")
                print(f"  Positions: {len(self.positions)} | Realized P&L: ${realized_pnl:.2f} | Unrealized P&L: ${unrealized_pnl:.2f}")
                if len(self.positions) > 0:
                    print(f"  Open Positions:")
                    for symbol, pos in self.positions.items():
                        pair = pos["pair"]
                        if pair in current_prices:
                            current_price = current_prices[pair]
                            entry_price = pos["entry_price"]
                            amount = pos["amount"]
                            position_value = amount * current_price
                            position_pnl = (current_price - entry_price) * amount
                            position_pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                            print(f"    {pair}: {amount:.6f} @ ${current_price:.4f} | Entry: ${entry_price:.4f} | "
                                  f"Value: ${position_value:.2f} | P&L: ${position_pnl:.2f} ({position_pnl_pct:.2f}%)")
                if len(signals) > 0:
                    print(f"  Generated {len(signals)} LONG signals (prob > 0.5):")
                    for pair, sig in list(signals.items())[:5]:  # Show first 5 signals
                        exp_ret = signal_expected_returns.get(pair, 0) if signal_expected_returns else 0
                        print(f"    {pair}: conf={sig.get('confidence', 0):.3f}, prob={sig.get('probability', 0):.3f}, exp_ret={exp_ret:.4f}")
                if len(high_confidence_signals) > 0:
                    print(f"  ⚡ High-confidence signals (conf > 0.6): {len(high_confidence_signals)} - REBALANCING")
                if len(positions_to_close) > 0:
                    print(f"  Closing {len(positions_to_close)} positions (stop/target hit)")
                
                # Priority 1: High-confidence rebalancing (if confidence > 0.6)
                # Only trade/rebalance when we have high-confidence signals to avoid equity drops
                if len(high_confidence_signals) > 0:
                    # High-confidence rebalancing: allocate based on expected return priority
                    print(f"  [REBALANCE] Allocating capital to {len(signals)} high-confidence signals (conf > 0.6) by expected return")
                    
                    # Calculate expected returns for allocation weights
                    # Use all high-confidence signals for rebalancing
                    total_expected_return = sum(signal_expected_returns.values())
                    if total_expected_return > 0:
                        # Normalize weights by expected return (higher expected return = larger weight)
                        weights = {
                            pair: signal_expected_returns[pair] / total_expected_return
                            for pair in signals.keys()
                        }
                    else:
                        # Equal weights if no expected return (shouldn't happen, but safety)
                        weights = {pair: 1.0 / len(signals) for pair in signals.keys()}
                    
                    # Log expected returns for debugging
                    print(f"  Expected Returns:")
                    for pair, exp_ret in sorted(signal_expected_returns.items(), key=lambda x: x[1], reverse=True)[:5]:
                        weight = weights.get(pair, 0)
                        print(f"    {pair}: exp_ret={exp_ret:.4f} ({exp_ret*100:.2f}%), weight={weight:.2%}")
                    
                    # Allocate capital (95% of equity)
                    total_equity = self.calculate_equity(current_prices)
                    max_allocation = self.param_overrides.get("max_portfolio_allocation", 0.95)
                    available_capital = total_equity * max_allocation
                    
                    # Allocate to each signal based on expected return weight
                    for pair, signal in signals.items():
                        weight = weights.get(pair, 0)
                        if weight > 0.01:  # At least 1% allocation
                            symbol = pair.replace("USD", "")
                            price = current_prices.get(pair, 0)
                            if price > 0:
                                position_value = available_capital * weight
                                amount = position_value / price
                                
                                if amount > 0:
                                    if symbol in self.positions:
                                        # Update existing position to target weight
                                        existing_pos = self.positions[symbol]
                                        target_value = position_value
                                        current_value = existing_pos["amount"] * price
                                        
                                        if abs(target_value - current_value) > 10:  # Only adjust if difference > $10
                                            if target_value > current_value:
                                                # Add to position
                                                add_value = target_value - current_value
                                                add_amount = add_value / price
                                                old_amount = existing_pos["amount"]
                                                existing_pos["amount"] += add_amount
                                                existing_pos["entry_price"] = (existing_pos["entry_price"] * old_amount + price * add_amount) / existing_pos["amount"]
                                                cost = add_value * (1 + 0.001)
                                                self.capital -= cost
                                                print(f"  [REBALANCE ADD] {pair}: +{add_amount:.6f} @ {price:.4f} | Value: ${add_value:.2f} | Balance: ${self.capital:.2f}")
                                            elif target_value < current_value:
                                                # Reduce position (partial close)
                                                reduce_value = current_value - target_value
                                                reduce_amount = reduce_value / price
                                                existing_pos["amount"] -= reduce_amount
                                                proceeds = reduce_value * (1 - 0.001)
                                                self.capital += proceeds
                                                print(f"  [REBALANCE REDUCE] {pair}: -{reduce_amount:.6f} @ {price:.4f} | Value: ${reduce_value:.2f} | Balance: ${self.capital:.2f}")
                                    else:
                                        # New position
                                        self.open_position(pair, signal, price, timestamp)
                
                # If no high-confidence signals, don't trade (avoid equity drops)
                elif len(signals) > 0:
                    print(f"  [SKIP] {len(signals)} signals but none have confidence > 0.6 - skipping to avoid equity drop")
                
                # Priority 2: Portfolio optimization if we have multiple signals (but no high-confidence)
                # NOTE: This path should rarely execute since we require confidence > 0.6 to trade
                elif len(signals) >= 2 and len(high_confidence_signals) == 0:
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
                    # AGGRESSIVE: Allocate all available capital
                    if len(signals) > 0:
                        print(f"  Processing {len(signals)} signal(s) - allocating all available capital")
                        
                        # Sort signals by probability (best first)
                        sorted_signals = sorted(signals.items(), key=lambda x: x[1].get('probability', 0), reverse=True)
                        
                        max_positions = self.param_overrides.get("max_simultaneous_positions", self.config.max_simultaneous_positions)
                        total_equity = self.calculate_equity(current_prices)
                        max_allocation = self.param_overrides.get("max_portfolio_allocation", 0.95)  # 95% allocation
                        target_allocated = total_equity * max_allocation
                        
                        # Calculate current allocation
                        current_allocated = sum(
                            pos["amount"] * current_prices.get(f"{symbol}USD", pos["entry_price"])
                            for symbol, pos in self.positions.items()
                        )
                        available_for_new = target_allocated - current_allocated
                        
                        # Allocate to new positions
                        for pair, signal in sorted_signals:
                            if len(self.positions) >= max_positions:
                                break
                            
                            symbol = pair.replace("USD", "")
                            if symbol not in self.positions:
                                price = current_prices.get(pair, 0)
                                if price > 0 and available_for_new > 100:  # At least $100 available
                                    self.open_position(pair, signal, price, timestamp)
                                    # Update available capital
                                    current_allocated = sum(
                                        pos["amount"] * current_prices.get(f"{symbol}USD", pos["entry_price"])
                                        for symbol, pos in self.positions.items()
                                    )
                                    available_for_new = target_allocated - current_allocated
                                elif price <= 0:
                                    print(f"    Warning: Invalid price {price} for {pair}")
                            else:
                                # Already have position - add to it if we have cash
                                if available_for_new > 100:
                                    # Add to existing position
                                    existing_pos = self.positions[symbol]
                                    add_value = min(available_for_new * 0.2, available_for_new)  # Add up to 20% of available
                                    add_amount = add_value / price
                                    
                                    if add_amount > 0:
                                        # Update position
                                        old_amount = existing_pos["amount"]
                                        existing_pos["amount"] += add_amount
                                        existing_pos["entry_price"] = (existing_pos["entry_price"] * old_amount + price * add_amount) / existing_pos["amount"]
                                        cost = add_value * (1 + 0.001)
                                        self.capital -= cost
                                        
                                        print(f"  [ADD] {pair}: +{add_amount:.6f} @ {price:.4f} | Value: ${add_value:.2f} | Balance: ${self.capital:.2f}")
                                        
                                        # Update available
                                        current_allocated = sum(
                                            pos["amount"] * current_prices.get(f"{symbol}USD", pos["entry_price"])
                                            for symbol, pos in self.positions.items()
                                        )
                                        available_for_new = target_allocated - current_allocated
                    
                    # If no new signals but we have cash, add to existing positions
                    elif len(signals) == 0 and len(self.positions) > 0 and self.capital > 100:
                        # Allocate remaining cash to existing positions based on signal strength
                        total_equity = self.calculate_equity(current_prices)
                        max_allocation = self.param_overrides.get("max_portfolio_allocation", 0.95)
                        target_allocated = total_equity * max_allocation
                        
                        current_allocated = sum(
                            pos["amount"] * current_prices.get(f"{symbol}USD", pos["entry_price"])
                            for symbol, pos in self.positions.items()
                        )
                        available = target_allocated - current_allocated
                        
                        if available > 100:
                            # Distribute proportionally to existing positions
                            for symbol, pos in self.positions.items():
                                pair = pos["pair"]
                                if pair in current_prices:
                                    price = current_prices[pair]
                                    # Add proportional amount
                                    position_value = pos["amount"] * price
                                    weight = position_value / current_allocated if current_allocated > 0 else 1.0 / len(self.positions)
                                    add_value = available * weight
                                    add_amount = add_value / price
                                    
                                    if add_amount > 0:
                                        old_amount = pos["amount"]
                                        pos["amount"] += add_amount
                                        pos["entry_price"] = (pos["entry_price"] * old_amount + price * add_amount) / pos["amount"]
                                        cost = add_value * (1 + 0.001)
                                        self.capital -= cost
                                        
                                        print(f"  [ADD] {pair}: +{add_amount:.6f} @ {price:.4f} | Value: ${add_value:.2f} | Balance: ${self.capital:.2f}")
            
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
    
    def generate_fallback_signals(self, all_data: Dict, timestamp: pd.Timestamp, prices: Dict[str, float]) -> Dict:
        """Generate simple momentum/trend-based signals when ensemble is not available.
        
        This is a fallback strategy that uses simple technical indicators to generate trades.
        """
        signals = {}
        
        for pair, df in all_data.items():
            if timestamp not in df.index:
                continue
            
            try:
                # Get historical data up to timestamp
                df_hist = df[df.index <= timestamp].tail(200)
                if len(df_hist) < 50:  # Need at least 50 data points
                    continue
                
                # Calculate momentum
                momentum = self.calculate_momentum(df_hist)
                momentum_20 = momentum.get("momentum_20", 0.0)
                momentum_50 = momentum.get("momentum_50", 0.0)
                
                # Simple moving averages
                prices_series = df_hist["price"]
                if len(prices_series) >= 20:
                    sma_20 = prices_series.tail(20).mean()
                    sma_50 = prices_series.tail(min(50, len(prices_series))).mean() if len(prices_series) >= 50 else sma_20
                    current_price = prices_series.iloc[-1]
                    
                    # Simple trend following strategy
                    # Long: price above SMA20, positive momentum, SMA20 > SMA50
                    # Short: price below SMA20, negative momentum, SMA20 < SMA50
                    
                    long_signal = False
                    confidence = 0.5
                    prob_up = 0.5
                    
                    # Very lenient conditions for LONG trading only
                    if current_price > sma_20 * 0.98 and momentum_20 > -0.01:  # Price near or above SMA20, not too negative momentum
                        if len(prices_series) >= 50:
                            if sma_20 > sma_50 * 0.99:  # Uptrend (SMA20 >= SMA50)
                                long_signal = True
                                confidence = min(0.6 + abs(momentum_20) * 5, 0.9)
                                prob_up = 0.5 + momentum_20 * 2  # Scale momentum to probability
                        else:
                            # Not enough data for SMA50, just use momentum
                            if momentum_20 > 0.001:  # Positive momentum
                                long_signal = True
                                confidence = min(0.5 + abs(momentum_20) * 5, 0.8)
                                prob_up = 0.5 + momentum_20 * 2
                    
                    # Only create LONG signals (no shorts)
                    # Only trade if probability > 0.5 (favorable for long)
                    if long_signal and prob_up > 0.5:
                        signals[pair] = {
                            "side": "long",
                            "confidence": float(confidence),
                            "probability": float(prob_up),
                            "regime": "fallback",
                            "momentum_20": momentum_20,
                            "momentum_50": momentum_50
                        }
            except Exception as e:
                continue
        
        return signals
    
    def generate_signals(self, all_data: Dict, timestamp: pd.Timestamp, prices: Dict[str, float]) -> Dict:
        """Generate trading signals with long and short support."""
        signals = {}
        
        # Get thresholds from overrides or config
        # Use VERY relaxed defaults to generate more signals
        min_confidence = self.param_overrides.get("min_confidence", 0.10)  # Even lower - just need minimal confidence
        # Accept either specific long/short thresholds or a generic 'high_confidence_threshold'
        generic_high_conf = self.param_overrides.get("high_confidence_threshold", 0.30)  # Lowered even further
        long_confidence_threshold = self.param_overrides.get(
            "long_confidence_threshold",
            generic_high_conf
        )
        short_confidence_threshold = self.param_overrides.get(
            "short_confidence_threshold",
            generic_high_conf
        )
        momentum_threshold = self.param_overrides.get("momentum_threshold", 0.00005)  # Even lower threshold (0.005%)
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
                
                # Generate LONG signal ONLY (no shorts)
                # Only trade when prob_up > 0.5 (favorable probability)
                long_signal = False
                if confidence >= min_confidence:
                    # Only trade if probability is favorable (> 0.5)
                    prob_condition = prob_up > 0.5
                    momentum_condition = not use_momentum_filter or momentum_20 > momentum_threshold
                    if prob_condition and momentum_condition:
                        long_signal = True
                
                # Create LONG signal only (no shorts)
                if long_signal:
                    signals[pair] = {
                        "side": "long",
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
        # Only long positions
        
        # Calculate position size with maximum capital deployment (95%+)
        total_equity = self.calculate_equity({pair: price})
        max_allocation = self.param_overrides.get("max_portfolio_allocation", 0.95)  # 95% deployment
        
        # Calculate how much capital is already allocated
        allocated_value = 0.0
        for symbol, pos in self.positions.items():
            pos_pair = f"{symbol}USD"
            if pos_pair in {pair: price}:
                allocated_value += pos["amount"] * price
            else:
                allocated_value += pos["amount"] * pos["entry_price"]
        
        available_capital = total_equity * max_allocation - allocated_value
        
        # Base position size - VERY aggressive to ensure trades happen
        volatility = 0.02  # Simplified
        base_size = self.param_overrides.get("base_position_pct", 0.30)  # 30% base (increased)
        
        # Calculate position size, but ensure minimum trade size
        position_value = self.portfolio_manager.calculate_position_size(
            base_size=base_size,
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            total_equity=total_equity,
            param_overrides=self.param_overrides
        )
        
        # Ensure minimum position size (at least 5% of capital or $1000, whichever is smaller)
        min_position_value = min(total_equity * 0.05, 1000.0)
        position_value = max(position_value, min_position_value)
        
        # Ensure we don't exceed available capital
        position_value = min(position_value, available_capital)
        
        if position_value <= 0:
            print(f"Warning: No available capital for {pair} (available: {available_capital:.2f}, total_equity: {total_equity:.2f})")
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
        
        # Only long positions (no shorts)
        
        # Deduct capital (with fees) - same for long and short
        cost = position_value * (1 + 0.001)  # 0.1% fee
        if cost > self.capital:
            return
        
        self.capital -= cost
        
        self.positions[symbol] = {
            "pair": pair,
            "side": "long",
            "entry_price": price,
            "amount": amount,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "regime": regime,
            "confidence": confidence,
            "entry_time": timestamp
        }
        
        # Log opening trade
        print(f"  [OPEN] {pair}: {amount:.6f} @ {price:.4f} | Value: ${position_value:.2f} | "
              f"Conf: {confidence:.3f}, Prob: {signal.get('probability', 0):.3f} | "
              f"Balance: ${self.capital:.2f}")
        
        self.trades.append({
            "timestamp": timestamp,
            "action": "open",
            "pair": pair,
            "side": "long",
            "price": price,
            "amount": amount,
            "value": position_value,
            "confidence": confidence,
            "probability": signal.get("probability", 0),
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
        entry_price = pos["entry_price"]
        amount = pos["amount"]
        entry_time = pos.get("entry_time", timestamp)
        hold_time = (timestamp - entry_time).total_seconds() / 3600 if hasattr(timestamp - entry_time, 'total_seconds') else 0
        
        # Calculate PnL for long position
        proceeds = amount * price * (1 - 0.001)  # 0.1% fee on exit
        cost = amount * entry_price * (1 + 0.001)  # Entry fee
        pnl = proceeds - cost
        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
        
        self.capital += proceeds  # Add proceeds from closing
        
        # Log closing trade
        print(f"  [CLOSE] {pos['pair']}: {amount:.6f} @ {price:.4f} | Entry: {entry_price:.4f} | "
              f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Hold: {hold_time:.1f}h | Balance: ${self.capital:.2f}")
        
        self.trades.append({
            "timestamp": timestamp,
            "action": "close",
            "pair": pos["pair"],
            "side": "long",
            "price": price,
            "entry_price": entry_price,
            "amount": amount,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_time_hours": hold_time,
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
                
                # Only long positions
                equity += amount * current_price
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
    
    # Check if data is in nested data/data directory
    from pathlib import Path
    default_data_dir = Path("data")
    nested_data_dir = default_data_dir / "data"
    
    # Use nested directory if it exists and has CSV files
    csv_count_nested = len(list(nested_data_dir.glob("*.csv"))) if nested_data_dir.exists() else 0
    csv_count_default = len(list(default_data_dir.glob("*.csv"))) if default_data_dir.exists() else 0
    
    if csv_count_nested > csv_count_default:
        data_dir = nested_data_dir
        print(f"Using nested data directory: {data_dir} ({csv_count_nested} CSV files)")
    elif csv_count_default > 0:
        data_dir = default_data_dir
        print(f"Using default data directory: {data_dir} ({csv_count_default} CSV files)")
    else:
        # Try to find data directory
        possible_dirs = [nested_data_dir, default_data_dir, Path("TradingBot/data/data"), Path("TradingBot/data")]
        for possible_dir in possible_dirs:
            if possible_dir.exists() and len(list(possible_dir.glob("*.csv"))) > 0:
                data_dir = possible_dir
                print(f"Found data in: {data_dir}")
                break
        else:
            data_dir = default_data_dir
            print(f"Warning: No CSV files found, using default: {data_dir}")
    
    datastore = DataStore(data_dir=data_dir)
    backtest = RegimeEnsembleBacktest(initial_capital=args.capital, data_dir=data_dir)
    
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
    
    # Save detailed trades to CSV
    if backtest.trades:
        trades_file = Path("figures") / "backtest_regime_ensemble_trades.csv"
        trades_df = pd.DataFrame(backtest.trades)
        trades_df.to_csv(trades_file, index=False)
        print(f"\nDetailed trades saved to {trades_file} ({len(trades_df)} trades)")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

