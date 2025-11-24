#!/usr/bin/env python3
"""Backtest volatility expansion strategy with batch entries and staggered exits."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from bot.config import get_config
from bot.datastore import DataStore
from bot.signals_volatility_expansion import VolatilityExpansionSignalGenerator, VolatilityExpansionSignal
from bot.position_manager import PositionManager, ManagedPosition, BatchEntry


@dataclass
class SimulatedBatchPosition:
    """Track a position with batch entries."""
    pair: str
    strategy_type: str
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit: float
    quality: float
    atr_value: float
    rsi_value: float
    macd_signal: str
    batch_entries: List[Tuple[float, float]]  # (entry_price, amount)
    total_amount: float = 0.0
    average_entry_price: float = 0.0
    exit_1_filled: bool = False
    exit_2_filled: bool = False
    exit_3_filled: bool = False
    highest_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_stop_active: bool = False
    weak_signal_mode: bool = False


class VolatilityExpansionBacktest:
    """Backtest volatility expansion strategy."""
    
    def __init__(
        self,
        data_dir: Path,
        initial_capital: float = 50000.0,
        risk_per_trade_pct: float = 0.015,  # Will be overridden by config at runtime
        max_positions: int = 10
    ):
        """Initialize backtest."""
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.config = get_config()
        # Use config-driven risk by default
        self.risk_per_trade_pct = self.config.risk_per_trade_pct
        self.max_positions = max_positions
        
        # Cool-off rule: Track last stop-out time per pair
        self.last_stopout_time: Dict[str, pd.Timestamp] = {}
        
        self.signal_generator = VolatilityExpansionSignalGenerator(
            rsi_period=self.config.rsi_period,
            rsi_overbought=self.config.rsi_overbought,
            rsi_oversold=self.config.rsi_oversold,
            macd_fast=self.config.macd_fast,
            macd_slow=self.config.macd_slow,
            macd_signal=self.config.macd_signal,
            bb_period=self.config.bb_period,
            bb_std_dev=self.config.bb_std_dev,
            atr_period=self.config.atr_period,
            volume_ma_period=self.config.volume_ma_period,
            volume_spike_threshold=self.config.volume_spike_threshold,
            squeeze_threshold=self.config.squeeze_threshold,
            batch_count=self.config.batch_entry_count,
            batch_spacing_pct=self.config.batch_spacing_pct,
            stop_loss_atr_multiplier=self.config.atr_stop_multiplier
        )
        
        # Trading costs
        self.trading_fee = 0.001
        self.slippage = 0.0005
    
    def load_historical_data(
        self,
        days: Optional[int] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """Load historical data."""
        all_data = {}
        all_timestamps = set()
        
        for csv_file in self.data_dir.glob("*.csv"):
            if csv_file.stem == "state":
                continue
            
            pair = csv_file.stem.replace("_", "")
            try:
                df = pd.read_csv(csv_file)
                if "timestamp" not in df.columns or "price" not in df.columns:
                    continue
                
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.sort_values("timestamp")
                df = df.drop_duplicates(subset=["timestamp"], keep="last")
                df = df.set_index("timestamp")
                df = df.sort_index()
                
                if len(df) > 0:
                    if start_date is not None:
                        df = df[df.index >= start_date]
                    if end_date is not None:
                        df = df[df.index <= end_date]
                    
                    if days is not None and start_date is None and end_date is None:
                        end_date_filter = df.index.max()
                        start_date_filter = end_date_filter - timedelta(days=days)
                        df = df[df.index >= start_date_filter]
                    
                    if len(df) > 72:  # Need at least 72 records (minimum for indicators)
                        all_data[pair] = df
                        all_timestamps.update(df.index)
                        print(f"Loaded {pair}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {pair}: {e}")
                continue
        
        timestamps = pd.DatetimeIndex(sorted(all_timestamps))
        print(f"\nLoaded {len(all_data)} pairs")
        if len(timestamps) > 0:
            print(f"Date range: {timestamps.min()} to {timestamps.max()}")
        
        return all_data, timestamps
    
    def calculate_trade_cost(self, value: float) -> float:
        """Calculate trading cost."""
        return value * (self.trading_fee + self.slippage)
    
    def calculate_dynamic_max_position_pct(
        self,
        current_positions: int,
        max_positions: int,
        current_utilization: float,
        available_cash_pct: float,
        signal_quality: float = 0.5,
        base_max_pct: float = 0.09
    ) -> float:
        """Calculate dynamic maximum position size percentage.
        
        The cap adapts based on:
        - Number of positions: Fewer positions = higher cap per position
        - Utilization: Lower utilization = higher cap per position  
        - Available cash: More cash = higher cap per position
        - Signal quality: Higher quality = slightly higher cap
        
        Args:
            current_positions: Current number of open positions
            max_positions: Maximum allowed positions
            current_utilization: Current portfolio utilization (0.0 to 1.0)
            available_cash_pct: Available cash as percentage of portfolio
            signal_quality: Signal quality (0.0 to 1.0)
            base_max_pct: Base maximum position percentage (default 9%)
            
        Returns:
            Dynamic maximum position size percentage
        """
        # Position count factor: fewer positions = higher cap
        # If we have 1 position out of 10, we can use up to 15% per position
        # If we have 9 positions out of 10, we should use ~5% per position
        position_slots_used = current_positions / max(max_positions, 1)
        position_factor = 1.0 - (position_slots_used * 0.4)  # 1.0 when empty, 0.6 when full
        
        # Utilization factor: lower utilization = higher cap
        # If utilization is 20%, we can use more per position
        # If utilization is 90%, we should use less per position
        utilization_factor = 1.0 - (current_utilization * 0.3)  # 1.0 at 0%, 0.7 at 100%
        
        # Cash availability factor: more cash = higher cap
        # If we have 50% cash, we can use more per position
        # If we have 5% cash, we should use less per position
        cash_factor = 0.7 + (available_cash_pct * 0.6)  # 0.7 at 0% cash, 1.3 at 100% cash
        cash_factor = min(cash_factor, 1.2)  # Cap at 1.2x
        
        # Quality factor: higher quality = slightly higher cap (small effect)
        quality_factor = 0.95 + (signal_quality * 0.1)  # 0.95 to 1.05
        
        # Combine factors
        dynamic_cap = base_max_pct * position_factor * utilization_factor * cash_factor * quality_factor
        
        # Set reasonable bounds: minimum 3%, maximum 20%
        dynamic_cap = max(0.03, min(dynamic_cap, 0.20))
        
        return dynamic_cap
    
    def get_current_prices(
        self,
        all_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp,
        pre_populated_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Get current prices.
        
        Uses latest available price for each pair, with fallback to pre-populated prices.
        This ensures ALL pairs are checked from the start, even if they have different start times.
        
        Args:
            all_data: Dictionary of pair -> DataFrame
            timestamp: Current timestamp
            pre_populated_prices: Pre-populated prices to use as fallback (from first available data)
        """
        prices = {}
        
        for pair, df in all_data.items():
            if len(df) == 0:
                continue
                
            # Strategy: Use the latest available price up to timestamp
            # If no data up to timestamp, fall back to pre-populated price (first available)
            available = df[df.index <= timestamp]
            
            if len(available) > 0:
                # Use the latest available price (even if it's hours before timestamp)
                # This handles timestamp misalignment between pairs
                latest_price = available["price"].iloc[-1]
                prices[pair] = latest_price
            else:
                # No data up to timestamp - this pair starts later
                # Use pre-populated price (first available price) as fallback
                # This ensures all pairs are always checked, regardless of when their data starts
                if pre_populated_prices and pair in pre_populated_prices:
                    prices[pair] = pre_populated_prices[pair]
        
        return prices
    
    def fill_batch_entries(
        self,
        position: SimulatedBatchPosition,
        current_price: float,
        timestamp: pd.Timestamp,
        portfolio_value: float,
        cash: float
    ) -> Tuple[float, float]:
        """Simulate filling batch entries based on price movement.
        
        Returns:
            Tuple of (updated_cash, total_cost)
        """
        total_cost = 0.0
        
        for i, (entry_price, _) in enumerate(position.batch_entries):
            # Check if batch should fill (price reached entry level)
            if position.batch_entries[i][1] > 0:  # Already filled
                continue
            
            # For mean reversion longs: entries are at/below current (buy on dips)
            # For breakout longs: entries are at/above current (buy on breakouts)
            # First batch always fills immediately
            should_fill = False
            if position.macd_signal in ["bullish", "neutral"]:
                if i == 0:
                    should_fill = True  # Always fill first batch immediately
                elif entry_price <= current_price:
                    # Mean reversion: fill when price dips to entry (with tolerance)
                    should_fill = current_price <= entry_price * 1.001
                else:
                    # Breakout: fill when price rises to entry
                    should_fill = current_price >= entry_price
            else:  # Short
                should_fill = current_price <= entry_price
            
            # Safety check: Skip subsequent batches if price is falling significantly
            # Only fill batch 2+ if price is stabilizing or recovering (not falling further)
            if i > 0:  # Not the first batch
                # Check if price has fallen more than 2% below previous batch entry
                prev_batch_price = position.batch_entries[i-1][0] if position.batch_entries[i-1][1] > 0 else current_price
                if current_price < prev_batch_price * 0.98:  # Price fallen > 2%
                    # Skip this batch - price is still falling
                    continue
            
            if should_fill:
                # Calculate amount for this batch using CURRENT portfolio value
                # Utilization-aware dynamic risk + quality scaling
                base_position_pct = self.risk_per_trade_pct
                if hasattr(position, 'weak_signal_mode') and position.weak_signal_mode:
                    base_position_pct = base_position_pct * 0.5  # 50% reduction for weak signals
                # Compute current utilization
                invested_value = portfolio_value - cash
                current_utilization = (invested_value / portfolio_value) if portfolio_value > 0 else 0.0
                target_u = getattr(self.config, "target_utilization", 0.70)
                low_u = getattr(self.config, "utilization_low_threshold", 0.50)
                high_u = getattr(self.config, "utilization_high_threshold", 0.70)
                dynamic_risk = base_position_pct * (1.0 + max(0.0, (target_u - current_utilization)) / max(target_u, 1e-6))
                if current_utilization < low_u:
                    dynamic_risk = max(dynamic_risk, 0.025)
                elif current_utilization > high_u:
                    dynamic_risk = min(dynamic_risk, 0.010)
                quality_multiplier = 0.7 + (position.quality * 0.6)  # Scale 0.7x to 1.3x based on quality
                
                # Calculate dynamic max position cap based on portfolio state
                available_cash_pct = cash / portfolio_value if portfolio_value > 0 else 0.0
                base_max_pct = getattr(self.config, "max_position_pct", 0.09)
                dynamic_cap_pct = self.calculate_dynamic_max_position_pct(
                    current_positions=len(positions),
                    max_positions=self.max_positions,
                    current_utilization=current_utilization,
                    available_cash_pct=available_cash_pct,
                    signal_quality=position.quality,
                    base_max_pct=base_max_pct
                )
                position_pct = min(dynamic_risk * quality_multiplier, dynamic_cap_pct)
                batch_value = (portfolio_value * position_pct) / len(position.batch_entries)
                amount = batch_value / entry_price
                
                # Calculate cost (purchase value + fees)
                purchase_value = amount * entry_price
                cost = self.calculate_trade_cost(purchase_value)
                total_cost += (purchase_value + cost)
                
                # Check if we have enough cash
                if cash >= (purchase_value + cost):
                    position.batch_entries[i] = (entry_price, amount)
                    position.total_amount += amount
                    cash -= (purchase_value + cost)
                else:
                    # Not enough cash - skip this batch
                    continue
        
        # Calculate average entry price
        if position.total_amount > 0:
            total_value = sum(price * amount for price, amount in position.batch_entries if amount > 0)
            position.average_entry_price = total_value / position.total_amount
        
        return cash, total_cost
    
    def check_staggered_exit(
        self,
        position: SimulatedBatchPosition,
        current_price: float
    ) -> Tuple[bool, float, str]:
        """Check if position should exit at staggered profit levels."""
        if position.average_entry_price == 0:
            return False, 0.0, ""
        
        profit_pct = ((current_price - position.average_entry_price) / position.average_entry_price) * 100
        
        # Use scaled exits - optimized for 14-day competition (7%, 15%, 25%)
        if self.config.use_scaled_exits:
            exit_1_level = self.config.exit_1_level  # 7% (optimized for faster exits)
            exit_2_level = self.config.exit_2_level  # 15% (optimized for faster exits)
            exit_3_level = self.config.exit_3_level  # 25% (optimized for faster exits)
            exit_1_pct = self.config.exit_1_pct
            exit_2_pct = self.config.exit_2_pct
            exit_3_pct = self.config.exit_3_pct
        else:
            exit_1_level = 50.0
            exit_2_level = 30.0
            exit_3_level = 20.0
            exit_1_pct = 0.50
            exit_2_pct = 0.30
            exit_3_pct = 0.20
        
        # Exit 1
        if not position.exit_1_filled and profit_pct >= exit_1_level:
            exit_amount = position.total_amount * exit_1_pct
            position.exit_1_filled = True
            return True, exit_amount, f"Staggered exit 1: {profit_pct:.2f}%"
        
        # Exit 2
        if not position.exit_2_filled and profit_pct >= exit_2_level:
            exit_amount = position.total_amount * exit_2_pct
            position.exit_2_filled = True
            return True, exit_amount, f"Staggered exit 2: {profit_pct:.2f}%"
        
        # Exit 3
        if not position.exit_3_filled and profit_pct >= exit_3_level:
            exit_amount = position.total_amount * exit_3_pct
            position.exit_3_filled = True
            return True, exit_amount, f"Staggered exit 3: {profit_pct:.2f}%"
        
        return False, 0.0, ""
    
    def run_backtest(
        self,
        days: Optional[int] = 15,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, List[Dict], Dict]:
        """Run volatility expansion strategy backtest."""
        print(f"\n{'='*60}")
        print(f"Running VOLATILITY EXPANSION Backtest - Last {days} Days")
        print(f"{'='*60}\n")
        
        # Load data
        all_data, timestamps = self.load_historical_data(
            days=days, start_date=start_date, end_date=end_date
        )
        if len(all_data) == 0:
            raise ValueError("No data available")
        
        # Initialize portfolio
        cash = self.initial_capital
        positions: Dict[str, SimulatedBatchPosition] = {}
        
        # Pre-populate prices for all pairs using their first available price
        # This ensures all pairs are checked from the start, even if their data starts later
        pre_populated_prices = {}
        for pair, df in all_data.items():
            if len(df) > 0:
                # Use first available price for pre-population
                pre_populated_prices[pair] = df["price"].iloc[0]
        print(f"Pre-populated prices for {len(pre_populated_prices)} pairs\n")
        
        # Mock datastore - filters data up to current timestamp
        class MockDataStore:
            def __init__(self, historical_data):
                self.historical_data = historical_data
                self.current_timestamp = None  # Will be set before each use
            
            def set_timestamp(self, timestamp):
                """Set the current timestamp for filtering."""
                self.current_timestamp = timestamp
            
            def read_minute_bars(self, pair: str, limit: Optional[int] = None) -> pd.DataFrame:
                if pair in self.historical_data:
                    df = self.historical_data[pair].copy()
                    # Filter to only include data up to current timestamp
                    if self.current_timestamp is not None:
                        df = df[df.index <= self.current_timestamp]
                    if limit:
                        # Get the last 'limit' rows from the filtered data
                        return df.tail(limit)
                    return df
                return pd.DataFrame(columns=["timestamp", "price", "volume"])
        
        datastore = MockDataStore(all_data)
        
        # Results tracking
        results = []
        trades = []
        
        # Sample timestamps (every 5 minutes for minute-level data)
        if len(timestamps) > 0:
            time_diff = (timestamps[1] - timestamps[0]).total_seconds() / 60 if len(timestamps) > 1 else 60
            if time_diff <= 5:
                test_timestamps = timestamps[::5]
            else:
                test_timestamps = timestamps[::1]
        else:
            test_timestamps = timestamps
        
        print(f"Simulating {len(test_timestamps)} trading cycles...\n")
        
        peak_equity = self.initial_capital
        
        # Dynamic start: begin when enough pairs have sufficient history up to the timestamp
        # Need at least 4800 minutes (80 hours = 3.33 days) for 20 four-hour bars
        # (20 * 240 minutes = 4800 minutes) - required for BB (20 period) + RSI (14 period)
        min_data_required = 4800  # minutes
        total_pairs = len(all_data)
        # Require between 20 and 30 pairs (or fewer if dataset smaller)
        required_pairs = max(10, min(30, total_pairs // 2))  # half the universe, capped at 30, floored at 10
        start_idx = 0
        for idx, ts in enumerate(test_timestamps):
            # Count pairs with enough minute bars up to this timestamp
            pairs_with_enough = 0
            for pair, df in all_data.items():
                if len(df) > 0:
                    available_data = df[df.index <= ts]
                    if len(available_data) >= min_data_required:
                        pairs_with_enough += 1
            if pairs_with_enough >= required_pairs:
                start_idx = idx
                print(f"Starting when {pairs_with_enough}/{total_pairs} pairs have ≥ {min_data_required} minute bars (threshold={required_pairs})")
                break
        
        if start_idx > 0:
            print(f"Skipping first {start_idx} cycles (insufficient historical data)")
            test_timestamps = test_timestamps[start_idx:]
            print(f"Starting backtest from cycle {start_idx+1} with {len(test_timestamps)} cycles remaining\n")
        
        for i, timestamp in enumerate(test_timestamps):
            # Update datastore timestamp for this cycle
            datastore.set_timestamp(timestamp)
            
            # Get current prices (with pre-populated fallback)
            prices = self.get_current_prices(all_data, timestamp, pre_populated_prices)
            if not prices:
                continue
            
            # Calculate portfolio value BEFORE any trades
            portfolio_value = cash
            for pair, pos in positions.items():
                if pair in prices:
                    portfolio_value += pos.total_amount * prices[pair]
            
            if portfolio_value > peak_equity:
                peak_equity = portfolio_value
            
            # Calculate current utilization (fraction of equity deployed)
            invested_value = portfolio_value - cash
            current_utilization = (invested_value / portfolio_value) if portfolio_value > 0 else 0.0
            # Expose utilization to signal generator (used for scout mode/diagnostics)
            try:
                setattr(self.signal_generator, "_current_utilization", current_utilization)
            except Exception:
                pass
            
            # Check exit conditions
            for pair, pos in list(positions.items()):
                if pair not in prices:
                    continue
                
                current_price = prices[pair]
                
                # Fill batch entries (deducts cash when batches fill)
                # Use portfolio_value from start of cycle for consistent sizing
                cash, batch_cost = self.fill_batch_entries(pos, current_price, timestamp, portfolio_value, cash)
                
                # Recalculate portfolio value after batch fills (cash changed)
                portfolio_value = cash
                for p, position in positions.items():
                    if p in prices:
                        portfolio_value += position.total_amount * prices[p]
                
                if pos.total_amount == 0:
                    continue  # No filled batches yet
                
                # Update highest price and trailing stop
                if current_price > pos.highest_price:
                    pos.highest_price = current_price
                    # Update trailing stop if active (after first exit)
                    if pos.trailing_stop_active and self.config.use_trailing_stop:
                        pos.trailing_stop_price = pos.highest_price * (1 - self.config.trailing_stop_distance_pct)
                
                # Check trailing stop first (if active)
                if pos.trailing_stop_active and current_price < pos.trailing_stop_price:
                    sell_value = pos.total_amount * current_price
                    cost = self.calculate_trade_cost(sell_value)
                    cash += (sell_value - cost)
                    avg_entry = pos.average_entry_price if pos.average_entry_price > 0 else pos.entry_price
                    pnl = sell_value - (pos.total_amount * avg_entry) - cost
                    
                    trades.append({
                        "time": timestamp,
                        "action": "SELL",
                        "pair": pair,
                        "amount": pos.total_amount,
                        "price": current_price,
                        "value": sell_value,
                        "pnl": pnl,
                        "pnl_pct": (pnl / (pos.total_amount * avg_entry)) * 100 if avg_entry > 0 else 0,
                        "reason": "Trailing Stop"
                    })
                    del positions[pair]
                    # Recalculate portfolio value after exit
                    portfolio_value = cash
                    for p, position in positions.items():
                        if p in prices:
                            portfolio_value += position.total_amount * prices[p]
                    continue
                
                # Check stop loss - use average entry price for P&L calculation
                avg_entry = pos.average_entry_price if pos.average_entry_price > 0 else pos.entry_price
                if current_price <= pos.stop_loss:
                    sell_value = pos.total_amount * current_price
                    cost = self.calculate_trade_cost(sell_value)
                    cash += (sell_value - cost)
                    pnl = sell_value - (pos.total_amount * avg_entry) - cost
                    
                    trades.append({
                        "time": timestamp,
                        "action": "SELL",
                        "pair": pair,
                        "amount": pos.total_amount,
                        "price": current_price,
                        "value": sell_value,
                        "pnl": pnl,
                        "pnl_pct": (pnl / (pos.total_amount * pos.average_entry_price)) * 100 if pos.average_entry_price > 0 else 0,
                        "reason": "Stop Loss"
                    })
                    # Record stop-out time for cool-off rule
                    self.last_stopout_time[pair] = timestamp
                    
                    del positions[pair]
                    # Recalculate portfolio value after exit
                    portfolio_value = cash
                    for p, position in positions.items():
                        if p in prices:
                            portfolio_value += position.total_amount * prices[p]
                    continue
                
                # Check take profit - use average entry price for calculation
                # Take profit should be 25% above average entry (not signal price)
                avg_entry = pos.average_entry_price if pos.average_entry_price > 0 else pos.entry_price
                take_profit_price = avg_entry * 1.25  # 25% above average entry
                if current_price >= take_profit_price:
                    sell_value = pos.total_amount * current_price
                    cost = self.calculate_trade_cost(sell_value)
                    cash += (sell_value - cost)
                    pnl = sell_value - (pos.total_amount * avg_entry) - cost
                    
                    trades.append({
                        "time": timestamp,
                        "action": "SELL",
                        "pair": pair,
                        "amount": pos.total_amount,
                        "price": current_price,
                        "value": sell_value,
                        "pnl": pnl,
                        "pnl_pct": (pnl / (pos.total_amount * pos.average_entry_price)) * 100 if pos.average_entry_price > 0 else 0,
                        "reason": "Take Profit"
                    })
                    del positions[pair]
                    # Recalculate portfolio value after exit
                    portfolio_value = cash
                    for p, position in positions.items():
                        if p in prices:
                            portfolio_value += position.total_amount * prices[p]
                    continue
                
                # Check staggered exits
                should_exit, exit_amount, reason = self.check_staggered_exit(pos, current_price)
                if should_exit and exit_amount > 0:
                    sell_value = exit_amount * current_price
                    cost = self.calculate_trade_cost(sell_value)
                    cash += (sell_value - cost)
                    pnl = sell_value - (exit_amount * pos.average_entry_price) - cost
                    
                    pos.total_amount -= exit_amount
                    
                    trades.append({
                        "time": timestamp,
                        "action": "SELL",
                        "pair": pair,
                        "amount": exit_amount,
                        "price": current_price,
                        "value": sell_value,
                        "pnl": pnl,
                        "pnl_pct": (pnl / (exit_amount * pos.average_entry_price)) * 100 if pos.average_entry_price > 0 else 0,
                        "reason": reason
                    })
                    
                    # Activate trailing stop after first exit (5% profit)
                    if reason.startswith("Staggered exit 1") and self.config.use_trailing_stop and not pos.trailing_stop_active:
                        pos.trailing_stop_price = current_price * (1 - self.config.trailing_stop_distance_pct)
                        pos.trailing_stop_active = True
                    
                    # If position fully closed
                    if pos.total_amount <= 0:
                        del positions[pair]
                    
                    # Recalculate portfolio value after partial/full exit
                    portfolio_value = cash
                    for p, position in positions.items():
                        if p in prices:
                            portfolio_value += position.total_amount * prices[p]
            
            # Recalculate portfolio value before generating new signals (after all exits/batch fills)
            portfolio_value = cash
            for pair, pos in positions.items():
                if pair in prices:
                    portfolio_value += pos.total_amount * prices[pair]
            
            # Generate signals
            if len(positions) < self.max_positions:
                current_data = {}
                for pair, df in all_data.items():
                    current_data[pair] = df[df.index <= timestamp]
                datastore.historical_data = current_data
                
                ticker_data = []
                for pair, price in prices.items():
                    ticker_data.append({
                        "pair": pair,
                        "price": price,
                        "volume_24h": 1000000.0,
                        "bid": price * 0.9999,
                        "ask": price * 1.0001
                    })
                
                all_signals = {}
                signals_checked = 0
                signals_found = 0
                min_quality_threshold = 0.60  # STEP 1: Lower threshold for minimal conditions (was 0.75)
                signal_errors = []
                
                for ticker in ticker_data:
                    pair = ticker["pair"]
                    if pair in positions:
                        continue
                    
                    signals_checked += 1
                    try:
                        # Enable debug for first few pairs only (to avoid spam)
                        debug_mode = (signals_checked <= 3 and i < 10)
                        signal = self.signal_generator.compute_signal(
                            pair, datastore, ticker, debug=debug_mode
                        )
                        # Only allow long positions - filter out short positions (compliance: no shorting)
                        # Filter by quality - only take high quality signals
                        if signal and signal.signal == "buy" and signal.quality >= min_quality_threshold:
                            all_signals[pair] = signal
                            signals_found += 1
                    except Exception as e:
                        # Collect errors for debugging
                        error_msg = f"{pair}: {str(e)[:80]}"
                        if error_msg not in signal_errors:
                            signal_errors.append(error_msg)
                        # Always show first few errors
                        if len(signal_errors) <= 10:
                            import traceback
                            print(f"  ERROR {pair}: {str(e)}")
                            if signals_checked <= 3:
                                print(f"    Traceback: {traceback.format_exc()[:200]}")
                        continue
                
                # Debug output every 500 cycles or when signals found
                if (i + 1) % 500 == 0 or signals_found > 0:
                    # Get diagnostic statistics from signal generator
                    diag = getattr(self.signal_generator, '_diagnostic_counter', {})
                    early = getattr(self.signal_generator, '_early_return_counter', {})
                    diag_str = ""
                    if diag:
                        total = diag.get('total_checked', 0)
                        if total > 0:
                            rsi_30_pct = (diag.get('rsi_oversold', 0) / total) * 100
                            rsi_25_pct = (diag.get('rsi_25', 0) / total) * 100
                            price_pct = (diag.get('price_near_lower', 0) / total) * 100
                            both_pct = (diag.get('both_met', 0) / total) * 100
                            diag_str = f" | Diagnostics: RSI<30={rsi_30_pct:.1f}% (RSI<25={rsi_25_pct:.1f}%), PriceNearBB={price_pct:.1f}%, Both={both_pct:.1f}%"
                    early_str = ""
                    if early:
                        total_early = sum(early.values())
                        if total_early > 0:
                            data_pct = (early.get('insufficient_data', 0) / total_early) * 100
                            h4_pct = (early.get('insufficient_4h', 0) / total_early) * 100
                            reached_pct = (early.get('reached_mean_rev', 0) / total_early) * 100
                            early_str = f" | Early returns: NoData={data_pct:.1f}%, No4H={h4_pct:.1f}%, ReachedMR={reached_pct:.1f}%"
                    print(f"  Cycle {i+1}: Checked {signals_checked} pairs (from {len(ticker_data)} total, {len(prices)} prices available), found {signals_found} signals{diag_str}{early_str}")
                    if signal_errors and (i + 1) % 500 == 0:
                        print(f"  Sample errors ({len(signal_errors)} total):")
                        for err in signal_errors[:5]:
                            print(f"    - {err}")
                
                # Rank signals
                ranked_signals = self.signal_generator.rank_signals(all_signals)
                
                # Debug when we have signals but aren't entering
                if ranked_signals and len(positions) < self.max_positions:
                    print(f"  Found {len(ranked_signals)} ranked signals, current positions: {len(positions)}")
                
                # Enter top signals (can hold multiple positions simultaneously)
                entered_count = 0
                breakout_positions_count = sum(1 for p in positions.values() if p.strategy_type == "breakout")
                max_breakout_positions = int(self.max_positions * self.config.breakout_position_pct)  # Max 50% breakout positions
                
                for pair, signal in ranked_signals:
                    # Check if we've reached max positions
                    if len(positions) >= self.max_positions:
                        break
                    
                    # Limit breakout positions to max_breakout_positions
                    if signal.strategy_type == "breakout" and breakout_positions_count >= max_breakout_positions:
                        continue  # Skip - already at max breakout positions
                    
                    # Cool-off rule: Skip if we had a stop-out recently (within cooldown period)
                    if pair in self.last_stopout_time:
                        time_since_stopout = timestamp - self.last_stopout_time[pair]
                        cooldown_hours = self.config.cooldown_after_stopout_hours
                        if time_since_stopout.total_seconds() < (cooldown_hours * 3600):
                            continue  # Skip this pair - still in cooldown
                    
                    # Calculate position size using CURRENT portfolio value (utilization-aware dynamic risk)
                    target_u = getattr(self.config, "target_utilization", 0.70)
                    low_u = getattr(self.config, "utilization_low_threshold", 0.50)
                    high_u = getattr(self.config, "utilization_high_threshold", 0.70)
                    invested_value_now = portfolio_value - cash
                    current_utilization_now = (invested_value_now / portfolio_value) if portfolio_value > 0 else 0.0
                    base_risk = self.risk_per_trade_pct
                    dynamic_risk = base_risk * (1.0 + max(0.0, (target_u - current_utilization_now)) / max(target_u, 1e-6))
                    if current_utilization_now < low_u:
                        dynamic_risk = max(dynamic_risk, 0.025)
                    elif current_utilization_now > high_u:
                        dynamic_risk = min(dynamic_risk, 0.010)
                    quality_multiplier = 0.7 + (signal.quality * 0.6)
                    
                    # Calculate dynamic max position cap based on portfolio state
                    available_cash_pct = cash / portfolio_value if portfolio_value > 0 else 0.0
                    base_max_pct = getattr(self.config, "max_position_pct", 0.09)
                    dynamic_cap_pct = self.calculate_dynamic_max_position_pct(
                        current_positions=len(positions),
                        max_positions=self.max_positions,
                        current_utilization=current_utilization_now,
                        available_cash_pct=available_cash_pct,
                        signal_quality=signal.quality,
                        base_max_pct=base_max_pct
                    )
                    position_pct = min(dynamic_risk * quality_multiplier, dynamic_cap_pct)
                    position_value = portfolio_value * position_pct
                    
                    # Estimate total cost for all batches (with fees)
                    estimated_total_cost = position_value * (1 + self.trading_fee + self.slippage)
                    
                    # Check if we have enough cash available
                    if cash < estimated_total_cost:
                        # Not enough cash - skip this signal
                        continue
                    
                    # Create position with batch entries
                    batch_entries = [(price, 0.0) for price in signal.batch_entries]
                    
                    # Check if this is a weak signal mode entry
                    weak_signal_mode = "weak_mode=True" in signal.reason or "weak_mode=1" in signal.reason
                    
                    positions[pair] = SimulatedBatchPosition(
                        pair=pair,
                        strategy_type=signal.strategy_type,
                        entry_time=timestamp,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        quality=signal.quality,
                        atr_value=signal.atr_value,
                        rsi_value=signal.rsi_value,
                        macd_signal=signal.macd_signal,
                        batch_entries=batch_entries,
                        weak_signal_mode=weak_signal_mode
                    )
                    
                    # Update breakout count if this is a breakout
                    if signal.strategy_type == "breakout":
                        breakout_positions_count += 1
                    
                    entered_count += 1
                    print(f"  ENTERED {pair}: {signal.strategy_type} signal (quality: {signal.quality:.2f}, reason: {signal.reason[:50]})")
                    
                    trades.append({
                        "time": timestamp,
                        "action": "BUY",
                        "pair": pair,
                        "amount": 0.0,  # Will be filled as batches execute
                        "price": signal.entry_price,
                        "value": position_value,
                        "reason": f"{signal.strategy_type}: {signal.reason}"
                    })
                
                if entered_count > 0:
                    print(f"  Opened {entered_count} new positions. Total positions: {len(positions)}")
            
            # Record snapshot
            results.append({
                "timestamp": timestamp,
                "cash": cash,
                "portfolio_value": portfolio_value,
                "positions": len(positions),
                "equity": portfolio_value
            })
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(test_timestamps)} cycles, "
                      f"Portfolio: ${portfolio_value:,.2f}, Positions: {len(positions)}")
        
        # Value remaining positions
        final_prices = self.get_current_prices(all_data, test_timestamps[-1])
        final_value = cash
        for pair, pos in list(positions.items()):
            if pair in final_prices:
                final_value += pos.total_amount * final_prices[pair]
        
        # Calculate metrics
        results_df = pd.DataFrame(results)
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        returns = results_df["equity"].pct_change().dropna()
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0001
        sortino_ratio = (returns.mean() / downside_std * np.sqrt(252 * 24)) if downside_std > 0 else 0
        
        # Sharpe ratio
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252 * 24)) if returns.std() > 0 else 0
        
        # Calmar ratio
        max_dd = ((results_df["equity"] / results_df["equity"].expanding().max()) - 1).min() * 100
        annualized_return = (1 + total_return / 100) ** (365 / len(results_df)) - 1 if len(results_df) > 0 else 0
        calmar_ratio = (annualized_return / abs(max_dd / 100)) if max_dd != 0 else 0
        
        # Competition score
        competition_score = 0.4 * sortino_ratio + 0.3 * sharpe_ratio + 0.3 * calmar_ratio
        
        # Print summary
        print(f"\n{'='*60}")
        print("VOLATILITY EXPANSION BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        print(f"Sortino Ratio: {sortino_ratio:.3f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Calmar Ratio: {calmar_ratio:.3f}")
        print(f"Competition Score: {competition_score:.3f}")
        print(f"Max Drawdown: {max_dd:.2f}%")
        
        if trades:
            buy_trades = [t for t in trades if t["action"] == "BUY"]
            sell_trades = [t for t in trades if t["action"] == "SELL"]
            print(f"Buy Trades: {len(buy_trades)}")
            print(f"Sell Trades: {len(sell_trades)}")
            
            if sell_trades:
                winning_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
                win_rate = (len(winning_trades) / len(sell_trades)) * 100
                avg_pnl = np.mean([t.get("pnl", 0) for t in sell_trades])
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Average P&L per trade: ${avg_pnl:.2f}")
        
        # Compile metrics
        metrics = {
            "start_date": test_timestamps[0] if len(test_timestamps) > 0 else None,
            "end_date": test_timestamps[-1] if len(test_timestamps) > 0 else None,
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "sortino_ratio": sortino_ratio,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "competition_score": competition_score,
            "max_drawdown": max_dd,
            "n_trades": len(trades),
            "win_rate": (len([t for t in trades if t["action"] == "SELL" and t.get("pnl", 0) > 0]) / 
                        len([t for t in trades if t["action"] == "SELL"]) * 100) if len([t for t in trades if t["action"] == "SELL"]) > 0 else 0
        }
        
        return results_df, trades, metrics


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest volatility expansion strategy")
    parser.add_argument("--days", type=int, default=15, help="Number of days to backtest")
    parser.add_argument("--capital", type=float, default=50000.0, help="Initial capital")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    
    args = parser.parse_args()
    
    backtest = VolatilityExpansionBacktest(args.data_dir, args.capital, max_positions=10)
    results_df, trades, metrics = backtest.run_backtest(days=args.days)
    
    # Save results
    output_dir = args.data_dir.parent
    results_df.to_csv(output_dir / "backtest_volatility_expansion_results.csv", index=False)
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(output_dir / "backtest_volatility_expansion_trades.csv", index=False)
        print(f"\nResults saved to:")
        print(f"  - backtest_volatility_expansion_results.csv")
        print(f"  - backtest_volatility_expansion_trades.csv")


if __name__ == "__main__":
    main()

