"""
Linear Momentum Trading Strategy for Bitcoin
Uses a linear momentum model to predict BTC price at t+1 and trade accordingly.
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from decimal import Decimal, ROUND_DOWN

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    # Note: statsmodels not available - will use provided coefficients only

from bot.roostoo_v3 import RoostooV3Client, ExchangeInfo
from bot.config import get_config


class LinearMomentumModel:
    """Linear momentum model for BTC price prediction."""
    
    def __init__(self, fit_model: bool = False, df: Optional[pd.DataFrame] = None):
        """Initialize with the provided model coefficients or fit from data.
        
        Args:
            fit_model: If True, fit model from data. If False, use provided coefficients.
            df: DataFrame with price data (required if fit_model=True)
        """
        if fit_model and df is not None:
            self._fit_model(df)
        else:
            # Use provided coefficients
            self.intercept = 1166.079278
            self.coef_close = 0.988583
            self.coef_change = 3309.553529
            self.coef_accel = -0.036025
    
    def _fit_model(self, df: pd.DataFrame):
        """Fit linear regression model from data."""
        if not HAS_STATSMODELS:
            print("Note: statsmodels not available, using provided coefficients")
            self.intercept = 1166.079278
            self.coef_close = 0.988583
            self.coef_change = 3309.553529
            self.coef_accel = -0.036025
            return
        
        print("\nFitting linear momentum model...")
        
        # Calculate features
        df_features = self.calculate_features(df)
        
        # Prepare data (remove NaN rows)
        df_clean = df_features.dropna(subset=['close', 'price_change_pct', 'acceleration', 'target_price'])
        
        if len(df_clean) < 10:
            print("⚠ Not enough data to fit model, using provided coefficients")
            self.intercept = 1166.079278
            self.coef_close = 0.988583
            self.coef_change = 3309.553529
            self.coef_accel = -0.036025
            return
        
        # Prepare features and target
        X = df_clean[['close', 'price_change_pct', 'acceleration']]
        y = df_clean['target_price']
        
        # Add intercept
        X_with_const = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X_with_const).fit()
        
        # Extract coefficients
        self.intercept = model.params['const']
        self.coef_close = model.params['close']
        self.coef_change = model.params['price_change_pct']
        self.coef_accel = model.params['acceleration']
        
        # Print model summary
        print("\n" + "="*60)
        print("LINEAR MOMENTUM MODEL SUMMARY")
        print("="*60)
        print(model.summary())
        print("="*60)
        print(f"\nModel Equation:")
        print(f"target_price = {self.intercept:.6f} + {self.coef_close:.6f}*close + "
              f"{self.coef_change:.6f}*price_change_pct + {self.coef_accel:.6f}*acceleration")
        print(f"R-squared: {model.rsquared:.4f}")
        print("="*60 + "\n")
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price_change_pct and acceleration from close prices."""
        df = df.copy()
        
        # Calculate price change percentage (beta)
        df['price_change_pct'] = df['close'].pct_change() * 100  # Convert to percentage
        
        # Calculate acceleration (derivative of price_change_pct)
        df['acceleration'] = df['price_change_pct'].diff()
        
        # Create target (price at t+1)
        df['target_price'] = df['close'].shift(-1)
        
        return df
    
    def predict(self, close: float, price_change_pct: float, acceleration: float) -> float:
        """Predict price at t+1."""
        prediction = (
            self.intercept +
            self.coef_close * close +
            self.coef_change * price_change_pct +
            self.coef_accel * acceleration
        )
        return prediction
    
    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict prices for entire dataframe."""
        predictions = []
        
        for idx in range(len(df)):
            if pd.isna(df.iloc[idx]['price_change_pct']) or pd.isna(df.iloc[idx]['acceleration']):
                predictions.append(np.nan)
            else:
                pred = self.predict(
                    df.iloc[idx]['close'],
                    df.iloc[idx]['price_change_pct'],
                    df.iloc[idx]['acceleration']
                )
                predictions.append(pred)
        
        return pd.Series(predictions, index=df.index)


class MomentumTrader:
    """Trading bot using linear momentum predictions."""
    
    def __init__(
        self,
        initial_capital: float = 50000.0,
        transaction_fee: float = 0.001,
        enable_live_orders: bool = False,
        trading_pair: str = "BTC/USD",
        min_trade_interval_minutes: int = 5
    ):
        """Initialize trader.
        
        Args:
            initial_capital: Starting capital in USD
            transaction_fee: Transaction fee as decimal (0.001 = 0.1%)
            enable_live_orders: If True, will forward trades to Roostoo API
            trading_pair: Trading pair symbol used on Roostoo
            min_trade_interval_minutes: Minimum minutes between trades (default: 5)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.btc_held = 0.0
        self.transaction_fee = transaction_fee
        self.enable_live_orders = enable_live_orders
        self.trading_pair = trading_pair
        
        self.model = LinearMomentumModel()
        self.client: Optional[RoostooV3Client] = None
        
        # Trading history
        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []
        
        # Current position
        self.position: Optional[str] = None  # 'LONG', 'SHORT', or None
        self.entry_price: Optional[float] = None
        self.entry_timestamp: Optional[pd.Timestamp] = None
        
        # Risk management
        self.stop_loss_multiplier: float = 0.8  # Stop loss at 0.8x entry (20% loss)
        self.take_profit_multiplier: float = 1.5  # Take profit at 1.5x entry (50% gain)
        
        # Trade frequency controls
        self.min_trade_interval_minutes: int = min_trade_interval_minutes  # Configurable minimum minutes between trades
        self.cooldown_minutes: int = 1  # 1 minute cooldown after any trade
        self.last_trade_time: Optional[pd.Timestamp] = None
        
        # Exchange precision / rules (defaults aligned with doc)
        self.price_precision: int = 2
        self.amount_precision: int = 6
        self.min_order_value: float = 1.0  # MiniOrder (price * quantity)
        self.quantity_step: float = 10 ** (-self.amount_precision)
    
    def _round_down(self, value: float, precision: int) -> float:
        """Round value down to required precision using Decimal."""
        if precision < 0:
            return value
        quantizer = Decimal("1").scaleb(-precision)
        rounded = Decimal(str(value)).quantize(quantizer, rounding=ROUND_DOWN)
        return float(rounded)
    
    def _load_exchange_rules(self):
        """Fetch exchange metadata (precision, min order) from Roostoo."""
        if self.client is None:
            return
        
        try:
            infos = self.client.exchange_info()
            target_pair = next(
                (info for info in infos if isinstance(info, ExchangeInfo) and info.pair == self.trading_pair),
                None
            )
            if target_pair is None:
                print(f"[WARN] Pair {self.trading_pair} not found in exchange info. Using defaults.")
                return
            
            self.price_precision = target_pair.price_precision
            self.amount_precision = target_pair.amount_precision
            self.min_order_value = float(target_pair.mini_order)
            self.quantity_step = 10 ** (-self.amount_precision)
            
            print(
                "[OK] Loaded exchange rules:",
                f"price_precision={self.price_precision},",
                f"amount_precision={self.amount_precision},",
                f"min_order_value={self.min_order_value}"
            )
        except Exception as exc:
            print(f"[WARN] Could not load exchange info: {exc}. Using default precision.")
        
    def initialize_api(self):
        """Initialize Roostoo API client."""
        try:
            config = get_config()
            self.client = RoostooV3Client(
                api_key=config.api_key,
                api_secret=config.api_secret,
                base_url=config.base_url
            )
            print("[OK] Roostoo API initialized")
            self._load_exchange_rules()
        except Exception as e:
            print(f"[WARN] Could not initialize Roostoo API: {e}")
            print("  Running in simulation mode only")
            self.client = None
    
    def get_current_price(self) -> Optional[float]:
        """Get current BTC price from Roostoo API."""
        if self.client is None:
            return None
        
        try:
            ticker = self.client.ticker("BTC/USD")
            return ticker.last_price
        except Exception as e:
            print(f"[WARN] Error fetching price: {e}")
            return None
    
    def get_portfolio_value(self, current_price: Optional[float] = None) -> float:
        """Calculate total portfolio value in USD."""
        if current_price is None:
            current_price = self.get_current_price()
            if current_price is None:
                # Fallback: use last known price or estimate
                if self.equity_history:
                    return self.equity_history[-1]['equity']
                return self.cash
        
        return self.cash + (self.btc_held * current_price)
    
    def _can_trade(self, current_time: pd.Timestamp) -> bool:
        """Check if enough time has passed since last trade."""
        if self.last_trade_time is None:
            return True
        
        time_since_last = current_time - self.last_trade_time
        minutes_since_last = time_since_last.total_seconds() / 60.0
        
        # Must wait at least min_trade_interval_minutes between trades
        return minutes_since_last >= self.min_trade_interval_minutes
    
    def _check_stop_loss_take_profit(self, current_price: float, current_time: pd.Timestamp) -> Optional[str]:
        """Check if stop loss or take profit should trigger.
        
        Returns:
            'STOP_LOSS' if stop loss triggered
            'TAKE_PROFIT' if take profit triggered
            None otherwise
        """
        if self.position != 'LONG' or self.entry_price is None:
            return None
        
        # Check stop loss (0.8x entry = 20% loss)
        if current_price <= self.entry_price * self.stop_loss_multiplier:
            return 'STOP_LOSS'
        
        # Check take profit (1.5x entry = 50% gain)
        if current_price >= self.entry_price * self.take_profit_multiplier:
            return 'TAKE_PROFIT'
        
        return None
    
    def should_buy(self, current_price: float, predicted_price: float, current_time: pd.Timestamp) -> bool:
        """Determine if we should buy based on prediction and trade frequency."""
        if pd.isna(predicted_price):
            return False
        
        # Can't buy if already in a position
        if self.position == 'LONG':
            return False
        
        # Check trade frequency limits
        if not self._can_trade(current_time):
            return False
        
        # Buy if predicted price is higher than current (expecting price increase)
        expected_return = (predicted_price - current_price) / current_price
        
        # Only buy if expected return exceeds transaction costs
        min_return = self.transaction_fee * 2  # Round trip cost
        return expected_return > min_return
    
    def should_sell(self, current_price: float, predicted_price: float, current_time: pd.Timestamp) -> bool:
        """Determine if we should sell based on prediction and trade frequency."""
        if pd.isna(predicted_price):
            return False
        
        # Can't sell if not in a position
        if self.position != 'LONG':
            return False
        
        # Check trade frequency limits (but allow stop loss/take profit to override)
        # We'll check this in the main loop for regular sells, but stop loss/take profit bypass this
        
        # Sell if predicted price is lower than current (expecting price decrease)
        expected_return = (current_price - predicted_price) / current_price
        
        # Only sell if expected return exceeds transaction costs
        min_return = self.transaction_fee * 2  # Round trip cost
        return expected_return > min_return
    
    def _clip_quantity_to_rules(self, quantity: float) -> float:
        """Adjust quantity to obey precision and minimum size."""
        if quantity <= 0:
            return 0.0
        
        quantized = self._round_down(quantity, self.amount_precision)
        # Ensure quantized value respects minimum step
        if quantized < self.quantity_step:
            return 0.0
        return quantized
    
    def execute_buy(self, price: float, timestamp: pd.Timestamp) -> bool:
        """Execute buy order (simulated or real)."""
        # Calculate max notional we can allocate while accounting for fees
        max_notional = self.cash * 0.99 / (1 + self.transaction_fee)
        if max_notional <= 0:
            return False
        
        raw_quantity = max_notional / price
        quantity = self._clip_quantity_to_rules(raw_quantity)
        notional = quantity * price
        
        if quantity <= 0 or notional < self.min_order_value:
            return False
        
        fee_amount = notional * self.transaction_fee
        total_spent = notional + fee_amount
        
        if total_spent > self.cash:
            return False
        
        # Update portfolio
        self.cash -= total_spent
        self.btc_held += quantity
        self.position = 'LONG'
        self.entry_price = price
        self.entry_timestamp = timestamp
        self.last_trade_time = timestamp  # Update last trade time
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'action': 'BUY',
            'price': price,
            'quantity': quantity,
            'fee': fee_amount,
            'cash_after': self.cash,
            'btc_after': self.btc_held,
            'entry_price': price
        })
        
        # Try to execute on exchange if API available
        if self.client is not None and self.enable_live_orders:
            try:
                order = self.client.place_order(
                    pair=self.trading_pair,
                    side="BUY",
                    type="MARKET",
                    quantity=quantity
                )
                print(f"  [OK] Real order placed: {order.order_id}")
            except Exception as e:
                print(f"  [WARN] Real order failed: {e} (using simulation)")
        
        return True
    
    def execute_sell(self, price: float, timestamp: pd.Timestamp) -> bool:
        """Execute sell order (simulated or real)."""
        if self.btc_held <= 0:
            return False
        
        # Sell available BTC respecting precision
        quantity = self._clip_quantity_to_rules(self.btc_held)
        if quantity <= 0:
            return False
        
        gross_proceeds = quantity * price
        fee_amount = gross_proceeds * self.transaction_fee
        net_proceeds = gross_proceeds - fee_amount
        
        # Update portfolio
        self.cash += net_proceeds
        self.btc_held = max(self.btc_held - quantity, 0.0)
        if self.btc_held < self.quantity_step:
            self.btc_held = 0.0  # Clear dust
        self.position = None
        
        # Calculate P&L
        entry_price_for_pnl = self.entry_price if self.entry_price else price
        pnl = (price - entry_price_for_pnl) * quantity - fee_amount
        pnl_pct = ((price - entry_price_for_pnl) / entry_price_for_pnl) * 100 if entry_price_for_pnl > 0 else 0
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'action': 'SELL',
            'price': price,
            'quantity': quantity,
            'fee': fee_amount,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_price': entry_price_for_pnl,
            'cash_after': self.cash,
            'btc_after': self.btc_held
        })
        
        # Reset position tracking
        self.entry_price = None
        self.entry_timestamp = None
        self.last_trade_time = timestamp  # Update last trade time
        
        # Try to execute on exchange if API available
        if self.client is not None and self.enable_live_orders:
            try:
                order = self.client.place_order(
                    pair=self.trading_pair,
                    side="SELL",
                    type="MARKET",
                    quantity=quantity
                )
                print(f"  [OK] Real order placed: {order.order_id}")
            except Exception as e:
                print(f"  [WARN] Real order failed: {e} (using simulation)")
        
        return True
    
    def backtest(self, df: pd.DataFrame, live_plot: bool = False, plot_update_interval: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Backtest strategy on historical data.
        
        Args:
            df: Input dataframe with price data
            live_plot: If True, update plots in real-time during backtest
            plot_update_interval: Update plot every N trades
            
        Returns:
            Tuple of (results_df, df_with_predictions)
        """
        print("\n" + "="*60)
        print("BACKTESTING LINEAR MOMENTUM STRATEGY")
        print("="*60)
        
        # Calculate features and predictions (keep a copy for plotting)
        df_with_features = self.model.calculate_features(df.copy())
        df_with_features['predicted_price'] = self.model.predict_batch(df_with_features)
        
        # Reset portfolio
        self.cash = self.initial_capital
        self.btc_held = 0.0
        self.position = None
        self.entry_price = None
        self.entry_timestamp = None
        self.last_trade_time = None
        self.trades = []
        self.equity_history = []
        
        # Setup live plotting if requested
        if live_plot:
            plt.ion()  # Turn on interactive mode
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            ax1, ax2, ax3 = axes
            ax2_twin = ax2.twinx()
            
            # Initialize plots
            price_line, = ax1.plot([], [], label='Actual BTC Price', linewidth=2, alpha=0.7)
            pred_line, = ax1.plot([], [], label='Predicted Price (t+1)', linewidth=1.5, alpha=0.8, linestyle='--')
            buy_markers = ax1.scatter([], [], color='green', marker='^', s=100, zorder=5, label='Buy')
            sell_markers = ax1.scatter([], [], color='red', marker='v', s=100, zorder=5, label='Sell')
            
            change_line, = ax2.plot([], [], label='Price Change % (β)', color='blue', linewidth=1.5, alpha=0.7)
            accel_line, = ax2_twin.plot([], [], label='Acceleration (dβ/dt)', color='orange', linewidth=1.5, alpha=0.7)
            
            equity_line, = ax3.plot([], [], label='Portfolio Value', linewidth=2, color='green')
            initial_line = ax3.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                                      label=f'Initial Capital (${self.initial_capital:,.0f})', alpha=0.7)
            
            ax1.set_title('Bitcoin Price: Actual vs Predicted (Live)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price (USD)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_title('Momentum Features', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Price Change %', color='blue')
            ax2_twin.set_ylabel('Acceleration', color='orange')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2_twin.tick_params(axis='y', labelcolor='orange')
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            ax3.set_title('Portfolio Equity Curve (Live)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Portfolio Value (USD)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show(block=False)
        
        # Iterate through data
        for idx in range(1, len(df_with_features)):  # Start from 1 to have previous values
            row = df_with_features.iloc[idx]
            prev_row = df_with_features.iloc[idx-1]
            
            current_price = row['close']
            predicted_price = prev_row['predicted_price']  # Use prediction from previous step
            
            # Skip if we don't have a valid prediction
            if pd.isna(predicted_price):
                continue
            
            current_time = row.name
            
            # First, check stop loss and take profit (these bypass trade frequency limits)
            sl_tp_result = self._check_stop_loss_take_profit(current_price, current_time)
            if sl_tp_result:
                # Capture entry price before sell resets it
                entry_price_for_msg = self.entry_price
                if sl_tp_result == 'STOP_LOSS':
                    if self.execute_sell(current_price, current_time):
                        pnl_pct = ((current_price - entry_price_for_msg) / entry_price_for_msg) * 100 if entry_price_for_msg else 0
                        print(f"[{current_time}] STOP LOSS @ ${current_price:.2f} | Entry: ${entry_price_for_msg:.2f} | Loss: {pnl_pct:.2f}%")
                elif sl_tp_result == 'TAKE_PROFIT':
                    if self.execute_sell(current_price, current_time):
                        pnl_pct = ((current_price - entry_price_for_msg) / entry_price_for_msg) * 100 if entry_price_for_msg else 0
                        print(f"[{current_time}] TAKE PROFIT @ ${current_price:.2f} | Entry: ${entry_price_for_msg:.2f} | Gain: {pnl_pct:.2f}%")
            else:
                # Regular trading logic (subject to frequency limits)
                if self.should_buy(current_price, predicted_price, current_time):
                    if self.execute_buy(current_price, current_time):
                        print(f"[{current_time}] BUY @ ${current_price:.2f} | Predicted: ${predicted_price:.2f}")
                
                elif self.should_sell(current_price, predicted_price, current_time):
                    # Check cooldown for regular sells (stop loss/take profit bypass this)
                    if self._can_trade(current_time):
                        if self.execute_sell(current_price, current_time):
                            entry_price = self.trades[-2].get('entry_price', current_price) if len(self.trades) >= 2 else current_price
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                            print(f"[{current_time}] SELL @ ${current_price:.2f} | Predicted: ${predicted_price:.2f} | P&L: {pnl_pct:.2f}%")
            
            # Record equity
            portfolio_value = self.get_portfolio_value(current_price)
            self.equity_history.append({
                'timestamp': row.name,
                'equity': portfolio_value,
                'cash': self.cash,
                'btc_value': self.btc_held * current_price,
                'price': current_price
            })
            
            # Update live plot
            if live_plot and len(self.trades) % plot_update_interval == 0:
                # Update price plot
                timestamps = df_with_features.index[:idx+1]
                price_line.set_data(timestamps, df_with_features['close'].iloc[:idx+1])
                pred_line.set_data(timestamps, df_with_features['predicted_price'].iloc[:idx+1])
                
                # Update trade markers
                buy_times = [t['timestamp'] for t in self.trades if t['action'] == 'BUY']
                buy_prices = [t['price'] for t in self.trades if t['action'] == 'BUY']
                sell_times = [t['timestamp'] for t in self.trades if t['action'] == 'SELL']
                sell_prices = [t['price'] for t in self.trades if t['action'] == 'SELL']
                
                if buy_times:
                    buy_markers.set_offsets(np.column_stack([pd.to_datetime(buy_times), buy_prices]))
                if sell_times:
                    sell_markers.set_offsets(np.column_stack([pd.to_datetime(sell_times), sell_prices]))
                
                # Update momentum features
                change_line.set_data(timestamps, df_with_features['price_change_pct'].iloc[:idx+1])
                accel_line.set_data(timestamps, df_with_features['acceleration'].iloc[:idx+1])
                
                # Update equity curve
                equity_times = [e['timestamp'] for e in self.equity_history]
                equity_values = [e['equity'] for e in self.equity_history]
                equity_line.set_data(equity_times, equity_values)
                
                # Update axis limits
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                ax2_twin.relim()
                ax2_twin.autoscale_view()
                ax3.relim()
                ax3.autoscale_view()
                
                plt.draw()
                plt.pause(0.01)  # Small pause to allow GUI update
        
        # Final sell if still holding
        if self.btc_held > 0:
            final_price = df_with_features.iloc[-1]['close']
            self.execute_sell(final_price, df_with_features.index[-1])
            print(f"[{df_with_features.index[-1]}] FINAL SELL @ ${final_price:.2f}")
        
        # Create results dataframe
        results_df = pd.DataFrame(self.equity_history)
        results_df.set_index('timestamp', inplace=True)
        
        if live_plot:
            plt.ioff()  # Turn off interactive mode
            print("\n[OK] Live plotting complete. Close plot window to continue...")
            plt.show(block=True)  # Block until window is closed
        
        return results_df, df_with_features
    
    def plot_results(self, df: pd.DataFrame, results_df: pd.DataFrame, save_path: Optional[Path] = None):
        """Plot predictions and trading results."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Price and Predictions
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Actual BTC Price', linewidth=2, alpha=0.7)
        ax1.plot(df.index, df['predicted_price'], label='Predicted Price (t+1)', 
                linewidth=1.5, alpha=0.8, linestyle='--')
        
        # Mark trades
        for trade in self.trades:
            if trade['action'] == 'BUY':
                ax1.scatter(trade['timestamp'], trade['price'], color='green', 
                           marker='^', s=100, zorder=5, label='Buy' if trade == self.trades[0] else '')
            elif trade['action'] == 'SELL':
                ax1.scatter(trade['timestamp'], trade['price'], color='red', 
                           marker='v', s=100, zorder=5, label='Sell' if self.trades.count(trade) == 1 else '')
        
        ax1.set_title('Bitcoin Price: Actual vs Predicted (Linear Momentum Model)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Momentum Features
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(df.index, df['price_change_pct'], label='Price Change % (β)', 
                color='blue', linewidth=1.5, alpha=0.7)
        ax2_twin.plot(df.index, df['acceleration'], label='Acceleration (dβ/dt)', 
                     color='orange', linewidth=1.5, alpha=0.7)
        
        ax2.set_title('Momentum Features', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price Change %', color='blue')
        ax2_twin.set_ylabel('Acceleration', color='orange')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Equity Curve
        ax3 = axes[2]
        ax3.plot(results_df.index, results_df['equity'], label='Portfolio Value', 
                linewidth=2, color='green')
        ax3.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                   label=f'Initial Capital (${self.initial_capital:,.0f})', alpha=0.7)
        
        ax3.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Portfolio Value (USD)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print trading summary."""
        final_equity = results_df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        total_trades = len(self.trades)
        buy_trades = sum(1 for t in self.trades if t['action'] == 'BUY')
        sell_trades = sum(1 for t in self.trades if t['action'] == 'SELL')
        
        total_fees = sum(t.get('fee', 0) for t in self.trades)
        
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        win_rate = (len(winning_trades) / sell_trades * 100) if sell_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Calculate total profit and total loss
        total_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        total_loss = sum([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
        
        print("\n" + "="*60)
        print("TRADING SUMMARY")
        print("="*60)
        print(f"Initial Capital:      ${self.initial_capital:,.2f}")
        print(f"Final Equity:          ${final_equity:,.2f}")
        print(f"Total Return:          {total_return:+.2f}%")
        print(f"Total Fees Paid:       ${total_fees:,.2f}")
        print(f"\nTotal Trades:          {total_trades} ({buy_trades} buys, {sell_trades} sells)")
        print(f"Win Rate:              {win_rate:.1f}%")
        if winning_trades:
            print(f"Winning Trades:        {len(winning_trades)}")
            print(f"Total Profit:          ${total_profit:,.2f}")
            print(f"Average Win:           ${avg_win:,.2f}")
        if losing_trades:
            print(f"Losing Trades:         {len(losing_trades)}")
            print(f"Total Loss:            ${total_loss:,.2f}")
            print(f"Average Loss:          ${avg_loss:,.2f}")
        print("="*60)

        sharpe, calmar, cagr = compute_performance_metrics(results_df['equity'])
        print("Performance Ratios:")
        if np.isnan(sharpe):
            print("  Sharpe Ratio:        n/a")
        else:
            print(f"  Sharpe Ratio:        {sharpe:.3f}")
        if np.isnan(calmar):
            print("  Calmar Ratio:        n/a")
        else:
            print(f"  Calmar Ratio:        {calmar:.3f}")
        if np.isnan(cagr):
            print("  CAGR:                n/a")
        else:
            print(f"  CAGR:                {cagr*100:+.2f}%")
        print("="*60)


def compute_performance_metrics(equity_series: pd.Series) -> Tuple[float, float, float]:
    """Compute Sharpe ratio, Calmar ratio, and CAGR from an equity curve."""
    equity = equity_series.astype(float)
    if len(equity) < 2:
        return np.nan, np.nan, np.nan

    returns = equity.pct_change().dropna()
    if returns.empty:
        return np.nan, np.nan, np.nan

    intervals = equity.index.to_series().diff().dropna()
    if not intervals.empty:
        seconds_per_period = intervals.dt.total_seconds().median()
    else:
        seconds_per_period = 300.0  # default to 5-minute data

    seconds_per_year = 365 * 24 * 60 * 60
    periods_per_year = seconds_per_year / seconds_per_period if seconds_per_period > 0 else 105120.0

    mean_return = returns.mean()
    std_return = returns.std(ddof=0)
    sharpe = np.nan
    if std_return > 0:
        sharpe = np.sqrt(periods_per_year) * mean_return / std_return

    duration_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = duration_seconds / seconds_per_year if duration_seconds > 0 else np.nan
    cagr = np.nan
    if years and years > 0:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    max_drawdown = drawdowns.min()
    calmar = np.nan
    if cagr is not np.nan and max_drawdown < 0:
        calmar = cagr / abs(max_drawdown)

    return sharpe, calmar, cagr


def fetch_data_from_api(
    client: RoostooV3Client,
    pair: str,
    days: int = 15,
    start_date: Optional[pd.Timestamp] = None,
) -> Optional[pd.DataFrame]:
    """Fetch historical data from Roostoo API for the last N days."""
    range_text = f"{days} days"
    if start_date is not None:
        range_text = f"{days} days starting {start_date.date()}"
    print(f"\nFetching {range_text} of historical data from Roostoo API...")
    
    try:
        # Calculate time range
        if start_date is not None:
            start_time_dt = start_date.tz_localize("UTC") if start_date.tzinfo is None else start_date
            end_time_dt = start_time_dt + pd.Timedelta(days=days)
        else:
            end_time_dt = pd.Timestamp.utcnow().tz_localize("UTC")
            start_time_dt = end_time_dt - pd.Timedelta(days=days)
        
        end_time = int(end_time_dt.timestamp() * 1000)
        start_time = int(start_time_dt.timestamp() * 1000)
        
        # Fetch klines
        klines = client.get_klines(
            pair=pair,
            interval="5m",
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Request large limit to get all data
        )
        
        if not klines:
            print("[WARN] No klines data returned from API")
            return None
        
        # Convert to DataFrame
        data = []
        for k in klines:
            data.append({
                'open_time': pd.to_datetime(k['open_time'], unit='ms', utc=True),
                'open': k['open'],
                'high': k['high'],
                'low': k['low'],
                'close': k['close'],
                'volume': k['volume']
            })
        
        df = pd.DataFrame(data)
        if df.empty:
            return None
        
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"[OK] Fetched {len(df)} data points from API")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        print(f"[WARN] Failed to fetch data from API: {e}")
        return None


def main():
    """Main execution function."""
    # Initialize API client
    client = None
    try:
        client = RoostooV3Client()
        print("[OK] Roostoo API client initialized")
    except Exception as e:
        print(f"[WARN] Could not initialize API client: {e}")
        print("[WARN] Will attempt to use CSV data as fallback")
    
    # Ask user for lookback window
    print("\n" + "="*60)
    print("DATA SOURCE SETTINGS")
    print("="*60)
    print("How many days back should the simulation cover? (default=15)")
    print("The bot will attempt to fetch real data from Roostoo API.")
    try:
        lookback_input = input("\nEnter lookback window in days (default=15): ").strip()
        lookback_days = int(lookback_input) if lookback_input else 15
        if lookback_days < 1:
            print("[WARN] Lookback days must be at least 1. Using 15 days.")
            lookback_days = 15
    except (ValueError, KeyboardInterrupt, EOFError):
        lookback_days = 15
        print("\nUsing default: 15 days")

    print("\nWould you like to analyse a specific historical window?")
    print("Leave blank to use the most recent period.")
    start_input = input("Enter custom start date (YYYY-MM-DD) or press Enter: ").strip()
    custom_start: Optional[pd.Timestamp] = None
    if start_input:
        try:
            custom_start = pd.to_datetime(start_input)
            print(f"[OK] Using custom start date: {custom_start.date()}")
        except ValueError:
            print("[WARN] Could not parse start date. Using most recent period.")
            custom_start = None
    
    # Try to fetch from API first
    df = None
    if client:
        df = fetch_data_from_api(client, "BTC/USD", days=lookback_days, start_date=custom_start)
    
    # Fallback to CSV if API fails
    if df is None or df.empty:
        print("\n[INFO] Falling back to CSV data...")
        possible_paths = [
            Path("data/historical/BTC_5m.csv"),
            Path("TradingBot/data/historical/BTC_5m.csv"),
            Path(__file__).parent / "data/historical/BTC_5m.csv"
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None or not data_path.exists():
            print(f"Error: Data file not found. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            return
        
        print("Loading historical BTC data from CSV...")
        df = pd.read_csv(data_path)
        
        # Parse timestamps
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)
        
        # Filter to last N days
        if custom_start is not None:
            end_time = custom_start + pd.Timedelta(days=lookback_days)
            df = df[(df.index >= custom_start) & (df.index < end_time)]
            print(f"\n[OK] Using window {custom_start} to {end_time}")
        elif lookback_days:
            cutoff_time = df.index.max() - pd.Timedelta(days=lookback_days)
            df = df[df.index >= cutoff_time]
            print(f"\n[OK] Using last {lookback_days} day(s) of CSV data:")
        else:
            print("\n[OK] Using entire CSV data history")
    
    print(f"Loaded {len(df)} data points")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Calculate simulation duration
    duration = df.index.max() - df.index.min()
    print(f"\nSimulation Duration: {duration.days} days, {duration.seconds // 3600} hours")
    print(f"Data Points: {len(df)} (5-minute intervals)")
    print(f"Estimated Backtest Time: ~{len(df) * 0.01:.1f} seconds")
    
    # Ask user for trading frequency
    print("\n" + "="*60)
    print("TRADING FREQUENCY SETTINGS")
    print("="*60)
    print("How often should the bot check for trading opportunities?")
    print("(This is the minimum time between trades)")
    print("\nExamples:")
    print("  - 5 minutes: Standard setting (default)")
    print("  - 10 minutes: Less frequent trading")
    print("  - 15 minutes: Conservative approach")
    print("  - 1 minute: More frequent (higher transaction costs)")
    
    try:
        freq_input = input("\nEnter minimum trade interval in minutes (default=5): ").strip()
        min_trade_interval = int(freq_input) if freq_input else 5
        if min_trade_interval < 1:
            print("[WARN] Minimum interval must be at least 1 minute. Using 1 minute.")
            min_trade_interval = 1
    except (ValueError, KeyboardInterrupt, EOFError):
        min_trade_interval = 5
        print("\nUsing default: 5 minutes")
    
    print(f"\n[OK] Trading frequency set to: {min_trade_interval} minutes between trades")
    
    # Ask user if they want live plotting
    print("\n" + "="*60)
    print("PLOTTING OPTIONS")
    print("="*60)
    print("1. Run with live plotting (see trades as they happen)")
    print("2. Run without live plotting (faster, plot at end)")
    
    try:
        choice = input("\nEnter choice (1 or 2, default=2): ").strip() or "2"
        live_plot = choice == "1"
    except (KeyboardInterrupt, EOFError):
        live_plot = False
        print("\nUsing default: no live plotting")
    
    # Initialize trader with configurable frequency
    trader = MomentumTrader(
        initial_capital=50000.0, 
        transaction_fee=0.001,
        min_trade_interval_minutes=min_trade_interval
    )
    trader.initialize_api()
    
    # Run backtest
    results_df, df_with_predictions = trader.backtest(df, live_plot=live_plot)
    
    # Print summary
    trader.print_summary(results_df)
    
    # Plot results (final plot)
    figures_dir = Path("TradingBot/figures")
    figures_dir.mkdir(exist_ok=True)
    plot_path = figures_dir / "linear_momentum_trading_results.png"
    
    trader.plot_results(df_with_predictions, results_df, save_path=plot_path)
    
    # Save results
    results_path = figures_dir / "linear_momentum_trading_equity.csv"
    results_df.to_csv(results_path)
    print(f"\n[OK] Equity history saved to {results_path}")
    
    trades_path = figures_dir / "linear_momentum_trading_trades.json"
    with open(trades_path, 'w') as f:
        json.dump(trader.trades, f, indent=2, default=str)
    print(f"[OK] Trades saved to {trades_path}")


if __name__ == "__main__":
    main()

