#!/usr/bin/env python3
"""Backtest the enhanced trading bot on 15 days of historical data."""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from bot.config import get_config
from bot.datastore import DataStore
from bot.signals_enhanced import EnhancedSignalGenerator
from bot.risk import RiskManager


@dataclass
class SimulatedPosition:
    """Track a position in the simulation."""
    symbol: str
    entry_price: float
    amount: float
    entry_time: pd.Timestamp
    stop_loss: float = 0.0
    take_profit: float = 0.0


class EnhancedBacktest:
    """Backtest the enhanced trading engine on historical data."""
    
    def __init__(self, data_dir: Path, initial_capital: float = 50000.0):
        """Initialize backtest.
        
        Args:
            data_dir: Directory containing historical data
            initial_capital: Starting capital in USD
        """
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.config = get_config()
        
        # Initialize signal generator with same params as enhanced engine
        self.signal_generator = EnhancedSignalGenerator(
            trend_window_long=self.config.trend_window_long,
            trend_window_short=self.config.trend_window_short,
            entry_window=self.config.entry_window,
            volume_window=self.config.volume_window,
            support_resistance_days=self.config.support_resistance_days,
            breakout_threshold=self.config.breakout_threshold,
            volume_surge_multiplier=self.config.volume_surge_multiplier
        )
        
        self.risk_manager = RiskManager(max_position_pct=self.config.max_position_pct)
        
        # Trading costs
        self.trading_fee = 0.001  # 0.1% per trade
        self.slippage = 0.0005    # 0.05% slippage
        
    def load_historical_data(self, days: int = 15) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """Load historical data for the last N days.
        
        Args:
            days: Number of days to load
            
        Returns:
            Tuple of (data_dict, timestamps)
        """
        all_data = {}
        all_timestamps = set()
        
        # Load all CSV files
        for csv_file in self.data_dir.glob("*.csv"):
            if csv_file.stem == "state":
                continue
            
            pair = csv_file.stem.replace("_", "")  # Handle any formatting
            try:
                df = pd.read_csv(csv_file)
                if "timestamp" not in df.columns or "price" not in df.columns:
                    continue
                    
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("timestamp").sort_index()
                
                # Filter to last N days
                if len(df) > 0:
                    end_date = df.index.max()
                    start_date = end_date - timedelta(days=days)
                    df = df[df.index >= start_date]
                    
                    if len(df) > self.config.trend_window_long:  # Need enough data
                        all_data[pair] = df
                        all_timestamps.update(df.index)
                        print(f"Loaded {pair}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {csv_file.stem}: {e}")
                continue
        
        # Get sorted timestamps (minute-by-minute)
        timestamps = pd.DatetimeIndex(sorted(all_timestamps))
        
        print(f"\nLoaded {len(all_data)} pairs")
        print(f"Date range: {timestamps.min()} to {timestamps.max()}")
        print(f"Total timestamps: {len(timestamps)}")
        
        return all_data, timestamps
    
    def calculate_trade_cost(self, value: float) -> float:
        """Calculate trading cost (fees + slippage)."""
        return value * (self.trading_fee + self.slippage)
    
    def get_current_prices(self, all_data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> Dict[str, float]:
        """Get current prices for all pairs at a given timestamp."""
        prices = {}
        for pair, df in all_data.items():
            # Get the most recent price up to this timestamp
            available = df[df.index <= timestamp]
            if len(available) > 0:
                prices[pair] = available["price"].iloc[-1]
        return prices
    
    def get_ticker_data(self, all_data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Dict]:
        """Generate ticker data for signal generation."""
        ticker_list = []
        prices = self.get_current_prices(all_data, timestamp)
        
        for pair, price in prices.items():
            # Estimate volume from 24h average (simplified)
            ticker_list.append({
                "pair": pair,
                "price": price,
                "volume_24h": 1000000.0,  # Placeholder - would need actual volume data
                "bid": price * 0.9999,
                "ask": price * 1.0001
            })
        
        return ticker_list
    
    def calculate_portfolio_value(self, cash: float, positions: Dict[str, SimulatedPosition], 
                                  prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total = cash
        for symbol, pos in positions.items():
            # Try both symbol and symbolUSD format
            price_key = symbol if symbol in prices else f"{symbol}USD"
            if price_key in prices:
                total += pos.amount * prices[price_key]
        return total
    
    def run_backtest(self, days: int = 15) -> pd.DataFrame:
        """Run the backtest simulation.
        
        Args:
            days: Number of days to backtest
            
        Returns:
            DataFrame with backtest results
        """
        print(f"\n{'='*60}")
        print(f"Running Enhanced Bot Backtest - Last {days} Days")
        print(f"{'='*60}\n")
        
        # Load data
        all_data, timestamps = self.load_historical_data(days)
        if len(all_data) == 0:
            raise ValueError("No data available for backtesting")
        
        # Initialize portfolio
        cash = self.initial_capital
        positions: Dict[str, SimulatedPosition] = {}
        
        # Create a mock datastore that uses our historical data
        class MockDataStore:
            def __init__(self, historical_data):
                self.historical_data = historical_data
                
            def read_minute_bars(self, pair: str, limit: Optional[int] = None) -> pd.DataFrame:
                """Read historical data up to current point."""
                if pair in self.historical_data:
                    df = self.historical_data[pair].copy()
                    if limit:
                        return df.tail(limit)
                    return df
                return pd.DataFrame(columns=["timestamp", "price", "volume"])
        
        datastore = MockDataStore(all_data)
        
        # Results tracking
        results = []
        trades = []
        
        # Get timestamps at 1-minute intervals (simulating real-time trading)
        # Use every minute for detailed simulation
        test_timestamps = timestamps[::1]  # Every minute
        
        print(f"Simulating {len(test_timestamps)} trading cycles...\n")
        
        last_trade_time = None
        min_trade_interval = timedelta(minutes=1)  # 1 minute cooldown
        
        for i, timestamp in enumerate(test_timestamps):
            # Get current prices
            prices = self.get_current_prices(all_data, timestamp)
            if not prices:
                continue
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value(cash, positions, prices)
            
            # Check stop losses and take profits for existing positions
            for symbol, pos in list(positions.items()):
                # Try both symbol and symbolUSD format
                price_key = symbol if symbol in prices else f"{symbol}USD"
                if price_key in prices:
                    current_price = prices[price_key]
                    # Use position's stop loss/take profit, or defaults
                    stop_loss = pos.stop_loss if pos.stop_loss > 0 else pos.entry_price * 0.95
                    take_profit = pos.take_profit if pos.take_profit > 0 else pos.entry_price * 1.03
                    
                    if current_price <= stop_loss:
                        # Stop loss hit
                        sell_value = pos.amount * current_price
                        cost = self.calculate_trade_cost(sell_value)
                        cash += (sell_value - cost)
                        pnl = sell_value - (pos.amount * pos.entry_price) - cost
                        
                        trades.append({
                            "time": timestamp,
                            "action": "SELL",
                            "pair": f"{symbol}USD",
                            "amount": pos.amount,
                            "price": current_price,
                            "value": sell_value,
                            "pnl": pnl,
                            "pnl_pct": (pnl / (pos.amount * pos.entry_price)) * 100,
                            "hold_time": (timestamp - pos.entry_time).total_seconds() / 3600,
                            "reason": "Stop Loss"
                        })
                        del positions[symbol]
                        portfolio_value = self.calculate_portfolio_value(cash, positions, prices)
                    elif current_price >= take_profit:
                        # Take profit hit
                        sell_value = pos.amount * current_price
                        cost = self.calculate_trade_cost(sell_value)
                        cash += (sell_value - cost)
                        pnl = sell_value - (pos.amount * pos.entry_price) - cost
                        
                        trades.append({
                            "time": timestamp,
                            "action": "SELL",
                            "pair": f"{symbol}USD",
                            "amount": pos.amount,
                            "price": current_price,
                            "value": sell_value,
                            "pnl": pnl,
                            "pnl_pct": (pnl / (pos.amount * pos.entry_price)) * 100,
                            "hold_time": (timestamp - pos.entry_time).total_seconds() / 3600,
                            "reason": "Take Profit"
                        })
                        del positions[symbol]
                        portfolio_value = self.calculate_portfolio_value(cash, positions, prices)
            
            # Check trade cooldown
            can_trade = True
            if last_trade_time:
                time_since_trade = timestamp - last_trade_time
                if time_since_trade < min_trade_interval:
                    can_trade = False
            
            # Generate signals (only if we can trade)
            if can_trade:
                # Update mock datastore to only include data up to current timestamp
                current_data = {}
                for pair, df in all_data.items():
                    current_data[pair] = df[df.index <= timestamp]
                datastore.historical_data = current_data
                
                ticker_data = self.get_ticker_data(all_data, timestamp)
                
                # Generate signals for all pairs
                all_signals = {}
                for pair in prices.keys():
                    if pair in current_data and len(current_data[pair]) >= self.config.trend_window_long:
                        ticker_entry = {
                            "pair": pair,
                            "price": prices[pair],
                            "volume_24h": 1000000.0,
                            "bid": prices[pair] * 0.9999,
                            "ask": prices[pair] * 1.0001
                        }
                        
                        try:
                            signal = self.signal_generator.compute_trading_signal(
                                pair, datastore, ticker_entry
                            )
                            all_signals[pair] = signal
                        except Exception as e:
                            # Skip if error
                            continue
                
                # First, check for sell signals on existing positions
                for symbol, pos in list(positions.items()):
                    pos_pair = f"{symbol}USD"
                    if pos_pair in all_signals:
                        pos_signal = all_signals[pos_pair]
                        if pos_signal["signal"] == "sell":
                            # Close position
                            sell_value = pos.amount * prices.get(symbol, pos.entry_price)
                            cost = self.calculate_trade_cost(sell_value)
                            cash += (sell_value - cost)
                            
                            # Calculate P&L
                            pnl = sell_value - (pos.amount * pos.entry_price) - cost
                            pnl_pct = (pnl / (pos.amount * pos.entry_price)) * 100
                            
                            trades.append({
                                "time": timestamp,
                                "action": "SELL",
                                "pair": pos_pair,
                                "amount": pos.amount,
                                "price": prices.get(symbol, pos.entry_price),
                                "value": sell_value,
                                "pnl": pnl,
                                "pnl_pct": pnl_pct,
                                "hold_time": (timestamp - pos.entry_time).total_seconds() / 3600
                            })
                            
                            del positions[symbol]
                            last_trade_time = timestamp
                            portfolio_value = self.calculate_portfolio_value(cash, positions, prices)
                
                # Then, check for new buy opportunities
                ranked_signals = self.signal_generator.rank_trading_opportunities(all_signals)
                
                if ranked_signals:
                    best_pair, best_signal = ranked_signals[0]
                    base_currency = best_pair.replace("USD", "")
                    
                    # Calculate total equity and capital allocation
                    total_equity = portfolio_value
                    total_allocated = sum(
                        pos.amount * prices.get(symbol, 0) 
                        for symbol, pos in positions.items()
                    )
                    allocated_pct = total_allocated / total_equity if total_equity > 0 else 0
                    max_allocated_pct = 0.85  # Keep 15% cash buffer
                    available_capital_pct = max_allocated_pct - allocated_pct
                    
                    # Handle buy signals
                    if best_signal["signal"] == "buy":
                        # Check if we already have this position
                        if base_currency in positions:
                            # Could add more if below target and signal is strong
                            current_pos = positions[base_currency]
                            current_value = current_pos.amount * best_signal["current_price"]
                            current_pct = current_value / total_equity if total_equity > 0 else 0
                            target_pct = self.config.max_position_pct * 0.8
                            
                            if current_pct < target_pct and best_signal["entry_quality"] > 0.6 and available_capital_pct > 0.05:
                                # Add to position
                                add_value = min(total_equity * 0.1, cash * 0.9)  # Add 10% of equity or 90% of cash
                                if add_value > 100:
                                    amount = add_value / best_signal["current_price"]
                                    cost = self.calculate_trade_cost(add_value)
                                    if cash >= add_value + cost:
                                        cash -= (add_value + cost)
                                        current_pos.amount += amount
                                        # Update entry price (weighted average)
                                        total_amount = current_pos.amount
                                        current_pos.entry_price = (
                                            (current_pos.entry_price * (total_amount - amount) + 
                                             best_signal["current_price"] * amount) / total_amount
                                        )
                                        last_trade_time = timestamp
                        else:
                            # New position - check capital constraints
                            if available_capital_pct >= 0.05 and best_signal["entry_quality"] >= 0.6:
                                # Calculate position size (matching engine logic)
                                position_scale = 0.5 + (best_signal.get("entry_quality", 0.5) - 0.5) * 0.5
                                position_value = total_equity * self.config.max_position_pct * position_scale
                                position_value = min(position_value, cash * 0.9, total_equity * available_capital_pct)
                                
                                if position_value > 100:  # Minimum trade size
                                    amount = position_value / best_signal["current_price"]
                                    cost = self.calculate_trade_cost(position_value)
                                    
                                    if cash >= position_value + cost:
                                        # Execute trade
                                        cash -= (position_value + cost)
                                        positions[base_currency] = SimulatedPosition(
                                            symbol=base_currency,
                                            entry_price=best_signal["current_price"],
                                            amount=amount,
                                            entry_time=timestamp,
                                            stop_loss=best_signal.get("stop_loss", best_signal["current_price"] * 0.95),
                                            take_profit=best_signal.get("take_profit", best_signal["current_price"] * 1.03)
                                        )
                                        last_trade_time = timestamp
                                        
                                        trades.append({
                                            "time": timestamp,
                                            "action": "BUY",
                                            "pair": best_pair,
                                            "amount": amount,
                                            "price": best_signal["current_price"],
                                            "value": position_value,
                                            "quality": best_signal.get("entry_quality", 0),
                                            "reason": best_signal.get("reason", "")
                                        })
            
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
        
        # Close all positions at end
        final_prices = self.get_current_prices(all_data, test_timestamps[-1])
        for symbol, pos in list(positions.items()):
            # Try both symbol formats
            price_key = symbol if symbol in final_prices else f"{symbol}USD"
            if price_key in final_prices:
                sell_value = pos.amount * final_prices[price_key]
                cost = self.calculate_trade_cost(sell_value)
                cash += (sell_value - cost)
                
                # Record final closing trade
                trades.append({
                    "time": test_timestamps[-1],
                    "action": "SELL",
                    "pair": f"{symbol}USD",
                    "amount": pos.amount,
                    "price": final_prices[price_key],
                    "value": sell_value,
                    "pnl": sell_value - (pos.amount * pos.entry_price) - cost,
                    "pnl_pct": ((sell_value - (pos.amount * pos.entry_price) - cost) / (pos.amount * pos.entry_price)) * 100,
                    "hold_time": (test_timestamps[-1] - pos.entry_time).total_seconds() / 3600,
                    "reason": "End of Backtest"
                })
        
        final_value = cash
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        returns = results_df["equity"].pct_change().dropna()
        
        if len(returns) > 0:
            sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)) if returns.std() > 0 else 0
            max_dd = ((results_df["equity"] / results_df["equity"].expanding().max()) - 1).min() * 100
        else:
            sharpe = 0
            max_dd = 0
        
        # Print summary
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        print(f"Sharpe Ratio: {sharpe:.3f}")
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
        
        # Compare to buy-and-hold BTC
        if "BTCUSD" in all_data:
            btc_data = all_data["BTCUSD"]
            btc_start = btc_data[btc_data.index <= test_timestamps[0]]
            btc_end = btc_data[btc_data.index <= test_timestamps[-1]]
            if len(btc_start) > 0 and len(btc_end) > 0:
                btc_start_price = btc_start["price"].iloc[-1]
                btc_end_price = btc_end["price"].iloc[-1]
                btc_return = ((btc_end_price - btc_start_price) / btc_start_price) * 100
                print(f"\nBuy-and-Hold BTC Return: {btc_return:.2f}%")
                print(f"Bot vs BTC: {total_return - btc_return:.2f}% difference")
        
        # Save results
        results_df.to_csv(self.data_dir.parent / "backtest_enhanced_results.csv", index=False)
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(self.data_dir.parent / "backtest_enhanced_trades.csv", index=False)
            print(f"\nResults saved to:")
            print(f"  - backtest_enhanced_results.csv")
            print(f"  - backtest_enhanced_trades.csv")
        
        return results_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest enhanced trading bot")
    parser.add_argument("--days", type=int, default=15, help="Number of days to backtest")
    parser.add_argument("--capital", type=float, default=50000.0, help="Initial capital")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    
    args = parser.parse_args()
    
    backtest = EnhancedBacktest(args.data_dir, args.capital)
    backtest.run_backtest(days=args.days)


if __name__ == "__main__":
    main()

