#!/usr/bin/env python3
"""Backtest optimized Sharpe mode with proper stop loss/take profit."""

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
from bot.signals_sharpe_optimized import OptimizedSharpeSignalGenerator, OptimizedSharpeSignal


@dataclass
class SimulatedPosition:
    """Track a position."""
    symbol: str
    entry_price: float
    amount: float
    entry_time: pd.Timestamp
    stop_loss: float = 0.0
    take_profit: float = 0.0
    quality: float = 0.5


class OptimizedSharpeBacktest:
    """Backtest optimized Sharpe mode."""
    
    def __init__(self, data_dir: Path, initial_capital: float = 50000.0, 
                 position_size_pct: float = 0.75, max_positions: int = 20):
        """Initialize backtest."""
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.config = get_config()
        
        self.signal_generator = OptimizedSharpeSignalGenerator(
            window_size=72,
            momentum_lookback=12,
            min_sharpe=0.05,
            momentum_threshold=0.01
        )
        
        # Position size (configurable)
        self.position_size = position_size_pct
        
        # Maximum number of simultaneous positions
        self.max_positions = max_positions
        
        # Market regime filter (BTC trend)
        self.use_market_filter = True
        
        # Track performance for dynamic sizing
        self.pair_performance = {}  # Track P&L per pair
        
        # Trading costs
        self.trading_fee = 0.001
        self.slippage = 0.0005
    
    def load_historical_data(
        self, 
        days: Optional[int] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex, Optional[pd.Series]]:
        """Load historical data and BTC prices for market filter."""
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
                # CRITICAL: Sort and remove duplicates BEFORE setting index
                df = df.sort_values("timestamp")
                df = df.drop_duplicates(subset=["timestamp"], keep="last")  # Keep last if duplicates
                df = df.set_index("timestamp")
                # Ensure it's still sorted after setting index
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
                    
                    if len(df) > 72:
                        all_data[pair] = df
                        all_timestamps.update(df.index)
                        print(f"Loaded {pair}: {len(df)} records")
            except Exception as e:
                continue
        
        timestamps = pd.DatetimeIndex(sorted(all_timestamps))
        
        # Get BTC prices for market filter
        btc_prices = None
        if "BTCUSD" in all_data:
            btc_df = all_data["BTCUSD"]
            btc_prices = btc_df["price"]
            print(f"BTC data available for market filter")
        
        print(f"\nLoaded {len(all_data)} pairs")
        if len(timestamps) > 0:
            print(f"Date range: {timestamps.min()} to {timestamps.max()}")
        
        return all_data, timestamps, btc_prices
    
    def calculate_trade_cost(self, value: float) -> float:
        """Calculate trading cost."""
        return value * (self.trading_fee + self.slippage)
    
    def get_current_prices(self, all_data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> Dict[str, float]:
        """Get current prices."""
        prices = {}
        for pair, df in all_data.items():
            available = df[df.index <= timestamp]
            if len(available) > 0:
                prices[pair] = available["price"].iloc[-1]
        return prices
    
    def get_ticker_data(self, all_data: Dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> List[Dict]:
        """Generate ticker data."""
        ticker_list = []
        prices = self.get_current_prices(all_data, timestamp)
        
        for pair, price in prices.items():
            ticker_list.append({
                "pair": pair,
                "price": price,
                "volume_24h": 1000000.0,
                "bid": price * 0.9999,
                "ask": price * 1.0001
            })
        
        return ticker_list
    
    def calculate_portfolio_value(self, cash: float, positions: Dict[str, SimulatedPosition], 
                                  prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total = cash
        for symbol, pos in positions.items():
            price_key = symbol if symbol in prices else f"{symbol}USD"
            if price_key in prices:
                total += pos.amount * prices[price_key]
        return total
    
    def run_backtest(
        self, 
        days: Optional[int] = 15,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, List[Dict], Dict]:
        """Run optimized Sharpe mode backtest."""
        print(f"\n{'='*60}")
        print(f"Running OPTIMIZED SHARPE MODE Backtest - Last {days} Days")
        print(f"{'='*60}\n")
        
        # Load data
        all_data, timestamps, btc_prices = self.load_historical_data(
            days=days, start_date=start_date, end_date=end_date
        )
        if len(all_data) == 0:
            raise ValueError("No data available")
        
        # Initialize portfolio
        cash = self.initial_capital
        positions: Dict[str, SimulatedPosition] = {}
        
        # Reset performance tracking for this backtest
        self.pair_performance = {}
        
        # Mock datastore
        class MockDataStore:
            def __init__(self, historical_data):
                self.historical_data = historical_data
            
            def read_minute_bars(self, pair: str, limit: Optional[int] = None) -> pd.DataFrame:
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
        
        # Check every 5 minutes for more trading opportunities
        # If data is hourly, use every hour; if minute-level, use every 5 minutes
        if len(timestamps) > 0:
            # Estimate data frequency
            time_diff = (timestamps[1] - timestamps[0]).total_seconds() / 60 if len(timestamps) > 1 else 60
            if time_diff <= 5:  # Minute-level data
                test_timestamps = timestamps[::5]  # Every 5 minutes
            elif time_diff <= 60:  # 5-60 minute data
                test_timestamps = timestamps[::1]  # Every data point
            else:  # Hourly or longer
                test_timestamps = timestamps[::1]  # Every data point
        else:
            test_timestamps = timestamps
        
        print(f"Simulating {len(test_timestamps)} trading cycles...\n")
        
        peak_equity = self.initial_capital
        
        for i, timestamp in enumerate(test_timestamps):
            # Get current prices
            prices = self.get_current_prices(all_data, timestamp)
            if not prices:
                continue
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value(cash, positions, prices)
            if portfolio_value > peak_equity:
                peak_equity = portfolio_value
            
            # Check exit conditions - PRIORITIZE TAKE PROFIT
            for symbol, pos in list(positions.items()):
                price_key = symbol if symbol in prices else f"{symbol}USD"
                if price_key in prices:
                    current_price = prices[price_key]
                    
                    # Take profit FIRST (lock in gains, minimize downside)
                    if current_price >= pos.take_profit * 1.001:  # Small buffer
                        sell_value = pos.amount * current_price
                        cost = self.calculate_trade_cost(sell_value)
                        cash += (sell_value - cost)
                        pnl = sell_value - (pos.amount * pos.entry_price) - cost
                        
                        # Track performance for this pair
                        if price_key not in self.pair_performance:
                            self.pair_performance[price_key] = 0
                        self.pair_performance[price_key] += pnl
                        
                        trades.append({
                            "time": timestamp,
                            "action": "SELL",
                            "pair": price_key,
                            "amount": pos.amount,
                            "price": current_price,
                            "value": sell_value,
                            "pnl": pnl,
                            "pnl_pct": (pnl / (pos.amount * pos.entry_price)) * 100,
                            "hold_time": (timestamp - pos.entry_time).total_seconds() / 3600,
                            "reason": "Take Profit",
                            "quality": pos.quality
                        })
                        del positions[symbol]
                        portfolio_value = self.calculate_portfolio_value(cash, positions, prices)
                        continue
                    
                    # Stop loss (minimize downside volatility)
                    if current_price <= pos.stop_loss * 0.999:  # Small buffer
                        sell_value = pos.amount * current_price
                        cost = self.calculate_trade_cost(sell_value)
                        cash += (sell_value - cost)
                        pnl = sell_value - (pos.amount * pos.entry_price) - cost
                        
                        # Track performance for this pair
                        if price_key not in self.pair_performance:
                            self.pair_performance[price_key] = 0
                        self.pair_performance[price_key] += pnl
                        
                        trades.append({
                            "time": timestamp,
                            "action": "SELL",
                            "pair": price_key,
                            "amount": pos.amount,
                            "price": current_price,
                            "value": sell_value,
                            "pnl": pnl,
                            "pnl_pct": (pnl / (pos.amount * pos.entry_price)) * 100,
                            "hold_time": (timestamp - pos.entry_time).total_seconds() / 3600,
                            "reason": "Stop Loss",
                            "quality": pos.quality
                        })
                        del positions[symbol]
                        portfolio_value = self.calculate_portfolio_value(cash, positions, prices)
            
            # Generate signals
            current_data = {}
            for pair, df in all_data.items():
                current_data[pair] = df[df.index <= timestamp]
            datastore.historical_data = current_data
            
            ticker_data = self.get_ticker_data(all_data, timestamp)
            
            # Market regime filter: Check BTC trend (BALANCED - allow trades in reasonable conditions)
            market_ok = True
            if self.use_market_filter and btc_prices is not None:
                btc_available = btc_prices[btc_prices.index <= timestamp]
                # Reduced threshold from 72 to 24 to allow earlier trading in Period 1
                if len(btc_available) > 24:
                    btc_current = btc_available.iloc[-1]
                    btc_ma12 = btc_available.rolling(window=12).mean().iloc[-1] if len(btc_available) >= 12 else btc_current
                    btc_ma24 = btc_available.rolling(window=24).mean().iloc[-1] if len(btc_available) >= 24 else btc_current
                    
                    # BTC market filter (BALANCED - block only severe downtrends)
                    # Allow trades unless BTC is in severe downtrend
                    if len(btc_available) > 12:
                        btc_momentum = (btc_current - btc_available.iloc[-12]) / btc_available.iloc[-12]
                        # Block if BTC down >3% in 12h (moderate threshold)
                        if btc_momentum < -0.03:
                            market_ok = False
                    
                    # Check recent BTC trend (24h) - block if down >5%
                    if len(btc_available) > 24:
                        btc_24h_trend = (btc_current - btc_available.iloc[-24]) / btc_available.iloc[-24]
                        if btc_24h_trend < -0.05:  # BTC down >5% in 24h = bad market
                            market_ok = False
            
            # CRITICAL: Analyze ALL pairs at EVERY timestamp (if market is OK)
            # This ensures we don't miss any opportunities
            if market_ok:
                all_signals = {}
                pairs_checked = 0
                pairs_with_signals = 0
                
                # Check ALL pairs in ticker_data (not just when no position)
                for ticker in ticker_data:
                    pair = ticker["pair"]
                    if pair == "BTCUSD":  # Skip BTC itself (used for market filter)
                        continue
                    
                    pairs_checked += 1
                    try:
                        signal = self.signal_generator.compute_signal(
                            pair=pair,
                            datastore=datastore,
                            ticker_data=ticker
                        )
                        if signal and signal.signal != "neutral":
                            all_signals[pair] = signal
                            pairs_with_signals += 1
                    except Exception as e:
                        # Silently skip errors (data issues, etc.)
                        continue
                
                # Log periodically to show we're checking all pairs
                if i % 100 == 0 and pairs_checked > 0:
                    print(f"  [{i+1}/{len(test_timestamps)}] Checked {pairs_checked} pairs, found {pairs_with_signals} signals")
                
                # Rank signals by quality
                ranked_signals = self.signal_generator.rank_signals(all_signals)
                
                # Execute MULTIPLE positions - enter as many as we can with available cash
                # Don't limit by max_positions if we have cash - use it all!
                if ranked_signals:
                    # QUALITY-BASED POSITION SIZING: Allocate more to higher quality signals
                    # Calculate total quality score for normalization (use all available signals)
                    available_signals = [sig for _, sig in ranked_signals if sig.pair.replace("USD", "") not in positions]
                    total_quality = sum(sig.quality for sig in available_signals)
                    if total_quality == 0:
                        total_quality = 1.0  # Avoid division by zero
                    
                    # Try to enter multiple positions - keep entering until we run out of cash
                    for pair, signal in ranked_signals:
                        # Skip if we already have a position in this pair
                        symbol = pair.replace("USD", "")
                        if symbol in positions:
                            continue
                        
                        # Check if we have enough cash (more important than max positions)
                        if cash < 1000:  # Need at least $1000 to enter a new position
                            break
                        
                        # QUALITY-BASED ALLOCATION: Higher quality = MUCH larger position
                        # Base allocation: 10% of total position size budget
                        # Quality bonus: up to 90% additional based on relative quality
                        quality_ratio = signal.quality / total_quality if total_quality > 0 else 1.0
                        base_allocation = self.position_size * 0.1  # 10% base
                        quality_bonus = self.position_size * 0.9 * quality_ratio  # 90% quality-based
                        
                        # Performance bonus: allocate MORE to pairs that have performed well
                        performance_multiplier = 1.0
                        if pair in self.pair_performance:
                            historical_pnl = self.pair_performance[pair]
                            if historical_pnl > 500:  # If profitable >$500, allocate 50% more
                                performance_multiplier = 1.5
                            elif historical_pnl > 0:  # If profitable, allocate 30% more
                                performance_multiplier = 1.3
                            elif historical_pnl < -500:  # If lost >$500, reduce allocation
                                performance_multiplier = 0.6
                        
                        position_pct = (base_allocation + quality_bonus) * performance_multiplier
                        
                        # Position size: quality and performance-based (AGGRESSIVE for 10-15% target)
                        position_value = portfolio_value * position_pct * (0.8 + signal.quality * 0.4)  # Scale by quality
                        
                        # Use ALMOST ALL available cash (very aggressive - target 98%+ utilization)
                        # Only keep minimal buffer for fees (0.5% of portfolio)
                        cash_buffer = portfolio_value * 0.005  # Only 0.5% buffer for fees
                        available_cash = max(0, cash - cash_buffer)
                        
                        # If we have lots of cash, use more per position to deploy capital faster
                        if cash > portfolio_value * 0.1:  # If we have >10% cash, be more aggressive
                            position_value = min(position_value, available_cash * 1.2)  # Use 20% more
                        else:
                            position_value = min(position_value, available_cash)
                        
                        # Minimum position size: $500, Maximum: 30% of portfolio per position (increased for maximum capital deployment)
                        position_value = max(position_value, 500)
                        position_value = min(position_value, portfolio_value * 0.30)
                        
                        if position_value > 100:
                            amount = position_value / signal.entry_price
                            cost = self.calculate_trade_cost(position_value)
                            
                            if cash >= position_value + cost:
                                cash -= (position_value + cost)
                                positions[symbol] = SimulatedPosition(
                                    symbol=symbol,
                                    entry_price=signal.entry_price,
                                    amount=amount,
                                    entry_time=timestamp,
                                    stop_loss=signal.stop_loss,
                                    take_profit=signal.take_profit,
                                    quality=signal.quality
                                )
                                
                                # Get reason from signal
                                reason = getattr(signal, 'reason', 'Signal generated')
                                if hasattr(signal, 'momentum') and hasattr(signal, 'sharpe'):
                                    reason = f"Valid entry: momentum {signal.momentum*100:.1f}%, Sharpe {signal.sharpe:.3f}"
                                
                                trades.append({
                                    "time": timestamp,
                                    "action": "BUY",
                                    "pair": pair,
                                    "amount": amount,
                                    "price": signal.entry_price,
                                    "value": position_value,
                                    "quality": signal.quality,
                                    "sharpe": getattr(signal, 'sharpe', None) or getattr(signal, 'sharpe_ratio', None),
                                    "momentum": getattr(signal, 'momentum', None),
                                    "reason": reason
                                })
                                
                                portfolio_value = self.calculate_portfolio_value(cash, positions, prices)
            
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
        
        # Value remaining positions (don't close them - they're still open)
        # This allows buys != sells, which is more realistic
        final_prices = self.get_current_prices(all_data, test_timestamps[-1])
        final_value = cash
        for symbol, pos in list(positions.items()):
            price_key = symbol if symbol in final_prices else f"{symbol}USD"
            if price_key in final_prices:
                # Value the position but don't create a trade entry
                final_value += pos.amount * final_prices[price_key]
        
        # Note: We're NOT closing positions at the end, so buys != sells is expected
        # This is more realistic - positions can remain open
        
        # Calculate metrics
        results_df = pd.DataFrame(results)
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        returns = results_df["equity"].pct_change().dropna()
        
        # Sortino ratio (only downside volatility)
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
        print("OPTIMIZED SHARPE MODE BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        
        # Calculate cash utilization
        final_cash_pct = (cash / final_value) * 100 if final_value > 0 else 0
        invested_pct = 100 - final_cash_pct
        print(f"Cash Utilization: {invested_pct:.1f}% invested, {final_cash_pct:.1f}% cash")
        print(f"Open Positions at End: {len(positions)}")
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
                avg_win = np.mean([t.get("pnl", 0) for t in sell_trades if t.get("pnl", 0) > 0]) if winning_trades else 0
                avg_loss = np.mean([t.get("pnl", 0) for t in sell_trades if t.get("pnl", 0) < 0]) if [t for t in sell_trades if t.get("pnl", 0) < 0] else 0
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Average P&L per trade: ${avg_pnl:.2f}")
                print(f"Average Win: ${avg_win:.2f}")
                print(f"Average Loss: ${avg_loss:.2f}")
        
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
            "n_buy_trades": len([t for t in trades if t["action"] == "BUY"]),
            "n_sell_trades": len([t for t in trades if t["action"] == "SELL"]),
            "win_rate": (len([t for t in trades if t["action"] == "SELL" and t.get("pnl", 0) > 0]) / 
                        len([t for t in trades if t["action"] == "SELL"]) * 100) if len([t for t in trades if t["action"] == "SELL"]) > 0 else 0,
            "avg_pnl": np.mean([t.get("pnl", 0) for t in trades if t["action"] == "SELL"]) if len([t for t in trades if t["action"] == "SELL"]) > 0 else 0
        }
        
        return results_df, trades, metrics


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest optimized Sharpe mode")
    parser.add_argument("--days", type=int, default=15, help="Number of days to backtest")
    parser.add_argument("--capital", type=float, default=50000.0, help="Initial capital")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    
    args = parser.parse_args()
    
    backtest = OptimizedSharpeBacktest(args.data_dir, args.capital, max_positions=15)
    results_df, trades, metrics = backtest.run_backtest(days=args.days)
    
    # Save results
    output_dir = args.data_dir.parent
    results_df.to_csv(output_dir / "backtest_sharpe_optimized_results.csv", index=False)
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(output_dir / "backtest_sharpe_optimized_trades.csv", index=False)
        print(f"\nResults saved to:")
        print(f"  - backtest_sharpe_optimized_results.csv")
        print(f"  - backtest_sharpe_optimized_trades.csv")


if __name__ == "__main__":
    main()

