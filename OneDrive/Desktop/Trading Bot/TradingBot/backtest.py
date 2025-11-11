#!/usr/bin/env python3
"""Backtesting framework for the trading strategy."""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bot.config import Config
from bot.datastore import DataStore
from bot.signals import SignalGenerator, TangencyPortfolioSignals
from bot.signals_enhanced import EnhancedSignalGenerator
from bot.risk import RiskManager


class BacktestEngine:
    """Backtesting engine for strategy evaluation."""
    
    def __init__(
        self,
        data_dir: Path,
        window_size: int = 240,
        momentum_lookback: int = 20,
        max_position_pct: float = 0.4,
        min_sharpe: float = 0.0,
        mode: str = "sharpe",
        trade_frequency: int = 60,  # Trading frequency in minutes
        max_positions: int = 3  # Maximum number of positions to hold
    ):
        """Initialize backtest engine."""
        self.data_dir = data_dir
        self.window_size = window_size
        self.momentum_lookback = momentum_lookback
        self.max_position_pct = max_position_pct
        self.min_sharpe = min_sharpe
        self.mode = mode
        self.trade_frequency = trade_frequency
        self.max_positions = max_positions
        
        # Initialize components
        if mode == "enhanced":
            self.signal_generator = EnhancedSignalGenerator(
                trend_window_long=1440,  # 24h
                trend_window_short=240,  # 4h
                entry_window=60,  # 1h
                volume_window=480,  # 8h
                support_resistance_days=7,
                breakout_threshold=0.02,
                volume_surge_multiplier=2.0
            )
        elif mode == "tangency":
            self.signal_generator = TangencyPortfolioSignals(
                window_size=window_size,
                momentum_lookback=momentum_lookback
            )
        else:
            self.signal_generator = SignalGenerator(
                window_size=window_size,
                momentum_lookback=momentum_lookback
            )
        
        self.risk_manager = RiskManager(max_position_pct=max_position_pct)
        
        # Trading costs
        self.trading_fee = 0.001  # 0.1% per trade
        self.slippage = 0.0005    # 0.05% slippage
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available minute bar data.
        
        Returns:
            Dictionary mapping pair to DataFrame
        """
        all_data = {}
        
        for csv_file in self.data_dir.glob("*.csv"):
            if csv_file.stem == "state":
                continue
            
            pair = csv_file.stem.replace("_", "")  # BTCUSD stays BTCUSD
            df = pd.read_csv(csv_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp").sort_index()
            
            # For enhanced mode, need enough data for long-term analysis (24h = 24 records for hourly data)
            min_required = 24 if self.mode == "enhanced" else self.window_size
            
            if len(df) > min_required:
                all_data[pair] = df
                print(f"Loaded {pair}: {len(df)} records")
        
        return all_data
    
    def simulate_features(
        self,
        all_data: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp
    ) -> Dict[str, Dict[str, float]]:
        """Simulate feature computation at a given timestamp.
        
        Args:
            all_data: All historical data
            timestamp: Current timestamp
        
        Returns:
            Features for all pairs
        """
        features = {}
        
        # Determine minimum window based on mode
        if self.mode == "enhanced":
            min_window = 24  # 24 hours for daily trend
        else:
            min_window = self.window_size + 1
        
        for pair, df in all_data.items():
            # Get data up to current timestamp
            historical = df[df.index <= timestamp]
            
            if len(historical) < min_window:
                continue
            
            # Use appropriate window
            if self.mode == "enhanced":
                # Use last 24-48 hours for enhanced mode
                window_data = historical.tail(48) if len(historical) >= 48 else historical
            else:
                window_data = historical.tail(self.window_size + 1)
            
            # Compute log returns
            log_returns = np.log(window_data["price"] / window_data["price"].shift(1))
            
            # Compute features
            latest_price = window_data["price"].iloc[-1]
            latest_return = log_returns.iloc[-1] if len(log_returns) > 0 else 0
            rolling_mean = log_returns.iloc[1:].mean()  # Exclude first NaN
            rolling_std = log_returns.iloc[1:].std()
            sharpe = rolling_mean / (rolling_std + 1e-10) if rolling_std > 0 else 0
            
            # Compute momentum
            lookback = min(self.momentum_lookback, len(window_data) - 1)
            if len(window_data) > lookback:
                past_price = window_data["price"].iloc[-(lookback + 1)]
                momentum = (latest_price - past_price) / past_price if past_price > 0 else 0
            else:
                momentum = 0
            
            # Use actual volume from data
            latest_volume = window_data["volume"].iloc[-1] if "volume" in window_data.columns else 0
            avg_volume = window_data["volume"].mean() if "volume" in window_data.columns else 0
            liquidity_score = latest_volume if latest_volume > 0 else 10000
            
            features[pair] = {
                "pair": pair,
                "price": float(latest_price),
                "log_return": float(latest_return),
                "rolling_mean": float(rolling_mean),
                "rolling_std": float(rolling_std),
                "sharpe_ratio": float(sharpe),
                "momentum": float(momentum),
                "liquidity_score": float(liquidity_score),
                "has_sufficient_data": True
            }
        
        return features
    
    def select_positions(self, features: Dict[str, Dict[str, float]]) -> List[str]:
        """Select target positions based on features (can return multiple).
        
        Args:
            features: Feature dictionary
        
        Returns:
            List of target pairs (top N by score)
        """
        if self.mode == "enhanced":
            # Enhanced strategy: best risk-adjusted returns with strong momentum
            eligible = []
            
            for pair, feat in features.items():
                # Balanced filters - not too strict
                if (feat["sharpe_ratio"] > 0.05 and  # Small positive Sharpe
                    feat["momentum"] > 0.01 and  # At least 1% momentum
                    feat["rolling_mean"] > 0):  # Positive expected return
                    
                    # Calculate combined score
                    # Sharpe (risk-adjusted) + momentum (trend strength) + volume bonus
                    sharpe_score = feat["sharpe_ratio"] * 3.0  # Weight Sharpe heavily
                    momentum_score = feat["momentum"] * 8.0  # Scale momentum
                    volume_bonus = min(feat["liquidity_score"] / 1000000, 1.0)  # Bonus for liquid assets
                    
                    # Penalize high volatility but don't eliminate
                    vol_penalty = max(0, 1 - feat["rolling_std"] * 5)
                    
                    total_score = (sharpe_score + momentum_score + volume_bonus) * vol_penalty
                    
                    eligible.append((pair, feat, total_score))
            
            if not eligible:
                return []
            
            # Sort by score and take top N
            eligible.sort(key=lambda x: x[2], reverse=True)
            top_pairs = [pair for pair, _, _ in eligible[:self.max_positions]]
            
            return top_pairs
            
    def select_position(self, features: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Legacy single position selection.
        
        Args:
            features: Feature dictionary
        
        Returns:
            Target pair or None for flat
        """
        if self.mode == "enhanced":
            # Use multi-position selection, return top one for backward compatibility
            positions = self.select_positions(features)
            return positions[0] if positions else None
        
        elif self.mode == "sharpe":
            # Filter eligible pairs
            eligible = [
                (pair, feat) for pair, feat in features.items()
                if feat["sharpe_ratio"] > self.min_sharpe and feat["momentum"] > 0
            ]
            
            if not eligible:
                return None
            
            # Select highest Sharpe
            best_pair = max(eligible, key=lambda x: x[1]["sharpe_ratio"])[0]
            return best_pair
        
        elif self.mode == "tangency":
            # Simplified tangency - just pick best expected return
            eligible = [
                (pair, feat) for pair, feat in features.items()
                if feat["rolling_mean"] > 0
            ]
            
            if not eligible:
                return None
            
            best_pair = max(eligible, key=lambda x: x[1]["rolling_mean"] / (x[1]["rolling_std"] + 1e-10))[0]
            return best_pair
        
        return None
    
    def calculate_trade_cost(self, value: float) -> float:
        """Calculate trading cost including fees and slippage.
        
        Args:
            value: Trade value
        
        Returns:
            Total cost
        """
        return value * (self.trading_fee + self.slippage)
    
    def run_backtest(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """Run backtest simulation.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital in USD
        
        Returns:
            DataFrame with backtest results
        """
        # Load all data
        all_data = self.load_all_data()
        if not all_data:
            raise ValueError("No data available for backtesting")
        
        # Determine date range
        all_timestamps = []
        for df in all_data.values():
            all_timestamps.extend(df.index.tolist())
        all_timestamps = sorted(set(all_timestamps))
        
        if start_date:
            all_timestamps = [ts for ts in all_timestamps if ts >= pd.Timestamp(start_date)]
        if end_date:
            all_timestamps = [ts for ts in all_timestamps if ts <= pd.Timestamp(end_date)]
        
        # Need sufficient history
        if self.mode == "enhanced":
            min_history_idx = 24  # Need at least 24 hours for trend analysis
        else:
            min_history_idx = self.window_size + 1
            
        if len(all_timestamps) < min_history_idx:
            raise ValueError(f"Insufficient data for backtesting (need {min_history_idx}, have {len(all_timestamps)})")
        
        # Initialize portfolio
        portfolio_value = initial_capital
        cash = initial_capital
        position = None
        position_amount = 0
        
        # Results tracking
        results = []
        
        # Adjust trade frequency based on data interval
        # If data is hourly, trade every hour
        effective_trade_freq = 1 if self.mode == "enhanced" else self.trade_frequency
        
        # Run simulation
        for i in range(min_history_idx, len(all_timestamps), effective_trade_freq):
            if i >= len(all_timestamps):
                break
                
            timestamp = all_timestamps[i]
            
            # Compute features
            features = self.simulate_features(all_data, timestamp)
            
            if not features:
                continue
            
            # Select target position
            target = self.select_position(features)
            
            # Get current prices
            current_prices = {
                pair: features[pair]["price"] 
                for pair in features
            }
            
            # Calculate current portfolio value BEFORE any trades
            if position and position in current_prices:
                portfolio_value = cash + position_amount * current_prices[position]
            else:
                portfolio_value = cash
            
            # Execute trade if needed
            traded = False
            if target != position:
                # Close current position first
                if position and position in current_prices:
                    sell_value = position_amount * current_prices[position]
                    trade_cost = self.calculate_trade_cost(sell_value)
                    cash += (sell_value - trade_cost)
                    position_amount = 0
                    position = None
                    traded = True
                
                # Open new position
                if target and target in current_prices and cash > 100:
                    # Calculate position value (keep buffer for fees)
                    max_invest = min(cash * self.max_position_pct, cash - 10)
                    trade_cost = self.calculate_trade_cost(max_invest)
                    
                    if cash >= max_invest + trade_cost:
                        position_amount = max_invest / current_prices[target]
                        cash -= (max_invest + trade_cost)
                        position = target
                        traded = True
            
            # Calculate final portfolio value for this period
            if position and position in current_prices:
                portfolio_value = cash + position_amount * current_prices[position]
            else:
                portfolio_value = cash
            
            # Record results
            results.append({
                "timestamp": timestamp,
                "portfolio_value": float(portfolio_value),
                "cash": float(cash),
                "position": position,
                "position_amount": float(position_amount),
                "traded": traded
            })
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            results: Backtest results DataFrame
        
        Returns:
            Dictionary of performance metrics
        """
        if len(results) < 2:
            return {}
        
        # Calculate returns
        results["returns"] = results["portfolio_value"].pct_change()
        
        # Basic metrics
        total_return = (results["portfolio_value"].iloc[-1] / results["portfolio_value"].iloc[0]) - 1
        
        # Annualized metrics
        # Determine data frequency (hourly vs minute data)
        if len(results) > 1:
            time_diff = (results["timestamp"].iloc[1] - results["timestamp"].iloc[0]).total_seconds() / 3600
            periods_per_year = 252 * 24 / time_diff  # Trading days * hours per day / hours per period
        else:
            periods_per_year = 252 * 24  # Default to hourly
        
        n_periods = len(results)
        annualization_factor = periods_per_year / n_periods
        
        # Sharpe ratio
        mean_return = results["returns"].mean()
        std_return = results["returns"].std()
        sharpe_ratio = np.sqrt(annualization_factor) * mean_return / (std_return + 1e-10)
        
        # Sortino ratio (downside deviation)
        downside_returns = results["returns"][results["returns"] < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(annualization_factor) * mean_return / (downside_std + 1e-10)
        
        # Maximum drawdown
        cumulative = (1 + results["returns"]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        annualized_return = (1 + total_return) ** (annualization_factor / n_periods) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        positive_returns = results["returns"] > 0
        win_rate = positive_returns.sum() / len(results["returns"].dropna())
        
        # Number of trades
        n_trades = results["traded"].sum()
        
        # Competition score
        competition_score = 0.4 * sortino_ratio + 0.3 * sharpe_ratio + 0.3 * calmar_ratio
        
        return {
            "total_return": total_return * 100,  # Percentage
            "annualized_return": annualized_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown * 100,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate * 100,
            "n_trades": n_trades,
            "competition_score": competition_score
        }
    
    def plot_results(self, results: pd.DataFrame, save_dir: Path) -> None:
        """Generate and save plots.
        
        Args:
            results: Backtest results
            save_dir: Directory to save plots
        """
        save_dir.mkdir(exist_ok=True)
        
        # Equity curve
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(results["timestamp"], results["portfolio_value"], label="Portfolio Value")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Equity Curve")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "equity_curve.png", dpi=150)
        plt.close()
        
        # Returns distribution
        if "returns" in results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            returns = results["returns"].dropna()
            ax1.hist(returns, bins=50, alpha=0.7, edgecolor="black")
            ax1.axvline(returns.mean(), color="red", linestyle="--", label=f"Mean: {returns.mean():.4f}")
            ax1.set_xlabel("Returns")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Returns Distribution")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Risk-return scatter (if we have position data)
            # This would show mean vs std for different assets
            ax2.set_xlabel("Volatility (σ)")
            ax2.set_ylabel("Mean Return (μ)")
            ax2.set_title("Risk-Return Profile")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / "returns_analysis.png", dpi=150)
            plt.close()


def main():
    """Main entry point for backtesting."""
    parser = argparse.ArgumentParser(description="Backtest Trading Strategy")
    parser.add_argument(
        "--mode",
        choices=["sharpe", "tangency", "enhanced"],
        default="enhanced",
        help="Trading strategy mode (enhanced is recommended)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=240,
        help="Rolling window size for signals (minutes)"
    )
    parser.add_argument(
        "--momentum",
        type=int,
        default=20,
        help="Momentum lookback period (minutes)"
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.4,
        help="Maximum position size as fraction of equity"
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Minimum Sharpe ratio to enter position"
    )
    parser.add_argument(
        "--trade-frequency",
        type=int,
        default=1,
        help="Trading frequency in minutes"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=1,
        help="Maximum number of positions to hold simultaneously (1-5)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital in USD"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with historical data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for output plots"
    )
    
    args = parser.parse_args()
    
    # Initialize backtest engine
    engine = BacktestEngine(
        data_dir=args.data_dir,
        window_size=args.window,
        momentum_lookback=args.momentum,
        max_position_pct=args.max_position,
        min_sharpe=args.min_sharpe,
        mode=args.mode,
        trade_frequency=args.trade_frequency,
        max_positions=args.max_positions
    )
    
    print(f"Running backtest with {args.mode} strategy...")
    print(f"Window: {args.window}, Momentum: {args.momentum}, Max Position: {args.max_position}")
    print(f"Max Positions: {args.max_positions}, Trade Frequency: {args.trade_frequency}")
    
    try:
        # Run backtest
        results = engine.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital
        )
        
        # Calculate metrics
        metrics = engine.calculate_metrics(results)
        
        # Print results
        print("\n=== Backtest Results ===")
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Number of Trades: {metrics.get('n_trades', 0)}")
        print(f"\nCompetition Score: {metrics.get('competition_score', 0):.3f}")
        
        # Generate plots
        engine.plot_results(results, args.output_dir)
        print(f"\nPlots saved to {args.output_dir}")
        
        # Save detailed results
        results_file = args.output_dir / "backtest_results.csv"
        results.to_csv(results_file, index=False)
        
        metrics_file = args.output_dir / "backtest_metrics.json"
        # Convert numpy types to native Python for JSON serialization
        metrics_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                               for k, v in metrics.items()}
        with open(metrics_file, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
