"""Cost-aware backtesting harness with comprehensive metrics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: pd.Timestamp
    pair: str
    side: str  # "buy" or "sell"
    price: float
    amount: float
    value: float
    cost: float
    reason: str = ""


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    n_trades: int
    n_winning_trades: int
    n_losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Exposure and turnover
    avg_exposure: float
    max_exposure: float
    turnover: float
    
    # Costs
    total_costs: float
    costs_as_pct_of_returns: float
    
    # Hit rate
    hit_rate: float


class CostAwareBacktester:
    """Cost-aware backtesting with transaction costs and slippage."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_bps: float = 10.0,  # 0.1%
        slippage_bps: float = 5.0,  # 0.05%
        execution_model: str = "next_bar",  # "next_bar" or "same_bar"
        risk_free_rate: float = 0.0
    ):
        """Initialize cost-aware backtester.
        
        Args:
            initial_capital: Starting capital
            commission_bps: Commission in basis points (per trade)
            slippage_bps: Slippage in basis points (per trade)
            execution_model: "next_bar" (realistic) or "same_bar" (optimistic)
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps / 10000.0  # Convert to decimal
        self.slippage_bps = slippage_bps / 10000.0
        self.execution_model = execution_model
        self.risk_free_rate = risk_free_rate
        
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # pair -> amount
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.prices_history: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
    
    def calculate_trade_cost(self, value: float) -> float:
        """Calculate total trading cost (commission + slippage).
        
        Args:
            value: Trade value in USD
            
        Returns:
            Total cost in USD
        """
        commission = value * self.commission_bps
        slippage = value * self.slippage_bps
        return commission + slippage
    
    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        pair: str,
        side: str,
        amount: float,
        price: float,
        reason: str = ""
    ) -> bool:
        """Execute a trade with cost accounting.
        
        Args:
            timestamp: Trade timestamp
            pair: Trading pair
            side: "buy" or "sell"
            amount: Trade amount
            price: Execution price
            reason: Reason for trade
            
        Returns:
            True if trade executed successfully
        """
        value = amount * price
        cost = self.calculate_trade_cost(value)
        
        if side == "buy":
            if self.cash < value + cost:
                logger.warning(f"Insufficient cash for buy: need {value + cost:.2f}, have {self.cash:.2f}")
                return False
            
            self.cash -= (value + cost)
            self.positions[pair] = self.positions.get(pair, 0.0) + amount
            
        elif side == "sell":
            if pair not in self.positions or self.positions[pair] < amount:
                logger.warning(f"Insufficient position for sell: {pair}")
                return False
            
            self.cash += (value - cost)
            self.positions[pair] -= amount
            if self.positions[pair] < 1e-8:  # Essentially zero
                del self.positions[pair]
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            pair=pair,
            side=side,
            price=price,
            amount=amount,
            value=value,
            cost=cost,
            reason=reason
        )
        self.trades.append(trade)
        
        return True
    
    def get_portfolio_value(
        self,
        prices: Dict[str, float],
        timestamp: Optional[pd.Timestamp] = None
    ) -> float:
        """Calculate total portfolio value.
        
        Args:
            prices: Current prices for all pairs
            timestamp: Optional timestamp for recording
            
        Returns:
            Total portfolio value
        """
        total = self.cash
        
        for pair, amount in self.positions.items():
            if pair in prices:
                total += amount * prices[pair]
        
        # Record equity curve
        if timestamp is not None:
            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": total,
                "cash": self.cash,
                "positions_value": total - self.cash,
                "n_positions": len(self.positions)
            })
        
        return total
    
    def run_backtest(
        self,
        timestamps: pd.DatetimeIndex,
        prices: Dict[str, pd.Series],
        signal_generator: Callable[[pd.Timestamp, Dict[str, float]], Dict[str, Dict[str, Any]]],
        position_sizer: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Run backtest simulation.
        
        Args:
            timestamps: Timestamps to simulate
            prices: Dictionary of price series by pair
            signal_generator: Function that generates signals at each timestamp
            position_sizer: Optional function to calculate position sizes
            
        Returns:
            DataFrame with equity curve
        """
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Track current prices
        current_prices = {}
        for pair, price_series in prices.items():
            current_prices[pair] = price_series.iloc[0] if len(price_series) > 0 else 0.0
        
        # Initialize equity curve
        self.get_portfolio_value(current_prices, timestamps[0])
        
        for i, timestamp in enumerate(timestamps[1:], 1):
            # Update current prices (next-bar execution)
            if self.execution_model == "next_bar":
                for pair, price_series in prices.items():
                    if i < len(price_series):
                        current_prices[pair] = price_series.iloc[i]
            
            # Generate signals (using prices from previous bar)
            signals = signal_generator(timestamp, current_prices)
            
            # Execute trades based on signals
            for pair, signal in signals.items():
                if pair not in current_prices:
                    continue
                
                current_price = current_prices[pair]
                signal_type = signal.get("signal", "neutral")
                
                if signal_type == "buy":
                    # Calculate position size
                    if position_sizer:
                        amount = position_sizer(signal, current_price, self.get_portfolio_value(current_prices))
                    else:
                        # Default: use 20% of equity
                        portfolio_value = self.get_portfolio_value(current_prices)
                        target_value = portfolio_value * 0.2
                        amount = target_value / current_price
                    
                    if amount > 0:
                        self.execute_trade(
                            timestamp, pair, "buy", amount, current_price,
                            signal.get("reason", "buy signal")
                        )
                
                elif signal_type == "sell":
                    # Close position if we have one
                    if pair in self.positions and self.positions[pair] > 0:
                        amount = self.positions[pair]
                        self.execute_trade(
                            timestamp, pair, "sell", amount, current_price,
                            signal.get("reason", "sell signal")
                        )
            
            # Record equity
            self.get_portfolio_value(current_prices, timestamp)
        
        # Close all positions at end
        final_timestamp = timestamps[-1]
        for pair, amount in list(self.positions.items()):
            if pair in current_prices and amount > 0:
                self.execute_trade(
                    final_timestamp, pair, "sell", amount, current_prices[pair],
                    "end of backtest"
                )
        
        # Final equity recording
        self.get_portfolio_value(current_prices, final_timestamp)
        
        return pd.DataFrame(self.equity_curve)
    
    def calculate_metrics(self, equity_curve: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics.
        
        Args:
            equity_curve: DataFrame with equity curve
            
        Returns:
            BacktestMetrics object
        """
        if len(equity_curve) < 2:
            raise ValueError("Insufficient data for metrics calculation")
        
        equity = equity_curve["equity"].values
        returns = pd.Series(equity).pct_change().dropna()
        
        # Basic returns
        total_return = (equity[-1] / equity[0]) - 1
        n_periods = len(returns)
        
        # Determine annualization factor
        if len(equity_curve) > 1:
            time_diff = (equity_curve["timestamp"].iloc[-1] - equity_curve["timestamp"].iloc[0])
            total_seconds = time_diff.total_seconds()
            periods_per_year = (365.25 * 24 * 3600) / (total_seconds / n_periods) if total_seconds > 0 else 252
        else:
            periods_per_year = 252
        
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / (excess_returns.std() + 1e-10)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = np.sqrt(periods_per_year) * returns.mean() / (downside_std + 1e-10) if downside_std > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        max_dd_duration = 0
        current_dd_duration = 0
        for in_dd in in_drawdown:
            if in_dd:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        sell_trades = [t for t in self.trades if t.side == "sell"]
        n_trades = len(sell_trades)
        
        if n_trades > 0:
            # Calculate P&L for each trade
            trade_pnl = []
            position_entry = {}  # pair -> (entry_price, amount)
            
            for trade in self.trades:
                if trade.side == "buy":
                    position_entry[trade.pair] = (trade.price, trade.amount)
                elif trade.side == "sell":
                    if trade.pair in position_entry:
                        entry_price, entry_amount = position_entry[trade.pair]
                        pnl = (trade.price - entry_price) * min(trade.amount, entry_amount) - trade.cost
                        trade_pnl.append(pnl)
                        # Update position
                        remaining = entry_amount - trade.amount
                        if remaining > 1e-8:
                            position_entry[trade.pair] = (entry_price, remaining)
                        else:
                            del position_entry[trade.pair]
            
            winning_trades = [pnl for pnl in trade_pnl if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnl if pnl <= 0]
            
            n_winning = len(winning_trades)
            n_losing = len(losing_trades)
            win_rate = n_winning / n_trades if n_trades > 0 else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
        else:
            n_winning = 0
            n_losing = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Exposure and turnover
        if "positions_value" in equity_curve.columns:
            avg_exposure = equity_curve["positions_value"].mean()
            max_exposure = equity_curve["positions_value"].max()
        else:
            avg_exposure = 0
            max_exposure = 0
        
        # Turnover: total trade value / average equity
        total_trade_value = sum(abs(t.value) for t in self.trades)
        avg_equity = equity.mean()
        turnover = total_trade_value / avg_equity if avg_equity > 0 else 0
        
        # Costs
        total_costs = sum(t.cost for t in self.trades)
        costs_as_pct = (total_costs / (equity[-1] - equity[0] + total_costs)) * 100 if (equity[-1] - equity[0] + total_costs) != 0 else 0
        
        # Hit rate (percentage of periods with positive returns)
        hit_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        return BacktestMetrics(
            total_return=total_return * 100,
            annualized_return=annualized_return * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown * 100,
            max_drawdown_duration=max_dd_duration,
            n_trades=n_trades,
            n_winning_trades=n_winning,
            n_losing_trades=n_losing,
            win_rate=win_rate * 100,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_exposure=avg_exposure,
            max_exposure=max_exposure,
            turnover=turnover,
            total_costs=total_costs,
            costs_as_pct_of_returns=costs_as_pct,
            hit_rate=hit_rate * 100
        )

