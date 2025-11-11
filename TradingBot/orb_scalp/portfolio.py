"""Portfolio: tracks positions, equity, exposure, caps."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import date
from dataclasses import dataclass, field

from .risk import Position
from .broker import Fill


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    side: str
    signal_type: str
    entry_dt: pd.Timestamp
    exit_dt: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl_abs: float
    pnl_pct: float
    pnl_r: float  # P&L in R multiples
    fees: float
    exit_reason: str
    risk_dollars: float  # 1R in dollars


class Portfolio:
    """Manages portfolio state, positions, and equity."""
    
    def __init__(self, params: Dict, initial_capital: float = 10000.0):
        """Initialize portfolio.
        
        Args:
            params: Configuration parameters
            initial_capital: Starting capital
        """
        self.params = params
        self.risk_params = params.get('risk', {})
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # pos_id -> Position
        self.trades: List[Trade] = []
        self.equity_curve: List[tuple] = []  # (dt, equity)
        
        # Daily state
        self.day_start_equity: Dict[date, float] = {}
        self.trade_enabled: bool = True
        
        # Position counters
        self.positions_per_symbol: Dict[str, int] = {}
        self.next_pos_id = 1
        
        # Daily scalp counter (moved from signal stage)
        self.scalp_count_today: Dict[date, int] = {}
    
    def equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate current equity (cash + positions).
        
        Args:
            current_prices: Dictionary mapping symbol to current price
            
        Returns:
            Total equity
        """
        equity = self.cash
        
        for position in self.positions.values():
            symbol = position.symbol
            if symbol in current_prices:
                price = current_prices[symbol]
                # Calculate P&L only (not notional)
                if position.side == "long":
                    # Long P&L = (current - entry) * quantity
                    pnl = (price - position.entry_price) * position.quantity
                else:  # short
                    # Short P&L = (entry - current) * quantity
                    pnl = (position.entry_price - price) * position.quantity
                
                # Add P&L to equity (don't add notional, just the profit/loss)
                equity += pnl
        
        return equity
    
    def can_take_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float
    ) -> Tuple[bool, str]:
        """Check if we can take a new position.
        
        Args:
            symbol: Symbol name
            side: "long" or "short"
            quantity: Position quantity
            entry_price: Entry price
            
        Returns:
            (can_take, reason)
        """
        # Check max concurrent positions
        max_concurrent = self.risk_params.get('max_concurrent_positions', 4)
        if len(self.positions) >= max_concurrent:
            return False, "MAX_CONCURRENT_POS"
        
        # Check max positions per symbol
        max_per_symbol = self.risk_params.get('max_positions_per_symbol', 2)
        symbol_count = self.positions_per_symbol.get(symbol, 0)
        if symbol_count >= max_per_symbol:
            return False, "MAX_PER_SYMBOL_POS"
        
        # Check notional exposure
        notional = quantity * entry_price
        current_equity = self.equity({})
        max_notional = current_equity * self.risk_params.get('max_notional_exposure_mult', 1.2)
        
        total_notional = sum(
            pos.quantity * pos.entry_price for pos in self.positions.values()
        )
        
        if (total_notional + notional) > max_notional:
            return False, "MAX_NOTIONAL_EXPOSURE"
        
        # Check if trading is enabled (kill switch)
        if not self.trade_enabled:
            return False, "KILL_SWITCH"
        
        return True, ""
    
    def check_scalp_daily_limit(self, current_day: date) -> Tuple[bool, str]:
        """Check if daily scalp limit is reached.
        
        Args:
            current_day: Current day (date object)
            
        Returns:
            (can_trade, reason)
        """
        scalp_params = self.params.get('scalp', {})
        max_scalps = scalp_params.get('max_scalps_per_day_all_pairs', 8)
        scalp_count = self.scalp_count_today.get(current_day, 0)
        
        if scalp_count >= max_scalps:
            return False, "DAILY_SCALP_CAP"
        
        return True, ""
    
    def add_position(
        self,
        position: Position,
        fill: Fill
    ) -> str:
        """Add a new position.
        
        Args:
            position: Position object
            fill: Entry fill
            
        Returns:
            Position ID
        """
        pos_id = f"pos_{self.next_pos_id}"
        self.next_pos_id += 1
        
        # Increment daily scalp counter if this is a scalp trade
        if position.signal_type == "SCALP":
            # Convert entry_dt to date (handle both Timestamp and integer period)
            if isinstance(position.entry_dt, pd.Timestamp):
                trade_day = position.entry_dt.date()
            else:
                # If it's a period index, convert to day
                trade_day = date.fromordinal(int(position.entry_dt // 1440) + 1)  # Approximate
            self.scalp_count_today[trade_day] = self.scalp_count_today.get(trade_day, 0) + 1
        
        self.positions[pos_id] = position
        
        # Update counters
        symbol = position.symbol
        self.positions_per_symbol[symbol] = self.positions_per_symbol.get(symbol, 0) + 1
        
        # Deduct cash
        notional = fill.price * fill.quantity
        self.cash -= (notional + fill.fees)
        
        return pos_id
    
    def close_position(
        self,
        pos_id: str,
        fill: Fill,
        exit_reason: str
    ):
        """Close a position.
        
        Args:
            pos_id: Position ID
            fill: Exit fill
            exit_reason: Exit reason
        """
        if pos_id not in self.positions:
            return
        
        position = self.positions[pos_id]
        
        # Calculate P&L
        entry_value = position.entry_price * position.quantity
        exit_value = fill.price * fill.quantity
        
        if position.side == "long":
            pnl_abs = exit_value - entry_value - fill.fees
        else:  # short
            pnl_abs = entry_value - exit_value - fill.fees
        
        pnl_pct = (pnl_abs / entry_value) * 100 if entry_value > 0 else 0.0
        
        # Calculate R multiple
        pnl_r = pnl_abs / position.risk_dollars if position.risk_dollars > 0 else 0.0
        
        # Create trade record
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            signal_type=position.signal_type,
            entry_dt=position.entry_dt,
            exit_dt=fill.dt,
            entry_price=position.entry_price,
            exit_price=fill.price,
            quantity=position.quantity,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
            pnl_r=pnl_r,
            fees=fill.fees,
            exit_reason=exit_reason,
            risk_dollars=position.risk_dollars
        )
        
        self.trades.append(trade)
        
        # Add cash back
        self.cash += exit_value - fill.fees
        
        # Remove position
        symbol = position.symbol
        del self.positions[pos_id]
        
        # Update counters
        self.positions_per_symbol[symbol] = max(0, self.positions_per_symbol.get(symbol, 0) - 1)
    
    def update_daily_state(self, current_date: date, current_equity: float):
        """Update daily state (call at start of each day).
        
        Args:
            current_date: Current date
            current_equity: Current equity
        """
        # Set day start equity
        self.day_start_equity[current_date] = current_equity
        
        # Reset kill switch if new day
        if current_date not in self.day_start_equity:
            self.trade_enabled = True
    
    def check_kill_switch(self, current_date: date, current_equity: float) -> bool:
        """Check and update kill switch.
        
        Args:
            current_date: Current date
            current_equity: Current equity
            
        Returns:
            True if kill switch is active
        """
        if current_date not in self.day_start_equity:
            return False
        
        day_start = self.day_start_equity[current_date]
        max_daily_dd = self.risk_params.get('max_daily_dd', 0.03)
        
        daily_loss = (current_equity - day_start) / day_start if day_start > 0 else 0.0
        
        if daily_loss <= -max_daily_dd:
            self.trade_enabled = False
            return True
        
        return False
    
    def record_equity(self, dt: pd.Timestamp, equity: float):
        """Record equity point.
        
        Args:
            dt: Datetime
            equity: Equity value
        """
        self.equity_curve.append((dt, equity))

