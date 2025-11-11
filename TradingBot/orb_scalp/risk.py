"""RiskEngine: position sizing, trailing stops, daily kill-switch."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class Position:
    """Open position."""
    symbol: str
    side: str  # "long" or "short"
    signal_type: str  # "ORB" or "SCALP"
    entry_dt: pd.Timestamp
    entry_price: float
    quantity: float
    stop_init: float
    stop_current: float  # Current stop (may be trailing)
    stop_distance: float  # Initial stop distance in price units (1R)
    target: Optional[float] = None
    trail_state: str = "initial"  # "initial", "breakeven", "trailing"
    timeout_dt: Optional[pd.Timestamp] = None
    risk_dollars: float = 0.0  # 1R in dollars


class RiskEngine:
    """Handles position sizing, trailing stops, and risk controls."""
    
    def __init__(self, params: Dict):
        """Initialize risk engine.
        
        Args:
            params: Configuration parameters
        """
        self.params = params
        self.risk_params = params.get('risk', {})
        self.orb_params = params.get('orb', {})
        self.scalp_params = params.get('scalp', {})
    
    def size_trade(
        self,
        equity: float,
        stop_distance: float,
        signal_type: str
    ) -> float:
        """Calculate position size based on risk.
        
        Args:
            equity: Current equity
            stop_distance: Stop distance in price units
            signal_type: "ORB" or "SCALP"
            
        Returns:
            Quantity (position size)
        """
        if signal_type == "ORB":
            risk_frac = self.orb_params.get('risk_frac', 0.0075)
        else:  # SCALP
            risk_frac = self.scalp_params.get('risk_frac', 0.005)
        
        risk_dollars = equity * risk_frac
        
        if stop_distance <= 0:
            return 0.0
        
        quantity = risk_dollars / stop_distance
        
        return quantity
    
    def calculate_stops(
        self,
        entry_price: float,
        side: str,
        stop_distance: float,
        signal_type: str
    ) -> tuple:
        """Calculate initial stop and target prices.
        
        Args:
            entry_price: Entry price
            side: "long" or "short"
            stop_distance: Stop distance in price units
            signal_type: "ORB" or "SCALP"
            
        Returns:
            (stop_price, target_price)
        """
        if side == "long":
            stop_price = entry_price - stop_distance
            if signal_type == "SCALP":
                tp_pct = self.scalp_params.get('tp_pct', 0.012)
                target_price = entry_price * (1 + tp_pct)
            else:  # ORB: use 1.5R target
                target_price = entry_price + (stop_distance * 1.5)
        else:  # short
            stop_price = entry_price + stop_distance
            if signal_type == "SCALP":
                tp_pct = self.scalp_params.get('tp_pct', 0.012)
                target_price = entry_price * (1 - tp_pct)
            else:  # ORB: use 1.5R target
                target_price = entry_price - (stop_distance * 1.5)
        
        return stop_price, target_price
    
    def update_trailing(
        self,
        position: Position,
        current_price: float,
        atr: float,
        signal_type: str
    ) -> bool:
        """Update trailing stop. Returns True if stop was updated.
        
        Args:
            position: Position object
            current_price: Current market price
            atr: Current ATR value
            signal_type: "ORB" or "SCALP"
            
        Returns:
            True if stop was updated
        """
        if signal_type == "ORB":
            trail_atr_mult = self.orb_params.get('trail_atr_mult', 0.7)
            breakeven_at_r = 1.0
        else:  # SCALP
            trail_pct = self.scalp_params.get('trail_pct', 0.004)
            breakeven_at_r = 0.66  # Move to breakeven at 66% of TP
        
        # Calculate current R multiple
        if position.side == "long":
            profit = current_price - position.entry_price
        else:  # short
            profit = position.entry_price - current_price
        
        profit_r = profit / position.stop_distance if position.stop_distance > 0 else 0.0
        
        # Move to breakeven
        if position.trail_state == "initial" and profit_r >= breakeven_at_r:
            position.stop_current = position.entry_price
            position.trail_state = "breakeven"
            return True
        
        # Activate trailing stop
        if profit_r >= 1.0:  # At +1R or more
            if signal_type == "ORB":
                trail_distance = trail_atr_mult * atr
                if position.side == "long":
                    new_stop = current_price - trail_distance
                    # Never lower than entry (breakeven)
                    new_stop = max(new_stop, position.entry_price)
                    if new_stop > position.stop_current:
                        position.stop_current = new_stop
                        position.trail_state = "trailing"
                        return True
                else:  # short
                    new_stop = current_price + trail_distance
                    # Never higher than entry (breakeven)
                    new_stop = min(new_stop, position.entry_price)
                    if new_stop < position.stop_current:
                        position.stop_current = new_stop
                        position.trail_state = "trailing"
                        return True
            else:  # SCALP: use percentage-based trailing
                trail_distance = current_price * trail_pct
                if position.side == "long":
                    new_stop = current_price - trail_distance
                    new_stop = max(new_stop, position.entry_price)
                    if new_stop > position.stop_current:
                        position.stop_current = new_stop
                        position.trail_state = "trailing"
                        return True
                else:  # short
                    new_stop = current_price + trail_distance
                    new_stop = min(new_stop, position.entry_price)
                    if new_stop < position.stop_current:
                        position.stop_current = new_stop
                        position.trail_state = "trailing"
                        return True
        
        return False
    
    def check_timeout(
        self,
        position: Position,
        current_dt: pd.Timestamp
    ) -> bool:
        """Check if position should timeout.
        
        Args:
            position: Position object
            current_dt: Current datetime
            
        Returns:
            True if timeout
        """
        if position.timeout_dt is None:
            return False
        
        return current_dt >= position.timeout_dt
    
    def daily_kill_switch(
        self,
        equity_curve: pd.DataFrame,
        day_start_equity: float,
        current_equity: float
    ) -> bool:
        """Check if daily loss limit exceeded.
        
        Args:
            equity_curve: Equity curve DataFrame
            day_start_equity: Equity at start of day
            current_equity: Current equity
            
        Returns:
            True if kill switch should activate
        """
        max_daily_dd = self.risk_params.get('max_daily_dd', 0.03)
        
        daily_loss = (current_equity - day_start_equity) / day_start_equity
        
        return daily_loss <= -max_daily_dd

