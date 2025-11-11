"""BrokerSim: simulates order execution with fees and slippage."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class Fill:
    """Order fill."""
    symbol: str
    side: str
    price: float
    quantity: float
    fees: float
    fill_type: str  # "entry", "exit_stop", "exit_target", "exit_timeout"
    dt: pd.Timestamp


class BrokerSim:
    """Simulates broker with fees and slippage."""
    
    def __init__(self, params: Dict):
        """Initialize broker simulator.
        
        Args:
            params: Configuration parameters
        """
        self.params = params
        self.fees_slip = params.get('fees_slip', {})
        self.maker_fee = self.fees_slip.get('maker', 0.0002)
        self.taker_fee = self.fees_slip.get('taker', 0.0005)
        self.slippage_base = self.fees_slip.get('slippage_pct_base', 0.0001)
        self.vol_scaled = self.fees_slip.get('vol_scaled_slippage', True)
    
    def calculate_slippage(
        self,
        price: float,
        side: str,
        rv: Optional[float] = None
    ) -> float:
        """Calculate slippage as percentage.
        
        Args:
            price: Base price
            side: "long" or "short"
            rv: Realized volatility (optional, for scaling)
            
        Returns:
            Slippage as decimal (e.g., 0.0001 = 0.01%)
        """
        slippage = self.slippage_base
        
        if self.vol_scaled and rv is not None:
            # Scale slippage with volatility (higher vol = more slippage)
            # Normalize RV to reasonable range (0-0.5 = 1x to 3x base)
            rv_factor = min(3.0, 1.0 + (rv / 0.5))
            slippage = slippage * rv_factor
        
        return slippage
    
    def enter(
        self,
        symbol: str,
        side: str,
        signal_price: float,
        quantity: float,
        dt: pd.Timestamp,
        rv: Optional[float] = None
    ) -> Fill:
        """Simulate entry fill.
        
        Args:
            symbol: Symbol name
            side: "long" or "short"
            signal_price: Signal price (bar close)
            quantity: Position quantity
            dt: Entry datetime
            rv: Realized volatility (optional)
            
        Returns:
            Fill object
        """
        # Entry: use taker fee (market order)
        slippage = self.calculate_slippage(signal_price, side, rv)
        
        if side == "long":
            fill_price = signal_price * (1 + slippage)
        else:  # short
            fill_price = signal_price * (1 - slippage)
        
        notional = fill_price * quantity
        fees = notional * self.taker_fee
        
        return Fill(
            symbol=symbol,
            side=side,
            price=fill_price,
            quantity=quantity,
            fees=fees,
            fill_type="entry",
            dt=dt
        )
    
    def exit(
        self,
        symbol: str,
        side: str,
        exit_price: float,
        quantity: float,
        dt: pd.Timestamp,
        exit_type: str,
        rv: Optional[float] = None
    ) -> Fill:
        """Simulate exit fill.
        
        Args:
            symbol: Symbol name
            side: "long" or "short"
            exit_price: Exit price (bar close)
            quantity: Position quantity
            dt: Exit datetime
            exit_type: "exit_stop", "exit_target", "exit_timeout"
            rv: Realized volatility (optional)
            
        Returns:
            Fill object
        """
        # Exit: use taker fee
        slippage = self.calculate_slippage(exit_price, side, rv)
        
        if side == "long":
            fill_price = exit_price * (1 - slippage)  # Sell at lower price
        else:  # short
            fill_price = exit_price * (1 + slippage)  # Buy back at higher price
        
        notional = fill_price * quantity
        fees = notional * self.taker_fee
        
        return Fill(
            symbol=symbol,
            side=side,
            price=fill_price,
            quantity=quantity,
            fees=fees,
            fill_type=exit_type,
            dt=dt
        )
    
    def step_exits(
        self,
        positions: Dict,
        bar_data: Dict[str, pd.Series],
        current_dt: pd.Timestamp,
        prev_close: Optional[Dict[str, float]] = None
    ) -> list:
        """Check exits for all positions on current bar.
        
        Uses proxy high/low for close-only data to simulate intrabar touches.
        
        Args:
            positions: Dictionary of Position objects
            bar_data: Dictionary mapping symbol to current bar data (close, atr, rv)
            current_dt: Current datetime
            prev_close: Previous bar close prices (for return calculation)
            
        Returns:
            List of Fill objects for exits
        """
        fills = []
        
        for pos_id, position in positions.items():
            symbol = position.symbol
            if symbol not in bar_data:
                continue
            
            bar = bar_data[symbol]
            close = bar.get('close')
            atr = bar.get('atr', 0.0)
            rv = bar.get('rv', None)
            
            if close is None or pd.isna(close):
                continue
            
            # Calculate proxy high/low from returns dispersion
            # For close-only data, estimate intrabar range
            k = 1.5  # Multiplier for return dispersion
            if prev_close and symbol in prev_close:
                ret_1m = abs((close - prev_close[symbol]) / prev_close[symbol])
            else:
                ret_1m = abs(atr / close) if close > 0 else 0.001
            
            hi_proxy = close * (1 + k * ret_1m)
            lo_proxy = close * (1 - k * ret_1m)
            
            # Check stop first (conservative: stop before target)
            stop_hit = False
            exit_price = close
            
            if position.side == "long":
                # Long: stop hit if low proxy <= stop
                if lo_proxy <= position.stop_current:
                    stop_hit = True
                    exit_price = max(position.stop_current, close)  # Use stop or close, whichever is higher
            else:  # short
                # Short: stop hit if high proxy >= stop
                if hi_proxy >= position.stop_current:
                    stop_hit = True
                    exit_price = min(position.stop_current, close)  # Use stop or close, whichever is lower
            
            if stop_hit:
                fill = self.exit(
                    symbol=symbol,
                    side=position.side,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    dt=current_dt,
                    exit_type="exit_stop",
                    rv=rv
                )
                fills.append((pos_id, fill))
                continue
            
            # Check target
            if position.target is not None:
                target_hit = False
                
                if position.side == "long":
                    # Long: target hit if high proxy >= target
                    if hi_proxy >= position.target:
                        target_hit = True
                        exit_price = min(position.target, close)  # Use target or close, whichever is lower
                else:  # short
                    # Short: target hit if low proxy <= target
                    if lo_proxy <= position.target:
                        target_hit = True
                        exit_price = max(position.target, close)  # Use target or close, whichever is higher
                
                if target_hit:
                    fill = self.exit(
                        symbol=symbol,
                        side=position.side,
                        exit_price=exit_price,
                        quantity=position.quantity,
                        dt=current_dt,
                        exit_type="exit_target",
                        rv=rv
                    )
                    fills.append((pos_id, fill))
                    continue
        
        return fills

