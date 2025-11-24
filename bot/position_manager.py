"""Multi-position management with batch entry tracking and staggered exits."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from bot.config import get_config


@dataclass
class BatchEntry:
    """Single batch entry within a position."""
    entry_price: float
    amount: float
    filled: bool = False
    fill_time: Optional[datetime] = None


@dataclass
class ManagedPosition:
    """Managed position with batch entries and partial exit tracking."""
    pair: str
    strategy_type: str  # "breakout" or "mean_reversion"
    entry_time: datetime
    stop_loss: float
    take_profit: float
    quality: float
    atr_value: float
    rsi_value: float
    macd_signal: str
    
    # Batch entry tracking
    batch_entries: List[BatchEntry] = field(default_factory=list)
    target_batch_prices: List[float] = field(default_factory=list)
    batch_spacing_pct: float = 0.10  # 10% spacing
    
    # Staggered exit tracking (5-3-2 or scaled version)
    exit_1_pct: float = 0.50  # 50% at first exit level
    exit_2_pct: float = 0.30  # 30% at second exit level
    exit_3_pct: float = 0.20  # 20% at final exit level
    exit_1_level: float = 0.0  # +50% or +10% for scaled
    exit_2_level: float = 0.0  # +30% or +20% for scaled
    exit_3_level: float = 0.0  # +20% or +30% for scaled
    exit_1_filled: bool = False
    exit_2_filled: bool = False
    exit_3_filled: bool = False
    
    # Trailing stop
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    total_amount: float = 0.0  # Total position size across all batches
    average_entry_price: float = 0.0
    
    def calculate_average_entry(self) -> float:
        """Calculate weighted average entry price from filled batches."""
        if not self.batch_entries:
            return 0.0
        
        filled_batches = [b for b in self.batch_entries if b.filled]
        if not filled_batches:
            return 0.0
        
        total_value = sum(b.entry_price * b.amount for b in filled_batches)
        total_amount = sum(b.amount for b in filled_batches)
        
        if total_amount > 0:
            self.average_entry_price = total_value / total_amount
            self.total_amount = total_amount
            return self.average_entry_price
        
        return 0.0
    
    def update_pnl(self, current_price: float) -> None:
        """Update P&L based on current price."""
        self.current_price = current_price
        avg_entry = self.calculate_average_entry()
        
        if avg_entry > 0 and self.total_amount > 0:
            self.unrealized_pnl = (current_price - avg_entry) * self.total_amount
            self.unrealized_pnl_pct = ((current_price - avg_entry) / avg_entry) * 100
            
            # Update highest price for trailing stop
            if current_price > self.highest_price:
                self.highest_price = current_price
    
    def should_exit_staggered(self, current_price: float) -> Tuple[bool, float, str]:
        """Check if position should exit at staggered profit levels.
        
        Returns:
            Tuple of (should_exit, exit_amount, reason)
        """
        avg_entry = self.calculate_average_entry()
        if avg_entry == 0:
            return False, 0.0, ""
        
        profit_pct = ((current_price - avg_entry) / avg_entry) * 100
        
        # Exit 1: 33% at +8% gain (achievable target, was +15% - too high)
        if not self.exit_1_filled and profit_pct >= self.exit_1_level:
            exit_amount = self.total_amount * self.exit_1_pct
            self.exit_1_filled = True
            return True, exit_amount, f"Staggered exit 1: {profit_pct:.2f}% profit"
        
        # Exit 2: 33% at +15% gain (achievable target, was +25% - too high)
        if not self.exit_2_filled and profit_pct >= self.exit_2_level:
            exit_amount = self.total_amount * self.exit_2_pct
            self.exit_2_filled = True
            return True, exit_amount, f"Staggered exit 2: {profit_pct:.2f}% profit"
        
        # Exit 3: 34% at +25% gain (achievable target, was +40% - too high)
        if not self.exit_3_filled and profit_pct >= self.exit_3_level:
            exit_amount = self.total_amount * self.exit_3_pct
            self.exit_3_filled = True
            return True, exit_amount, f"Staggered exit 3: {profit_pct:.2f}% profit"
        
        return False, 0.0, ""
    
    def should_exit_stop_loss(self, current_price: float) -> bool:
        """Check if position should exit on stop loss."""
        avg_entry = self.calculate_average_entry()
        if avg_entry == 0:
            return False
        
        # Check stop loss
        if self.batch_entries and self.batch_entries[0].filled:  # Use first batch direction
            is_long = True  # Assume long for now
            if is_long and current_price <= self.stop_loss:
                return True
            elif not is_long and current_price >= self.stop_loss:
                return True
        
        return False
    
    def should_exit_trailing_stop(self, current_price: float, trailing_pct: float = 0.02) -> bool:
        """Check if position should exit on trailing stop."""
        if self.highest_price == 0:
            return False
        
        avg_entry = self.calculate_average_entry()
        if avg_entry == 0:
            return False
        
        # Update trailing stop
        if self.highest_price > 0:
            trailing_stop_price = self.highest_price * (1 - trailing_pct)
            if self.trailing_stop is None or trailing_stop_price > self.trailing_stop:
                self.trailing_stop = trailing_stop_price
        
        # Check if price hit trailing stop
        if self.trailing_stop and current_price <= self.trailing_stop:
            return True
        
        return False


class PositionManager:
    """Manage multiple positions with batch entry tracking and staggered exits."""
    
    def __init__(
        self,
        max_positions: int = 10,
        max_position_pct: float = 0.25,
        min_correlation: float = 0.7,
        use_scaled_exits: bool = True  # Use scaled exits (7%, 15%, 25%) optimized for 14-day competition
    ):
        """Initialize position manager.
        
        Args:
            max_positions: Maximum number of simultaneous positions
            max_position_pct: Maximum position size as fraction of equity
            min_correlation: Minimum correlation to consider positions correlated
            use_scaled_exits: Use scaled exit levels (7%, 15%, 25%) optimized for 14-day competition
        """
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.min_correlation = min_correlation
        self.use_scaled_exits = use_scaled_exits
        self.positions: Dict[str, ManagedPosition] = {}
        self.config = get_config()
    
    def add_position(
        self,
        pair: str,
        strategy_type: str,
        stop_loss: float,
        take_profit: float,
        quality: float,
        atr_value: float,
        rsi_value: float,
        macd_signal: str,
        batch_entry_prices: List[float],
        batch_spacing_pct: float = 0.10,
        use_scaled_exits: Optional[bool] = None
    ) -> bool:
        """Add a new position with batch entry prices.
        
        Args:
            pair: Trading pair
            strategy_type: Strategy type ("breakout" or "mean_reversion")
            stop_loss: Stop loss price
            take_profit: Take profit price
            quality: Signal quality
            atr_value: ATR value for stop loss
            rsi_value: Current RSI value
            macd_signal: MACD signal
            batch_entry_prices: List of batch entry prices (4 batches)
            batch_spacing_pct: Price spacing between batches
            use_scaled_exits: Override default scaled exits setting
            
        Returns:
            True if position added, False if max positions reached
        """
        if len(self.positions) >= self.max_positions:
            return False
        
        # Determine exit levels - use config values (optimized for 14-day competition)
        use_scaled = use_scaled_exits if use_scaled_exits is not None else self.config.use_scaled_exits
        
        if use_scaled:
            # Scaled version: 33% at +7%, 33% at +15%, 34% at +25% (optimized for 14-day)
            exit_1_level = self.config.exit_1_level  # 7%
            exit_2_level = self.config.exit_2_level  # 15%
            exit_3_level = self.config.exit_3_level  # 25%
            exit_1_pct = self.config.exit_1_pct
            exit_2_pct = self.config.exit_2_pct
            exit_3_pct = self.config.exit_3_pct
        else:
            # Original 5-3-2: 50% at +50%, 30% at +30%, 20% at +20%
            exit_1_level = 50.0
            exit_2_level = 30.0
            exit_3_level = 20.0
            exit_1_pct = 0.50
            exit_2_pct = 0.30
            exit_3_pct = 0.20
        
        # Create batch entries
        batch_entries = []
        for entry_price in batch_entry_prices:
            batch_entries.append(BatchEntry(entry_price=entry_price, amount=0.0, filled=False))
        
        position = ManagedPosition(
            pair=pair,
            strategy_type=strategy_type,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            quality=quality,
            atr_value=atr_value,
            rsi_value=rsi_value,
            macd_signal=macd_signal,
            batch_entries=batch_entries,
            target_batch_prices=batch_entry_prices,
            batch_spacing_pct=batch_spacing_pct,
            exit_1_pct=exit_1_pct,
            exit_2_pct=exit_2_pct,
            exit_3_pct=exit_3_pct,
            exit_1_level=exit_1_level,
            exit_2_level=exit_2_level,
            exit_3_level=exit_3_level
        )
        
        self.positions[pair] = position
        return True
    
    def fill_batch_entry(
        self,
        pair: str,
        batch_index: int,
        amount: float
    ) -> bool:
        """Mark a batch entry as filled.
        
        Args:
            pair: Trading pair
            batch_index: Index of batch entry (0-3)
            amount: Amount filled
            
        Returns:
            True if successful
        """
        if pair not in self.positions:
            return False
        
        position = self.positions[pair]
        if batch_index >= len(position.batch_entries):
            return False
        
        batch = position.batch_entries[batch_index]
        batch.amount = amount
        batch.filled = True
        batch.fill_time = datetime.now()
        
        # Recalculate average entry
        position.calculate_average_entry()
        
        return True
    
    def remove_position(self, pair: str) -> bool:
        """Remove a position.
        
        Args:
            pair: Trading pair to remove
            
        Returns:
            True if position removed, False if not found
        """
        if pair in self.positions:
            del self.positions[pair]
            return True
        return False
    
    def get_positions(self) -> Dict[str, ManagedPosition]:
        """Get all current positions.
        
        Returns:
            Dictionary of positions
        """
        return self.positions.copy()
    
    def get_position(self, pair: str) -> Optional[ManagedPosition]:
        """Get a specific position.
        
        Args:
            pair: Trading pair
            
        Returns:
            ManagedPosition or None
        """
        return self.positions.get(pair)
    
    def update_prices(self, current_prices: Dict[str, float]) -> None:
        """Update current prices for all positions.
        
        Args:
            current_prices: Dictionary of pair -> current price
        """
        for pair, position in self.positions.items():
            if pair in current_prices:
                position.update_pnl(current_prices[pair])
    
    def check_exits(
        self,
        current_prices: Dict[str, float],
        trailing_stop_pct: float = 0.02
    ) -> List[Tuple[str, float, str]]:
        """Check all positions for exit conditions.
        
        Args:
            current_prices: Dictionary of pair -> current price
            trailing_stop_pct: Trailing stop percentage
            
        Returns:
            List of (pair, exit_amount, reason) tuples for positions that should exit
        """
        exits = []
        
        for pair, position in self.positions.items():
            if pair not in current_prices:
                continue
            
            current_price = current_prices[pair]
            position.update_pnl(current_price)
            
            # Check stop loss
            if position.should_exit_stop_loss(current_price):
                avg_entry = position.calculate_average_entry()
                if avg_entry > 0:
                    exits.append((pair, position.total_amount, "Stop loss"))
                    continue
            
            # Check trailing stop
            if position.should_exit_trailing_stop(current_price, trailing_stop_pct):
                avg_entry = position.calculate_average_entry()
                if avg_entry > 0:
                    exits.append((pair, position.total_amount, "Trailing stop"))
                    continue
            
            # Check staggered exits
            should_exit, exit_amount, reason = position.should_exit_staggered(current_price)
            if should_exit and exit_amount > 0:
                exits.append((pair, exit_amount, reason))
        
        return exits
    
    def get_total_exposure(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio exposure.
        
        Args:
            current_prices: Current prices for all pairs
            
        Returns:
            Total exposure in USD
        """
        total = 0.0
        for pair, position in self.positions.items():
            if pair in current_prices:
                total += position.total_amount * current_prices[pair]
            else:
                avg_entry = position.calculate_average_entry()
                if avg_entry > 0:
                    total += position.total_amount * avg_entry
        
        return total
    
    def get_portfolio_pnl(self, current_prices: Dict[str, float]) -> Tuple[float, float]:
        """Calculate total portfolio P&L.
        
        Args:
            current_prices: Current prices for all pairs
            
        Returns:
            Tuple of (total_pnl, total_pnl_pct)
        """
        total_pnl = 0.0
        total_cost = 0.0
        
        for pair, position in self.positions.items():
            position.update_pnl(current_prices.get(pair, 0))
            total_pnl += position.unrealized_pnl
            avg_entry = position.calculate_average_entry()
            if avg_entry > 0:
                total_cost += position.total_amount * avg_entry
        
        pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
        
        return total_pnl, pnl_pct

