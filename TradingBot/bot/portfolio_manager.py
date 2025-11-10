"""Regime-adaptive portfolio management."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from bot.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about a position."""
    symbol: str
    entry_price: float
    amount: float
    current_value: float
    confidence: float
    regime: str


class RegimeAdaptivePortfolioManager:
    """Manages portfolio with regime-adaptive position sizing and risk management."""
    
    def __init__(self):
        """Initialize portfolio manager."""
        self.config = get_config()
    
    def calculate_position_size(
        self,
        base_size: float,
        regime: str,
        confidence: float,
        volatility: float,
        total_equity: float,
        param_overrides: Optional[Dict] = None
    ) -> float:
        """Calculate dynamic position size based on regime and confidence.
        
        Args:
            base_size: Base position size (default: 40% of equity)
            regime: Current regime (e.g., "calm_bullish", "volatile_bearish")
            confidence: Prediction confidence (0-1)
            volatility: Asset volatility (standard deviation)
            total_equity: Total account equity
        
        Returns:
            Position size in USD
        """
        # Start with base size
        position_value = base_size * total_equity
        
        # Adjust by regime (use overrides if provided)
        overrides = param_overrides or {}
        if "calm" in regime:
            multiplier = overrides.get("calm_multiplier", self.config.regime_calm_position_multiplier)
        elif "volatile" in regime:
            multiplier = overrides.get("volatile_multiplier", self.config.regime_volatile_position_multiplier)
        else:
            multiplier = 1.0
        
        if "bearish" in regime:
            bearish_mult = overrides.get("bearish_multiplier", self.config.regime_bearish_position_multiplier)
            multiplier *= bearish_mult
        elif "bullish" in regime:
            bullish_mult = overrides.get("bullish_multiplier", 1.0)
            multiplier *= bullish_mult
        
        position_value *= multiplier
        
        # Scale by confidence to power 1.2 (less aggressive reduction than squared)
        # This allows more position sizing while still rewarding high confidence
        confidence_multiplier = confidence ** 1.2
        position_value *= confidence_multiplier
        
        # Adjust inversely by volatility (higher vol = smaller position)
        if volatility > 0:
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)  # Scale volatility impact
            position_value *= volatility_adjustment
        
        # Cap at maximum position size (60% of equity - more aggressive)
        max_position_value = total_equity * 0.6
        position_value = min(position_value, max_position_value)
        
        # Ensure minimum viable position (reduced from 5% to 3%)
        min_position_value = total_equity * 0.03  # At least 3% to be meaningful
        if position_value < min_position_value:
            return 0.0
        
        return position_value
    
    def calculate_adaptive_stops(
        self,
        entry_price: float,
        regime: str,
        confidence: float,
        volatility: float,
        param_overrides: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """Calculate adaptive stop loss and take profit based on regime.
        
        Args:
            entry_price: Entry price
            regime: Current regime
            confidence: Prediction confidence
            volatility: Asset volatility
        
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Base stops (use overrides if provided, otherwise use config)
        overrides = param_overrides or {}
        base_stop_pct = overrides.get("base_stop_loss_pct", self.config.stop_loss_pct)
        base_tp_pct = overrides.get("base_take_profit_pct", self.config.take_profit_pct)
        
        # Adjust by regime
        if "calm" in regime:
            # Tighter stops in calm markets
            stop_pct = base_stop_pct * 0.8
            tp_pct = base_tp_pct * 0.9
        elif "volatile" in regime:
            # Wider stops in volatile markets
            stop_pct = base_stop_pct * 1.5
            tp_pct = base_tp_pct * 1.3
        else:
            stop_pct = base_stop_pct
            tp_pct = base_tp_pct
        
        # Adjust by confidence (higher confidence = tighter stop, higher target)
        if confidence > 0.8:
            stop_pct *= 0.9
            tp_pct *= 1.1
        elif confidence < 0.6:
            stop_pct *= 1.2
            tp_pct *= 0.9
        
        # Adjust by volatility
        if volatility > 0.03:  # High volatility
            stop_pct *= 1.3
            tp_pct *= 1.2
        
        stop_loss = entry_price * (1 - stop_pct)
        take_profit = entry_price * (1 + tp_pct)
        
        return stop_loss, take_profit
    
    def check_portfolio_allocation(
        self,
        current_positions: Dict[str, PositionInfo],
        total_equity: float
    ) -> Tuple[float, float, bool]:
        """Check current portfolio allocation.
        
        Args:
            current_positions: Dictionary of current positions
            total_equity: Total account equity
        
        Returns:
            Tuple of (allocated_value, allocated_pct, can_add_position)
        """
        allocated_value = sum(pos.current_value for pos in current_positions.values())
        allocated_pct = allocated_value / total_equity if total_equity > 0 else 0.0
        
        max_allocation = self.config.max_portfolio_allocation
        can_add = allocated_pct < max_allocation and len(current_positions) < self.config.max_simultaneous_positions
        
        return allocated_value, allocated_pct, can_add
    
    def rank_signals(
        self,
        signals: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float, Dict]]:
        """Rank trading signals by confidence and quality.
        
        Args:
            signals: Dictionary mapping pair to signal dict with 'confidence', 'regime', etc.
        
        Returns:
            Sorted list of (pair, score, signal_dict) tuples, best first
        """
        ranked = []
        
        for pair, signal in signals.items():
            confidence = signal.get("confidence", 0.0)
            regime = signal.get("regime", "unknown")
            
            # Base score is confidence
            score = confidence
            
            # Boost score for favorable regimes
            if "calm" in regime and "bullish" in regime:
                score *= 1.2
            elif "volatile" in regime or "bearish" in regime:
                score *= 0.8
            
            ranked.append((pair, score, signal))
        
        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def limit_positions(
        self,
        positions: Dict[str, PositionInfo],
        max_count: Optional[int] = None
    ) -> Dict[str, PositionInfo]:
        """Limit number of positions, keeping best ones.
        
        Args:
            positions: Current positions
            max_count: Maximum number of positions (default: from config)
        
        Returns:
            Filtered positions dictionary
        """
        if max_count is None:
            max_count = self.config.max_simultaneous_positions
        
        if len(positions) <= max_count:
            return positions
        
        # Sort by confidence and keep top N
        sorted_positions = sorted(
            positions.items(),
            key=lambda x: x[1].confidence,
            reverse=True
        )
        
        return dict(sorted_positions[:max_count])
    
    def should_close_position(
        self,
        position: PositionInfo,
        current_price: float,
        current_regime: str
    ) -> Tuple[bool, str]:
        """Determine if a position should be closed.
        
        Args:
            position: Position information
            current_price: Current market price
            current_regime: Current market regime
        
        Returns:
            Tuple of (should_close, reason)
        """
        # Check stop loss / take profit (handled elsewhere)
        
        # Check regime change
        if "bearish" in current_regime and "bullish" not in position.regime:
            return True, "Regime changed to bearish"
        
        # Check if confidence would be too low now
        # (This would need current prediction confidence, handled in trading engine)
        
        return False, ""

