"""Optimized Sharpe mode signal generator with better entry/exit logic."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from bot.config import get_config
from bot.datastore import DataStore


@dataclass
class OptimizedSharpeSignal:
    """Optimized Sharpe signal with stop loss and take profit."""
    pair: str
    signal: str  # "buy", "sell", "neutral"
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    sharpe_ratio: float
    momentum: float
    quality: float  # 0-1
    reason: str


class OptimizedSharpeSignalGenerator:
    """Optimized Sharpe mode with better entry conditions and risk management."""
    
    def __init__(
        self,
        window_size: int = 72,  # 3 days (hours)
        momentum_lookback: int = 12,  # 12 hours
        min_sharpe: float = 0.05,
        momentum_threshold: float = 0.01,  # Minimum momentum for entry
        stop_loss_pct: float = 0.015,  # 1.5% stop loss (tighter)
        take_profit_pct: float = 0.10  # 10% take profit (larger for 10-15% target)
    ):
        """Initialize optimized Sharpe signal generator."""
        self.window_size = window_size
        self.momentum_lookback = momentum_lookback
        self.min_sharpe = min_sharpe
        self.momentum_threshold = momentum_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.config = get_config()
    
    def compute_log_returns(self, prices: pd.Series) -> pd.Series:
        """Compute log returns."""
        return np.log(prices / prices.shift(1))
    
    def compute_sharpe_ratio(self, prices: pd.Series, window: int = 24) -> float:
        """Compute rolling Sharpe ratio."""
        if len(prices) < window + 1:
            return 0.0
        
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return 0.0
        
        rolling_mean = returns.tail(window).mean()
        rolling_std = returns.tail(window).std()
        
        if rolling_std > 0:
            return rolling_mean / rolling_std
        return 0.0
    
    def compute_momentum(self, prices: pd.Series, lookback: int = 12) -> float:
        """Compute momentum over lookback period."""
        if len(prices) < lookback + 1:
            return 0.0
        
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-(lookback + 1)]
        
        if past_price > 0:
            return (current_price - past_price) / past_price
        return 0.0
    
    def check_entry_conditions(
        self,
        prices: pd.Series,
        current_price: float
    ) -> Tuple[bool, float, str]:
        """Check if entry conditions are met.
        
        Returns:
            Tuple of (is_valid, quality_score, reason)
        """
        if len(prices) < self.window_size:
            return False, 0.0, "Insufficient data"
        
        # CONDITION 1: Price must be above moving averages (uptrend)
        ma12 = prices.rolling(window=12).mean().iloc[-1] if len(prices) >= 12 else current_price
        ma24 = prices.rolling(window=24).mean().iloc[-1] if len(prices) >= 24 else current_price
        ma72 = prices.rolling(window=72).mean().iloc[-1] if len(prices) >= 72 else current_price
        
        # RELAXED: Allow if price is above at least one MA (not all three)
        if not (current_price > ma12 or current_price > ma24):
            return False, 0.0, "Not in uptrend"
        
        # CONDITION 2: Positive momentum (VERY LOW for maximum opportunities)
        momentum = self.compute_momentum(prices, self.momentum_lookback)
        if momentum < self.momentum_threshold * 0.3:  # 70% lower threshold for maximum opportunities
            return False, 0.0, f"Insufficient momentum: {momentum*100:.2f}%"
        
        # CONDITION 3: Positive Sharpe ratio (VERY LOW for maximum opportunities)
        sharpe = self.compute_sharpe_ratio(prices, window=24)
        if sharpe < self.min_sharpe * 0.4:  # 60% lower threshold for maximum opportunities
            return False, 0.0, f"Low Sharpe: {sharpe:.3f}"
        
        # CONDITION 4: Not at recent high (MORE RELAXED - allow entries closer to high)
        if len(prices) > 24:
            recent_high = prices.tail(24).max()
            if current_price > recent_high * 0.995:  # Within 0.5% of recent high (was 1%)
                return False, 0.0, "Too close to recent high"
        
        # CONDITION 5: Low volatility (MORE RELAXED - allow more volatility)
        if len(prices) > 24:
            recent_std = prices.tail(24).std() / prices.tail(24).mean()
            if recent_std > 0.06:  # Volatility > 6% = too choppy (was 5%)
                return False, 0.0, f"Too volatile: {recent_std*100:.2f}%"
        
        # Calculate quality score (LOWERED for maximum trades)
        quality = 0.42  # Even lower base quality for more opportunities
        quality += 0.2 if momentum > 0.012 else 0.1  # Lower momentum threshold
        quality += 0.15 if sharpe > 0.06 else 0.1  # Lower Sharpe threshold
        quality += 0.1 if current_price > ma72 * 1.015 else 0.05  # Lower MA threshold
        quality += 0.05 if len(prices) > 24 and current_price < prices.tail(24).max() * 0.98 else 0.0  # More relaxed pullback
        
        return True, quality, f"Valid entry: momentum {momentum*100:.1f}%, Sharpe {sharpe:.3f}"
    
    def compute_signal(
        self,
        pair: str,
        datastore: DataStore,
        ticker_data: Dict[str, any]
    ) -> Optional[OptimizedSharpeSignal]:
        """Compute optimized Sharpe signal."""
        # Read historical data
        df = datastore.read_minute_bars(pair, limit=self.window_size + 50)
        
        if len(df) < self.window_size:
            return None
        
        prices = df["price"]
        current_price = ticker_data["price"]
        
        # Check entry conditions
        is_valid, quality, reason = self.check_entry_conditions(prices, current_price)
        
        if not is_valid:
            return None
        
        # Quality threshold (VERY LOW for maximum trades)
        if quality < 0.42:  # Very low threshold for maximum opportunities
            return None
        
        # Calculate Sharpe and momentum
        sharpe = self.compute_sharpe_ratio(prices, window=24)
        momentum = self.compute_momentum(prices, self.momentum_lookback)
        
        # Calculate stop and target (custom percentages)
        # Use recent low as stop reference
        if len(prices) > 24:
            recent_low = prices.tail(24).min()
            stop_loss = recent_low * (1 - self.stop_loss_pct)
        else:
            stop_loss = current_price * (1 - self.stop_loss_pct)
        
        # Take profit target (AGGRESSIVE for 10-15% target returns)
        # Scale take profit by quality: higher quality = MUCH larger target
        quality_multiplier = 1.0 + (quality - 0.45) * 0.5  # 1.0x to 1.3x based on quality
        take_profit = current_price * (1 + self.take_profit_pct * quality_multiplier)
        
        # Ensure excellent risk/reward (at least 4:1 for maximum returns)
        risk = current_price - stop_loss
        reward = take_profit - current_price
        if risk > 0 and reward / risk < 4.0:
            # Adjust take profit to maintain 4:1 R:R minimum (aggressive)
            take_profit = current_price + (risk * 4.0)
        
        return OptimizedSharpeSignal(
            pair=pair,
            signal="buy",
            current_price=current_price,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sharpe_ratio=sharpe,
            momentum=momentum,
            quality=quality,
            reason=reason
        )
    
    def rank_signals(self, signals: Dict[str, OptimizedSharpeSignal]) -> List[Tuple[str, OptimizedSharpeSignal]]:
        """Rank signals by quality (Sharpe * quality)."""
        actionable = [
            (pair, sig) for pair, sig in signals.items()
            if sig is not None and sig.signal != "neutral"
        ]
        
        # Sort by combined score (Sharpe * quality)
        actionable.sort(key=lambda x: x[1].sharpe_ratio * x[1].quality, reverse=True)
        
        return actionable

