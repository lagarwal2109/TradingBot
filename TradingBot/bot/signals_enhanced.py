"""Enhanced signal generation with volume analysis and breakout detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from bot.config import get_config
from bot.datastore import DataStore


@dataclass
class TrendInfo:
    """Trend information for multi-timeframe analysis."""
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1, higher means stronger trend
    ma_slope: float  # Moving average slope
    higher_highs: bool
    lower_lows: bool
    

@dataclass
class VolumeProfile:
    """Volume analysis results."""
    current_volume: float
    avg_volume: float
    volume_ratio: float  # current/average
    volume_trend: str  # "increasing", "decreasing", "stable"
    is_climax: bool  # Extreme volume spike
    is_divergence: bool  # Price/volume divergence
    

@dataclass
class BreakoutSignal:
    """Breakout detection results."""
    level: float  # Price level broken
    level_type: str  # "resistance", "support", "none"
    strength: float  # How significant is this level
    volume_confirmed: bool
    false_breakout_risk: float  # 0-1, higher means more likely fake


class EnhancedSignalGenerator:
    """Advanced signal generator with volume, breakout, and multi-timeframe analysis."""
    
    def __init__(
        self, 
        trend_window_long: int = 1440,  # 24 hours for daily trend
        trend_window_short: int = 240,  # 4 hours for short trend
        entry_window: int = 60,  # 1 hour for entry timing
        volume_window: int = 480,  # 8 hours for volume average
        support_resistance_days: int = 7,  # Look back 7 days for levels
        breakout_threshold: float = 0.02,  # 2% beyond level for breakout
        volume_surge_multiplier: float = 2.0  # 2x average volume for confirmation
    ):
        """Initialize enhanced signal generator."""
        self.trend_window_long = trend_window_long
        self.trend_window_short = trend_window_short
        self.entry_window = entry_window
        self.volume_window = volume_window
        self.support_resistance_days = support_resistance_days
        self.breakout_threshold = breakout_threshold
        self.volume_surge_multiplier = volume_surge_multiplier
        
    def analyze_trend(self, prices: pd.Series, window: int) -> TrendInfo:
        """Analyze trend over specified window.
        
        Args:
            prices: Price series
            window: Window size in minutes
            
        Returns:
            TrendInfo object with trend details
        """
        if len(prices) < window:
            return TrendInfo("neutral", 0, 0, False, False)
        
        # Get window data
        window_prices = prices.tail(window)
        
        # Calculate moving average and slope
        ma = window_prices.rolling(window=min(50, window//2)).mean()
        if len(ma) > 1:
            ma_slope = (ma.iloc[-1] - ma.iloc[-20]) / ma.iloc[-20] if len(ma) > 20 else 0
        else:
            ma_slope = 0
            
        # Identify higher highs and lower lows
        higher_highs = False
        lower_lows = False
        
        # Only check patterns if we have enough data
        if len(window_prices) > 40:
            highs = window_prices.rolling(window=min(20, window//4)).max()
            lows = window_prices.rolling(window=min(20, window//4)).min()
            
            recent_highs = highs.tail(min(20, len(highs)//2))
            older_highs = highs.iloc[-min(40, len(highs)):-min(20, len(highs)//2)] if len(highs) > min(20, len(highs)//2) else pd.Series()
            if len(older_highs) > 0:
                higher_highs = recent_highs.max() > older_highs.max()
            
            recent_lows = lows.tail(min(20, len(lows)//2))
            older_lows = lows.iloc[-min(40, len(lows)):-min(20, len(lows)//2)] if len(lows) > min(20, len(lows)//2) else pd.Series()
            if len(older_lows) > 0:
                lower_lows = recent_lows.min() < older_lows.min()
        
        # Determine trend direction and strength
        current_price = prices.iloc[-1]
        avg_price = window_prices.mean()
        
        # Relaxed trend detection - use slope as primary indicator
        # Higher highs/lower lows are nice-to-have but not required
        if ma_slope > 0.005:  # Reduced from 0.01 to 0.5% slope
            direction = "bullish"
            # Base strength on slope, boost if we have higher highs
            strength = min(abs(ma_slope) * 20, 1.0)  # Increased multiplier
            if higher_highs and not lower_lows:
                strength = min(strength * 1.3, 1.0)  # Bonus for pattern confirmation
        elif ma_slope < -0.005:  # Reduced from -0.01
            direction = "bearish" 
            strength = min(abs(ma_slope) * 20, 1.0)
            if lower_lows and not higher_highs:
                strength = min(strength * 1.3, 1.0)
        else:
            direction = "neutral"
            strength = 0.2
            
        # Adjust strength based on price position
        if direction == "bullish" and current_price > avg_price * 1.01:  # Reduced from 1.02
            strength = min(strength * 1.2, 1.0)
        elif direction == "bearish" and current_price < avg_price * 0.99:  # Reduced from 0.98
            strength = min(strength * 1.2, 1.0)
            
        return TrendInfo(direction, strength, ma_slope, higher_highs, lower_lows)
    
    def analyze_volume(self, volumes: pd.Series, prices: pd.Series) -> VolumeProfile:
        """Analyze volume patterns.
        
        Args:
            volumes: Volume series  
            prices: Price series (for divergence detection)
            
        Returns:
            VolumeProfile with volume analysis
        """
        if len(volumes) < self.volume_window:
            return VolumeProfile(0, 0, 1.0, "stable", False, False)
            
        # Current and average volume
        current_volume = volumes.iloc[-1] if len(volumes) > 0 else 0
        avg_volume = volumes.tail(self.volume_window).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume trend
        recent_avg = volumes.tail(60).mean()  # Last hour
        older_avg = volumes.iloc[-120:-60].mean() if len(volumes) > 120 else recent_avg
        
        if recent_avg > older_avg * 1.2:
            volume_trend = "increasing"
        elif recent_avg < older_avg * 0.8:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"
            
        # Climax detection - extreme volume spike
        is_climax = volume_ratio > 3.0  # 3x average is climactic
        
        # Divergence detection - price up but volume down or vice versa
        is_divergence = False
        if len(prices) > 20 and len(volumes) > 20:
            price_change = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]
            vol_change = (volumes.tail(5).mean() - volumes.iloc[-20:-15].mean()) / volumes.iloc[-20:-15].mean()
            
            # Divergence if price and volume move opposite directions significantly
            if (price_change > 0.02 and vol_change < -0.2) or (price_change < -0.02 and vol_change < -0.2):
                is_divergence = True
                
        return VolumeProfile(
            current_volume, avg_volume, volume_ratio, 
            volume_trend, is_climax, is_divergence
        )
    
    def find_support_resistance_levels(self, prices: pd.Series, days: int = 7) -> Dict[str, List[float]]:
        """Find key support and resistance levels from historical data.
        
        Args:
            prices: Price series
            days: Number of days to look back
            
        Returns:
            Dictionary with support and resistance levels
        """
        if len(prices) < days * 1440:  # Not enough data
            return {"support": [], "resistance": []}
            
        # Get data for specified days
        lookback_data = prices.tail(days * 1440)
        
        # Find local highs and lows using rolling windows
        window = 60  # 1-hour window
        highs = lookback_data.rolling(window=window, center=True).max()
        lows = lookback_data.rolling(window=window, center=True).min()
        
        # Identify peaks and troughs
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(lookback_data) - window):
            # Peak detection
            if lookback_data.iloc[i] == highs.iloc[i]:
                level = lookback_data.iloc[i]
                # Check if this level was tested multiple times
                touches = ((lookback_data > level * 0.99) & (lookback_data < level * 1.01)).sum()
                if touches >= 2:  # At least 2 touches
                    resistance_levels.append(level)
                    
            # Trough detection  
            if lookback_data.iloc[i] == lows.iloc[i]:
                level = lookback_data.iloc[i]
                touches = ((lookback_data > level * 0.99) & (lookback_data < level * 1.01)).sum()
                if touches >= 2:
                    support_levels.append(level)
        
        # Cluster nearby levels
        def cluster_levels(levels: List[float], threshold: float = 0.01) -> List[float]:
            if not levels:
                return []
            
            sorted_levels = sorted(set(levels))
            clustered = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if level <= current_cluster[-1] * (1 + threshold):
                    current_cluster.append(level)
                else:
                    # Add average of cluster
                    clustered.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
                    
            clustered.append(sum(current_cluster) / len(current_cluster))
            return clustered
        
        return {
            "support": cluster_levels(support_levels),
            "resistance": cluster_levels(resistance_levels)
        }
    
    def detect_breakout(
        self, 
        current_price: float, 
        prices: pd.Series,
        volume_profile: VolumeProfile,
        levels: Dict[str, List[float]]
    ) -> BreakoutSignal:
        """Detect if a breakout is occurring.
        
        Args:
            current_price: Current price
            prices: Recent price series
            volume_profile: Volume analysis
            levels: Support/resistance levels
            
        Returns:
            BreakoutSignal with breakout details
        """
        if len(prices) < 10:
            return BreakoutSignal(0, "none", 0, False, 0)
            
        recent_prices = prices.tail(10)
        prev_price = recent_prices.iloc[-2] if len(recent_prices) > 1 else current_price
        
        # Check resistance breakout
        for resistance in levels.get("resistance", []):
            if prev_price <= resistance and current_price > resistance * (1 + self.breakout_threshold):
                # Breakout detected
                volume_confirmed = volume_profile.volume_ratio >= self.volume_surge_multiplier
                
                # Calculate level strength based on number of previous touches
                touches = ((prices > resistance * 0.99) & (prices < resistance * 1.01)).sum()
                strength = min(touches / 10, 1.0)  # More touches = stronger level
                
                # False breakout risk
                false_risk = 0.3  # Base risk
                if not volume_confirmed:
                    false_risk += 0.3
                if volume_profile.is_divergence:
                    false_risk += 0.2
                if strength < 0.3:  # Weak level
                    false_risk -= 0.1
                    
                false_risk = min(max(false_risk, 0), 1)
                
                return BreakoutSignal(
                    resistance, "resistance", strength, 
                    volume_confirmed, false_risk
                )
        
        # Check support breakdown
        for support in levels.get("support", []):
            if prev_price >= support and current_price < support * (1 - self.breakout_threshold):
                volume_confirmed = volume_profile.volume_ratio >= self.volume_surge_multiplier
                
                touches = ((prices > support * 0.99) & (prices < support * 1.01)).sum()
                strength = min(touches / 10, 1.0)
                
                false_risk = 0.3
                if not volume_confirmed:
                    false_risk += 0.3
                if volume_profile.is_divergence:
                    false_risk += 0.2
                if strength < 0.3:
                    false_risk -= 0.1
                    
                false_risk = min(max(false_risk, 0), 1)
                
                return BreakoutSignal(
                    support, "support", strength,
                    volume_confirmed, false_risk
                )
                
        return BreakoutSignal(0, "none", 0, False, 0)
    
    def compute_trading_signal(
        self, 
        pair: str, 
        datastore: DataStore, 
        ticker_data: Dict[str, any]
    ) -> Dict[str, any]:
        """Compute comprehensive trading signal for a pair.
        
        Args:
            pair: Trading pair
            datastore: DataStore instance
            ticker_data: Current ticker data
            
        Returns:
            Dictionary with signal details
        """
        # Read historical data - use available data (reduced limit for available data)
        # Try to get as much as possible, but don't require full support_resistance_days
        max_limit = min(self.support_resistance_days * 1440, 10000)  # Cap at 10k to avoid memory issues
        df = datastore.read_minute_bars(pair, limit=max_limit)
        
        signal = {
            "pair": pair,
            "current_price": ticker_data["price"],
            "signal": "neutral",  # "buy", "sell", "neutral"
            "strength": 0.0,  # 0-1, signal strength
            "reason": "",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "volume_confirmed": False,
            "trend_aligned": False,
            "entry_quality": 0.0  # 0-1, overall quality score
        }
        
        # Need sufficient data - use minimum of trend_window_long or trend_window_short
        min_required = min(self.trend_window_long, self.trend_window_short + 20)  # At least short window + buffer
        if len(df) < min_required:
            signal["reason"] = f"Insufficient data: {len(df)} rows, need {min_required}"
            return signal
        
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"{pair}: {len(df)} rows available, analyzing trends...")
            
        prices = df["price"]
        
        # Multi-timeframe trend analysis
        long_trend = self.analyze_trend(prices, self.trend_window_long)
        short_trend = self.analyze_trend(prices, self.trend_window_short)
        entry_trend = self.analyze_trend(prices, self.entry_window)
        
        # Volume analysis (need volume data in ticker)
        if "volume" in df.columns and len(df["volume"]) > 0:
            volumes = df["volume"]
        else:
            # Use 24h volume as proxy, but scale it down to minute-level estimate
            # Assume volume is distributed evenly across 24h = 1440 minutes
            minute_volume_estimate = ticker_data.get("volume_24h", 0) / 1440 if ticker_data.get("volume_24h", 0) > 0 else 1.0
            volumes = pd.Series([minute_volume_estimate] * len(prices))
        volume_profile = self.analyze_volume(volumes, prices)
        
        # Support/resistance levels
        levels = self.find_support_resistance_levels(prices, self.support_resistance_days)
        
        # Breakout detection
        breakout = self.detect_breakout(
            ticker_data["price"], 
            prices, 
            volume_profile, 
            levels
        )
        
        # Debug: log trend analysis
        import logging
        logger = logging.getLogger(__name__)
        
        # Log detailed analysis for signals or sample of pairs
        # Show all signals, and sample 10% of neutral pairs for debugging
        import random
        should_log = (
            signal["signal"] != "neutral" or 
            pair in ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "ADAUSD", "XRPUSD"] or
            random.random() < 0.1  # 10% random sample
        )
        
        if should_log:
            # Always use INFO level so we can see what's happening
            logger.info(
                f"{pair}: long={long_trend.direction}({long_trend.strength:.2f}), "
                f"short={short_trend.direction}({short_trend.strength:.2f}), "
                f"entry={entry_trend.direction}({entry_trend.strength:.2f}), "
                f"breakout={breakout.level_type}, volume_ratio={volume_profile.volume_ratio:.2f}, "
                f"signal={signal['signal']}, quality={signal.get('entry_quality', 0):.2f}"
            )
        
        # Generate trading signal based on strategy rules
        
        # 1. Trend-following breakout BUY conditions (relaxed)
        if (long_trend.direction == "bullish" and 
            breakout.level_type == "resistance" and
            (breakout.volume_confirmed or volume_profile.volume_ratio > 1.2) and  # Relaxed volume requirement
            breakout.false_breakout_risk < 0.6):  # Relaxed false breakout threshold
            
            signal["signal"] = "buy"
            signal["strength"] = (
                long_trend.strength * 0.3 +
                short_trend.strength * 0.2 + 
                breakout.strength * 0.3 +
                (1 - breakout.false_breakout_risk) * 0.2
            )
            signal["reason"] = f"Bullish breakout above {breakout.level:.2f} with volume confirmation"
            signal["stop_loss"] = breakout.level * 0.98  # 2% below breakout level
            signal["take_profit"] = ticker_data["price"] * 1.05  # 5% target
            signal["volume_confirmed"] = True
            signal["trend_aligned"] = True
            
        # 2. Momentum continuation BUY (relaxed conditions)
        elif (long_trend.direction == "bullish" and
              (short_trend.direction == "bullish" or short_trend.strength > 0.3) and  # Reduced from 0.5
              entry_trend.strength > 0.3 and  # Reduced from 0.5
              (volume_profile.volume_trend == "increasing" or volume_profile.volume_ratio > 1.0) and  # Reduced from 1.1
              not volume_profile.is_divergence):
            
            signal["signal"] = "buy"
            signal["strength"] = (
                long_trend.strength * 0.4 +
                short_trend.strength * 0.3 + 
                entry_trend.strength * 0.3
            )
            signal["reason"] = "Strong bullish momentum with increasing volume"
            
            # Find nearest support for stop loss
            current_price = ticker_data["price"]
            support_below = [s for s in levels["support"] if s < current_price]
            if support_below:
                signal["stop_loss"] = max(support_below) * 0.98
            else:
                signal["stop_loss"] = current_price * 0.97  # 3% stop
                
            signal["take_profit"] = current_price * 1.05
            signal["volume_confirmed"] = volume_profile.volume_ratio > 1.2  # Reduced from 1.5
            signal["trend_aligned"] = True
            
        # 3. Simple momentum BUY (new - less strict)
        elif (long_trend.direction == "bullish" and
              long_trend.strength > 0.15 and  # Very low threshold - just need some bullishness
              (short_trend.direction != "bearish" or short_trend.strength > 0.15) and  # Not strongly bearish
              entry_trend.direction != "bearish" and  # Not declining
              not volume_profile.is_divergence):
            
            signal["signal"] = "buy"
            signal["strength"] = (
                long_trend.strength * 0.5 +
                short_trend.strength * 0.3 + 
                entry_trend.strength * 0.2
            )
            signal["reason"] = "Bullish momentum continuation"
            
            current_price = ticker_data["price"]
            support_below = [s for s in levels["support"] if s < current_price]
            if support_below:
                signal["stop_loss"] = max(support_below) * 0.98
            else:
                signal["stop_loss"] = current_price * 0.96  # 4% stop
                
            signal["take_profit"] = current_price * 1.04  # 4% target
            signal["volume_confirmed"] = volume_profile.volume_ratio > 1.0
            signal["trend_aligned"] = True
            
        # 4. Very simple trend-following BUY (most lenient)
        elif (long_trend.direction == "bullish" and
              long_trend.strength > 0.1 and  # Just need any bullish trend
              entry_trend.direction != "bearish" and  # Not declining right now
              not volume_profile.is_divergence):
            
            signal["signal"] = "buy"
            signal["strength"] = max(long_trend.strength, 0.3)  # Minimum 0.3 strength
            signal["reason"] = "Simple bullish trend following"
            
            current_price = ticker_data["price"]
            signal["stop_loss"] = current_price * 0.95  # 5% stop
            signal["take_profit"] = current_price * 1.03  # 3% target
            signal["volume_confirmed"] = volume_profile.volume_ratio > 0.8
            signal["trend_aligned"] = True
            
        # 5. Trend reversal SELL conditions (support breakdown)
        elif (breakout.level_type == "support" and
              (breakout.volume_confirmed or volume_profile.volume_ratio > 1.2) and  # Relaxed
              breakout.false_breakout_risk < 0.6):  # Relaxed
            
            signal["signal"] = "sell"
            signal["strength"] = (
                breakout.strength * 0.5 +
                (1 - breakout.false_breakout_risk) * 0.5
            )
            signal["reason"] = f"Support breakdown below {breakout.level:.2f} with volume"
            signal["stop_loss"] = breakout.level * 1.02  # 2% above broken support
            signal["take_profit"] = ticker_data["price"] * 0.95  # 5% target
            signal["volume_confirmed"] = True
            signal["trend_aligned"] = long_trend.direction != "bullish"
            
        # 6. Climax/exhaustion SELL
        elif (volume_profile.is_climax and 
              long_trend.direction == "bullish" and
              ticker_data["price"] > prices.tail(min(100, len(prices))).mean() * 1.08):  # Reduced from 1.1
            
            signal["signal"] = "sell"
            signal["strength"] = 0.6
            signal["reason"] = "Potential climax top with extreme volume"
            signal["stop_loss"] = ticker_data["price"] * 1.03
            signal["take_profit"] = ticker_data["price"] * 0.97
            signal["volume_confirmed"] = True
            signal["trend_aligned"] = False  # Counter-trend
            
        # 7. Bearish momentum SELL (new - less strict)
        elif (long_trend.direction == "bearish" and
              long_trend.strength > 0.4 and
              short_trend.strength > 0.3 and
              entry_trend.direction != "bullish" and
              not volume_profile.is_divergence):
            
            signal["signal"] = "sell"
            signal["strength"] = (
                long_trend.strength * 0.5 +
                short_trend.strength * 0.3 + 
                entry_trend.strength * 0.2
            )
            signal["reason"] = "Bearish momentum continuation"
            
            current_price = ticker_data["price"]
            resistance_above = [r for r in levels["resistance"] if r > current_price]
            if resistance_above:
                signal["stop_loss"] = min(resistance_above) * 1.02
            else:
                signal["stop_loss"] = current_price * 1.04  # 4% stop
                
            signal["take_profit"] = current_price * 0.96  # 4% target
            signal["volume_confirmed"] = volume_profile.volume_ratio > 1.0
            signal["trend_aligned"] = True
            
        # Calculate entry quality
        if signal["signal"] != "neutral":
            quality_factors = []
            
            # Trend alignment across timeframes
            if (long_trend.direction == short_trend.direction == entry_trend.direction):
                quality_factors.append(0.9)
            elif (long_trend.direction == short_trend.direction):
                quality_factors.append(0.7)
            elif long_trend.direction == entry_trend.direction:
                quality_factors.append(0.6)  # New: long and entry aligned
            else:
                quality_factors.append(0.5)  # Increased from 0.4
                
            # Volume confirmation - more lenient
            if signal["volume_confirmed"]:
                quality_factors.append(0.8)
            elif volume_profile.volume_ratio > 0.8:  # Not too low
                quality_factors.append(0.5)  # Partial credit
            else:
                quality_factors.append(0.4)  # Increased from 0.3
                
            # Breakout quality (optional - not required for all signals)
            if breakout.level_type != "none":
                quality_factors.append(1 - breakout.false_breakout_risk)
            else:
                # For momentum signals without breakouts, use trend strength
                quality_factors.append(0.5 + (long_trend.strength + short_trend.strength) / 4)
                
            # No divergence
            if not volume_profile.is_divergence:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.4)  # Increased from 0.2
            
            signal["entry_quality"] = sum(quality_factors) / len(quality_factors)
        else:
            # Log why signal is neutral for debugging (sample a few pairs)
            import logging
            logger = logging.getLogger(__name__)
            if pair in ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"]:
                logger.debug(
                    f"{pair} NEUTRAL: long={long_trend.direction}({long_trend.strength:.2f}), "
                    f"short={short_trend.direction}({short_trend.strength:.2f}), "
                    f"entry={entry_trend.direction}({entry_trend.strength:.2f}), "
                    f"breakout={breakout.level_type}, vol_ratio={volume_profile.volume_ratio:.2f}, "
                    f"divergence={volume_profile.is_divergence}"
                )
            
        return signal
    
    def rank_trading_opportunities(
        self, 
        signals: Dict[str, Dict[str, any]],
        prioritize_sells: bool = False
    ) -> List[Tuple[str, Dict[str, any]]]:
        """Rank trading opportunities by quality and strength.
        
        Args:
            signals: Dictionary of pair -> signal
            prioritize_sells: If True, prioritize sell signals over buys
            
        Returns:
            Sorted list of (pair, signal) tuples, best first
        """
        # Filter for actionable signals
        # Lower quality threshold to 0.2 to be even less strict
        actionable = [
            (pair, sig) for pair, sig in signals.items()
            if sig["signal"] != "neutral" and sig.get("entry_quality", 0) > 0.2
        ]
        
        # Sort by combined score
        def score_signal(pair_signal):
            pair, signal = pair_signal
            # Combine strength and quality, prefer trend-aligned trades
            score = signal["strength"] * 0.6 + signal["entry_quality"] * 0.4
            if signal["trend_aligned"]:
                score *= 1.2
            # Prioritize sells if requested (for risk management)
            if prioritize_sells and signal["signal"] == "sell":
                score *= 1.5  # Boost sell signals
            return score
            
        actionable.sort(key=score_signal, reverse=True)
        
        return actionable
