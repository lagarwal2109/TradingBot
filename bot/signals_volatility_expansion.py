"""Volatility Expansion + Mean Reversion signal generator with trend confirmation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from bot.config import get_config
from bot.datastore import DataStore
from bot.indicators import (
    calculate_rsi, calculate_macd, detect_rsi_divergence,
    calculate_bollinger_bands, detect_bollinger_squeeze,
    calculate_atr, detect_volume_spike, calculate_choppiness_index,
    calculate_adx, calculate_ema, detect_bullish_engulfing, detect_hammer_pattern
)


@dataclass
class VolatilityExpansionSignal:
    """Volatility expansion trading signal."""
    pair: str
    signal: str  # "buy", "sell", "neutral"
    strategy_type: str  # "breakout" or "mean_reversion"
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    quality: float  # 0-1
    confidence: float  # 0-1
    reason: str
    regime: str  # "trending" or "ranging"
    batch_entries: List[float]  # 4 batch entry prices with 10% spacing
    atr_value: float
    rsi_value: float
    macd_signal: str  # "bullish", "bearish", "neutral"


class VolatilityExpansionSignalGenerator:
    """Generate signals using volatility expansion breakout or mean reversion with divergence."""
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 80.0,  # Crypto-optimized: 80/20 instead of 70/30
        rsi_oversold: float = 20.0,
        macd_fast: int = 3,  # Crypto-optimized: 3-10-16
        macd_slow: int = 10,
        macd_signal: int = 16,
        bb_period: int = 20,
        bb_std_dev: float = 1.5,  # Crypto-optimized: 1.5-2.0
        atr_period: int = 14,
        volume_ma_period: int = 20,
        volume_spike_threshold: float = 1.5,  # 150% of average
        squeeze_threshold: float = 0.02,  # 2% band width
        batch_count: int = 4,
        batch_spacing_pct: float = 0.10,  # 10% price interval
        stop_loss_atr_multiplier: float = 2.0  # Balanced stops (2.0x ATR)
    ):
        """Initialize volatility expansion signal generator.
        
        Args:
            rsi_period: RSI calculation period
            rsi_overbought: RSI overbought threshold (crypto-optimized: 80)
            rsi_oversold: RSI oversold threshold (crypto-optimized: 20)
            macd_fast: MACD fast EMA period (crypto-optimized: 3)
            macd_slow: MACD slow EMA period (crypto-optimized: 10)
            macd_signal: MACD signal line period (crypto-optimized: 16)
            bb_period: Bollinger Band period
            bb_std_dev: Bollinger Band standard deviation multiplier
            atr_period: ATR period
            volume_ma_period: Volume moving average period
            volume_spike_threshold: Volume spike multiplier (1.5 = 150%)
            squeeze_threshold: Bollinger squeeze threshold (band width %)
            batch_count: Number of batch entries (default 4)
            batch_spacing_pct: Price spacing between batches (default 10%)
            stop_loss_atr_multiplier: ATR multiplier for stop loss (default 2.5 - wider to reduce false stops)
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.atr_period = atr_period
        self.volume_ma_period = volume_ma_period
        self.volume_spike_threshold = volume_spike_threshold
        self.squeeze_threshold = squeeze_threshold
        self.batch_count = batch_count
        self.batch_spacing_pct = batch_spacing_pct
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.config = get_config()
    
        # Initialize diagnostic counters
        self._diagnostic_counter = {'total_checked': 0, 'rsi_oversold': 0, 'rsi_25': 0, 'price_near_lower': 0, 'both_met': 0}
        self._early_return_counter = {'insufficient_data': 0, 'insufficient_4h': 0, 'reached_mean_rev': 0}
    
    def aggregate_to_timeframe(
        self,
        df: pd.DataFrame,
        timeframe_minutes: int
    ) -> pd.DataFrame:
        """Aggregate minute bars to higher timeframe.
        
        Args:
            df: DataFrame with minute bars (timestamp index, price, volume columns)
            timeframe_minutes: Target timeframe in minutes (60=1h, 240=4h, 1440=daily)
            
        Returns:
            Aggregated DataFrame with OHLCV data
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'price'])
        
        # Ensure timestamp is index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # More lenient: allow aggregation even if we don't have full timeframe
        # We'll get fewer bars, but that's okay for indicators
        try:
            # Resample to target timeframe (will create bars from available data)
            resampled = df.resample(f'{timeframe_minutes}min', label='right', closed='right').agg({
                'price': ['first', 'max', 'min', 'last'],
                'volume': 'sum'
            })
            
            # Flatten column names
            resampled.columns = ['open', 'high', 'low', 'close', 'volume']
            resampled['price'] = resampled['close']  # Use close as price
            
            # Drop any NaN rows
            resampled = resampled.dropna()
            
            return resampled
        except Exception as e:
            # Return empty DataFrame on error
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'price'])
    
    def detect_market_regime(
        self,
        daily_data: pd.DataFrame
    ) -> Tuple[str, str]:
        """Detect market regime using MACD and Bollinger Band width.
        
        Args:
            daily_data: Daily aggregated price data
            
        Returns:
            Tuple of (regime, macd_signal) where regime is "trending" or "ranging",
            macd_signal is "bullish", "bearish", or "neutral"
        """
        # More lenient: need at least some data for MACD
        if len(daily_data) < max(self.macd_slow, 5):  # Need at least 5 for basic MACD
            return "ranging", "neutral"
        
        prices = daily_data['price'] if 'price' in daily_data.columns else daily_data['close']
        
        # Calculate MACD on daily chart
        macd_line, signal_line, histogram = calculate_macd(
            prices, self.macd_fast, self.macd_slow, self.macd_signal
        )
        
        if len(macd_line) == 0:
            return "ranging", "neutral"
        
        # Determine MACD signal
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        if current_macd > current_signal:
            macd_signal = "bullish"
        elif current_macd < current_signal:
            macd_signal = "bearish"
        else:
            macd_signal = "neutral"
        
        # Check Bollinger Band width for consolidation/trending
        is_squeeze, band_width_pct = detect_bollinger_squeeze(
            prices, self.bb_period, self.bb_std_dev, self.squeeze_threshold
        )
        
        # Determine regime
        # If MACD is bullish AND bands are widening = trending (use breakout)
        # If bands are narrow (squeeze) = ranging (use mean reversion)
        if is_squeeze or band_width_pct < (self.squeeze_threshold * 100):
            regime = "ranging"
        elif macd_signal == "bullish" and not is_squeeze:
            regime = "trending"
        elif macd_signal == "bearish" and not is_squeeze:
            regime = "trending"  # Can still use breakout for bearish trends
        else:
            regime = "ranging"
        
        return regime, macd_signal
    
    def generate_breakout_signal(
        self,
        pair: str,
        prices: pd.Series,
        volumes: pd.Series,
        current_price: float,
        regime: str,
        macd_signal: str,
        debug: bool = False
    ) -> Optional[VolatilityExpansionSignal]:
        """Generate volatility expansion breakout signal (Strategy A).
        
        Entry Rules:
        - Bollinger Band squeeze (low volatility)
        - Volume spike (>150% of 20-day average)
        - RSI crosses above 50 (bullish) or below 50 (bearish)
        - Price breaks upper/lower Bollinger Band
        
        Args:
            pair: Trading pair
            prices: Price series (4-hour or hourly)
            volumes: Volume series
            current_price: Current price
            regime: Market regime ("trending" or "ranging")
            macd_signal: MACD signal ("bullish", "bearish", "neutral")
            
        Returns:
            VolatilityExpansionSignal or None
        """
        if len(prices) < max(self.bb_period, self.volume_ma_period) + 1:
            if debug:
                print(f"      {pair}: Insufficient price data ({len(prices)} < {max(self.bb_period, self.volume_ma_period) + 1})")
            return None
        
        # Calculate indicators
        rsi = calculate_rsi(prices, self.rsi_period)
        if len(rsi) == 0:
            if debug:
                print(f"      {pair}: RSI calculation returned empty")
            return None
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) >= 2 else current_rsi
        
        # Check for RSI crossing 50
        rsi_crossed_up = prev_rsi <= 50 and current_rsi > 50
        rsi_crossed_down = prev_rsi >= 50 and current_rsi < 50
        
        if debug:
            print(f"      {pair}: RSI={current_rsi:.1f} (prev={prev_rsi:.1f}), crossed_up={rsi_crossed_up}, crossed_down={rsi_crossed_down}")
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            prices, self.bb_period, self.bb_std_dev
        )
        if len(upper_band) == 0:
            return None
        
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Check for Bollinger squeeze
        is_squeeze, band_width_pct = detect_bollinger_squeeze(
            prices, self.bb_period, self.bb_std_dev, self.squeeze_threshold
        )
        
        # Check for volume spike
        is_volume_spike, volume_ratio = detect_volume_spike(
            volumes, self.volume_ma_period, self.volume_spike_threshold
        )
        if pd.isna(volume_ratio) or volume_ratio <= 0:
            volume_ratio = 0.0
        
        if debug:
            print(f"      {pair}: BB squeeze={is_squeeze}, volume_ratio={volume_ratio:.2f} (spike={is_volume_spike})")
        
        # Long entry: Price breaks/near upper band + (volume surge OR RSI momentum) + (squeeze OR RSI favorable)
        # AGGRESSIVE for 14-day: Relaxed conditions for faster entries
        price_breakout = current_price > current_upper * 0.95  # Within 5% of upper BB (was 2%)
        rsi_favorable = rsi_crossed_up or (current_rsi > 45 and current_rsi > prev_rsi)  # RSI above 45 and rising (was 50)
        volume_condition = is_volume_spike or volume_ratio >= 1.1  # Volume spike or at least 110% of average (was 120%)
        
        if debug:
            print(f"      {pair} BREAKOUT LONG: price_breakout={price_breakout} (price={current_price:.4f} near upper={current_upper:.4f}), rsi_favorable={rsi_favorable}, volume_condition={volume_condition}, squeeze={is_squeeze}")
        
        # AGGRESSIVE: Price near upper BB + (volume OR RSI favorable) - removed squeeze requirement
        # If volume is good, don't need squeeze. If RSI is favorable, don't need volume spike.
        if (price_breakout and (volume_condition or rsi_favorable)):
            
            # Calculate ATR for stop loss
            high = prices  # Use prices as high/low proxy if not available
            low = prices
            close = prices
            atr = calculate_atr(high, low, close, self.atr_period)
            current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calculate stop loss (1.5x ATR below entry)
            stop_loss = current_price - (current_atr * self.stop_loss_atr_multiplier)
            
            # Batch entries for breakout: 2 batches (not 4, breakouts are riskier)
            batch_count = 2  # Cap at 2 batches for breakouts
            batch_entries = []
            for i in range(batch_count):
                if i == 0:
                    entry_price = current_price  # First batch at current price
                else:
                    entry_price = current_price * (1 + self.batch_spacing_pct * i)  # Buy on breakouts (above current)
                batch_entries.append(entry_price)
            
            # Take profit target (scaled: 33% at +5%, 33% at +10%, 34% at +15% - lowered targets)
            take_profit = current_price * 1.15  # 15% target (lowered from 25%)
            
            # Quality score for breakout signals
            quality = 0.60  # Base quality for breakouts
            quality += 0.20 if is_volume_spike else 0.10  # Volume spike bonus
            quality += 0.15 if rsi_crossed_up else 0.05  # RSI crossing up bonus
            quality += 0.10 if is_squeeze else 0.0  # Squeeze bonus
            quality += 0.05 if current_rsi > 55 else 0.0  # Strong RSI bonus
            quality += 0.05 if macd_signal == "bullish" else 0.0  # MACD bonus
            
            return VolatilityExpansionSignal(
                pair=pair,
                signal="buy",
                strategy_type="breakout",
                current_price=current_price,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quality=quality,
                confidence=min(quality * 1.2, 1.0),
                reason=f"Breakout: BB squeeze={is_squeeze}, volume={volume_ratio:.2f}x, RSI={current_rsi:.1f}",
                regime=regime,
                batch_entries=batch_entries,
                atr_value=current_atr,
                rsi_value=current_rsi,
                macd_signal=macd_signal
            )
        
        # Short entry: Price breaks/near lower band + (volume surge OR RSI momentum) + (squeeze OR RSI favorable)
        price_breakdown = current_price < current_lower * 1.02  # Within 2% of lower BB
        rsi_favorable_short = rsi_crossed_down or (current_rsi < 50 and current_rsi < prev_rsi)  # RSI below 50 and falling
        volume_condition_short = is_volume_spike or volume_ratio >= 1.2  # Volume spike or at least 120% of average
        
        if debug:
            print(f"      {pair} BREAKOUT SHORT: price_breakdown={price_breakdown} (price={current_price:.4f} near lower={current_lower:.4f}), rsi_favorable={rsi_favorable_short}, volume_condition={volume_condition_short}, squeeze={is_squeeze}")
        
        # Practical strategy: Price near lower BB + volume surge + (squeeze OR RSI favorable)
        if (price_breakdown and volume_condition_short and (is_squeeze or rsi_favorable_short)):
            
            # Calculate ATR for stop loss
            high = prices
            low = prices
            close = prices
            atr = calculate_atr(high, low, close, self.atr_period)
            current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
            
            # Calculate stop loss (1.5x ATR above entry for shorts)
            stop_loss = current_price + (current_atr * self.stop_loss_atr_multiplier)
            
            # Calculate batch entry prices
            batch_entries = []
            for i in range(self.batch_count):
                entry_price = current_price * (1 - self.batch_spacing_pct * i)
                batch_entries.append(entry_price)
            
            # Take profit target
            take_profit = current_price * 0.70  # 30% target for shorts
            
            # Calculate quality score
            quality = 0.5
            quality += 0.2 if is_squeeze else 0.1
            quality += 0.15 if volume_ratio >= 1.8 else 0.1
            quality += 0.1 if current_rsi < 45 else 0.05
            quality += 0.05 if macd_signal == "bearish" else 0.0
            
            return VolatilityExpansionSignal(
                pair=pair,
                signal="sell",
                strategy_type="breakout",
                current_price=current_price,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quality=quality,
                confidence=min(quality * 1.2, 1.0),
                reason=f"Breakout short: BB squeeze={is_squeeze}, volume={volume_ratio:.2f}x, RSI={current_rsi:.1f}",
                regime=regime,
                batch_entries=batch_entries,
                atr_value=current_atr,
                rsi_value=current_rsi,
                macd_signal=macd_signal
            )
        
        return None
    
    def generate_mean_reversion_signal(
        self,
        pair: str,
        prices: pd.Series,
        volumes: pd.Series,
        current_price: float,
        regime: str,
        macd_signal: str,
        daily_data: Optional[pd.DataFrame] = None,
        four_hour_data: Optional[pd.DataFrame] = None,
        hourly_data: Optional[pd.DataFrame] = None,
        debug: bool = False
    ) -> Optional[VolatilityExpansionSignal]:
        """Generate mean reversion signal with divergence (Strategy B).
        
        Entry Rules:
        - RSI >70 (short) or <30 (long)
        - Price touches/breaches Bollinger Band
        - RSI divergence confirmation
        - Enter after RSI crosses 50-line
        
        Args:
            pair: Trading pair
            prices: Price series (4-hour or hourly)
            volumes: Volume series
            current_price: Current price
            regime: Market regime
            macd_signal: MACD signal
            
        Returns:
            VolatilityExpansionSignal or None
        """
        if len(prices) < max(self.bb_period, self.rsi_period) + 1:
            if debug:
                print(f"      {pair}: Mean reversion - Insufficient price data ({len(prices)} < {max(self.bb_period, self.rsi_period) + 1})")
            return None
        
        # Calculate RSI
        rsi = calculate_rsi(prices, self.rsi_period)
        if len(rsi) == 0:
            if debug:
                print(f"      {pair}: Mean reversion - RSI calculation returned empty")
            return None
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) >= 2 else current_rsi
        
        # Check for RSI crossing 50 (entry confirmation)
        rsi_crossed_up = prev_rsi <= 50 and current_rsi > 50
        rsi_crossed_down = prev_rsi >= 50 and current_rsi < 50
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = calculate_bollinger_bands(
            prices, self.bb_period, self.bb_std_dev
        )
        if len(upper_band) == 0:
            if debug:
                print(f"      {pair}: Mean reversion - Bollinger Bands calculation returned empty")
            return None
        
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = middle_band.iloc[-1] if len(middle_band) > 0 else (current_upper + current_lower) / 2
        
        # Detect RSI divergence
        divergence = detect_rsi_divergence(prices, rsi, lookback=20)
        
        # Check volume - For mean reversion, we want volume >= 80% (not too low) but don't require spike
        # Mean reversion often happens on low volume (selling exhaustion)
        is_volume_spike, volume_ratio = detect_volume_spike(
            volumes, self.volume_ma_period, self.volume_spike_threshold
        )
        if pd.isna(volume_ratio) or volume_ratio <= 0:
            volume_ratio = 0.0
        # More lenient for mean reversion: require volume >= 60% (not too low), but prefer >= 120% (spike)
        # Mean reversion often happens on low volume (selling exhaustion), so 60% is reasonable
        has_volume_confirmation = volume_ratio >= 0.6  # >= 60% of average (was 80% - still too strict for mean reversion)
        has_volume_spike = volume_ratio >= self.volume_spike_threshold  # >= 120% (bonus, not required)
        
        # ADX Filter: Only mean revert in ranging markets (ADX < 20)
        # Calculate ADX on 4-hour data - CRITICAL: Use actual OHLC data
        if four_hour_data is not None and len(four_hour_data) > 0:
            if 'high' in four_hour_data.columns and 'low' in four_hour_data.columns:
                high = four_hour_data['high']
                low = four_hour_data['low']
                close = four_hour_data['close'] if 'close' in four_hour_data.columns else four_hour_data['price']
            else:
                # Fallback: use prices as proxy (less accurate)
                high = prices
                low = prices
                close = prices
        else:
            # Fallback: use prices as proxy (less accurate)
            high = prices
            low = prices
            close = prices
        
        adx = calculate_adx(high, low, close, period=14)
        current_adx = adx.iloc[-1] if len(adx) > 0 and not pd.isna(adx.iloc[-1]) else None
        # More lenient: Allow weak trends (ADX < 25) for mean reversion, not just strict ranging (< 20)
        # Default to allowing trade if ADX can't be calculated (unknown state)
        adx_ranging = (current_adx is None) or (current_adx < 25.0)  # < 25 = ranging or weak trend (was < 20)
        
        # MA Trend Filter: Only mean revert in uptrends (price > 50 EMA on daily)
        ma_trend_ok = True  # Default to True if no daily data
        if daily_data is not None and len(daily_data) >= self.config.ma_trend_filter_period:
            daily_prices = daily_data['price'] if 'price' in daily_data.columns else daily_data['close']
            if len(daily_prices) >= self.config.ma_trend_filter_period:
                ema_50 = calculate_ema(daily_prices, self.config.ma_trend_filter_period)
                if len(ema_50) > 0 and not pd.isna(ema_50.iloc[-1]):
                    ma_trend_ok = current_price > ema_50.iloc[-1]  # Price above 50 EMA = uptrend
        
        # Volatility Expansion Requirement: Current ATR > 0.8 * median ATR
        # CRITICAL: Use same high/low/close from ADX calculation above
        atr = calculate_atr(high, low, close, self.atr_period)
        current_atr = atr.iloc[-1] if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else current_price * 0.02
        if len(atr) >= 20 and current_atr > 0:
            median_atr = atr.tail(20).median()
            # More lenient: require ATR > 0.3 * median (was 0.5) to allow more trades
            # Very low volatility (ATR < 0.3 * median) might indicate dead market, but allow most cases
            volatility_ok = current_atr > (0.3 * median_atr) if median_atr > 0 else True
        else:
            volatility_ok = True  # Default to OK if not enough data or ATR invalid
        
        # Guard against invalid prices/bands
        if current_price is None or current_price <= 0 or pd.isna(current_lower) or current_lower <= 0:
            if debug:
                print(f"      {pair}: Invalid price/band - skipping (price={current_price}, lower_bb={current_lower})")
            return None
        
        # STEP 1: MINIMAL CONDITIONS with utilization-aware scout mode
        # Base thresholds
        rsi_threshold = 30.0
        lower_bb_tolerance = 1.05  # Within 5%
        
        # Utilization-aware scout mode: relax thresholds when under-utilized
        try:
            # Peek utilization from diagnostic counter if available (set by backtest via debug context)
            current_utilization = getattr(self, '_current_utilization', None)
        except Exception:
            current_utilization = None
        
        # If backtest provides utilization via setter, use it; otherwise stay with base thresholds
        if current_utilization is not None:
            if current_utilization < getattr(self.config, 'utilization_low_threshold', 0.5):
                # Scout mode: seek more entries when we are under-utilized
                rsi_threshold = 35.0
                lower_bb_tolerance = 1.07  # within 7%
            elif current_utilization > getattr(self.config, 'utilization_high_threshold', 0.7):
                # Tighten slightly when over 70% utilized
                rsi_threshold = 28.0
                lower_bb_tolerance = 1.04
        
        rsi_oversold = current_rsi < rsi_threshold
        price_near_lower = current_price <= current_lower * lower_bb_tolerance
        
        # Calculate other conditions for debugging (but don't require them yet)
        has_divergence = divergence == "bullish_divergence"
        rsi_recovering = current_rsi > prev_rsi  # Just rising, no threshold
        
        # Check for confirmation candle (for debugging, not required in Step 1)
        has_confirmation_candle = False
        if self.config.confirmation_candle_required:
            candle_data = None
            if four_hour_data is not None and len(four_hour_data) >= 2:
                if all(col in four_hour_data.columns for col in ['open', 'high', 'low', 'close']):
                    candle_data = four_hour_data
            elif hourly_data is not None and len(hourly_data) >= 2:
                if all(col in hourly_data.columns for col in ['open', 'high', 'low', 'close']):
                    candle_data = hourly_data
            
            if candle_data is not None:
                has_confirmation_candle = (detect_bullish_engulfing(candle_data) or 
                                         detect_hammer_pattern(candle_data))
        
        # Check for weak signal mode (for debugging, not required in Step 1)
        is_weak_signal_mode = False
        if self.config.weak_signal_mode_enabled:
            if len(atr) >= 20 and current_atr > 0:
                median_atr = atr.tail(20).median()
                if median_atr > 0:
                    is_weak_signal_mode = current_atr < (self.config.weak_signal_atr_threshold * median_atr)
        
        # DIAGNOSTIC: Always log when we have valid data but conditions fail (not just in debug mode)
        # This helps us understand why signals aren't being generated
        diagnostic_log = False
        if not hasattr(self, '_diagnostic_counter'):
            self._diagnostic_counter = {'total_checked': 0, 'rsi_oversold': 0, 'price_near_lower': 0, 'both_met': 0}
        
        self._diagnostic_counter['total_checked'] += 1
        if rsi_oversold:
            self._diagnostic_counter['rsi_oversold'] += 1
        if price_near_lower:
            self._diagnostic_counter['price_near_lower'] += 1
        # Also track RSI < 25 for comparison
        if not hasattr(self._diagnostic_counter, 'rsi_25'):
            self._diagnostic_counter['rsi_25'] = 0
        if current_rsi < 25:
            self._diagnostic_counter['rsi_25'] += 1
        
        if debug:
            adx_str = f"{current_adx:.1f}" if current_adx is not None else "N/A"
            print(f"      {pair} MEAN REVERSION LONG (STEP 1 - MINIMAL): rsi_oversold={rsi_oversold} (RSI={current_rsi:.1f} < {rsi_threshold:.0f}), price_near_lower={price_near_lower} (price={current_price:.4f} <= lower_bb*{lower_bb_tolerance:.2f}={current_lower*lower_bb_tolerance:.4f}), has_divergence={has_divergence}, rsi_recovering={rsi_recovering}, has_confirmation_candle={has_confirmation_candle}, volume_ok={has_volume_confirmation} (ratio={volume_ratio:.2f}), adx_ranging={adx_ranging} (ADX={adx_str}), ma_trend_ok={ma_trend_ok}, volatility_ok={volatility_ok}, weak_signal_mode={is_weak_signal_mode}")
        
        # STEP 1: MINIMAL ENTRY - Only RSI < 25 and price within 5% of lower BB
        # All other conditions are optional (for debugging/monitoring)
        should_enter = rsi_oversold and price_near_lower
        
        if should_enter:
            self._diagnostic_counter['both_met'] += 1
        
        # Track filter status for debugging (but don't require them)
        filters_passed = sum([adx_ranging, ma_trend_ok, volatility_ok])
        
        # DIAGNOSTIC: Log detailed failure reason when conditions are close but not met
        if not should_enter:
            # Log when RSI is close to threshold (20-30) or price is close to lower BB (within 10%)
            rsi_close = 20 <= current_rsi <= 30
            price_close = current_price <= current_lower * 1.10  # Within 10% of lower BB
            if (rsi_close or price_close) and self._diagnostic_counter['total_checked'] % 100 == 0:
                # Sample diagnostic output every 100 checks
                print(f"      DIAGNOSTIC {pair}: RSI={current_rsi:.1f} (oversold={rsi_oversold}), price={current_price:.4f}, lower_bb={current_lower:.4f} (near={price_near_lower}, distance={((current_price/current_lower - 1) * 100):.2f}%)")
            
            if debug:
                print(f"      {pair}: Entry conditions not met - skipping (RSI={current_rsi:.1f}, price={current_price:.4f}, lower_bb={current_lower:.4f})")
            return None
        
        # Generate signal if we passed the conditions above
        # ATR already calculated above in volatility check (current_atr)
        
        # Stop loss (2.0x ATR below entry)
        stop_loss = current_price - (current_atr * self.stop_loss_atr_multiplier)
        
        # Conditional batch sizing based on RSI level
        # STEP 1: Simplified - use 2 batches for all entries
        rsi_very_oversold = current_rsi < 20
        rsi_moderately_oversold = 20 <= current_rsi < 25  # Original threshold
        if rsi_very_oversold:
            batch_count = 4  # 4 batches for very oversold
        elif rsi_moderately_oversold:
            batch_count = 3  # 3 batches for moderately oversold (RSI 20-25)
        else:
            batch_count = 2  # 2 batches for RSI 25-30 (relaxed threshold)
        
        # Batch entries - For mean reversion: buy on dips (below current price)
        # Batch 1: current price (immediate entry)
        # Batch 2-4: progressively lower prices (buy more if price dips)
        batch_entries = []
        for i in range(batch_count):
            if i == 0:
                entry_price = current_price  # First batch at current price
            else:
                entry_price = current_price * (1 - self.batch_spacing_pct * i)  # Buy on dips
            batch_entries.append(entry_price)
        
        # Take profit
        take_profit = current_price * 1.25  # 25% target (achievable)
        
        # Quality score - STEP 1: Simplified scoring (will enhance in later steps)
        quality = 0.60  # Base quality for minimal conditions
        quality += 0.20 if current_rsi < 20 else (0.15 if current_rsi < 25 else 0.10)  # Very oversold bonus (RSI < 20), moderate (20-25), relaxed (25-30)
        quality += 0.10 if current_price <= current_lower else 0.05  # Price at lower BB bonus
        quality += 0.10 if has_divergence else 0.0  # Divergence bonus
        quality += 0.05 if rsi_recovering else 0.0  # RSI recovery bonus
        quality += 0.05 if has_volume_confirmation else 0.0  # Volume confirmation bonus
        quality += 0.05 if has_volume_spike else 0.0  # Volume spike bonus
        quality += 0.05 if adx_ranging and current_adx is not None else 0.0  # ADX ranging bonus
        quality += 0.05 if ma_trend_ok else 0.0  # MA trend bonus
        quality += 0.05 if has_confirmation_candle else 0.0  # Confirmation candle bonus
        
        return VolatilityExpansionSignal(
            pair=pair,
            signal="buy",
            strategy_type="mean_reversion",
            current_price=current_price,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quality=quality,
            confidence=min(quality * 1.1, 1.0),
            reason=f"Mean reversion (STEP 1 - RSI<30): RSI={current_rsi:.1f}, price_near_lower={price_near_lower}, filters={filters_passed}/3, candle={has_confirmation_candle}, volume={has_volume_confirmation}",
            regime=regime,
            batch_entries=batch_entries,
            atr_value=current_atr,
            rsi_value=current_rsi,
            macd_signal=macd_signal
        )
        
        # Short entry: RSI overbought + price near upper BB + (divergence OR RSI declining)
        # More practical: RSI > 65 (moderately overbought) + price near upper BB + RSI declining
        rsi_overbought = current_rsi > 65  # More practical threshold
        price_near_upper = current_price >= current_upper * 0.98  # Within 2% of upper BB
        has_divergence_short = divergence == "bearish_divergence"
        rsi_declining = rsi_crossed_down or (current_rsi < prev_rsi and current_rsi < 70)  # RSI declining
        
        if debug:
            print(f"      {pair} MEAN REVERSION SHORT: rsi_overbought={rsi_overbought} (RSI={current_rsi:.1f}), price_near_upper={price_near_upper}, has_divergence={has_divergence_short}, rsi_declining={rsi_declining}")
        
        # Practical strategy: RSI overbought + price near upper BB + (divergence OR RSI declining)
        if (rsi_overbought and price_near_upper and (has_divergence_short or rsi_declining)):
            
            # Calculate ATR for stop loss
            high = prices
            low = prices
            close = prices
            atr = calculate_atr(high, low, close, self.atr_period)
            current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
            
            # Stop loss (1.5x ATR above entry)
            stop_loss = current_price + (current_atr * self.stop_loss_atr_multiplier)
            
            # Batch entries
            batch_entries = []
            for i in range(self.batch_count):
                entry_price = current_price * (1 - self.batch_spacing_pct * i)
                batch_entries.append(entry_price)
            
            # Take profit
            take_profit = current_price * 0.70  # 30% target for shorts
            
            # Quality score
            quality = 0.6
            quality += 0.2 if divergence == "bearish_divergence" else 0.0
            quality += 0.1 if current_rsi > 75 else 0.05
            quality += 0.1 if current_price >= current_upper else 0.05
            
            return VolatilityExpansionSignal(
                pair=pair,
                signal="sell",
                strategy_type="mean_reversion",
                current_price=current_price,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quality=quality,
                confidence=min(quality * 1.1, 1.0),
                reason=f"Mean reversion short: RSI={current_rsi:.1f}, divergence={divergence}",
                regime=regime,
                batch_entries=batch_entries,
                atr_value=current_atr,
                rsi_value=current_rsi,
                macd_signal=macd_signal
            )
        
        return None
    
    def compute_signal(
        self,
        pair: str,
        datastore: DataStore,
        ticker_data: Dict[str, any],
        debug: bool = False,
        utilization: float = 0.0
    ) -> Optional[VolatilityExpansionSignal]:
        """Compute volatility expansion signal.
        
        Args:
            pair: Trading pair
            datastore: DataStore instance
            ticker_data: Current ticker data
            debug: Enable debug output
            
        Returns:
            VolatilityExpansionSignal or None
        """
        # Read minute bars - need enough for indicators
        # For 4h bars: need at least 20 bars for BB (20 period) + RSI (14 period needs 15 bars)
        # 20 * 240 minutes = 4800 minutes minimum
        # Read more to be safe (6000 = 25 four-hour bars)
        df = datastore.read_minute_bars(pair, limit=6000)
        
        if debug:
            print(f"    {pair}: Read {len(df)} minute bars from datastore")
            if len(df) > 0:
                print(f"    {pair}: Data range: {df.index[0]} to {df.index[-1]}")
        
        if len(df) < 500:  # Minimum for basic aggregation
            # Don't print "Insufficient data" - it's expected for pairs that haven't started yet
            # Only print in debug mode if explicitly requested
            if not hasattr(self, '_early_return_counter'):
                self._early_return_counter = {'insufficient_data': 0, 'insufficient_4h': 0, 'reached_mean_rev': 0}
            self._early_return_counter['insufficient_data'] += 1
            return None
        
        # Ensure timestamp is index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # Aggregate to different timeframes
        daily_data = self.aggregate_to_timeframe(df, 1440)  # Daily
        hourly_data = self.aggregate_to_timeframe(df, 60)  # 1-hour
        four_hour_data = self.aggregate_to_timeframe(df, 240)  # 4-hour
        
        # Need at least 20 four-hour bars for indicators (BB 20 period + RSI 14 period needs 15 bars)
        # Reduced from 21 to 20 bars - BB needs 20, RSI needs 15, so 20 is sufficient
        if len(four_hour_data) < 20:
            if debug:
                print(f"    {pair}: Insufficient 4h data ({len(four_hour_data)} < 20, need 20 for indicators)")
            if not hasattr(self, '_early_return_counter'):
                self._early_return_counter = {'insufficient_data': 0, 'insufficient_4h': 0, 'reached_mean_rev': 0}
            self._early_return_counter['insufficient_4h'] += 1
            return None
        
        # Detect market regime - use daily if available, otherwise use 4h data
        if len(daily_data) >= 5:
            regime, macd_signal = self.detect_market_regime(daily_data)
        else:
            # Fallback: use 4h data for regime detection if daily not available
            # Aggregate 4h to "daily" by taking longer periods
            if len(four_hour_data) >= 6:
                # Use 4h data as proxy for daily (6 * 4h = 24h)
                regime, macd_signal = self.detect_market_regime(four_hour_data)
            else:
                # Default to ranging if we can't determine
                regime = "ranging"
                macd_signal = "neutral"
        
        # Get current price
        current_price = ticker_data["price"]
        
        if debug:
            print(f"    {pair}: Data check - 4h bars: {len(four_hour_data)}, columns: {list(four_hour_data.columns)}")
            if len(four_hour_data) > 0:
                print(f"    {pair}: Last 4h bar: {four_hour_data.iloc[-1].to_dict()}")
            print(f"    {pair}: Current price from ticker: {current_price}")
        
        # Use 4-hour data for position trades, 1-hour for entry optimization
        prices_4h = four_hour_data['price'] if 'price' in four_hour_data.columns else four_hour_data['close']
        volumes_4h = four_hour_data['volume'] if 'volume' in four_hour_data.columns else pd.Series([1.0] * len(four_hour_data))
        
        prices_1h = hourly_data['price'] if 'price' in hourly_data.columns else hourly_data['close']
        volumes_1h = hourly_data['volume'] if 'volume' in hourly_data.columns else pd.Series([1.0] * len(hourly_data))
        
        if debug:
            print(f"    {pair}: Prices 4h length: {len(prices_4h)}, last value: {prices_4h.iloc[-1] if len(prices_4h) > 0 else 'N/A'}")
            print(f"    {pair}: Volumes 4h length: {len(volumes_4h)}, last value: {volumes_4h.iloc[-1] if len(volumes_4h) > 0 else 'N/A'}")
            print(f"    {pair}: Regime: {regime}, MACD signal: {macd_signal}")
        
        # Regime-switching: Try both strategies, prefer appropriate one for regime
        all_signals_list = []
        
        if regime == "trending":
            # Trending: Try breakout first, then mean reversion as fallback
            breakout_signal = self.generate_breakout_signal(
                pair, prices_4h, volumes_4h, current_price, regime, macd_signal, debug=debug
            )
            if breakout_signal:
                all_signals_list.append(breakout_signal)
            
            # Also try mean reversion in trending markets (fallback)
            if not hasattr(self, '_early_return_counter'):
                self._early_return_counter = {'insufficient_data': 0, 'insufficient_4h': 0, 'reached_mean_rev': 0}
            self._early_return_counter['reached_mean_rev'] += 1
            mean_rev_signal = self.generate_mean_reversion_signal(
                pair, prices_4h, volumes_4h, current_price, regime, macd_signal, 
                daily_data=daily_data if len(daily_data) >= 5 else None,
                four_hour_data=four_hour_data,
                hourly_data=hourly_data, debug=debug
            )
            if mean_rev_signal:
                all_signals_list.append(mean_rev_signal)
            
            if debug:
                if not all_signals_list:
                    print(f"    {pair}: Trending regime, no signals found")
                else:
                    print(f"    {pair}: Trending regime, found {len(all_signals_list)} signals")
        else:
            # Ranging: Use mean reversion only
            if not hasattr(self, '_early_return_counter'):
                self._early_return_counter = {'insufficient_data': 0, 'insufficient_4h': 0, 'reached_mean_rev': 0}
            self._early_return_counter['reached_mean_rev'] += 1
            mean_rev_signal = self.generate_mean_reversion_signal(
                pair, prices_4h, volumes_4h, current_price, regime, macd_signal, 
                daily_data=daily_data if len(daily_data) >= 5 else None,
                four_hour_data=four_hour_data,
                hourly_data=hourly_data, debug=debug
            )
            if mean_rev_signal:
                all_signals_list.append(mean_rev_signal)
            
            if debug and not all_signals_list:
                print(f"    {pair}: Ranging regime, no mean reversion signal")
        
        # Select best signal (highest quality)
        signal = None
        if all_signals_list:
            # Sort by quality and take the best one
            all_signals_list.sort(key=lambda s: s.quality, reverse=True)
            signal = all_signals_list[0]
        
        if debug and signal is None:
            print(f"    {pair}: No signal generated for {regime} regime")
        
        if debug and signal:
            print(f"    {pair}: SIGNAL FOUND! {signal.strategy_type}, quality={signal.quality:.2f}")
        
        return signal
    
    def rank_signals(
        self,
        signals: Dict[str, VolatilityExpansionSignal]
    ) -> List[Tuple[str, VolatilityExpansionSignal]]:
        """Rank signals by quality and confidence.
        
        Args:
            signals: Dictionary of pair -> signal
            
        Returns:
            Sorted list of (pair, signal) tuples, best first
        """
        actionable = [
            (pair, sig) for pair, sig in signals.items()
            if sig is not None and sig.signal != "neutral"
        ]
        
        # Sort by combined score (quality * confidence)
        actionable.sort(key=lambda x: x[1].quality * x[1].confidence, reverse=True)
        
        return actionable

