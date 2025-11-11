"""MACD-based trading strategy implementation."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Pandas offset aliases (lowercase to avoid deprecation warnings)
H1 = "1h"
H4 = "4h"
MIN1 = "1min"
SEC1 = "1s"


@dataclass
class MACDSignal:
    """MACD signal data structure (also used for Donchian breakout)."""
    signal_type: str  # "long", "short", or "none"
    macd: float = 0.0
    signal_line: float = 0.0
    histogram: float = 0.0
    price: float = 0.0
    ma_200: float = 0.0
    trend: str = "neutral"  # "uptrend" or "downtrend"
    macd_above_zero: bool = False
    signal_above_zero: bool = False
    macd_cross_above: bool = False  # MACD crossed above signal
    macd_cross_below: bool = False  # MACD crossed below signal
    momentum: float = 0.0  # Current histogram value (momentum)
    momentum_derivative: float = 0.0  # Rate of change of momentum (growth rate)
    atr: float = 0.0  # Average True Range
    adx: float = 0.0  # Average Directional Index
    stop_distance: float = 0.0  # ATR-based stop distance
    # Donchian breakout fields
    donchian_high_20: float = 0.0  # Highest high over 20 bars
    donchian_low_20: float = 0.0  # Lowest low over 20 bars
    ema_50: float = 0.0  # EMA 50
    ema_200: float = 0.0  # EMA 200
    # Diagnostic gates (for debugging)
    b_cross: bool = False  # Cross detected (Donchian breakout for BATS v1)
    b_regime: bool = False  # Trend filter agrees (EMA200/EMA50 regime)
    b_adx: bool = False  # ADX >= threshold
    b_histmag: bool = False  # Histogram magnitude >= threshold (not used for Donchian)
    b_slope: bool = False  # MACD slope in correct direction (not used for Donchian)
    b_volwindow: bool = False  # ATR/Price in valid range
    b_warmup: bool = False  # Enough bars for indicators


class MACDStrategy:
    """MACD-based trading strategy with trend filter."""
    
    def __init__(
        self,
        fast_period: int = 12,  # Default: 12 periods (will be adaptive)
        slow_period: int = 26,  # Default: 26 periods (will be adaptive)
        signal_period: int = 9,  # Default: 9 periods (will be adaptive)
        trend_period: int = 50  # Default: 50 periods for trend (will be adaptive)
    ):
        """Initialize MACD strategy.
        
        Args:
            fast_period: Fast EMA period (adaptive based on data)
            slow_period: Slow EMA period (adaptive based on data)
            signal_period: Signal line EMA period (adaptive based on data)
            trend_period: Trend filter MA period (adaptive based on data)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.trend_period = trend_period
        
        # Store previous values for cross detection
        self.prev_macd: Dict[str, float] = {}
        self.prev_signal: Dict[str, float] = {}
        
        # Track last trade time per pair for cool-down
        self.last_trade_time: Dict[str, int] = defaultdict(int)  # pair -> timestamp index
        self.last_trade_direction: Dict[str, str] = {}  # pair -> "long" or "short"
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Ensure True Range is never zero (minimum 0.01% of close price)
        # Use element-wise comparison to avoid Timestamp/int comparison issues
        min_tr_values = (close * 0.0001).values  # 0.01% minimum
        tr_values = tr.values
        tr_clipped = pd.Series(np.maximum(tr_values, min_tr_values), index=tr.index)
        
        # ATR = EMA of True Range
        atr = tr_clipped.ewm(span=period, adjust=False).mean()
        
        # Final safety: ensure ATR is never zero (minimum 0.01% of close)
        # Use element-wise comparison
        min_atr_values = (close * 0.0001).values  # 0.01% minimum
        atr_values = atr.values
        atr_clipped = pd.Series(np.maximum(atr_values, min_atr_values), index=atr.index)
        
        return atr_clipped
    
    def calculate_atr_from_close(self, close: pd.Series, period: int = 14, min_ratio: float = 0.001) -> pd.Series:
        """Calculate ATR from close prices only (for price-only data).
        
        Args:
            close: Close price series
            period: ATR period (default 14)
            min_ratio: Minimum ATR as ratio of price (default 0.1%)
            
        Returns:
            ATR series with floor applied
        """
        # True Range = absolute change in close price
        tr = (close - close.shift(1)).abs()
        
        # ATR = EMA of True Range
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Ensure a floor so ATR never 0 (0.1% of price)
        floor = close * min_ratio
        atr = atr.clip(lower=floor)
        
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range
        tr = self.calculate_atr(high, low, close, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / tr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # ADX = EMA of DX
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def calculate_macd(
        self,
        prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal Line, and Histogram.
        
        Args:
            prices: Price series
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        # Calculate EMAs
        fast_ema = self.calculate_ema(prices, self.fast_period)
        slow_ema = self.calculate_ema(prices, self.slow_period)
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = fast_ema - slow_ema
        
        # Signal line = EMA of MACD line
        signal_line = self.calculate_ema(macd_line, self.signal_period)
        
        # Histogram = MACD - Signal
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _compute_regime(self, prices, current_price: float, current_ma_200: float) -> str:
        """Compute HTF regime (uptrend/downtrend) using 4H timeframe.
        
        Args:
            prices: Price series (should have DatetimeIndex for resampling)
            current_price: Current price
            current_ma_200: Current 200-period MA value
            
        Returns:
            "uptrend" or "downtrend"
        """
        # If prices is a Series, use it directly; if DataFrame, use price column
        if isinstance(prices, pd.Series):
            close_1h = prices
        else:
            close_1h = prices['price'] if 'price' in prices.columns else prices.iloc[:, -1]
        
        # Check if we have DatetimeIndex (required for resampling)
        if not isinstance(close_1h.index, pd.DatetimeIndex):
            # Fallback to simple MA comparison if no DatetimeIndex
            return "uptrend" if current_price > current_ma_200 else "downtrend"
        
        # Need enough data for HTF computation
        if len(close_1h) < 200:
            return "uptrend" if current_price > current_ma_200 else "downtrend"
        
        try:
            # Resample to 4H for HTF (use H4 constant)
            close_htf = close_1h.resample(H4).last()
            
            if len(close_htf) < 34:  # Need enough HTF bars (reduced from 50 for faster warmup)
                return "uptrend" if current_price > current_ma_200 else "downtrend"
            
            # Calculate EMA on HTF with proper min_periods (use 34 instead of 200 for faster warmup)
            ema_htf = close_htf.ewm(span=34, min_periods=34, adjust=False).mean()
            
            # Regime: EMA rising = uptrend (use diff().gt(0) for clarity)
            regime_htf = ema_htf.diff().gt(0)  # True if EMA rising
            
            # Forward-fill to 1H index
            regime_on_1h = regime_htf.reindex(close_1h.index, method='ffill').fillna(False)
            
            # Use last value (both HTF and LTF must agree)
            if len(regime_combined) > 0:
                is_uptrend = bool(regime_combined.iloc[-1])
                return "uptrend" if is_uptrend else "downtrend"
            else:
                return "uptrend" if current_price > current_ma_200 else "downtrend"
        except Exception:
            # Fallback on any error
            return "uptrend" if current_price > current_ma_200 else "downtrend"
    
    def detect_cross(
        self,
        pair: str,
        current_macd: float,
        current_signal: float
    ) -> Tuple[bool, bool]:
        """Detect MACD cross above or below signal line.
        
        Args:
            pair: Trading pair identifier
            current_macd: Current MACD value
            current_signal: Current signal line value
            
        Returns:
            Tuple of (cross_above, cross_below)
        """
        cross_above = False
        cross_below = False
        
        if pair in self.prev_macd and pair in self.prev_signal:
            prev_macd = self.prev_macd[pair]
            prev_signal = self.prev_signal[pair]
            
            # Cross above: MACD was below signal, now above
            if prev_macd <= prev_signal and current_macd > current_signal:
                cross_above = True
            
            # Cross below: MACD was above signal, now below
            if prev_macd >= prev_signal and current_macd < current_signal:
                cross_below = True
        
        # Update previous values
        self.prev_macd[pair] = current_macd
        self.prev_signal[pair] = current_signal
        
        return cross_above, cross_below
    
    def generate_signal(
        self,
        pair: str,
        prices: pd.Series
    ) -> MACDSignal:
        """Generate trading signal based on MACD strategy.
        
        Args:
            pair: Trading pair identifier
            prices: Price series (should have enough data for indicators)
            
        Returns:
            MACDSignal object
        """
        # Use adaptive periods based on available data
        data_points = len(prices)
        
        # Calculate adaptive periods first (needed for warmup calculation)
        if data_points < 100:
            # Very limited data: use tiny periods
            adaptive_fast = max(5, int(data_points * 0.1))
            adaptive_slow = max(10, int(data_points * 0.2))
            adaptive_signal = max(5, int(data_points * 0.1))
            adaptive_trend = max(20, int(data_points * 0.3))
        elif data_points < 500:
            # Limited data: use small periods
            adaptive_fast = 12
            adaptive_slow = 26
            adaptive_signal = 9
            adaptive_trend = max(50, int(data_points * 0.2))
        else:
            # More data available: use standard periods but still reasonable
            adaptive_fast = 12
            adaptive_slow = 26
            adaptive_signal = 9
            adaptive_trend = max(50, min(200, int(data_points * 0.15)))
        
        # Warmup should end after max of all windows used on this timeframe
        max_win = max(adaptive_fast, adaptive_slow, adaptive_signal, 14, 200)  # MACD+ADX+MAD window
        b_warmup = data_points < max_win
        if data_points < 50:
            return MACDSignal(
                signal_type="none",
                macd=0.0,
                signal_line=0.0,
                histogram=0.0,
                price=prices.iloc[-1] if len(prices) > 0 else 0.0,
                ma_200=0.0,
                trend="neutral",
                macd_above_zero=False,
                signal_above_zero=False,
                macd_cross_above=False,
                macd_cross_below=False,
                momentum=0.0,
                momentum_derivative=0.0,
                b_warmup=b_warmup
            )
        
        # Need at least slow_period + signal_period + buffer
        min_required = max(adaptive_slow + adaptive_signal + 10, max_win)
        
        if data_points < min_required:
            return MACDSignal(
                signal_type="none",
                macd=0.0,
                signal_line=0.0,
                histogram=0.0,
                price=prices.iloc[-1] if len(prices) > 0 else 0.0,
                ma_200=0.0,
                trend="neutral",
                macd_above_zero=False,
                signal_above_zero=False,
                macd_cross_above=False,
                macd_cross_below=False,
                momentum=0.0,
                momentum_derivative=0.0,
                b_warmup=b_warmup
            )
        
        # Calculate indicators with adaptive periods
        fast_ema = self.calculate_ema(prices, adaptive_fast)
        slow_ema = self.calculate_ema(prices, adaptive_slow)
        macd_line = fast_ema - slow_ema
        signal_line = self.calculate_ema(macd_line, adaptive_signal)
        
        # Normalize MACD by price to avoid scale bias (makes thresholds work across DOGE/BTC/BNB)
        # Use slow EMA as normalization factor
        norm = slow_ema.clip(lower=1e-9)  # Prevent division by zero
        histogram = (macd_line - signal_line) / norm
        ma_trend = self.calculate_ma(prices, adaptive_trend)
        
        # Calculate ATR and ADX (use real high/low if available, otherwise use close-only ATR)
        # Check if prices is a Series or DataFrame with OHLCV columns
        has_ohlcv = False
        if isinstance(prices, pd.DataFrame):
            # DataFrame with OHLCV columns
            if "high" in prices.columns and "low" in prices.columns:
                high_series = prices["high"]
                low_series = prices["low"]
                close_series = prices["price"] if "price" in prices.columns else prices["close"] if "close" in prices.columns else prices.iloc[:, -1]
                has_ohlcv = True
            else:
                # Price-only data: use close-only ATR
                close_series = prices["price"] if "price" in prices.columns else prices.iloc[:, -1]
                atr_series = self.calculate_atr_from_close(close_series, period=14, min_ratio=0.001)
                # For ADX, we still need high/low approximations
                high_series = close_series * 1.002
                low_series = close_series * 0.998
                adx_series = self.calculate_adx(high_series, low_series, close_series, period=14)
        else:
            # Series - price-only data: use close-only ATR
            close_series = prices
            atr_series = self.calculate_atr_from_close(close_series, period=14, min_ratio=0.001)
            # For ADX, we still need high/low approximations
            high_series = prices * 1.002
            low_series = prices * 0.998
            adx_series = self.calculate_adx(high_series, low_series, close_series, period=14)
        
        # If we have OHLCV, calculate ATR normally
        if has_ohlcv:
            atr_series = self.calculate_atr(high_series, low_series, close_series, period=14)
            adx_series = self.calculate_adx(high_series, low_series, close_series, period=14)
        
        # Ensure ATR series has no zeros (replace zeros with minimum value)
        if len(atr_series) > 0:
            min_atr_value = close_series.iloc[-1] * 0.001  # 0.1% minimum (increased from 0.05%)
            atr_series = atr_series.replace(0, min_atr_value)
            atr_series = atr_series.fillna(min_atr_value)
        
        # Get current values with validation
        current_price = prices.iloc[-1]
        
        # Per-signal guard: validate price (skip BONK and small-price assets cleanly)
        if current_price <= 0.05 or not np.isfinite(current_price):
            return MACDSignal(
                signal_type="none",
                macd=0.0,
                signal_line=0.0,
                histogram=0.0,
                price=current_price,
                ma_200=0.0,
                trend="neutral",
                macd_above_zero=False,
                signal_above_zero=False,
                macd_cross_above=False,
                macd_cross_below=False,
                momentum=0.0,
                momentum_derivative=0.0,
                atr=0.0,
                adx=0.0,
                stop_distance=0.0,
                b_warmup=b_warmup
            )
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        current_ma_200 = ma_trend.iloc[-1]
        
        # Get ATR with robust floor (per-signal guard)
        # Last-line floor right before creating signal
        if len(atr_series) > 0:
            raw_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else float('nan')
        else:
            raw_atr = float('nan')
        
        # Minimum ATR = 0.1% of price (never let it collapse)
        atr_floor = max(1e-9, current_price * 0.001)  # 0.1% of price
        
        # Ensure ATR is positive and finite, never zero
        if not np.isfinite(raw_atr) or raw_atr <= 0:
            current_atr = atr_floor
        else:
            # Use max of calculated ATR and minimum threshold
            current_atr = max(raw_atr, atr_floor)
        
        # Final validation - ATR must be positive and finite (don't emit signal if invalid)
        if current_atr <= 0 or not np.isfinite(current_atr):
            return MACDSignal(
                signal_type="none",
                macd=current_macd if not pd.isna(current_macd) else 0.0,
                signal_line=current_signal if not pd.isna(current_signal) else 0.0,
                histogram=current_histogram if not pd.isna(current_histogram) else 0.0,
                price=current_price,
                ma_200=current_ma_200 if not pd.isna(current_ma_200) else current_price,
                trend="neutral",
                macd_above_zero=False,
                signal_above_zero=False,
                macd_cross_above=False,
                macd_cross_below=False,
                momentum=0.0,
                momentum_derivative=0.0,
                atr=current_atr,
                adx=0.0,
                stop_distance=0.0,
                b_warmup=b_warmup
            )
        
        current_adx = adx_series.iloc[-1] if len(adx_series) > 0 and not pd.isna(adx_series.iloc[-1]) else 0.0
        
        # Calculate momentum derivative (growth rate of momentum)
        # Use last 5 periods to calculate rate of change
        if len(histogram) >= 5:
            recent_histogram = histogram.iloc[-5:]
            momentum_derivative = (recent_histogram.iloc[-1] - recent_histogram.iloc[0]) / len(recent_histogram)
        else:
            momentum_derivative = 0.0
        
        # Calculate MACD slope (change in MACD) - use proper diff to avoid double-shift
        if len(macd_line) >= 2:
            # Use diff(1) which gives current - previous, then take last value
            macd_diff = macd_line.diff(1)
            macd_slope = macd_diff.iloc[-1] if len(macd_diff) > 0 and not pd.isna(macd_diff.iloc[-1]) else 0.0
        else:
            macd_slope = 0.0
        
        # Calculate histogram statistics for quality gates (vectorized, fast)
        # Use rolling std as a proxy for robust scale (much faster than MAD with apply)
        roll = histogram.rolling(window=200, min_periods=100)
        robust_scale = roll.std(ddof=0)  # vectorized & fast
        
        if len(robust_scale) > 0 and not pd.isna(robust_scale.iloc[-1]):
            hist_std = float(robust_scale.iloc[-1])
        else:
            hist_std = histogram.std() if len(histogram) > 0 else abs(current_histogram)
        
        # Add epsilon to prevent division by zero in quiet markets
        hist_std = max(hist_std if not pd.isna(hist_std) else 0.0, 1e-8)
        
        # Stop distance based on ATR with clipping (very tight to limit losses to <15%)
        atr_mult = getattr(self, "atr_stop_multiplier", 1.2)  # Default 1.2×ATR (very tight)
        max_drawdown_pct = getattr(self, "max_drawdown_pct", 0.15)  # 15% hard limit
        
        stop_distance = atr_mult * current_atr
        # Clip stop distance: between 0.1% and max_drawdown_pct of price (15% hard limit)
        min_stop = current_price * 0.001  # 0.1% minimum
        max_stop = current_price * max_drawdown_pct  # 15% maximum (hard limit to prevent horrible trades)
        stop_distance = max(min_stop, min(stop_distance, max_stop))
        
        # Validate stop distance
        if stop_distance <= 0 or not np.isfinite(stop_distance):
            return MACDSignal(
                signal_type="none",
                macd=current_macd if not pd.isna(current_macd) else 0.0,
                signal_line=current_signal if not pd.isna(current_signal) else 0.0,
                histogram=current_histogram if not pd.isna(current_histogram) else 0.0,
                price=current_price,
                ma_200=current_ma_200 if not pd.isna(current_ma_200) else current_price,
                trend="neutral",
                macd_above_zero=False,
                signal_above_zero=False,
                macd_cross_above=False,
                macd_cross_below=False,
                momentum=0.0,
                momentum_derivative=0.0,
                atr=current_atr,
                adx=current_adx,
                stop_distance=0.0,
                b_warmup=b_warmup
            )
        
        # Check if values are valid (not NaN)
        if pd.isna(current_ma_200):
            if len(prices) >= 20:
                current_ma_200 = prices.iloc[-20:].mean()
            else:
                current_ma_200 = current_price
        
        # Skip if any indicator is NaN (insufficient warm-up)
        if pd.isna(current_macd) or pd.isna(current_signal) or pd.isna(current_atr) or pd.isna(current_adx):
            return MACDSignal(
                signal_type="none",
                macd=current_macd if not pd.isna(current_macd) else 0.0,
                signal_line=current_signal if not pd.isna(current_signal) else 0.0,
                histogram=current_histogram if not pd.isna(current_histogram) else 0.0,
                price=current_price,
                ma_200=current_ma_200,
                trend="neutral",
                macd_above_zero=False,
                signal_above_zero=False,
                macd_cross_above=False,
                macd_cross_below=False,
                momentum=0.0,
                momentum_derivative=0.0,
                atr=current_atr,
                adx=current_adx,
                stop_distance=stop_distance,
                b_warmup=False  # NaN indicates insufficient warm-up
            )
        
        # Determine trend using HTF regime (compute on HTF, forward-fill to LTF)
        trend = self._compute_regime(prices, current_price, current_ma_200)
        
        # Check zero line positions
        macd_above_zero = current_macd > 0
        signal_above_zero = current_signal > 0
        
        # Detect crosses (using previous bar comparison) - improved to prevent flip-flops
        # Use shift(1) to get previous bar values for proper cross detection
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            prev_hist = prev_macd - prev_signal
            curr_hist = current_macd - current_signal
            
            # Cross above: histogram was <= 0, now > 0
            cross_above = (prev_hist <= 0) and (curr_hist > 0)
            # Cross below: histogram was >= 0, now < 0
            cross_below = (prev_hist >= 0) and (curr_hist < 0)
        else:
            # Fallback to old method if not enough data
            cross_above, cross_below = self.detect_cross(pair, current_macd, current_signal)
        
        b_cross = cross_above or cross_below
        
        # Track diagnostic gates
        # Volatility guardrails - tightened band: 0.2% to 2.5%
        atr_ratio = current_atr / current_price if current_price > 0 else 0
        atr_band_min = getattr(self, "atr_band_min", 0.002)  # 0.2% minimum
        atr_band_max = getattr(self, "atr_band_max", 0.025)  # 2.5% maximum
        b_volwindow = pd.Series([atr_ratio]).between(atr_band_min, atr_band_max).iloc[0] if atr_ratio > 0 else False
        
        # Require min ATR/price >= 0.2% to avoid tiny-ATR chop
        if atr_ratio < atr_band_min:
            return MACDSignal(
                signal_type="none",
                macd=current_macd,
                signal_line=current_signal,
                histogram=current_histogram,
                price=current_price,
                ma_200=current_ma_200,
                trend="neutral",
                macd_above_zero=macd_above_zero,
                signal_above_zero=signal_above_zero,
                macd_cross_above=cross_above,
                macd_cross_below=cross_below,
                momentum=current_histogram,
                momentum_derivative=momentum_derivative,
                atr=current_atr,
                adx=current_adx,
                stop_distance=stop_distance,
                b_cross=b_cross,
                b_volwindow=False,
                b_warmup=b_warmup
            )
        
        if not b_volwindow:
            return MACDSignal(
                signal_type="none",
                macd=current_macd,
                signal_line=current_signal,
                histogram=current_histogram,
                price=current_price,
                ma_200=current_ma_200,
                trend=trend,
                macd_above_zero=macd_above_zero,
                signal_above_zero=signal_above_zero,
                macd_cross_above=cross_above,
                macd_cross_below=cross_below,
                momentum=current_histogram,
                momentum_derivative=momentum_derivative,
                atr=current_atr,
                adx=current_adx,
                stop_distance=stop_distance,
                b_cross=b_cross,
                b_volwindow=b_volwindow,
                b_warmup=b_warmup
            )
        
        # Generate signal based on QUALITY GATES (relaxed for testing)
        signal_type = "none"
        
        # Track gates for diagnostics
        b_regime = False
        b_adx = False
        b_histmag = False
        b_slope = False
        
        # Get configurable thresholds (tuned for better quality)
        hist_z_score_min = getattr(self, "hist_z_score_min", 0.7)  # Z-score >= 0.7 (loosened from 1.0)
        min_adx = getattr(self, "min_adx", 22.0)  # ADX >= 22 (tightened from 20)
        
        # Calculate histogram z-score (current value / std)
        hist_z_score = abs(current_histogram) / hist_std if hist_std > 0 else 0.0
        
        # ADX rising filter: ΔADX > 0 over last 5 bars
        adx_rising = False
        if len(adx_series) >= 5:
            adx_recent = adx_series.iloc[-5:]
            adx_rising = (adx_recent.iloc[-1] > adx_recent.iloc[0])
        
        # EMA(50) slope filter for trend confirmation
        ema_50_slope_ok = False
        if len(prices) >= 50:
            ema_50 = self.calculate_ema(prices, 50)
            if len(ema_50) >= 2:
                ema_50_slope = ema_50.diff(1).iloc[-1]
                if cross_above and not macd_above_zero:
                    ema_50_slope_ok = ema_50_slope >= 0  # Rising EMA for longs
                elif cross_below and macd_above_zero:
                    ema_50_slope_ok = ema_50_slope <= 0  # Falling EMA for shorts
                else:
                    ema_50_slope_ok = True  # Neutral if no cross
            else:
                ema_50_slope_ok = True
        else:
            ema_50_slope_ok = True  # Not enough data
        
        # Priority 1: STRICT CROSS with all quality gates (tuned thresholds)
        if cross_above and not macd_above_zero and not signal_above_zero:
            # Tuned quality gates:
            # 1. MTF regime alignment (both HTF and LTF agree)
            b_regime_long = (trend == "uptrend")
            # 2. Histogram z-score >= 0.7 (loosened from 1.0)
            b_histmag_long = (hist_z_score >= hist_z_score_min)
            # 3. ADX >= 22 AND rising
            b_adx_long = (current_adx >= min_adx) and adx_rising
            # 4. MACD slope: ΔMACD ≥ 0
            macd_slope_ok = macd_slope >= 0
            b_slope_long = macd_slope_ok
            # 5. EMA(50) slope > 0 for longs
            b_ema_slope_long = ema_50_slope_ok
            
            if b_regime_long and b_histmag_long and b_adx_long and b_slope_long and b_ema_slope_long:
                signal_type = "long"
                b_regime = True
                b_histmag = True
                b_adx = True
                b_slope = True
        
        if cross_below and macd_above_zero:
            # Tuned quality gates for short:
            b_regime_short = (trend == "downtrend")
            b_histmag_short = (hist_z_score >= hist_z_score_min)
            b_adx_short = (current_adx >= min_adx) and adx_rising
            macd_slope_ok = macd_slope <= 0
            b_slope_short = macd_slope_ok
            # EMA(50) slope < 0 for shorts
            b_ema_slope_short = ema_50_slope_ok
            
            if b_regime_short and b_histmag_short and b_adx_short and b_slope_short and b_ema_slope_short:
                signal_type = "short"
                b_regime = True
                b_histmag = True
                b_adx = True
                b_slope = True
        
        # Priority 2: Cross with ADX and histogram z-score (ADX >= 20, z-score >= 1.0)
        if signal_type == "none":
            if cross_above and trend == "uptrend":
                b_adx_long = current_adx >= min_adx
                b_histmag_long = hist_z_score >= hist_z_score_min
                if b_adx_long and b_histmag_long:
                    signal_type = "long"
                    b_regime = True
                    b_adx = True
                    b_histmag = True
            elif cross_below and trend == "downtrend":
                b_adx_short = current_adx >= min_adx
                b_histmag_short = hist_z_score >= hist_z_score_min
                if b_adx_short and b_histmag_short:
                    signal_type = "short"
                    b_regime = True
                    b_adx = True
                    b_histmag = True
        
        # Priority 3: Fallback - just cross with trend and ADX >= 20
        if signal_type == "none":
            if cross_above and trend == "uptrend":
                b_adx_long = current_adx >= min_adx
                if b_adx_long:
                    signal_type = "long"
                    b_regime = True
                    b_adx = True
            elif cross_below and trend == "downtrend":
                b_adx_short = current_adx >= min_adx
                if b_adx_short:
                    signal_type = "short"
                    b_regime = True
                    b_adx = True
        
        # Priorities 4-7 DISABLED - they cause over-trading
        
        return MACDSignal(
            signal_type=signal_type,
            macd=current_macd,
            signal_line=current_signal,
            histogram=current_histogram,
            price=current_price,
            ma_200=current_ma_200,
            trend=trend,
            macd_above_zero=macd_above_zero,
            signal_above_zero=signal_above_zero,
            macd_cross_above=cross_above,
            macd_cross_below=cross_below,
            momentum=current_histogram,
            momentum_derivative=momentum_derivative,
            atr=current_atr,
            adx=current_adx,
            stop_distance=stop_distance,
            b_cross=b_cross,
            b_regime=b_regime,
            b_adx=b_adx,
            b_histmag=b_histmag,
            b_slope=b_slope,
            b_volwindow=b_volwindow,
            b_warmup=b_warmup
        )
    
    def should_exit_long(
        self,
        entry_price: float,
        current_price: float,
        current_atr: float,
        stop_distance: float,
        bars_held: int,
        initial_stop: float,
        breakeven_stop: Optional[float] = None,
        trailing_stop: Optional[float] = None
    ) -> Tuple[bool, str, bool]:
        """Check if long position should be exited using ATR-based stops.
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            current_atr: Current ATR value
            stop_distance: Initial stop distance (2.5×ATR)
            bars_held: Number of bars position has been held
            initial_stop: Initial stop price
            breakeven_stop: Breakeven stop price (after 1R profit)
            trailing_stop: Trailing stop price
            
        Returns:
            Tuple of (should_exit, reason, is_partial)
        """
        # Calculate R (initial risk)
        risk = entry_price - initial_stop
        if risk <= 0:
            return False, "", False
        
        profit_r = (current_price - entry_price) / risk
        
        # Time stop: exit after 48 bars if not reached +1R
        time_stop_bars = getattr(self, "time_stop_bars", 48)
        if bars_held >= time_stop_bars and profit_r < 1.0:
            return True, "time_stop", False
        
        # Partial profit at +2R (take 25-33%, tighten stop to +0.5R)
        partial_profit_r = getattr(self, "partial_profit_r", 2.0)
        partial_profit_pct = getattr(self, "partial_profit_pct", 0.30)
        partial_stop_r = getattr(self, "partial_stop_r", 0.5)
        
        if profit_r >= partial_profit_r:
            # Check if we already took partial
            # This will be handled in backtester with scaled_out flag
            # For now, just signal partial exit
            return True, "take_profit_partial", True
        
        # Then check stops (hard max drawdown first, then trailing, then breakeven, then initial)
        # Hard max drawdown limit (15% from entry) - prevents horrible trades
        max_drawdown_pct = getattr(self, "max_drawdown_pct", 0.15)
        max_drawdown_stop = entry_price * (1 - max_drawdown_pct)
        if current_price <= max_drawdown_stop:
            return True, "max_drawdown_stop", False
        
        if trailing_stop is not None and current_price <= trailing_stop:
            return True, "trailing_stop", False
        
        if breakeven_stop is not None and current_price <= breakeven_stop:
            return True, "breakeven_stop", False
        
        if current_price <= initial_stop:
            return True, "initial_stop", False
        
        return False, "", False
    
    def should_exit_short(
        self,
        entry_price: float,
        current_price: float,
        current_atr: float,
        stop_distance: float,
        bars_held: int,
        initial_stop: float,
        breakeven_stop: Optional[float] = None,
        trailing_stop: Optional[float] = None
    ) -> Tuple[bool, str, bool]:
        """Check if short position should be exited using ATR-based stops.
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            current_atr: Current ATR value
            stop_distance: Initial stop distance (2.5×ATR)
            bars_held: Number of bars position has been held
            initial_stop: Initial stop price
            breakeven_stop: Breakeven stop price (after 1R profit)
            trailing_stop: Trailing stop price
            
        Returns:
            Tuple of (should_exit, reason, is_partial)
        """
        # Time stop: exit after 40 bars if no target/stop hit (increased from 35)
        if bars_held >= 40:
            return True, "time_stop", False
        
        # Exit order: profit targets first, then stops
        # Check partial profit target first (0.6R - earlier partial)
        risk = initial_stop - entry_price
        if risk > 0:
            target_partial = entry_price - (risk * 0.6)  # 0.6R profit target for short (from 0.8R)
            if current_price <= target_partial:
                return True, "take_profit_partial", True  # Partial exit
        
        # Check full profit target (1.5R)
        target_1_5r = entry_price - (risk * 1.5)
        if current_price <= target_1_5r:
            return True, "take_profit_1_5r", False
        
        # Then check stops (hard max drawdown first, then trailing, then breakeven, then initial)
        # Hard max drawdown limit (15% from entry) - prevents horrible trades
        max_drawdown_pct = getattr(self, "max_drawdown_pct", 0.15)
        max_drawdown_stop = entry_price * (1 + max_drawdown_pct)  # For shorts, stop is above entry
        if current_price >= max_drawdown_stop:
            return True, "max_drawdown_stop", False
        
        if trailing_stop is not None and current_price >= trailing_stop:
            return True, "trailing_stop", False
        
        if breakeven_stop is not None and current_price >= breakeven_stop:
            return True, "breakeven_stop", False
        
        if current_price >= initial_stop:
            return True, "initial_stop", False
        
        return False, "", False
    
    def generate_donchian_signal(
        self,
        pair: str,
        prices: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None
    ) -> MACDSignal:
        """Generate Donchian breakout signal (BATS v1).
        
        Args:
            pair: Trading pair identifier
            prices: Close price series
            high: High price series (optional, uses prices if not provided)
            low: Low price series (optional, uses prices if not provided)
            
        Returns:
            MACDSignal object with Donchian breakout logic
        """
        data_points = len(prices)
        current_price = prices.iloc[-1] if len(prices) > 0 else 0.0
        
        # Minimum required bars: max(200 for EMA200, 50 for EMA50, 20 for Donchian, 14 for ATR/ADX)
        min_required = 200
        b_warmup = data_points < min_required
        
        if b_warmup or current_price <= 0:
            return MACDSignal(
                signal_type="none",
                price=current_price,
                b_warmup=b_warmup
            )
        
        # Use prices as fallback for high/low if not provided
        # Ensure all series have the same index and length
        if high is None:
            high = prices * 1.001  # Approximate high
            high.index = prices.index  # Ensure same index
        if low is None:
            low = prices * 0.999  # Approximate low
            low.index = prices.index  # Ensure same index
        
        # Align all series to have the same index (in case of misalignment)
        high = high.reindex(prices.index, method='ffill').fillna(prices * 1.001)
        low = low.reindex(prices.index, method='ffill').fillna(prices * 0.999)
        
        # Calculate indicators
        # EMA 200 and EMA 50 for regime filter
        ema_200 = self.calculate_ema(prices, 200)
        ema_50 = self.calculate_ema(prices, 50)
        
        # Donchian channels (20-bar)
        donchian_high_20 = high.rolling(window=20, min_periods=20).max()
        donchian_low_20 = low.rolling(window=20, min_periods=20).min()
        
        # ATR and ADX (need high/low for proper calculation)
        # Ensure high/low/prices are aligned before calculation
        aligned_high = high.reindex(prices.index)
        aligned_low = low.reindex(prices.index)
        atr_series = self.calculate_atr(aligned_high, aligned_low, prices, period=14)
        adx_series = self.calculate_adx(aligned_high, aligned_low, prices, period=14)
        
        # Get current values
        current_ema_200 = ema_200.iloc[-1] if len(ema_200) > 0 and not pd.isna(ema_200.iloc[-1]) else current_price
        current_ema_50 = ema_50.iloc[-1] if len(ema_50) > 0 and not pd.isna(ema_50.iloc[-1]) else current_price
        current_donchian_high = donchian_high_20.iloc[-1] if len(donchian_high_20) > 0 and not pd.isna(donchian_high_20.iloc[-1]) else current_price
        current_donchian_low = donchian_low_20.iloc[-1] if len(donchian_low_20) > 0 and not pd.isna(donchian_low_20.iloc[-1]) else current_price
        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else current_price * 0.01
        current_adx = adx_series.iloc[-1] if len(adx_series) > 0 and not pd.isna(adx_series.iloc[-1]) else 0.0
        
        # Ensure ATR is never zero
        atr_floor = max(current_price * 0.001, current_atr)  # 0.1% floor
        current_atr = max(atr_floor, current_price * 0.0001)  # Minimum 0.01%
        
        # ATR% band check (use config values: 0.8% to 4.0% default)
        atr_ratio = current_atr / current_price
        atr_band_min = getattr(self, "atr_band_min", 0.008)  # 0.8% default
        atr_band_max = getattr(self, "atr_band_max", 0.04)  # 4.0% default
        b_volwindow = atr_band_min <= atr_ratio <= atr_band_max
        
        # ADX check (>= 20)
        min_adx = getattr(self, "min_adx", 20.0)
        b_adx = current_adx >= min_adx
        
        # Regime filter (BATS v1: binary EMA200/EMA50)
        # Longs: close > EMA200 AND EMA50 > EMA200
        # Shorts: close < EMA200 AND EMA50 < EMA200
        b_regime_long = (current_price > current_ema_200) and (current_ema_50 > current_ema_200)
        b_regime_short = (current_price < current_ema_200) and (current_ema_50 < current_ema_200)
        b_regime = b_regime_long or b_regime_short
        
        # Donchian breakout check
        # Long: close > highest(high, 20)
        # Short: close < lowest(low, 20)
        b_breakout_long = current_price > current_donchian_high
        b_breakout_short = current_price < current_donchian_low
        b_cross = b_breakout_long or b_breakout_short
        
        # Determine signal
        signal_type = "none"
        if b_cross and b_regime and b_adx and b_volwindow and not b_warmup:
            if b_breakout_long and b_regime_long:
                signal_type = "long"
            elif b_breakout_short and b_regime_short:
                signal_type = "short"
        
        # Calculate stop distance (3.0 × ATR for BATS v1)
        k_atr_stop = getattr(self, "k_atr_stop", 3.0)
        max_drawdown_pct = getattr(self, "max_drawdown_pct", 0.15)
        stop_distance = k_atr_stop * current_atr
        
        # Clip stop distance: between 0.1% and max_drawdown_pct of price
        min_stop = current_price * 0.001  # 0.1% minimum
        max_stop = current_price * max_drawdown_pct  # 15% maximum
        stop_distance = max(min_stop, min(stop_distance, max_stop))
        
        # Validate stop distance
        if stop_distance <= 0 or not np.isfinite(stop_distance):
            stop_distance = current_price * 0.01  # Fallback to 1%
        
        return MACDSignal(
            signal_type=signal_type,
            price=current_price,
            atr=current_atr,
            adx=current_adx,
            stop_distance=stop_distance,
            donchian_high_20=current_donchian_high,
            donchian_low_20=current_donchian_low,
            ema_50=current_ema_50,
            ema_200=current_ema_200,
            b_cross=b_cross,
            b_regime=b_regime,
            b_adx=b_adx,
            b_volwindow=b_volwindow,
            b_warmup=b_warmup
        )

