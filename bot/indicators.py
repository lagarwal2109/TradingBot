"""Centralized technical indicator calculations for volatility expansion strategy."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index).
    
    Args:
        prices: Price series
        period: RSI period (default 14, crypto-optimized uses 14 with 80/20 thresholds)
        
    Returns:
        RSI series (0-100)
    """
    if len(prices) < period + 1:
        return pd.Series(dtype=float)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 3,
    slow_period: int = 10,
    signal_period: int = 16
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence) - crypto-optimized settings.
    
    Args:
        prices: Price series
        fast_period: Fast EMA period (default 3 for crypto)
        slow_period: Slow EMA period (default 10 for crypto)
        signal_period: Signal line EMA period (default 16 for crypto)
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    if len(prices) < slow_period + signal_period:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow
    
    # Signal line = EMA of MACD line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram = MACD line - Signal line
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def detect_rsi_divergence(
    prices: pd.Series,
    rsi: pd.Series,
    lookback: int = 20
) -> Optional[str]:
    """Detect RSI divergence patterns.
    
    Args:
        prices: Price series
        rsi: RSI series
        lookback: Number of periods to look back for divergence
        
    Returns:
        'bullish_divergence', 'bearish_divergence', or None
    """
    if len(prices) < lookback + 1 or len(rsi) < lookback + 1:
        return None
    
    # Get recent data
    recent_prices = prices.tail(lookback)
    recent_rsi = rsi.tail(lookback)
    
    # Find local highs and lows
    price_highs = []
    price_lows = []
    rsi_highs = []
    rsi_lows = []
    
    for i in range(2, len(recent_prices) - 2):
        # Price peaks
        if (recent_prices.iloc[i] > recent_prices.iloc[i-1] and 
            recent_prices.iloc[i] > recent_prices.iloc[i+1] and
            recent_prices.iloc[i] > recent_prices.iloc[i-2] and
            recent_prices.iloc[i] > recent_prices.iloc[i+2]):
            price_highs.append((i, recent_prices.iloc[i]))
        
        # Price troughs
        if (recent_prices.iloc[i] < recent_prices.iloc[i-1] and 
            recent_prices.iloc[i] < recent_prices.iloc[i+1] and
            recent_prices.iloc[i] < recent_prices.iloc[i-2] and
            recent_prices.iloc[i] < recent_prices.iloc[i+2]):
            price_lows.append((i, recent_prices.iloc[i]))
        
        # RSI peaks
        if (recent_rsi.iloc[i] > recent_rsi.iloc[i-1] and 
            recent_rsi.iloc[i] > recent_rsi.iloc[i+1]):
            rsi_highs.append((i, recent_rsi.iloc[i]))
        
        # RSI troughs
        if (recent_rsi.iloc[i] < recent_rsi.iloc[i-1] and 
            recent_rsi.iloc[i] < recent_rsi.iloc[i+1]):
            rsi_lows.append((i, recent_rsi.iloc[i]))
    
    # Check for bearish divergence (price makes higher high, RSI makes lower high)
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        # Get last two price highs
        price_high1_idx, price_high1_val = price_highs[-2]
        price_high2_idx, price_high2_val = price_highs[-1]
        
        # Find corresponding RSI highs
        rsi_high1 = next((v for idx, v in rsi_highs if abs(idx - price_high1_idx) <= 3), None)
        rsi_high2 = next((v for idx, v in rsi_highs if abs(idx - price_high2_idx) <= 3), None)
        
        if (rsi_high1 is not None and rsi_high2 is not None and
            price_high2_val > price_high1_val and rsi_high2 < rsi_high1):
            return 'bearish_divergence'
    
    # Check for bullish divergence (price makes lower low, RSI makes higher low)
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        # Get last two price lows
        price_low1_idx, price_low1_val = price_lows[-2]
        price_low2_idx, price_low2_val = price_lows[-1]
        
        # Find corresponding RSI lows
        rsi_low1 = next((v for idx, v in rsi_lows if abs(idx - price_low1_idx) <= 3), None)
        rsi_low2 = next((v for idx, v in rsi_lows if abs(idx - price_low2_idx) <= 3), None)
        
        if (rsi_low1 is not None and rsi_low2 is not None and
            price_low2_val < price_low1_val and rsi_low2 > rsi_low1):
            return 'bullish_divergence'
    
    return None


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.
    
    Args:
        prices: Price series
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0, crypto-optimized uses 1.5-2.0)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(prices) < period:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def detect_bollinger_squeeze(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
    squeeze_threshold: float = 0.02
) -> Tuple[bool, float]:
    """Detect Bollinger Band squeeze (low volatility consolidation).
    
    Args:
        prices: Price series
        period: Bollinger Band period
        std_dev: Standard deviation multiplier
        squeeze_threshold: Band width as percentage of price (default 2%)
        
    Returns:
        Tuple of (is_squeeze, band_width_pct)
    """
    if len(prices) < period:
        return False, 0.0
    
    upper_band, middle_band, lower_band = calculate_bollinger_bands(prices, period, std_dev)
    
    if len(upper_band) == 0 or len(lower_band) == 0:
        return False, 0.0
    
    # Calculate band width as percentage of price
    current_price = prices.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    
    if current_price > 0:
        band_width_pct = ((current_upper - current_lower) / current_price) * 100
        is_squeeze = band_width_pct < (squeeze_threshold * 100)
        return is_squeeze, band_width_pct
    
    return False, 0.0


def calculate_choppiness_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Choppiness Index (CI) for consolidation detection.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: CI period (default 14)
        
    Returns:
        Choppiness Index series (0-100, higher = more choppy/ranging)
    """
    if len(high) < period + 1:
        return pd.Series(dtype=float)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    atr_sum = tr.rolling(window=period).sum()
    
    # Highest High - Lowest Low over period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    hl_range = highest_high - lowest_low
    
    # Choppiness Index
    ci = 100 * np.log10(atr_sum / (hl_range + 1e-10)) / np.log10(period)
    
    return ci


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)
        
    Returns:
        ATR series
    """
    if len(high) < period + 1:
        return pd.Series(dtype=float)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Price series
        period: EMA period
        
    Returns:
        EMA series
    """
    if len(prices) < period:
        return pd.Series(dtype=float)
    
    return prices.ewm(span=period, adjust=False).mean()


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average Directional Index (ADX).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default 14)
        
    Returns:
        ADX series (0-100, where < 20 = ranging, > 25 = trending)
    """
    if len(high) < period + 1:
        return pd.Series(dtype=float)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    # Filter directional movement
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0
    
    # Smooth TR and DM
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = dx.rolling(window=period).mean()
    
    return adx


def calculate_volume_ma(volumes: pd.Series, period: int = 20) -> pd.Series:
    """Calculate volume moving average.
    
    Args:
        volumes: Volume series
        period: MA period (default 20)
        
    Returns:
        Volume MA series
    """
    if len(volumes) < period:
        return pd.Series(dtype=float)
    
    return volumes.rolling(window=period).mean()


def detect_volume_spike(
    volumes: pd.Series,
    period: int = 20,
    spike_multiplier: float = 1.5
) -> Tuple[bool, float]:
    """Detect volume spike (e.g., >150% of average).
    
    Args:
        volumes: Volume series
        period: Period for average calculation
        spike_multiplier: Multiplier threshold (default 1.5 = 150%)
        
    Returns:
        Tuple of (is_spike, volume_ratio)
    """
    if len(volumes) < period + 1:
        return False, 1.0
    
    current_volume = volumes.iloc[-1]
    avg_volume = volumes.tail(period).mean()
    
    if avg_volume > 0:
        volume_ratio = current_volume / avg_volume
        is_spike = volume_ratio >= spike_multiplier
        return is_spike, volume_ratio
    
    return False, 1.0


def detect_bullish_engulfing(
    df: pd.DataFrame,
    lookback: int = 2
) -> bool:
    """Detect bullish engulfing candlestick pattern.
    
    A bullish engulfing pattern occurs when:
    - Current candle is bullish (close > open)
    - Current candle body engulfs previous candle body
    - Current open < previous close
    - Current close > previous open
    - Current body > previous body * 1.2 (significant engulfing)
    
    Args:
        df: DataFrame with OHLC data (columns: 'open', 'high', 'low', 'close')
        lookback: Number of candles to check (default 2, checks last 2 candles)
        
    Returns:
        True if bullish engulfing pattern detected, False otherwise
    """
    if len(df) < 2:
        return False
    
    # Need OHLC data
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        return False
    
    # Check last 2 candles
    prev_candle = df.iloc[-2]
    curr_candle = df.iloc[-1]
    
    # Previous candle body
    prev_body = abs(prev_candle['close'] - prev_candle['open'])
    
    # Current candle body
    curr_body = abs(curr_candle['close'] - curr_candle['open'])
    curr_is_bullish = curr_candle['close'] > curr_candle['open']
    
    # Bullish engulfing conditions
    if not curr_is_bullish:
        return False
    
    # Current open should be below previous close
    if curr_candle['open'] >= prev_candle['close']:
        return False
    
    # Current close should be above previous open
    if curr_candle['close'] <= prev_candle['open']:
        return False
    
    # Current body should be significantly larger (at least 1.2x)
    if prev_body > 0 and curr_body < (prev_body * 1.2):
        return False
    
    return True


def detect_hammer_pattern(
    df: pd.DataFrame,
    lookback: int = 3
) -> bool:
    """Detect hammer candlestick pattern.
    
    A hammer pattern occurs when:
    - Lower wick > 2x body size
    - Upper wick < 0.3x body size (small or no upper wick)
    - Close near high (within 10% of candle range)
    - Body can be bullish or bearish (both valid for hammer)
    
    Args:
        df: DataFrame with OHLC data (columns: 'open', 'high', 'low', 'close')
        lookback: Number of candles to check (default 3, checks last 3 candles)
        
    Returns:
        True if hammer pattern detected in any of the last candles, False otherwise
    """
    if len(df) < 1:
        return False
    
    # Need OHLC data
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        return False
    
    # Check last few candles for hammer pattern
    for i in range(min(lookback, len(df))):
        candle = df.iloc[-(i+1)]
        
        # Calculate candle components
        body = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            continue
        
        # Lower wick (distance from low to body)
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        # Upper wick (distance from body to high)
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        
        # Hammer conditions
        # 1. Lower wick > 2x body size
        if body > 0 and lower_wick < (body * 2):
            continue
        
        # 2. Upper wick < 0.3x body size (or very small)
        if body > 0 and upper_wick > (body * 0.3):
            continue
        
        # 3. Close near high (within 10% of candle range)
        close_position = (candle['close'] - candle['low']) / candle_range
        if close_position < 0.9:  # Close should be in top 10% of range
            continue
        
        # All conditions met - hammer pattern detected
        return True
    
    return False
