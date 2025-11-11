"""FeatureEngine: computes EMA, RSI, ATR_close_only, RV, ORH/ORL, vol_spike."""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average.
    
    Args:
        series: Price series
        span: EMA span
        
    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing.
    
    Args:
        close: Close price series
        length: RSI period (default 14)
        
    Returns:
        RSI series (0-100)
    """
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder's smoothing: EMA with alpha = 1/length
    avg_gain = gain.ewm(alpha=1.0/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def rolling_median_vol(volume: pd.Series, length: int = 20) -> pd.Series:
    """Calculate rolling median of volume.
    
    Args:
        volume: Volume series
        length: Window length
        
    Returns:
        Rolling median series
    """
    return volume.rolling(window=length, min_periods=1).median()


def realized_vol(close: pd.Series, lookback: int = 30) -> pd.Series:
    """Calculate realized volatility (std of log returns scaled to hourly).
    
    Args:
        close: Close price series
        lookback: Lookback window
        
    Returns:
        Realized volatility series (annualized)
    """
    log_returns = np.log(close / close.shift(1))
    std_1min = log_returns.rolling(window=lookback, min_periods=1).std()
    
    # Scale to hourly: sqrt(60) for 1-min to 1-hour
    # Then annualize: sqrt(252 * 24) for hourly to annual
    rv = std_1min * np.sqrt(60) * np.sqrt(252 * 24)
    
    return rv.fillna(0.0)


def atr_close_only(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate ATR from close-only data using True Range approximation.
    
    TR_t = abs(close_t - close_{t-1})
    ATR = Wilder's smoothing of TR
    
    Args:
        close: Close price series
        length: ATR period (default 14)
        
    Returns:
        ATR series
    """
    tr = close.diff().abs()
    
    # Wilder's smoothing: EMA with alpha = 1/length
    atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
    
    # Floor: minimum 0.01% of price to avoid zero ATR
    atr_floor = close * 0.0001
    atr = atr.clip(lower=atr_floor)
    
    return atr.fillna(0.0)


class FeatureEngine:
    """Computes all features needed for signals."""
    
    def __init__(self, df: pd.DataFrame, params: Dict):
        """Initialize feature engine.
        
        Args:
            df: DataFrame with columns: dt, close, volume, day, minute_idx
            params: Configuration parameters
        """
        self.df = df.copy()
        self.params = params
        self.features = {}
    
    def compute_all(self) -> pd.DataFrame:
        """Compute all features.
        
        Returns:
            DataFrame with original columns plus all features
        """
        df = self.df.copy()
        
        # Basic indicators
        df['ema5'] = ema(df['close'], 5)
        df['ema8'] = ema(df['close'], 8)
        df['rsi14'] = rsi(df['close'], 14)
        df['atr14'] = atr_close_only(df['close'], 14)
        df['rv30'] = realized_vol(df['close'], 30)
        
        # Volume features
        df['vol_med20'] = rolling_median_vol(df['volume'], 20)
        vol_spike_mult = self.params.get('orb', {}).get('vol_spike_mult', 1.2)
        df['vol_spike'] = df['volume'] >= (vol_spike_mult * df['vol_med20'])
        
        # Opening Range (ORH/ORL)
        open_range_min = self.params.get('session', {}).get('open_range_min', 15)
        
        # Compute ORH/ORL per day
        opening_mask = df['minute_idx'] < open_range_min
        
        # Group by day and compute max/min of opening range
        or_high_series = df[opening_mask].groupby('day')['close'].max()
        or_low_series = df[opening_mask].groupby('day')['close'].min()
        
        # Map back to dataframe
        df['or_high'] = df['day'].map(or_high_series)
        df['or_low'] = df['day'].map(or_low_series)
        
        # Forward-fill ORH/ORL within each day
        df['or_high'] = df.groupby('day')['or_high'].ffill()
        df['or_low'] = df.groupby('day')['or_low'].ffill()
        
        # Realized volatility percentile floor (per day)
        rv_pctile_floor = self.params.get('scalp', {}).get('rv_pctile_floor', 10)
        df['rv_floor'] = df.groupby('day')['rv30'].transform(
            lambda x: x.quantile(rv_pctile_floor / 100.0) if len(x) > 0 else 0.0
        )
        df['rv_ok'] = df['rv30'] >= df['rv_floor']
        
        # Trade hours filter
        trade_hours = self.params.get('session', {}).get('trade_hours_utc', [8, 20])
        hour_start, hour_end = trade_hours[0], trade_hours[1]
        df['trade_hours'] = (df['dt'].dt.hour >= hour_start) & (df['dt'].dt.hour < hour_end)
        
        # Store features
        self.features = {
            'ema5': df['ema5'],
            'ema8': df['ema8'],
            'rsi14': df['rsi14'],
            'atr14': df['atr14'],
            'rv30': df['rv30'],
            'vol_spike': df['vol_spike'],
            'or_high': df['or_high'],
            'or_low': df['or_low'],
            'rv_ok': df['rv_ok'],
            'trade_hours': df['trade_hours']
        }
        
        return df

