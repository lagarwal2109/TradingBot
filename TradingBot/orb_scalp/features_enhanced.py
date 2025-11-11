"""Enhanced FeatureEngine with gate tracking and adaptive filters."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .features import ema, rsi, rolling_median_vol, realized_vol, atr_close_only


class FeatureEngineEnhanced:
    """Enhanced feature engine with gate tracking and adaptive filters."""
    
    def __init__(self, df: pd.DataFrame, params: Dict):
        """Initialize enhanced feature engine.
        
        Args:
            df: DataFrame with columns: dt, close, volume, day, minute_idx
            params: Configuration parameters
        """
        self.df = df.copy()
        self.params = params
        self.features = {}
        self.diagnostic_mode = params.get('diagnostic_mode', False)
        self.gap_max_minutes_orb = params.get('gaps', {}).get('max_minutes_orb', 5)
        self.gap_max_minutes_scalp = params.get('gaps', {}).get('max_minutes_scalp', 30)
        self.warmup_bars = params.get('warmup_bars', 200)
        if self.diagnostic_mode:
            self.warmup_bars = params.get('warmup_bars', 50)
    
    def compute_all(self) -> pd.DataFrame:
        """Compute all features with gate tracking.
        
        Returns:
            DataFrame with original columns plus all features and gate flags
        """
        df = self.df.copy()
        
        # Basic indicators
        df['ema5'] = ema(df['close'], 5)
        df['ema8'] = ema(df['close'], 8)
        df['rsi14'] = rsi(df['close'], 14)
        df['atr14'] = atr_close_only(df['close'], 14)
        df['rv30'] = realized_vol(df['close'], 30)
        
        # Volume features with adaptive spike
        df['vol_med20'] = rolling_median_vol(df['volume'], 20)
        df['vol_std20'] = df['volume'].rolling(window=20, min_periods=1).std().fillna(0)
        
        orb_params = self.params.get('orb', {})
        vol_spike_mult = orb_params.get('vol_spike_mult', 1.03)  # Default to 1.03
        
        # Adaptive volume spike: max(mult * median, median + z * std)
        z = 0.5
        df['vol_spike_threshold'] = np.maximum(
            vol_spike_mult * df['vol_med20'],
            df['vol_med20'] + z * df['vol_std20']
        )
        df['vol_spike'] = df['volume'] >= df['vol_spike_threshold']
        
        # Opening Range (ORH/ORL) - improved with minimum bars check
        open_range_min = self.params.get('session', {}).get('open_range_min', 15)
        min_or_bars = self.params.get('orb', {}).get('min_or_bars', 10)
        
        # Compute ORH/ORL per day, only if enough opening bars exist
        # Consider OR ready when >= min_or_bars of the first open_range_min minutes exist
        def compute_or_safe(group):
            opening_mask = group['minute_idx'] < open_range_min
            opening_bars = group[opening_mask]
            if len(opening_bars) >= min_or_bars:
                or_high = opening_bars['close'].max()
                or_low = opening_bars['close'].min()
            else:
                # Defer ORB for this day - mark as invalid
                or_high = np.nan
                or_low = np.nan
            return pd.Series([or_high, or_low], index=['or_high', 'or_low'])
        
        or_values = df.groupby('day', group_keys=False).apply(compute_or_safe)
        
        # Map back to dataframe
        df['or_high'] = df['day'].map(or_values['or_high'])
        df['or_low'] = df['day'].map(or_values['or_low'])
        
        # Forward-fill ORH/ORL within each day ONLY after they're formed
        # Don't forward-fill NaN values
        df['or_high'] = df.groupby('day')['or_high'].ffill()
        df['or_low'] = df.groupby('day')['or_low'].ffill()
        
        # Mark days with valid opening range (>=90% should be valid if data is good)
        df['or_valid'] = ~(pd.isna(df['or_high']) | pd.isna(df['or_low']))
        
        # Realized volatility percentile floor - use 7-day rolling window
        scalp_params = self.params.get('scalp', {})
        rv_pctile_floor = scalp_params.get('rv_pctile_floor', 5)  # Default to 5th percentile
        
        # Use 7-day rolling window (7 days * 1440 minutes per day)
        # min_periods=1440 means need at least 1 day of data
        df['rv_floor_7d'] = df['rv30'].rolling(
            window=7*1440, 
            min_periods=1440
        ).quantile(
            rv_pctile_floor / 100.0,
            interpolation='linear'
        ).fillna(0.0)
        df['rv_ok'] = df['rv30'] >= df['rv_floor_7d']
        
        # Trade hours filter - with fallback option
        trade_hours = self.params.get('session', {}).get('trade_hours_utc', [8, 20])
        if self.diagnostic_mode:
            trade_hours = [0, 24]  # All day in diagnostic mode
        
        hour_start, hour_end = trade_hours[0], trade_hours[1]
        # Compute hour of day from minute index (since we use period-based indexing)
        df['hour_of_day'] = (df['minute_idx'] // 60).astype(int)
        df['trade_hours'] = (df['hour_of_day'] >= hour_start) & (df['hour_of_day'] < hour_end)
        
        # Hours fallback: if no ORB by 16:00, allow outside hours for that day
        df['hours_fallback'] = False
        if not self.diagnostic_mode:
            for day in df['day'].unique():
                day_mask = df['day'] == day
                day_data = df[day_mask]
                
                # Check if any ORB signal fired before 16:00
                before_16 = day_data[day_data['hour_of_day'] < 16]
                if len(before_16) > 0:
                    # Check if there was a breakout
                    orb_fired = (
                        (before_16['close'] > before_16['or_high']).any() or
                        (before_16['close'] < before_16['or_low']).any()
                    )
                    if not orb_fired:
                        # Allow trading after 16:00 for this day
                        after_16 = day_data[day_data['hour_of_day'] >= 16]
                        df.loc[after_16.index, 'hours_fallback'] = True
        
        # Combined hours check
        df['hours_ok'] = df['trade_hours'] | df['hours_fallback']
        
        # Gap detection - different thresholds for ORB vs Scalp
        df['gap_ok_orb'] = ~self._detect_gaps(df['dt'], self.gap_max_minutes_orb)
        df['gap_ok_scalp'] = ~self._detect_gaps(df['dt'], self.gap_max_minutes_scalp)
        
        # Warmup check
        df['warmup_ok'] = True  # Will be set per-bar based on history length
        
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
            'or_valid': df['or_valid'],
            'rv_ok': df['rv_ok'],
            'trade_hours': df['trade_hours'],
            'hours_ok': df['hours_ok'],
            'gap_ok_orb': df['gap_ok_orb'],
            'gap_ok_scalp': df['gap_ok_scalp'],
            'warmup_ok': df['warmup_ok']
        }
        
        return df
    
    def _detect_gaps(self, dt_series: pd.Series, max_minutes: float) -> pd.Series:
        """Detect gaps exceeding max_minutes.
        
        Args:
            dt_series: Period index series (integers)
            max_minutes: Maximum allowed gap in minutes (bars)
            
        Returns:
            Boolean series: True if gap detected
        """
        if len(dt_series) < 2:
            return pd.Series([False] * len(dt_series), index=dt_series.index)
        
        time_diffs = dt_series.diff().fillna(1)
        gaps = time_diffs > max_minutes
        
        return gaps.fillna(False)

