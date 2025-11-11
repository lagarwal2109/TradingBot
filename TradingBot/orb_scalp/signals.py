"""SignalEngine: generates ORB and Scalp signals."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, date


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    dt: datetime
    side: str  # "long" or "short"
    signal_type: str  # "ORB" or "SCALP"
    price: float
    atr: float
    stop_distance: float
    reason: str = ""  # Rejection reason if invalid
    valid: bool = True


class SignalEngine:
    """Generates ORB and Scalp signals."""
    
    def __init__(self, params: Dict):
        """Initialize signal engine.
        
        Args:
            params: Configuration parameters
        """
        self.params = params
        self.orb_retry_counts: Dict[tuple, int] = {}  # (symbol, day, side) -> count
        self.scalp_count_today: Dict[date, int] = {}  # day -> count
        self.last_scalp_price: Dict[tuple, float] = {}  # (symbol, day) -> price
    
    def orb_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate ORB signals.
        
        Args:
            df: DataFrame with features computed
            symbol: Symbol name
            
        Returns:
            DataFrame with signals (empty if none)
        """
        orb_params = self.params.get('orb', {})
        
        rsi_long_min = orb_params.get('rsi_long_min', 55)
        rsi_short_max = orb_params.get('rsi_short_max', 45)
        vol_spike_mult = orb_params.get('vol_spike_mult', 1.2)
        stop_atr_mult = orb_params.get('stop_atr_mult', 0.8)
        pct_stop_floor = orb_params.get('pct_stop_floor', 0.005)
        max_retries = orb_params.get('max_retries_per_side', 2)
        
        signals = []
        
        for idx, row in df.iterrows():
            if pd.isna(row['or_high']) or pd.isna(row['or_low']):
                continue  # No opening range for this day
            
            if not row['trade_hours']:
                continue
            
            if row['gap_flag']:
                continue  # Skip if gap detected
            
            current_day = row['day']
            atr = row['atr14']
            close = row['close']
            rsi_val = row['rsi14']
            vol_spike = row['vol_spike']
            
            # Calculate stop distance
            stop_distance = max(pct_stop_floor * close, stop_atr_mult * atr)
            
            # Long signal: close > ORH
            if close > row['or_high']:
                retry_key = (symbol, current_day, 'long')
                retry_count = self.orb_retry_counts.get(retry_key, 0)
                
                if retry_count >= max_retries:
                    continue  # Max retries reached
                
                if vol_spike and rsi_val >= rsi_long_min:
                    signal = Signal(
                        symbol=symbol,
                        dt=row['dt'],
                        side='long',
                        signal_type='ORB',
                        price=close,
                        atr=atr,
                        stop_distance=stop_distance,
                        valid=True
                    )
                    signals.append(signal)
                    self.orb_retry_counts[retry_key] = retry_count + 1
            
            # Short signal: close < ORL
            elif close < row['or_low']:
                retry_key = (symbol, current_day, 'short')
                retry_count = self.orb_retry_counts.get(retry_key, 0)
                
                if retry_count >= max_retries:
                    continue  # Max retries reached
                
                if vol_spike and rsi_val <= rsi_short_max:
                    signal = Signal(
                        symbol=symbol,
                        dt=row['dt'],
                        side='short',
                        signal_type='ORB',
                        price=close,
                        atr=atr,
                        stop_distance=stop_distance,
                        valid=True
                    )
                    signals.append(signal)
                    self.orb_retry_counts[retry_key] = retry_count + 1
        
        if not signals:
            return pd.DataFrame()
        
        # Convert to DataFrame
        signals_df = pd.DataFrame([
            {
                'symbol': s.symbol,
                'dt': s.dt,
                'side': s.side,
                'signal_type': s.signal_type,
                'price': s.price,
                'atr': s.atr,
                'stop_distance': s.stop_distance,
                'valid': s.valid
            }
            for s in signals
        ])
        
        return signals_df
    
    def scalp_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate Scalp signals (EMA 5/8).
        
        Args:
            df: DataFrame with features computed
            symbol: Symbol name
            
        Returns:
            DataFrame with signals (empty if none)
        """
        scalp_params = self.params.get('scalp', {})
        
        rsi_exhaust_long = scalp_params.get('rsi_exhaust_long', 75)
        rsi_exhaust_short = scalp_params.get('rsi_exhaust_short', 25)
        pct_stop_floor = scalp_params.get('pct_stop_floor', 0.005)
        atr_mult_stop = scalp_params.get('atr_mult_stop', 0.6)
        tp_pct = scalp_params.get('tp_pct', 0.012)
        no_chase = scalp_params.get('no_chase', 0.0035)
        max_scalps = scalp_params.get('max_scalps_per_day_all_pairs', 8)
        
        signals = []
        
        for idx, row in df.iterrows():
            if not row['trade_hours']:
                continue
            
            if not row['rv_ok']:
                continue  # RV too low
            
            if row['gap_flag']:
                continue
            
            current_day = row['day']
            close = row['close']
            ema5 = row['ema5']
            ema8 = row['ema8']
            rsi_val = row['rsi14']
            atr = row['atr14']
            
            # Check daily scalp limit
            scalp_count = self.scalp_count_today.get(current_day, 0)
            if scalp_count >= max_scalps:
                continue
            
            # Calculate stop distance
            stop_distance = max(pct_stop_floor * close, atr_mult_stop * atr)
            
            # Long: close > ema5, ema5 > ema8, RSI not exhausted
            if close > ema5 and ema5 > ema8 and rsi_val <= rsi_exhaust_long:
                # No-chase check
                chase_key = (symbol, current_day)
                last_price = self.last_scalp_price.get(chase_key, close)
                price_move = (close - last_price) / last_price if last_price > 0 else 0
                
                if price_move <= no_chase:
                    signal = Signal(
                        symbol=symbol,
                        dt=row['dt'],
                        side='long',
                        signal_type='SCALP',
                        price=close,
                        atr=atr,
                        stop_distance=stop_distance,
                        valid=True
                    )
                    signals.append(signal)
                    self.scalp_count_today[current_day] = scalp_count + 1
                    self.last_scalp_price[chase_key] = close
            
            # Short: close < ema5, ema5 < ema8, RSI not exhausted
            elif close < ema5 and ema5 < ema8 and rsi_val >= rsi_exhaust_short:
                # No-chase check
                chase_key = (symbol, current_day)
                last_price = self.last_scalp_price.get(chase_key, close)
                price_move = (last_price - close) / last_price if last_price > 0 else 0
                
                if price_move <= no_chase:
                    signal = Signal(
                        symbol=symbol,
                        dt=row['dt'],
                        side='short',
                        signal_type='SCALP',
                        price=close,
                        atr=atr,
                        stop_distance=stop_distance,
                        valid=True
                    )
                    signals.append(signal)
                    self.scalp_count_today[current_day] = scalp_count + 1
                    self.last_scalp_price[chase_key] = close
        
        if not signals:
            return pd.DataFrame()
        
        # Convert to DataFrame
        signals_df = pd.DataFrame([
            {
                'symbol': s.symbol,
                'dt': s.dt,
                'side': s.side,
                'signal_type': s.signal_type,
                'price': s.price,
                'atr': s.atr,
                'stop_distance': s.stop_distance,
                'valid': s.valid
            }
            for s in signals
        ])
        
        return signals_df
    
    def reset_daily_counters(self, current_date: date):
        """Reset daily counters (call at start of each day).
        
        Args:
            current_date: Current date
        """
        # Reset scalp counter for previous days
        keys_to_remove = [d for d in self.scalp_count_today.keys() if d < current_date]
        for key in keys_to_remove:
            del self.scalp_count_today[key]
        
        # Reset last scalp price for previous days
        keys_to_remove = [k for k in self.last_scalp_price.keys() if k[1] < current_date]
        for key in keys_to_remove:
            del self.last_scalp_price[key]



