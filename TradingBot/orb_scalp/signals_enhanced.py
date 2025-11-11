"""Enhanced SignalEngine with gate tracking and diagnostic mode."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date
from collections import defaultdict

from .signals import Signal


class SignalEngineEnhanced:
    """Enhanced signal engine with gate tracking and adaptive filters."""
    
    def __init__(self, params: Dict):
        """Initialize enhanced signal engine.
        
        Args:
            params: Configuration parameters
        """
        self.params = params
        self.diagnostic_mode = params.get('diagnostic_mode', False)
        self.orb_retry_counts: Dict[tuple, int] = {}
        self.scalp_count_today: Dict[date, int] = {}
        self.last_scalp_price: Dict[tuple, float] = {}
        
        # Gate tracking for diagnostics
        self.gate_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.failed_signals: List[Dict] = []
    
    def orb_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        track_gates: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate ORB signals with gate tracking.
        
        Args:
            df: DataFrame with features computed
            symbol: Symbol name
            track_gates: Whether to track gate statistics
            
        Returns:
            Tuple of (signals_df, gate_stats_df)
        """
        orb_params = self.params.get('orb', {})
        
        rsi_long_min = orb_params.get('rsi_long_min', 55)
        rsi_short_max = orb_params.get('rsi_short_max', 45)
        max_retries = orb_params.get('max_retries_per_side', 2)
        
        if self.diagnostic_mode:
            rsi_long_min = orb_params.get('rsi_long_min', 52)
            rsi_short_max = orb_params.get('rsi_short_max', 48)
            max_retries = orb_params.get('max_retries_per_side', 3)
        
        signals = []
        gate_stats_list = []
        
        rows_to_process = df if track_gates else df.tail(1)
        
        for idx, row in rows_to_process.iterrows():
            current_day = row['day']
            close = row['close']
            rsi_val = row['rsi14']
            
            # Gate 1: Basic filters
            gate1 = (
                row.get('hours_ok', row.get('trade_hours', False)) and
                row.get('gap_ok_orb', True) and
                row.get('warmup_ok', True)
            )
            
            if track_gates:
                gate_stats_list.append({
                    'dt': row['dt'],
                    'symbol': symbol,
                    'gate1_basic': gate1
                })
            
            if not gate1:
                if track_gates:
                    self.gate_stats[symbol]['orb_gate1_fail'] += 1
                continue
            
            # Gate 2: Opening range valid
            gate2 = gate1 and row.get('or_valid', False)
            
            if track_gates:
                gate_stats_list[-1]['gate2_or_valid'] = gate2
            
            if not gate2:
                if track_gates:
                    self.gate_stats[symbol]['orb_gate2_fail'] += 1
                continue
            
            # Gate 3: Breakout detected
            orb_break_long = close > row.get('or_high', 0)
            orb_break_short = close < row.get('or_low', float('inf'))
            gate3 = gate2 and (orb_break_long or orb_break_short)
            
            if track_gates:
                gate_stats_list[-1]['gate3_breakout'] = gate3
                gate_stats_list[-1]['breakout_long'] = orb_break_long
                gate_stats_list[-1]['breakout_short'] = orb_break_short
            
            if not gate3:
                if track_gates:
                    self.gate_stats[symbol]['orb_gate3_fail'] += 1
                continue
            
            # Gate 4: Volume spike
            gate4 = gate3 and row.get('vol_spike', False)
            
            if track_gates:
                gate_stats_list[-1]['gate4_vol_spike'] = gate4
            
            if not gate4:
                if track_gates:
                    self.gate_stats[symbol]['orb_gate4_fail'] += 1
                continue
            
            # Gate 5: RSI filter
            rsi_long_ok = orb_break_long and rsi_val >= rsi_long_min
            rsi_short_ok = orb_break_short and rsi_val <= rsi_short_max
            gate5 = gate4 and (rsi_long_ok or rsi_short_ok)
            
            if track_gates:
                gate_stats_list[-1]['gate5_rsi'] = gate5
                gate_stats_list[-1]['rsi_val'] = rsi_val
            
            if not gate5:
                if track_gates:
                    self.gate_stats[symbol]['orb_gate5_fail'] += 1
                continue
            
            # Gate 6: Retry limit
            side = 'long' if orb_break_long else 'short'
            retry_key = (symbol, current_day, side)
            retry_count = self.orb_retry_counts.get(retry_key, 0)
            gate6 = gate5 and (retry_count < max_retries)
            
            if track_gates:
                gate_stats_list[-1]['gate6_retry'] = gate6
                gate_stats_list[-1]['retry_count'] = retry_count
            
            if not gate6:
                if track_gates:
                    self.gate_stats[symbol]['orb_gate6_fail'] += 1
                continue
            
            # All gates passed - create signal
            atr = row.get('atr14', 0.0)
            stop_atr_mult = orb_params.get('stop_atr_mult', 0.8)
            pct_stop_floor = orb_params.get('pct_stop_floor', 0.005)
            stop_distance = max(pct_stop_floor * close, stop_atr_mult * atr)
            
            signal = Signal(
                symbol=symbol,
                dt=row['dt'],
                side=side,
                signal_type='ORB',
                price=close,
                atr=atr,
                stop_distance=stop_distance,
                valid=True
            )
            signals.append(signal)
            self.orb_retry_counts[retry_key] = retry_count + 1
            
            if track_gates:
                gate_stats_list[-1]['signal_generated'] = True
                self.gate_stats[symbol]['orb_signals'] += 1
        
        if not signals:
            signals_df = pd.DataFrame()
        else:
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
        
        gate_stats_df = pd.DataFrame(gate_stats_list) if gate_stats_list else pd.DataFrame()
        
        return signals_df, gate_stats_df
    
    def scalp_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        track_gates: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate Scalp signals with gate tracking.
        
        Args:
            df: DataFrame with features computed
            symbol: Symbol name
            track_gates: Whether to track gate statistics
            
        Returns:
            Tuple of (signals_df, gate_stats_df)
        """
        scalp_params = self.params.get('scalp', {})
        
        rsi_exhaust_long = scalp_params.get('rsi_exhaust_long', 75)
        rsi_exhaust_short = scalp_params.get('rsi_exhaust_short', 25)
        no_chase = scalp_params.get('no_chase', 0.006)  # 0.6% - crypto routinely jumps >0.35% in a minute
        
        signals = []
        gate_stats_list = []
        
        rows_to_process = df if track_gates else df.tail(1)
        
        for idx, row in rows_to_process.iterrows():
            current_day = row['day']
            close = row['close']
            ema5 = row.get('ema5', 0)
            ema8 = row.get('ema8', 0)
            rsi_val = row.get('rsi14', 50)
            atr = row.get('atr14', 0.0)
            
            # Gate 1: Basic filters
            gate1 = (
                row.get('hours_ok', row.get('trade_hours', False)) and
                row.get('gap_ok_scalp', True) and
                row.get('warmup_ok', True)
            )
            
            if track_gates:
                gate_stats_list.append({
                    'dt': row['dt'],
                    'symbol': symbol,
                    'gate1_basic': gate1
                })
            
            if not gate1:
                if track_gates:
                    self.gate_stats[symbol]['scalp_gate1_fail'] += 1
                continue
            
            # Gate 2: RV check
            gate2 = gate1 and row.get('rv_ok', False)
            
            if track_gates:
                gate_stats_list[-1]['gate2_rv'] = gate2
            
            if not gate2:
                if track_gates:
                    self.gate_stats[symbol]['scalp_gate2_fail'] += 1
                continue
            
            # Gate 3: EMA stack alignment (removed daily limit - moved to portfolio layer)
            ema_stack_long = close > ema5 and ema5 > ema8
            ema_stack_short = close < ema5 and ema5 < ema8
            gate3 = gate2 and (ema_stack_long or ema_stack_short)
            
            if track_gates:
                gate_stats_list[-1]['gate3_ema_stack'] = gate3
                gate_stats_list[-1]['ema_stack_long'] = ema_stack_long
                gate_stats_list[-1]['ema_stack_short'] = ema_stack_short
            
            if not gate3:
                if track_gates:
                    self.gate_stats[symbol]['scalp_gate3_fail'] += 1
                continue
            
            # Gate 4: RSI exhaustion check
            rsi_long_ok = ema_stack_long and rsi_val <= rsi_exhaust_long
            rsi_short_ok = ema_stack_short and rsi_val >= rsi_exhaust_short
            gate4 = gate3 and (rsi_long_ok or rsi_short_ok)
            
            if track_gates:
                gate_stats_list[-1]['gate4_rsi'] = gate4
                gate_stats_list[-1]['rsi_val'] = rsi_val
            
            if not gate4:
                if track_gates:
                    self.gate_stats[symbol]['scalp_gate4_fail'] += 1
                continue
            
            # Gate 5: No-chase check
            side = 'long' if ema_stack_long else 'short'
            chase_key = (symbol, current_day)
            last_price = self.last_scalp_price.get(chase_key, close)
            
            if side == 'long':
                price_move = (close - last_price) / last_price if last_price > 0 else 0
            else:
                price_move = (last_price - close) / last_price if last_price > 0 else 0
            
            gate5 = gate4 and (price_move <= no_chase)
            
            if track_gates:
                gate_stats_list[-1]['gate5_no_chase'] = gate5
                gate_stats_list[-1]['price_move'] = price_move
            
            if not gate5:
                if track_gates:
                    self.gate_stats[symbol]['scalp_gate5_fail'] += 1
                continue
            
            # All gates passed - create signal
            pct_stop_floor = scalp_params.get('pct_stop_floor', 0.005)
            atr_mult_stop = scalp_params.get('atr_mult_stop', 0.6)
            stop_distance = max(pct_stop_floor * close, atr_mult_stop * atr)
            
            signal = Signal(
                symbol=symbol,
                dt=row['dt'],
                side=side,
                signal_type='SCALP',
                price=close,
                atr=atr,
                stop_distance=stop_distance,
                valid=True
            )
            signals.append(signal)
            # Don't increment scalp_count_today here - moved to portfolio layer
            self.last_scalp_price[chase_key] = close
            
            if track_gates:
                gate_stats_list[-1]['signal_generated'] = True
                self.gate_stats[symbol]['scalp_signals'] += 1
        
        if not signals:
            signals_df = pd.DataFrame()
        else:
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
        
        gate_stats_df = pd.DataFrame(gate_stats_list) if gate_stats_list else pd.DataFrame()
        
        return signals_df, gate_stats_df
    
    def reset_daily_counters(self, current_date: date):
        """Reset daily counters."""
        keys_to_remove = [d for d in self.scalp_count_today.keys() if d < current_date]
        for key in keys_to_remove:
            del self.scalp_count_today[key]
        
        keys_to_remove = [k for k in self.last_scalp_price.keys() if k[1] < current_date]
        for key in keys_to_remove:
            del self.last_scalp_price[key]
    
    def get_gate_summary(self) -> Dict:
        """Get summary of gate statistics.
        
        Returns:
            Dictionary with gate pass/fail counts
        """
        return dict(self.gate_stats)

