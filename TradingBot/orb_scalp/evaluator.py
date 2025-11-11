"""Evaluator: computes metrics, walk-forward, stress tests."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class Evaluator:
    """Computes performance metrics and runs walk-forward tests."""
    
    def __init__(self, trades: List, equity_curve: List):
        """Initialize evaluator.
        
        Args:
            trades: List of Trade objects
            equity_curve: List of (dt, equity) tuples
        """
        self.trades = trades
        self.equity_curve = equity_curve
    
    def compute_metrics(self) -> Dict:
        """Compute all performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        if not self.trades:
            return self._empty_metrics()
        
        # Convert to DataFrames
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'side': t.side,
                'signal_type': t.signal_type,
                'entry_dt': t.entry_dt,
                'exit_dt': t.exit_dt,
                'pnl_abs': t.pnl_abs,
                'pnl_pct': t.pnl_pct,
                'pnl_r': t.pnl_r,
                'fees': t.fees,
                'exit_reason': t.exit_reason
            }
            for t in self.trades
        ])
        
        equity_df = pd.DataFrame(
            self.equity_curve,
            columns=['dt', 'equity']
        )
        
        if equity_df.empty:
            return self._empty_metrics()
        
        # Normalize timeline to datetimelike (support numeric periods)
        equity_df['dt'] = self._normalize_time_series(equity_df['dt'], unit='m')
        equity_df = equity_df.sort_values('dt').set_index('dt')
        
        # Basic metrics
        initial_capital = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['returns'].dropna()
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_daily = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_daily = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_daily = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_daily = sharpe_daily if sharpe_daily > 0 else 0.0
        
        # Max drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Calmar ratio
        period_years = self._compute_period_years(equity_df.index)
        if period_years > 0:
            annualized_return = ((final_equity / initial_capital) ** (1 / period_years) - 1) * 100
        else:
            annualized_return = total_return
        
        calmar = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0
        
        # Trade metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_abs'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_abs'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        avg_win = trades_df[trades_df['pnl_abs'] > 0]['pnl_abs'].mean() if winning_trades > 0 else 0.0
        avg_loss = trades_df[trades_df['pnl_abs'] < 0]['pnl_abs'].mean() if losing_trades > 0 else 0.0
        
        # Profit factor
        total_profit = trades_df[trades_df['pnl_abs'] > 0]['pnl_abs'].sum()
        total_loss = abs(trades_df[trades_df['pnl_abs'] < 0]['pnl_abs'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Average R
        avg_r = trades_df['pnl_r'].mean()
        median_r = trades_df['pnl_r'].median()
        
        # 5% CVaR (Conditional Value at Risk)
        if len(daily_returns) > 0:
            cvar_5 = daily_returns.quantile(0.05)
        else:
            cvar_5 = 0.0
        
        # Attribution
        orb_trades = trades_df[trades_df['signal_type'] == 'ORB']
        scalp_trades = trades_df[trades_df['signal_type'] == 'SCALP']
        
        orb_pnl = orb_trades['pnl_abs'].sum() if len(orb_trades) > 0 else 0.0
        scalp_pnl = scalp_trades['pnl_abs'].sum() if len(scalp_trades) > 0 else 0.0
        
        # By symbol
        symbol_pnl = trades_df.groupby('symbol')['pnl_abs'].sum().to_dict()
        
        # Trades per day
        if not trades_df.empty:
            trades_df['entry_dt'] = self._normalize_time_series(trades_df['entry_dt'], unit='m')
            trades_df['exit_dt'] = self._normalize_time_series(trades_df['exit_dt'], unit='m')
            
            if pd.api.types.is_timedelta64_dtype(trades_df['entry_dt']):
                trades_df['day'] = trades_df['entry_dt'].dt.components.days
            else:
                trades_df['day'] = trades_df['entry_dt'].dt.floor('D')
            trades_per_day = trades_df.groupby('day').size().mean() if len(trades_df) > 0 else 0.0
        else:
            trades_per_day = 0.0
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'sharpe_daily': sharpe_daily,
            'sortino_daily': sortino_daily,
            'calmar_ratio': calmar,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_r': avg_r,
            'median_r': median_r,
            'cvar_5_pct': cvar_5 * 100,
            'trades_per_day': trades_per_day,
            'orb_pnl': orb_pnl,
            'scalp_pnl': scalp_pnl,
            'symbol_pnl': symbol_pnl,
            'total_fees': trades_df['fees'].sum() if len(trades_df) > 0 else 0.0
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'initial_capital': 0.0,
            'final_equity': 0.0,
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'sharpe_daily': 0.0,
            'sortino_daily': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'avg_r': 0.0,
            'median_r': 0.0,
            'cvar_5_pct': 0.0,
            'trades_per_day': 0.0,
            'orb_pnl': 0.0,
            'scalp_pnl': 0.0,
            'symbol_pnl': {},
            'total_fees': 0.0
        }

    def _normalize_time_series(self, series: pd.Series, unit: str = 'm') -> pd.Series:
        """Convert numeric sequences to timedeltas and ensure datetimelike series."""
        if series.empty:
            return series
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series)
        
        if pd.api.types.is_timedelta64_dtype(series):
            return series
        
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().all():
            return pd.to_timedelta(numeric, unit=unit)
        
        # Try datetime conversion
        converted_datetime = pd.to_datetime(series, errors='coerce')
        if converted_datetime.notna().all():
            return converted_datetime
        
        # Fallback: mix of numeric and datetime-like
        result = pd.Series(index=series.index, dtype='timedelta64[ns]')
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            result.loc[numeric_mask] = pd.to_timedelta(numeric[numeric_mask], unit=unit)
        datetime_mask = ~numeric_mask
        if datetime_mask.any():
            result.loc[datetime_mask] = converted_datetime[datetime_mask]
        
        return result
    
    def _compute_period_years(self, index: pd.Index) -> float:
        """Compute span in years from a timeline index."""
        if len(index) < 2:
            return 0.0
        
        if isinstance(index, pd.DatetimeIndex):
            total_seconds = (index[-1] - index[0]).total_seconds()
            return total_seconds / (60 * 60 * 24 * 365.25)
        
        if isinstance(index, pd.TimedeltaIndex):
            total_seconds = (index[-1] - index[0]).total_seconds()
            return total_seconds / (60 * 60 * 24 * 365.25)
        
        # Numeric or other types (assume 1-minute bars)
        try:
            values = index.astype(float)
            total_minutes = values[-1] - values[0]
        except Exception:
            total_minutes = len(index)
        
        bars_per_year = 60 * 24 * 365.25
        return total_minutes / bars_per_year

