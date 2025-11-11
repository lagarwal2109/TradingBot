"""DataLoader: streams 1-min data, enforces UTC, detects gaps."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class DataLoader:
    """Loads and normalizes close-only 1-minute candle data."""
    
    def __init__(self, data_dir: Path):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing CSV files (timestamp, price, volume)
        """
        self.data_dir = Path(data_dir)
    
    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load and normalize data for a symbol.
        
        Args:
            symbol: Symbol name (e.g., "BTCUSD")
            
        Returns:
            DataFrame with columns: dt, close, volume, day, minute_idx
        """
        csv_path = self.data_dir / f"{symbol}.csv"
        
        if not csv_path.exists():
            return pd.DataFrame()
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Validate columns
        if 'timestamp' not in df.columns or 'price' not in df.columns:
            raise ValueError(f"{symbol}: Missing required columns (timestamp, price)")
        
        # Convert timestamp (ms) to datetime UTC (optional)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Rename price to close
        df['close'] = df['price'].astype(float)
        
        # Volume (default to 0 if missing)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
        else:
            df['volume'] = 0.0
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates (keep last)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Period index based on row number (treat as bar number)
        df['period'] = np.arange(len(df), dtype=int)
        
        minutes_per_day = 1440
        df['day'] = (df['period'] // minutes_per_day).astype(int)
        df['minute_idx'] = df['period'] % minutes_per_day
        
        # Use period as dt reference (integer)
        df['dt'] = df['period']
        
        # Validate: price must be positive and finite
        df = df[(df['close'] > 0) & np.isfinite(df['close'])]
        
        # Detect gaps (period jumps > 1)
        df['gap_flag'] = df['period'].diff().fillna(1).astype(int) > 1
        
        result = df[['dt', 'timestamp', 'close', 'volume', 'day', 'minute_idx', 'period', 'gap_flag']].copy()
        
        return result
    
    def _detect_gaps(self, dt_series: pd.Series) -> pd.Series:
        """Detect gaps of more than 5 consecutive minutes.
        
        Args:
            dt_series: Datetime series
            
        Returns:
            Boolean series: True if gap detected
        """
        if len(dt_series) < 2:
            return pd.Series([False] * len(dt_series), index=dt_series.index)
        
        # Calculate time differences (should be ~1 minute)
        time_diffs = dt_series.diff().dt.total_seconds() / 60.0
        
        # Gap if difference > 5 minutes
        gaps = time_diffs > 5.0
        
        return gaps.fillna(False)
    
    def load_multiple(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols.
        
        Args:
            symbols: List of symbol names
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            df = self.load_symbol(symbol)
            if len(df) > 0:
                data[symbol] = df
        return data
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from data directory.
        
        Returns:
            List of symbol names (without .csv extension)
        """
        symbols = []
        for csv_file in self.data_dir.glob("*.csv"):
            if csv_file.stem != "state":  # Skip state file
                symbols.append(csv_file.stem)
        return sorted(symbols)

