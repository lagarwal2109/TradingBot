"""Data persistence and state management for the trading bot."""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel
from bot.config import get_config


class BotState(BaseModel):
    """Bot state model."""
    current_position: Optional[str] = None  # Current coin symbol or None for USD
    last_trade_ts: int = 0  # Last trade timestamp in milliseconds
    consecutive_errors: int = 0
    last_exchange_info_update: int = 0  # Timestamp of last exchange info update
    

class DataStore:
    """Manages data persistence for minute bars and bot state."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the data store."""
        config = get_config()
        self.data_dir = data_dir or config.data_dir
        self.data_dir.mkdir(exist_ok=True)
        
        # State file path
        self.state_file = self.data_dir / "state.json"
        
        # Initialize or load state
        self.state = self._load_state()
    
    def _load_state(self) -> BotState:
        """Load bot state from file or create new."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                return BotState(**data)
            except Exception as e:
                print(f"Error loading state, creating new: {e}")
        
        return BotState()
    
    def save_state(self) -> None:
        """Save current bot state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state.dict(), f, indent=2)
    
    def update_state(self, **kwargs) -> None:
        """Update state fields and save."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        self.save_state()
    
    def get_csv_path(self, pair: str) -> Path:
        """Get CSV file path for a trading pair."""
        # Clean pair name for filename
        safe_pair = pair.replace("/", "_").replace("\\", "_")
        return self.data_dir / f"{safe_pair}.csv"
    
    def append_minute_bar(
        self, 
        pair: str, 
        timestamp: int, 
        price: float, 
        volume: float = 0,
        open: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None
    ) -> None:
        """Append a minute bar to the pair's CSV file.
        
        Args:
            pair: Trading pair
            timestamp: Timestamp in milliseconds
            price: Close price
            volume: Volume
            open: Open price (optional, for OHLCV format)
            high: High price (optional, for OHLCV format)
            low: Low price (optional, for OHLCV format)
        """
        csv_path = self.get_csv_path(pair)
        
        # Check if file exists and what format it uses
        file_exists = csv_path.exists()
        has_ohlcv = False
        
        if file_exists:
            # Check if file has OHLCV format by reading first line
            try:
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header and "open" in header and "high" in header and "low" in header:
                        has_ohlcv = True
            except Exception:
                pass  # If we can't read, assume old format
        
        # Append data (file is automatically flushed when context exits)
        with open(csv_path, "a", newline="", buffering=1) as f:  # Line buffering for immediate write
            writer = csv.writer(f)
            
            # Write header if new file
            if not file_exists:
                # Use OHLCV format if open/high/low provided, otherwise use old format
                if open is not None and high is not None and low is not None:
                    writer.writerow(["timestamp", "open", "high", "low", "price", "volume"])
                    has_ohlcv = True
                else:
                    writer.writerow(["timestamp", "price", "volume"])
            
            # Write data row - match existing format
            if has_ohlcv:
                # Use provided OHLCV or fill with price if not available
                open_val = open if open is not None else price
                high_val = high if high is not None else price
                low_val = low if low is not None else price
                writer.writerow([timestamp, open_val, high_val, low_val, price, volume])
            else:
                # Old format: just timestamp, price, volume
                writer.writerow([timestamp, price, volume])
            f.flush()  # Ensure data is written immediately
    
    def collect_minute_bars(self, ticker_data: List[Dict[str, any]]) -> None:
        """Collect and store minute bars for all pairs."""
        timestamp = int(datetime.now().timestamp() * 1000)
        
        for ticker in ticker_data:
            pair = ticker["pair"]
            price = ticker["price"]
            volume = ticker.get("volume_24h", 0)  # Use 24h volume as proxy
            self.append_minute_bar(pair, timestamp, price, volume)
    
    def load_and_clean(self, pair: str, bar: str = '1h') -> pd.DataFrame:
        """Load and clean data with uniform bars and monotonic time.
        
        Args:
            pair: Trading pair
            bar: Resample frequency (default '1h')
            
        Returns:
            Cleaned DataFrame with OHLCV (DatetimeIndex preserved for resampling)
        """
        csv_path = self.get_csv_path(pair)
        
        if not csv_path.exists():
            return pd.DataFrame()
        
        raw = pd.read_csv(csv_path)
        
        # Type conversion
        raw['timestamp'] = pd.to_numeric(raw['timestamp'], errors='coerce')
        raw['price'] = pd.to_numeric(raw['price'], errors='coerce')
        if 'volume' in raw.columns:
            raw['volume'] = pd.to_numeric(raw['volume'], errors='coerce').fillna(0.0)
        else:
            raw['volume'] = 0.0
        
        # Handle OHLCV if available
        if 'open' in raw.columns:
            raw['open'] = pd.to_numeric(raw['open'], errors='coerce')
        if 'high' in raw.columns:
            raw['high'] = pd.to_numeric(raw['high'], errors='coerce')
        if 'low' in raw.columns:
            raw['low'] = pd.to_numeric(raw['low'], errors='coerce')
        
        # ms epoch → UTC and strict ordering - KEEP DATETIME INDEX for resampling
        df = (
            raw.assign(ts=pd.to_datetime(raw['timestamp'], unit='ms', utc=True, errors='coerce'))
            .dropna(subset=['ts', 'price'])
            .sort_values('ts')
            .drop_duplicates('ts', keep='last')
            .set_index('ts')
        )
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Uniform grid - resample to bar frequency (requires DatetimeIndex)
        if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
            # Full OHLCV available
            ohlc = df[['open', 'high', 'low', 'price']].resample(bar).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'price': 'last'
            })
            ohlc.columns = ['open', 'high', 'low', 'close']
        else:
            # Price-only: create OHLC from price
            ohlc = df['price'].resample(bar).ohlc()
        
        vol = df['volume'].resample(bar).sum(min_count=1) if 'volume' in df.columns else pd.Series(0.0, index=ohlc.index)
        out = ohlc.join(vol.rename('volume'))
        
        # Rename close to price for compatibility
        if 'close' in out.columns:
            out['price'] = out['close']
        
        # Widen synthetic ranges where missing/flat
        m = out['high'].isna() | out['low'].isna() | (out['high'] == out['low'])
        if m.any():
            close = out.loc[m, 'price'] if 'price' in out.columns else out.loc[m, 'close']
            rv = close.pct_change(fill_method=None).abs().rolling(20, min_periods=5).mean().fillna(0)
            span = (0.002 + 2.5 * rv).clip(0.002, 0.05)  # 0.2%..5%
            out.loc[m, 'open'] = close
            out.loc[m, 'high'] = close * (1 + span)
            out.loc[m, 'low'] = close * (1 - span)
            if 'price' not in out.columns:
                out['price'] = out['close']
        
        # Final validity check
        out = out.dropna(subset=['price'])
        out = out[(out['price'] > 0) & (out['high'] > 0) & (out['low'] > 0)]
        
        # NOTE: Keep DatetimeIndex for HTF regime computation
        # Convert to sequential periods AFTER all resampling is done (in backtester)
        
        return out
    
    def read_minute_bars(self, pair: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Read minute bars for a pair with cleaning and resampling.
        
        Args:
            pair: Trading pair
            limit: Number of most recent bars to return
        
        Returns:
            DataFrame with timestamp, price, and optionally open/high/low/volume columns
            (backward compatible with price-only format)
        """
        # Use load_and_clean for better data quality
        df = self.load_and_clean(pair, bar='1h')  # Resample to 1h bars
        
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "price"])
        
        # Return limited rows if specified
        if limit:
            return df.tail(limit)
        
        return df
    
    def get_all_pairs_with_data(self) -> List[str]:
        """Get list of all pairs that have data files."""
        pairs = []
        for csv_file in self.data_dir.glob("*.csv"):
            if csv_file.stem != "state":  # Exclude state file
                # Convert filename back to pair name
                pair = csv_file.stem.replace("_", "/")
                pairs.append(pair)
        return pairs
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get the latest price for all pairs with data."""
        prices = {}
        
        for pair in self.get_all_pairs_with_data():
            df = self.read_minute_bars(pair, limit=1)
            if not df.empty:
                prices[pair] = df["price"].iloc[-1]
        
        return prices
    
    def check_trade_allowed(self) -> bool:
        """Check if a trade is allowed based on the one-minute rule."""
        current_time = int(datetime.now().timestamp() * 1000)
        time_since_last_trade = (current_time - self.state.last_trade_ts) / 1000  # seconds
        
        # Allow trade if at least 60 seconds have passed
        return time_since_last_trade >= 60
    
    def record_trade(self, position: Optional[str] = None) -> None:
        """Record that a trade was made."""
        self.update_state(
            current_position=position,
            last_trade_ts=int(datetime.now().timestamp() * 1000)
        )
    
    def increment_error_count(self) -> int:
        """Increment consecutive error count and return new value."""
        new_count = self.state.consecutive_errors + 1
        self.update_state(consecutive_errors=new_count)
        return new_count
    
    def reset_error_count(self) -> None:
        """Reset consecutive error count."""
        self.update_state(consecutive_errors=0)
    
    def should_update_exchange_info(self) -> bool:
        """Check if exchange info should be updated (weekly)."""
        current_time = int(datetime.now().timestamp() * 1000)
        time_since_update = (current_time - self.state.last_exchange_info_update) / 1000  # seconds
        
        # Update if more than 7 days have passed
        return time_since_update >= (7 * 24 * 60 * 60)
    
    def mark_exchange_info_updated(self) -> None:
        """Mark that exchange info was updated."""
        self.update_state(last_exchange_info_update=int(datetime.now().timestamp() * 1000))
    
    def read_aggregated_bars(self, pair: str, interval: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Read minute bars and aggregate to specified interval.
        
        Args:
            pair: Trading pair
            interval: Aggregation interval ('4h', '1d', or minutes as int)
            limit: Number of most recent aggregated bars to return
        
        Returns:
            DataFrame with timestamp, price, volume columns aggregated to interval
        """
        # Read minute bars
        df = self.read_minute_bars(pair, limit=None)
        
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "price", "volume"])
        
        # Ensure volume column exists
        if "volume" not in df.columns:
            df["volume"] = 0.0
        
        # Convert interval to timedelta
        if interval == "4h":
            freq = "4h"  # Use lowercase 'h' instead of 'H' for pandas compatibility
        elif interval == "1d" or interval == "1D":
            freq = "1D"
        elif isinstance(interval, int):
            freq = f"{interval}min"
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # Resample to aggregated interval
        # Use OHLC aggregation: open (first), high (max), low (min), close (last)
        # Volume: sum
        aggregated = df.resample(freq).agg({
            "price": ["first", "max", "min", "last"],
            "volume": "sum"
        })
        
        # Flatten column names - handle MultiIndex columns
        if isinstance(aggregated.columns, pd.MultiIndex):
            # Flatten MultiIndex to simple column names
            aggregated.columns = ["open", "high", "low", "close", "volume"]
        else:
            # If already flat, check what we have
            if len(aggregated.columns) >= 4:
                # Assume first 4 are OHLC, last is volume
                col_names = list(aggregated.columns)
                if "close" not in col_names and len(col_names) >= 4:
                    aggregated.columns = ["open", "high", "low", "close", "volume"][:len(col_names)]
        
        # Use close price as primary price
        if "close" in aggregated.columns:
            # Convert to Series to avoid assignment issues
            aggregated = aggregated.copy()
            aggregated["price"] = pd.Series(aggregated["close"].values, index=aggregated.index)
        elif len(aggregated.columns) >= 4:
            # Fallback: use last numeric column as price
            numeric_cols = aggregated.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                aggregated = aggregated.copy()
                aggregated["price"] = pd.Series(aggregated[numeric_cols[-1]].values, index=aggregated.index)
            else:
                aggregated = aggregated.copy()
                aggregated["price"] = pd.Series([0.0] * len(aggregated), index=aggregated.index)
        
        # Ensure volume column exists
        if "volume" not in aggregated.columns:
            aggregated["volume"] = pd.Series([0.0] * len(aggregated), index=aggregated.index)
        
        # Select columns
        result = aggregated[["price", "volume"]].copy()
        result.index.name = "timestamp"
        result = result.reset_index()
        
        # Return limited rows if specified
        if limit:
            return result.tail(limit)
        
        return result
    
    def get_data_range(self, pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get data for a specific date range.
        
        Args:
            pair: Trading pair
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            DataFrame with data in the specified range
        """
        df = self.read_minute_bars(pair, limit=None)
        
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "price", "volume"])
        
        # Ensure volume column exists
        if "volume" not in df.columns:
            df["volume"] = 0.0
        
        # Filter by date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df[mask]