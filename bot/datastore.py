"""Data persistence and state management for the trading bot."""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel
from bot.config import get_config


class BotState(BaseModel):
    """Bot state model."""
    current_position: Optional[str] = None  # Current coin symbol or None for USD
    last_trade_ts: int = 0  # Last trade timestamp in milliseconds
    consecutive_errors: int = 0
    last_exchange_info_update: int = 0  # Timestamp of last exchange info update
    # Track recent trade timestamps (ms) for rate-limiting (e.g., max 5 trades per minute)
    recent_trades: List[int] = []
    

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
    
    def preload_from_data_folder(self, max_rows: int = 6000) -> int:
        """Preload/normalize existing CSVs into minute-bar schema for immediate use.
        
        This ensures files under data_dir use the expected columns: timestamp (ms), price, volume.
        If a file already has these columns, it is left as-is. If it has OHLCV columns
        (e.g., open_time, open, high, low, close, volume, close_time), it will be converted
        to minute-bar format using open_time as timestamp and close as price.
        
        Returns:
            Number of files normalized/preloaded.
        """
        normalized = 0
        for csv_file in self.data_dir.glob("*.csv"):
            if csv_file.name == "state.json":
                continue
            try:
                df = pd.read_csv(csv_file)
                # Already in expected format
                if {"timestamp", "price"}.issubset(set(df.columns)):
                    # Optionally trim to last max_rows to reduce footprint
                    if len(df) > max_rows:
                        df = df.tail(max_rows)
                        df.to_csv(csv_file, index=False)
                        normalized += 1
                    continue
                
                # Try Binance-style schema
                if "open_time" in df.columns and "close" in df.columns:
                    ts_col = "open_time"
                    price_col = "close"
                    vol_col = "volume" if "volume" in df.columns else None
                    
                    out = pd.DataFrame({
                        "timestamp": df[ts_col].astype("int64"),
                        "price": df[price_col].astype("float64"),
                    })
                    if vol_col and vol_col in df.columns:
                        out["volume"] = df[vol_col].astype("float64")
                    else:
                        out["volume"] = 0.0
                    
                    if len(out) > max_rows:
                        out = out.tail(max_rows)
                    out.to_csv(csv_file, index=False)
                    normalized += 1
                    continue
                
                # Fallback: if a 'price' like column exists, try to synthesize
                for candidate in ["close", "last_price", "price"]:
                    if candidate in df.columns:
                        price_series = df[candidate].astype("float64")
                        # Timestamp fallback: try 'timestamp' (ms) or index-as-seq
                        if "timestamp" in df.columns:
                            ts_series = df["timestamp"].astype("int64")
                        elif "close_time" in df.columns:
                            ts_series = df["close_time"].astype("int64")
                        elif "open_time" in df.columns:
                            ts_series = df["open_time"].astype("int64")
                        else:
                            # Generate synthetic timestamps: 1 min apart ending now
                            end_ms = int(datetime.now().timestamp() * 1000)
                            count = len(df)
                            ts_series = pd.Series([end_ms - 60_000 * (count - i) for i in range(count)], dtype="int64")
                        
                        out = pd.DataFrame({
                            "timestamp": ts_series,
                            "price": price_series,
                            "volume": 0.0
                        })
                        if len(out) > max_rows:
                            out = out.tail(max_rows)
                        out.to_csv(csv_file, index=False)
                        normalized += 1
                        break
                
            except Exception as e:
                print(f"Preload warning for {csv_file.name}: {e}")
                continue
        return normalized
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
    
    def append_minute_bar(self, pair: str, timestamp: int, price: float, volume: float = 0) -> None:
        """Append a minute bar to the pair's CSV file."""
        csv_path = self.get_csv_path(pair)
        
        # Check if file exists to determine if we need headers
        file_exists = csv_path.exists()
        
        # Append data (file is automatically flushed when context exits)
        with open(csv_path, "a", newline="", buffering=1) as f:  # Line buffering for immediate write
            writer = csv.writer(f)
            
            # Write header if new file
            if not file_exists:
                writer.writerow(["timestamp", "price", "volume"])
            
            # Write data row
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
    
    def read_minute_bars(self, pair: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Read minute bars for a pair.
        
        Args:
            pair: Trading pair
            limit: Number of most recent bars to return
        
        Returns:
            DataFrame with timestamp and price columns
        """
        csv_path = self.get_csv_path(pair)
        
        if not csv_path.exists():
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["timestamp", "price"])
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp").sort_index()
        
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
    
    def check_trade_allowed(self, max_trades_per_minute: int = 5) -> bool:
        """Check if a trade is allowed based on rate limits (e.g., 5 per rolling minute)."""
        current_time = int(datetime.now().timestamp() * 1000)
        window_start = current_time - 60_000  # last 60 seconds
        
        # Prune old entries
        recent = [ts for ts in (self.state.recent_trades or []) if ts >= window_start]
        self.state.recent_trades = recent
        self.save_state()
        
        return len(recent) < max_trades_per_minute
    
    def record_trade(self, position: Optional[str] = None) -> None:
        """Record that a trade was made (updates timestamps for rate limiting)."""
        now_ms = int(datetime.now().timestamp() * 1000)
        window_start = now_ms - 60_000
        recent = [ts for ts in (self.state.recent_trades or []) if ts >= window_start]
        recent.append(now_ms)
        self.update_state(
            current_position=position,
            last_trade_ts=now_ms,
            recent_trades=recent
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
