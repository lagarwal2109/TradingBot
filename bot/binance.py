"""Binance public API client for historical data (no authentication required)."""

import time
import requests
from typing import Dict, List, Optional
from pydantic import BaseModel


class BinanceKline(BaseModel):
    """Binance kline/candlestick data."""
    timestamp: int  # Open time
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int


class BinancePublicClient:
    """Public Binance API client for historical market data (no auth needed)."""
    
    def __init__(self, base_url: str = "https://api.binance.com"):
        """Initialize Binance public client."""
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json"
        })
    
    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[BinanceKline]:
        """Get kline/candlestick data.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval - "1m", "5m", "15m", "1h", "4h", "1d", etc.
            start_time: Start time in milliseconds
            end_time: End time in milliseconds  
            limit: Number of klines (max 1000)
            
        Returns:
            List of kline data
        """
        endpoint = "/api/v3/klines"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            
            klines = []
            for k in response.json():
                try:
                    klines.append(BinanceKline(
                        timestamp=int(k[0]),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                        close_time=int(k[6]),
                        quote_volume=float(k[7]),
                        trades=int(k[8])
                    ))
                except Exception as e:
                    print(f"Error parsing kline: {e}")
                    continue
                    
            return klines
            
        except requests.exceptions.RequestException as e:
            print(f"Binance API error: {e}")
            return []
    
    def get_historical_data(
        self,
        symbol: str,
        days: int = 7,
        interval: str = "1h"
    ) -> List[BinanceKline]:
        """Get historical data for specified days.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            days: Number of days
            interval: Data interval
            
        Returns:
            List of klines
        """
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_klines = []
        current_start = start_time
        
        # Binance limits to 1000 klines per request
        while current_start < end_time:
            klines = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000
            )
            
            if not klines:
                break
                
            all_klines.extend(klines)
            
            # Update start time
            if klines:
                current_start = klines[-1].close_time + 1
            else:
                break
                
        return all_klines
    
    def backfill_multiple_assets(
        self,
        assets: List[str],
        days: int = 7,
        interval: str = "1h"
    ) -> Dict[str, List[BinanceKline]]:
        """Backfill data for multiple assets.
        
        Args:
            assets: List of base assets (e.g., ["BTC", "ETH"])
            days: Days to backfill
            interval: Data interval
            
        Returns:
            Dictionary mapping symbol to klines
        """
        all_data = {}
        
        for asset in assets:
            symbol = f"{asset}USDT"  # Binance uses USDT pairs
            
            try:
                print(f"Fetching {symbol}...", end=" ", flush=True)
                klines = self.get_historical_data(symbol, days, interval)
                
                if klines:
                    all_data[asset] = klines
                    print(f"✓ {len(klines)} candles")
                else:
                    print("❌ No data")
                    
                # Respect rate limits
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
                
        return all_data
