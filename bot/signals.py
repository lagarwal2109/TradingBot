"""Signal generation and feature computation for trading strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from bot.config import get_config
from bot.datastore import DataStore


class SignalGenerator:
    """Generate trading signals based on price data."""
    
    def __init__(self, window_size: int = 240, momentum_lookback: int = 20):
        """Initialize signal generator.
        
        Args:
            window_size: Rolling window size for calculations (minutes)
            momentum_lookback: Lookback period for momentum calculation
        """
        self.window_size = window_size
        self.momentum_lookback = momentum_lookback
    
    def compute_log_returns(self, prices: pd.Series) -> pd.Series:
        """Compute log returns from price series."""
        return np.log(prices / prices.shift(1))
    
    def compute_rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling Sharpe ratio (no annualization).
        
        Args:
            returns: Return series
            window: Rolling window size
        
        Returns:
            Rolling Sharpe ratio series
        """
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        # Avoid division by zero
        sharpe = rolling_mean / (rolling_std + 1e-10)
        return sharpe
    
    def compute_momentum(self, prices: pd.Series, lookback: int) -> float:
        """Compute momentum as percentage change over lookback period.
        
        Args:
            prices: Price series
            lookback: Number of periods to look back
        
        Returns:
            Momentum value (percentage change)
        """
        if len(prices) < lookback + 1:
            return 0.0
        
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-(lookback + 1)]
        
        if past_price > 0:
            return (current_price - past_price) / past_price
        return 0.0
    
    def compute_liquidity_proxy(self, ticker_data: Dict[str, float]) -> float:
        """Compute simple liquidity proxy from ticker data.
        
        Args:
            ticker_data: Dictionary with bid, ask, and volume_24h
        
        Returns:
            Liquidity score (higher is better)
        """
        volume_24h = ticker_data.get("volume_24h", 0)
        bid = ticker_data.get("bid", 0)
        ask = ticker_data.get("ask", 0)
        
        # Simple liquidity score based on volume and spread
        if bid > 0 and ask > 0 and bid < ask:
            spread_pct = (ask - bid) / ask
            # Penalize wide spreads
            liquidity_score = volume_24h * (1 - spread_pct)
        else:
            liquidity_score = volume_24h * 0.5  # Penalty for missing bid/ask
        
        return liquidity_score
    
    def compute_features(self, pair: str, datastore: DataStore, ticker_data: Dict[str, any]) -> Dict[str, float]:
        """Compute all features for a trading pair.
        
        Args:
            pair: Trading pair
            datastore: DataStore instance
            ticker_data: Current ticker data for the pair
        
        Returns:
            Dictionary of computed features
        """
        # Read minute bars
        df = datastore.read_minute_bars(pair, limit=self.window_size + 1)
        
        features = {
            "pair": pair,
            "price": ticker_data["price"],
            "volume_24h": ticker_data.get("volume_24h", 0),
            "liquidity_score": self.compute_liquidity_proxy(ticker_data),
            "log_return": 0.0,
            "rolling_mean": 0.0,
            "rolling_std": 0.0,
            "sharpe_ratio": 0.0,
            "momentum": 0.0,
            "has_sufficient_data": False
        }
        
        # Need at least window_size + 1 data points
        if len(df) < self.window_size + 1:
            # Log why insufficient data for debugging
            import logging
            logger = logging.getLogger(__name__)
            if len(df) == 0:
                logger.debug(f"{pair}: No data file found")
            else:
                logger.debug(f"{pair}: {len(df)} data points, need {self.window_size + 1}")
            return features
        
        # Compute log returns
        log_returns = self.compute_log_returns(df["price"])
        
        # Get latest values
        latest_return = log_returns.iloc[-1]
        rolling_mean = log_returns.rolling(window=self.window_size).mean().iloc[-1]
        rolling_std = log_returns.rolling(window=self.window_size).std().iloc[-1]
        
        # Compute Sharpe ratio
        sharpe = rolling_mean / (rolling_std + 1e-10) if rolling_std > 0 else 0.0
        
        # Compute momentum
        momentum = self.compute_momentum(df["price"], self.momentum_lookback)
        
        # Update features
        features.update({
            "log_return": latest_return,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "sharpe_ratio": sharpe,
            "momentum": momentum,
            "has_sufficient_data": True
        })
        
        return features
    
    def compute_all_features(self, datastore: DataStore, ticker_data: List[Dict[str, any]]) -> Dict[str, Dict[str, float]]:
        """Compute features for all trading pairs.
        
        Args:
            datastore: DataStore instance
            ticker_data: List of ticker data for all pairs
        
        Returns:
            Dictionary mapping pair to features
        """
        all_features = {}
        
        # Create ticker lookup
        ticker_lookup = {t["pair"]: t for t in ticker_data}
        
        # Process each pair from ticker data (not just those with existing data files)
        # This ensures we compute features for all available pairs
        for pair, ticker_info in ticker_lookup.items():
            try:
                features = self.compute_features(pair, datastore, ticker_info)
                all_features[pair] = features
            except Exception as e:
                # Skip pairs that fail (e.g., no data yet)
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Skipping {pair}: {e}")
                continue
        
        return all_features
    
    def filter_liquid_pairs(self, features: Dict[str, Dict[str, float]], min_liquidity: float = 1000.0) -> Dict[str, Dict[str, float]]:
        """Filter pairs by liquidity threshold.
        
        Args:
            features: Dictionary of pair features
            min_liquidity: Minimum liquidity score
        
        Returns:
            Filtered dictionary with only liquid pairs
        """
        return {
            pair: feat for pair, feat in features.items()
            if feat["liquidity_score"] >= min_liquidity
        }
    
    def rank_by_sharpe(self, features: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Rank pairs by Sharpe ratio.
        
        Args:
            features: Dictionary of pair features
        
        Returns:
            List of (pair, sharpe_ratio) tuples sorted by Sharpe (highest first)
        """
        pairs_with_sharpe = [
            (pair, feat["sharpe_ratio"]) 
            for pair, feat in features.items()
            if feat["has_sufficient_data"]
        ]
        
        # Sort by Sharpe ratio (descending)
        pairs_with_sharpe.sort(key=lambda x: x[1], reverse=True)
        
        return pairs_with_sharpe


class TangencyPortfolioSignals(SignalGenerator):
    """Alternative signal generator using tangency portfolio approach."""
    
    def compute_covariance_matrix(self, returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compute covariance matrix from returns."""
        # Create DataFrame from returns dict
        returns_df = pd.DataFrame(returns_dict)
        
        # Compute covariance matrix
        return returns_df.cov()
    
    def compute_tangency_weights(self, features: Dict[str, Dict[str, float]], top_n: int = 10) -> Dict[str, float]:
        """Compute tangency portfolio weights (max Sharpe long-only).
        
        Args:
            features: Dictionary of pair features
            top_n: Number of top liquid coins to consider
        
        Returns:
            Dictionary mapping pair to portfolio weight
        """
        # Filter liquid pairs with positive expected returns
        eligible_pairs = [
            pair for pair, feat in features.items()
            if feat["has_sufficient_data"] and 
               feat["rolling_mean"] > 0 and 
               feat["liquidity_score"] > 1000
        ]
        
        if not eligible_pairs:
            return {}
        
        # Sort by liquidity and take top N
        eligible_pairs.sort(key=lambda p: features[p]["liquidity_score"], reverse=True)
        selected_pairs = eligible_pairs[:top_n]
        
        if len(selected_pairs) < 2:
            # Not enough pairs for portfolio
            if selected_pairs:
                return {selected_pairs[0]: 1.0}
            return {}
        
        # Get expected returns and covariances
        expected_returns = np.array([features[p]["rolling_mean"] for p in selected_pairs])
        
        # Simplified covariance estimation using individual variances
        # (Full covariance would require synchronized returns data)
        variances = np.array([features[p]["rolling_std"]**2 for p in selected_pairs])
        cov_matrix = np.diag(variances)
        
        try:
            # Compute inverse covariance matrix
            inv_cov = np.linalg.inv(cov_matrix)
            
            # Tangency portfolio weights (unnormalized)
            weights = inv_cov @ expected_returns
            
            # Make weights long-only
            weights = np.maximum(weights, 0)
            
            # Normalize to sum to 1
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                # Fall back to equal weighting
                weights = np.ones(len(selected_pairs)) / len(selected_pairs)
            
            # Create weight dictionary
            return {pair: float(w) for pair, w in zip(selected_pairs, weights)}
            
        except np.linalg.LinAlgError:
            # Singular matrix, fall back to equal weighting
            equal_weight = 1.0 / len(selected_pairs)
            return {pair: equal_weight for pair in selected_pairs}
