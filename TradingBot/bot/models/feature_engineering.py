"""Feature engineering for trading models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    import ta
except ImportError:
    ta = None
    logging.warning("ta library not available, some technical indicators may be unavailable")

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features from price and volume data for ML models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators from price data.
        
        Args:
            df: DataFrame with 'price' column (and optionally 'volume')
        
        Returns:
            DataFrame with added technical indicator columns
        """
        result = df.copy()
        
        if len(result) < 2:
            return result
        
        prices = result["price"]
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(prices) >= period:
                result[f"sma_{period}"] = prices.rolling(window=period).mean()
                result[f"ema_{period}"] = prices.ewm(span=period, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        if ta is not None and len(prices) >= 14:
            try:
                result["rsi_14"] = ta.momentum.RSIIndicator(prices, window=14).rsi()
            except:
                # Fallback manual calculation
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                result["rsi_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(prices) >= 26:
            ema_12 = prices.ewm(span=12, adjust=False).mean()
            ema_26 = prices.ewm(span=26, adjust=False).mean()
            result["macd"] = ema_12 - ema_26
            result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
            result["macd_histogram"] = result["macd"] - result["macd_signal"]
        
        # Bollinger Bands
        if len(prices) >= 20:
            sma_20 = prices.rolling(window=20).mean()
            std_20 = prices.rolling(window=20).std()
            result["bb_upper"] = sma_20 + (std_20 * 2)
            result["bb_lower"] = sma_20 - (std_20 * 2)
            result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / sma_20
            result["bb_position"] = (prices - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])
        
        # Stochastic Oscillator
        if len(prices) >= 14:
            low_14 = prices.rolling(window=14).min()
            high_14 = prices.rolling(window=14).max()
            result["stoch_k"] = 100 * ((prices - low_14) / (high_14 - low_14))
            result["stoch_d"] = result["stoch_k"].rolling(window=3).mean()
        
        # ATR (Average True Range) - need high/low, approximate with price
        if len(prices) >= 14:
            high = prices * 1.01  # Approximate high
            low = prices * 0.99   # Approximate low
            tr1 = high - low
            tr2 = abs(high - prices.shift())
            tr3 = abs(low - prices.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result["atr_14"] = tr.rolling(window=14).mean()
        
        return result
    
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features.
        
        Args:
            df: DataFrame with 'price' and 'volume' columns
        
        Returns:
            DataFrame with added volume features
        """
        result = df.copy()
        
        if "volume" not in result.columns:
            result["volume"] = 0.0
        
        if len(result) < 2:
            return result
        
        volume = result["volume"]
        prices = result["price"]
        
        # Volume moving averages
        for period in [10, 20, 50]:
            if len(volume) >= period:
                result[f"volume_sma_{period}"] = volume.rolling(window=period).mean()
        
        # Volume ratios
        if len(volume) >= 20:
            volume_sma_20 = volume.rolling(window=20).mean()
            result["volume_ratio"] = volume / (volume_sma_20 + 1e-10)
        
        # OBV (On-Balance Volume)
        price_change = prices.diff()
        obv = (price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * volume).cumsum()
        result["obv"] = obv
        
        # Volume volatility
        if len(volume) >= 20:
            result["volume_volatility"] = volume.rolling(window=20).std()
            result["volume_volatility_ratio"] = result["volume_volatility"] / (volume_sma_20 + 1e-10)
        
        # Price-Volume correlation
        if len(result) >= 20:
            result["price_volume_corr"] = prices.rolling(window=20).corr(volume)
        
        # Volume-weighted average price (VWAP) approximation
        if len(result) >= 20:
            typical_price = prices  # (high + low + close) / 3, approximated
            result["vwap_20"] = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        return result
    
    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based features.
        
        Args:
            df: DataFrame with 'price' column
        
        Returns:
            DataFrame with added volatility features
        """
        result = df.copy()
        
        if len(result) < 2:
            return result
        
        prices = result["price"]
        
        # Log returns
        log_returns = np.log(prices / prices.shift(1))
        result["log_return"] = log_returns
        
        # Realized volatility (rolling standard deviation of returns)
        for window in [5, 10, 20, 60, 240]:
            if len(log_returns) >= window:
                result[f"realized_vol_{window}"] = log_returns.rolling(window=window).std() * np.sqrt(window)
        
        # Parkinson volatility estimator (using high/low approximation)
        if len(prices) >= 20:
            high = prices * 1.01
            low = prices * 0.99
            parkinson = np.sqrt((1 / (4 * np.log(2))) * np.log(high / low) ** 2)
            result["parkinson_vol_20"] = parkinson.rolling(window=20).mean()
        
        # GARCH-like features (simplified)
        if len(log_returns) >= 20:
            # Conditional variance approximation
            squared_returns = log_returns ** 2
            result["conditional_variance"] = squared_returns.ewm(alpha=0.1, adjust=False).mean()
            result["conditional_volatility"] = np.sqrt(result["conditional_variance"])
        
        # Return skewness and kurtosis
        for window in [20, 60]:
            if len(log_returns) >= window:
                result[f"return_skewness_{window}"] = log_returns.rolling(window=window).skew()
                result[f"return_kurtosis_{window}"] = log_returns.rolling(window=window).kurt()
        
        return result
    
    def compute_multi_timeframe_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute returns at multiple timeframes.
        
        Args:
            df: DataFrame with 'price' column indexed by timestamp
        
        Returns:
            DataFrame with added return columns
        """
        result = df.copy()
        
        if len(result) < 2:
            return result
        
        prices = result["price"]
        
        # Returns at different lookback periods (in minutes if minute data)
        # For minute data: 1m, 5m, 15m, 60m, 240m, 1440m
        # For aggregated data, adjust accordingly
        return_periods = {
            "return_1m": 1,
            "return_5m": 5,
            "return_15m": 15,
            "return_60m": 60,
            "return_240m": 240,
            "return_1440m": 1440,
        }
        
        for col_name, periods in return_periods.items():
            if len(prices) >= periods:
                result[col_name] = (prices / prices.shift(periods) - 1)
        
        # Cumulative returns over different windows
        log_returns = np.log(prices / prices.shift(1)).fillna(0)
        for window in [5, 10, 20, 60]:
            if len(log_returns) >= window:
                cumulative = log_returns.rolling(window=window).sum()
                # Ensure proper assignment by converting to Series with matching index
                result = result.copy()
                result[f"cumulative_return_{window}"] = pd.Series(cumulative.values, index=result.index)
        
        return result
    
    def compute_regime_features(self, df: pd.DataFrame, regime_probs: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Add regime probabilities as features.
        
        Args:
            df: DataFrame
            regime_probs: Dictionary of regime probabilities (e.g., {"calm": 0.7, "volatile": 0.3})
        
        Returns:
            DataFrame with added regime features
        """
        result = df.copy()
        
        # Always add the same regime feature columns for consistency
        # Define standard regime feature names
        standard_regime_features = {
            "regime_prob_calm": 0.5,
            "regime_prob_volatile": 0.5,
            "regime_prob_bearish": 0.33,
            "regime_prob_neutral": 0.33,
            "regime_prob_bullish": 0.34,
            # Also include combined regime probabilities if available
            "regime_prob_calm_bearish": 0.0,
            "regime_prob_calm_neutral": 0.0,
            "regime_prob_calm_bullish": 0.0,
            "regime_prob_volatile_bearish": 0.0,
            "regime_prob_volatile_neutral": 0.0,
            "regime_prob_volatile_bullish": 0.0,
        }
        
        if regime_probs:
            # Update with actual probabilities if provided
            for regime, prob in regime_probs.items():
                feature_name = f"regime_prob_{regime}"
                if feature_name in standard_regime_features:
                    standard_regime_features[feature_name] = prob
                else:
                    # Add any additional regime features
                    standard_regime_features[feature_name] = prob
        
        # Always set all standard regime features
        for feature_name, default_value in standard_regime_features.items():
            result[feature_name] = default_value
        
        return result
    
    def create_feature_matrix(
        self,
        pairs_data: Dict[str, pd.DataFrame],
        regime_detector: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create feature matrix for all pairs.
        
        Args:
            pairs_data: Dictionary mapping pair names to DataFrames with price/volume
            regime_detector: Optional regime detector to get regime probabilities
        
        Returns:
            Tuple of (feature_matrix DataFrame, list of feature names)
        """
        all_features = []
        feature_names = []
        
        for pair, df in pairs_data.items():
            if df.empty or len(df) < 2:
                continue
            
            # Ensure we have the latest row
            latest_row = df.iloc[-1:].copy()
            
            # Compute all features
            df_with_features = self.compute_technical_indicators(df)
            df_with_features = self.compute_volume_features(df_with_features)
            df_with_features = self.compute_volatility_features(df_with_features)
            df_with_features = self.compute_multi_timeframe_returns(df_with_features)
            
            # Get regime probabilities if detector available
            if regime_detector is not None:
                try:
                    regime_probs = regime_detector.predict_proba(df)
                    df_with_features = self.compute_regime_features(df_with_features, regime_probs)
                except Exception as e:
                    logger.warning(f"Failed to get regime probabilities for {pair}: {e}")
                    df_with_features = self.compute_regime_features(df_with_features)
            else:
                df_with_features = self.compute_regime_features(df_with_features)
            
            # Get latest row with all features
            latest_features = df_with_features.iloc[-1:].copy()
            
            # Add pair identifier
            latest_features["pair"] = pair
            
            # Select only feature columns (exclude original price/volume if desired, or keep them)
            # Keep price and volume as they might be useful
            feature_cols = [col for col in latest_features.columns if col not in ["pair"]]
            
            if not feature_names:
                feature_names = feature_cols
            
            all_features.append(latest_features)
        
        if not all_features:
            return pd.DataFrame(), []
        
        # Combine all pairs
        feature_matrix = pd.concat(all_features, ignore_index=True)
        
        # Ensure consistent columns
        for col in feature_names:
            if col not in feature_matrix.columns:
                feature_matrix[col] = 0.0
        
        # Select and order columns
        feature_matrix = feature_matrix[["pair"] + feature_names]
        
        return feature_matrix, feature_names
    
    def prepare_sequential_features(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60
    ) -> np.ndarray:
        """Prepare sequential features for LSTM.
        
        Args:
            df: DataFrame with features
            sequence_length: Length of sequence (number of time steps)
        
        Returns:
            Numpy array of shape (1, sequence_length, n_features)
        """
        # Compute all features
        df_features = self.compute_technical_indicators(df)
        df_features = self.compute_volume_features(df_features)
        df_features = self.compute_volatility_features(df_features)
        df_features = self.compute_multi_timeframe_returns(df_features)
        
        # Select numeric feature columns (exclude pair, timestamp if present)
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Get last sequence_length rows
        if len(df_features) < sequence_length:
            # Pad with zeros if not enough data
            padding = pd.DataFrame(0, index=range(sequence_length - len(df_features)), columns=numeric_cols)
            df_features = pd.concat([padding, df_features], ignore_index=True)
        
        sequence_data = df_features[numeric_cols].tail(sequence_length).values
        
        # Reshape to (1, sequence_length, n_features) for LSTM
        return sequence_data.reshape(1, sequence_length, -1)

