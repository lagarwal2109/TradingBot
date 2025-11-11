"""Regime detection using GMM and HMM models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
import logging

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available. HMM functionality will be limited. Install with: pip install hmmlearn")

from bot.models.model_storage import ModelStorage

logger = logging.getLogger(__name__)


class GMMRegimeDetector:
    """Gaussian Mixture Model for microstructure regime detection (calm vs volatile)."""
    
    def __init__(self, n_components: int = 2, model_storage: Optional[ModelStorage] = None):
        """Initialize GMM regime detector.
        
        Args:
            n_components: Number of mixture components (2 for calm/volatile)
            model_storage: Optional ModelStorage instance for saving/loading
        """
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.model_storage = model_storage or ModelStorage()
        self.is_fitted = False
    
    def extract_microstructure_features(self, df: pd.DataFrame, window_minutes: int = 60) -> np.ndarray:
        """Extract microstructure features for regime detection.
        
        Args:
            df: DataFrame with price and volume columns
            window_minutes: Rolling window size in minutes
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if len(df) < window_minutes:
            # Return zeros if insufficient data
            return np.zeros((1, 4))
        
        prices = df["price"].values
        volumes = df["volume"].values if "volume" in df.columns else np.zeros_like(prices)
        
        features = []
        
        # Rolling window features
        for i in range(window_minutes, len(prices) + 1):
            window_prices = prices[i - window_minutes:i]
            window_volumes = volumes[i - window_minutes:i] if len(volumes) > 0 else np.zeros_like(window_prices)
            
            # Realized volatility (standard deviation of returns)
            returns = np.diff(np.log(window_prices))
            realized_vol = np.std(returns) * np.sqrt(window_minutes) if len(returns) > 0 else 0.0
            
            # Volume volatility
            volume_vol = np.std(window_volumes) / (np.mean(window_volumes) + 1e-10) if len(window_volumes) > 0 else 0.0
            
            # Price range (high - low) / mean
            price_range = (np.max(window_prices) - np.min(window_prices)) / (np.mean(window_prices) + 1e-10)
            
            # Return skewness
            return_skewness = np.mean((returns - np.mean(returns)) ** 3) / (np.std(returns) ** 3 + 1e-10) if len(returns) > 1 else 0.0
            
            features.append([realized_vol, volume_vol, price_range, return_skewness])
        
        if not features:
            return np.zeros((1, 4))
        
        return np.array(features)
    
    def fit(self, data: pd.DataFrame, window_hours: int = 24) -> None:
        """Train GMM on historical data.
        
        Args:
            data: Historical price/volume data
            window_hours: Training window size in hours (default: 24)
        """
        window_minutes = window_hours * 60
        
        # Extract features
        features = self.extract_microstructure_features(data, window_minutes=window_minutes)
        
        if len(features) < self.n_components:
            logger.warning(f"Insufficient data for GMM training: {len(features)} samples, need at least {self.n_components}")
            return
        
        # Fit GMM
        self.model.fit(features)
        self.is_fitted = True
        
        logger.info(f"GMM fitted on {len(features)} samples with {self.n_components} components")
    
    def predict_proba(self, data: pd.DataFrame) -> Dict[str, float]:
        """Predict regime probabilities.
        
        Args:
            data: Current price/volume data
        
        Returns:
            Dictionary mapping regime names to probabilities
        """
        if not self.is_fitted:
            # Return default probabilities if not fitted
            return {"calm": 0.5, "volatile": 0.5}
        
        # Extract features for latest window
        features = self.extract_microstructure_features(data, window_minutes=60)
        
        if len(features) == 0:
            return {"calm": 0.5, "volatile": 0.5}
        
        # Get probabilities for latest features
        latest_features = features[-1:].reshape(1, -1)
        proba = self.model.predict_proba(latest_features)[0]
        
        # Map to regime names
        # Component 0 is typically calm (lower volatility), component 1 is volatile
        # We determine this by checking mean volatility of each component
        if self.n_components == 2:
            # Check which component has higher mean volatility
            means = self.model.means_
            if means[0, 0] < means[1, 0]:  # Compare realized volatility (first feature)
                return {"calm": float(proba[0]), "volatile": float(proba[1])}
            else:
                return {"calm": float(proba[1]), "volatile": float(proba[0])}
        else:
            # For more components, use generic names
            return {f"regime_{i}": float(p) for i, p in enumerate(proba)}
    
    def get_dominant_regime(self, proba: Dict[str, float]) -> str:
        """Get dominant regime from probabilities.
        
        Args:
            proba: Regime probabilities
        
        Returns:
            Dominant regime name
        """
        return max(proba.items(), key=lambda x: x[1])[0]
    
    def update_model(self, new_data: pd.DataFrame) -> None:
        """Update model with new data (incremental training).
        
        Args:
            new_data: New price/volume data
        """
        # For simplicity, refit on recent data
        # In production, could implement online learning
        self.fit(new_data, window_hours=24)
    
    def save(self, name: str = "gmm_regime", version: Optional[str] = None) -> str:
        """Save model to storage.
        
        Args:
            name: Model name
            version: Optional version string
        
        Returns:
            Version string
        """
        metadata = {
            "n_components": self.n_components,
            "is_fitted": self.is_fitted
        }
        return self.model_storage.save_model(self.model, name, version, metadata)
    
    def load(self, name: str = "gmm_regime", version: Optional[str] = None) -> None:
        """Load model from storage.
        
        Args:
            name: Model name
            version: Optional version string
        """
        self.model = self.model_storage.load_model(name, version)
        metadata = self.model_storage.load_metadata(name, version)
        self.is_fitted = metadata.get("is_fitted", False)


class HMMTrendDetector:
    """Hidden Markov Model for trend regime detection (bearish, neutral, bullish)."""
    
    def __init__(self, n_states: int = 3, model_storage: Optional[ModelStorage] = None):
        """Initialize HMM trend detector.
        
        Args:
            n_states: Number of hidden states (3 for bearish/neutral/bullish)
            model_storage: Optional ModelStorage instance
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required for HMMTrendDetector. Install with: pip install hmmlearn")
        
        self.n_states = n_states
        # Use "diag" covariance for better numerical stability, or "full" with min_covar
        # "diag" is more stable but less flexible; "full" with min_covar handles edge cases
        self.model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="diag",  # Changed from "full" to "diag" for numerical stability
            n_iter=100, 
            random_state=42,
            min_covar=1e-3  # Minimum covariance to ensure positive-definite
        )
        self.model_storage = model_storage or ModelStorage()
        self.is_fitted = False
        self.state_names = ["bearish", "neutral", "bullish"] if n_states == 3 else [f"state_{i}" for i in range(n_states)]
        self.feature_mean = None
        self.feature_std = None
    
    def prepare_trend_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data for HMM training (returns and volatility).
        
        Args:
            df: Aggregated price data (4h or daily)
        
        Returns:
            Feature matrix (n_samples, 2) - [returns, volatility]
        """
        if len(df) < 2:
            return np.zeros((1, 2))
        
        # Ensure we have price column
        if "price" not in df.columns:
            if len(df.columns) > 0:
                # Use first numeric column as price
                price_col = df.select_dtypes(include=[np.number]).columns[0]
                prices = df[price_col].values
            else:
                return np.zeros((1, 2))
        else:
            prices = df["price"].values
        
        if len(prices) < 2:
            return np.zeros((1, 2))
        
        # Compute returns
        returns = np.diff(np.log(prices + 1e-10))  # Add small epsilon to avoid log(0)
        
        # Compute rolling volatility (simplified)
        if len(returns) < 5:
            volatilities = np.abs(returns)
        else:
            returns_series = pd.Series(returns)
            rolling_std = returns_series.rolling(window=min(5, len(returns))).std()
            # Fill NaN values with absolute returns as Series
            abs_returns_series = pd.Series(np.abs(returns), index=returns_series.index)
            volatilities = rolling_std.fillna(abs_returns_series).values
        
        # Stack returns and volatilities
        # Need to align lengths
        min_len = min(len(returns), len(volatilities))
        if min_len == 0:
            return np.zeros((1, 2))
        
        features = np.column_stack([returns[:min_len], volatilities[:min_len]])
        
        # Ensure 2D array
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
        return features
    
    def fit(self, data: pd.DataFrame, window_days: int = 30) -> None:
        """Train HMM on historical data.
        
        Args:
            data: Aggregated price data (4h or daily)
            window_days: Training window size in days
        """
        try:
            # Ensure data is a proper DataFrame
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Expected DataFrame, got {type(data)}")
                return
            
            # Ensure we have price column
            if "price" not in data.columns:
                logger.error(f"DataFrame missing 'price' column. Columns: {data.columns.tolist()}")
                return
            
            # Prepare features
            features = self.prepare_trend_data(data)
            
            if len(features) < self.n_states:
                logger.warning(f"Insufficient data for HMM training: {len(features)} samples, need at least {self.n_states}")
                return
            
            # Fit HMM - hmmlearn expects 2D array with shape (n_samples, n_features)
            # features should already be in this format from prepare_trend_data
            # Ensure features is 2D array with correct shape
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
            
            # Ensure we have at least n_states samples
            if len(features) < self.n_states:
                logger.warning(f"Insufficient samples for HMM: {len(features)} < {self.n_states}")
                return
            
            # Convert to numpy array and ensure float type
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            features = features.astype(np.float64)
            
            # Remove any NaN or Inf values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure features are well-conditioned (no perfect correlation)
            # Add small random noise if features are too similar (helps with numerical stability)
            if features.shape[1] > 1:
                # Check if features are too correlated
                corr = np.corrcoef(features.T)
                if np.any(np.abs(corr - np.eye(corr.shape[0])) > 0.99):
                    # Add tiny noise to break perfect correlation
                    noise_scale = np.std(features) * 1e-6
                    features = features + np.random.normal(0, noise_scale, features.shape)
            
            # Normalize features to improve numerical stability
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
            features_normalized = (features - feature_mean) / feature_std
            
            # Store normalization parameters for prediction
            self.feature_mean = feature_mean
            self.feature_std = feature_std
            
            # hmmlearn.fit expects X as numpy array
            self.model.fit(features_normalized)
            self.is_fitted = True
            logger.info(f"HMM fitted on {len(features)} samples with {self.n_states} states")
        except Exception as e:
            logger.error(f"Error fitting HMM: {e}", exc_info=True)
            self.is_fitted = False
    
    def predict_proba(self, data: pd.DataFrame) -> Dict[str, float]:
        """Predict trend state probabilities.
        
        Args:
            data: Current aggregated price data
        
        Returns:
            Dictionary mapping trend names to probabilities
        """
        if not self.is_fitted:
            # Return default probabilities
            return {state: 1.0 / self.n_states for state in self.state_names}
        
        # Prepare features
        features = self.prepare_trend_data(data)
        
        if len(features) == 0:
            return {state: 1.0 / self.n_states for state in self.state_names}
        
        # Get probabilities for latest observation
        # Ensure features is 2D array
        if len(features.shape) == 1:
            latest_features = features.reshape(1, -1)
        else:
            latest_features = features[-1:].reshape(1, -1)
        
        try:
            # Convert to numpy array and ensure float type
            if not isinstance(latest_features, np.ndarray):
                latest_features = np.array(latest_features)
            latest_features = latest_features.astype(np.float64)
            latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize using same parameters as training
            if hasattr(self, 'feature_mean') and hasattr(self, 'feature_std') and self.feature_mean is not None:
                latest_features = (latest_features - self.feature_mean) / self.feature_std
            
            # hmmlearn doesn't have predict_proba, use score_samples
            # score_samples returns (logprob, posteriors) tuple
            score_result = self.model.score_samples(latest_features)
            
            # Handle tuple return (logprob, posteriors) or just array
            if isinstance(score_result, tuple):
                # score_samples returns (logprob, posteriors)
                # posteriors is shape (n_samples, n_states)
                log_probs = score_result[1]  # Get posteriors
                if len(log_probs.shape) == 2:
                    log_probs = log_probs[0]  # Get first sample
            else:
                # If it's just an array
                log_probs = score_result
                if len(log_probs.shape) == 2:
                    log_probs = log_probs[0]  # Get first sample
            
            # Convert log probabilities to probabilities
            # Subtract max for numerical stability
            log_probs = log_probs - np.max(log_probs)
            probs = np.exp(log_probs)
            probs = probs / probs.sum()  # Normalize
            
            # Map to state names
            result = {}
            for i, state_name in enumerate(self.state_names):
                if i < len(probs):
                    result[state_name] = float(probs[i])
                else:
                    result[state_name] = 0.0
            
            return result
        except Exception as e:
            logger.error(f"Error predicting HMM probabilities: {e}", exc_info=True)
            return {state: 1.0 / self.n_states for state in self.state_names}
    
    def get_dominant_trend(self, proba: Dict[str, float]) -> str:
        """Get dominant trend from probabilities.
        
        Args:
            proba: Trend probabilities
        
        Returns:
            Dominant trend name
        """
        return max(proba.items(), key=lambda x: x[1])[0]
    
    def update_model(self, new_data: pd.DataFrame) -> None:
        """Update model with new data.
        
        Args:
            new_data: New aggregated price data
        """
        # Refit on recent data
        self.fit(new_data, window_days=30)
    
    def save(self, name: str = "hmm_trend", version: Optional[str] = None) -> str:
        """Save model to storage.
        
        Args:
            name: Model name
            version: Optional version string
        
        Returns:
            Version string
        """
        metadata = {
            "n_states": self.n_states,
            "is_fitted": self.is_fitted,
            "state_names": self.state_names
        }
        return self.model_storage.save_model(self.model, name, version, metadata)
    
    def load(self, name: str = "hmm_trend", version: Optional[str] = None) -> None:
        """Load model from storage.
        
        Args:
            name: Model name
            version: Optional version string
        """
        self.model = self.model_storage.load_model(name, version)
        metadata = self.model_storage.load_metadata(name, version)
        self.is_fitted = metadata.get("is_fitted", False)
        self.state_names = metadata.get("state_names", self.state_names)


class RegimeFusion:
    """Fuses GMM and HMM regime outputs."""
    
    def __init__(self, gmm_weight: float = 0.6, hmm_weight: float = 0.4):
        """Initialize regime fusion.
        
        Args:
            gmm_weight: Weight for GMM (microstructure) regime
            hmm_weight: Weight for HMM (trend) regime
        """
        self.gmm_weight = gmm_weight
        self.hmm_weight = hmm_weight
        
        # Normalize weights
        total = gmm_weight + hmm_weight
        if total > 0:
            self.gmm_weight /= total
            self.hmm_weight /= total
    
    def fuse_regimes(
        self,
        gmm_proba: Dict[str, float],
        hmm_proba: Dict[str, float]
    ) -> Dict[str, float]:
        """Fuse regime probabilities from GMM and HMM.
        
        Args:
            gmm_proba: GMM regime probabilities (calm/volatile)
            hmm_proba: HMM trend probabilities (bearish/neutral/bullish)
        
        Returns:
            Combined regime probabilities
        """
        # Create combined regime space
        combined = {}
        
        # Combine microstructure and trend
        for gmm_regime, gmm_p in gmm_proba.items():
            for hmm_regime, hmm_p in hmm_proba.items():
                combined_name = f"{gmm_regime}_{hmm_regime}"
                combined[combined_name] = (self.gmm_weight * gmm_p) + (self.hmm_weight * hmm_p)
        
        # Also return individual components for easier access
        combined["gmm_calm"] = gmm_proba.get("calm", 0.5)
        combined["gmm_volatile"] = gmm_proba.get("volatile", 0.5)
        combined["hmm_bearish"] = hmm_proba.get("bearish", 0.33)
        combined["hmm_neutral"] = hmm_proba.get("neutral", 0.33)
        combined["hmm_bullish"] = hmm_proba.get("bullish", 0.34)
        
        return combined
    
    def get_combined_regime(self, gmm_regime: str, hmm_regime: str) -> str:
        """Get combined regime name.
        
        Args:
            gmm_regime: Dominant GMM regime
            hmm_regime: Dominant HMM regime
        
        Returns:
            Combined regime name
        """
        return f"{gmm_regime}_{hmm_regime}"
    
    def get_regime_features(self, gmm_proba: Dict[str, float], hmm_proba: Dict[str, float]) -> np.ndarray:
        """Get regime probabilities as feature vector.
        
        Args:
            gmm_proba: GMM probabilities
            hmm_proba: HMM probabilities
        
        Returns:
            Feature vector
        """
        features = []
        
        # Add GMM probabilities
        features.append(gmm_proba.get("calm", 0.5))
        features.append(gmm_proba.get("volatile", 0.5))
        
        # Add HMM probabilities
        features.append(hmm_proba.get("bearish", 0.33))
        features.append(hmm_proba.get("neutral", 0.33))
        features.append(hmm_proba.get("bullish", 0.34))
        
        return np.array(features)

