"""Configuration management for the trading bot."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    """Configuration settings for the Roostoo trading bot."""
    
    # Roostoo API Settings (Trading)
    api_key: str = Field(..., description="Roostoo API key")
    api_secret: str = Field(..., description="Roostoo API secret")
    base_url: str = Field(default="https://mock-api.roostoo.com", description="Roostoo API base URL")
    
    # Trading Parameters - Optimized for Competition
    # Note: window_size is in hours but converted to minutes for minute-by-minute data
    window_size: int = Field(default=72, description="Rolling window size for signals (minutes, not hours - will be converted)")
    momentum_lookback: int = Field(default=12, description="Momentum lookback period (minutes, not hours - will be converted)")
    max_position_pct: float = Field(default=0.40, description="Maximum position size as fraction of equity")
    min_sharpe: float = Field(default=0.05, description="Minimum Sharpe ratio to enter position")
    
    # Enhanced Strategy Parameters
    strategy_mode: str = Field(default="enhanced", description="Strategy mode: 'sharpe', 'tangency', or 'enhanced'")
    trend_window_long: int = Field(default=72, description="Long-term trend window (minutes) - reduced for available data")
    trend_window_short: int = Field(default=24, description="Short-term trend window (minutes) - reduced for available data")
    entry_window: int = Field(default=12, description="Entry timing window (minutes) - reduced for available data")
    volume_window: int = Field(default=480, description="Volume averaging window (minutes) - 8 hours")
    support_resistance_days: int = Field(default=7, description="Days to look back for support/resistance")
    breakout_threshold: float = Field(default=0.02, description="Percentage beyond level for breakout confirmation")
    volume_surge_multiplier: float = Field(default=2.0, description="Volume multiplier for breakout confirmation")
    min_entry_quality: float = Field(default=0.25, description="Minimum signal quality to take trade (lowered for more trades)")
    return_focused_mode: bool = Field(default=True, description="Optimize for returns rather than pure Sharpe")
    min_expected_return_pct: float = Field(default=0.005, description="Minimum expected return (0.5%) to take trade")
    position_size_by_return: bool = Field(default=True, description="Scale position size by expected return")
    
    # Risk Management
    max_consecutive_errors: int = Field(default=5, description="Max errors before kill switch")
    order_timeout_minutes: int = Field(default=5, description="Order timeout period")
    stop_loss_pct: float = Field(default=0.02, description="Default stop loss percentage")
    take_profit_pct: float = Field(default=0.05, description="Default take profit percentage")
    use_trailing_stop: bool = Field(default=True, description="Enable trailing stop loss")
    trailing_stop_pct: float = Field(default=0.02, description="Trailing stop distance")
    
    # Data Storage
    data_dir: Path = Field(default=Path("data"), description="Directory for market data")
    log_dir: Path = Field(default=Path("logs"), description="Directory for logs")
    figures_dir: Path = Field(default=Path("figures"), description="Directory for plots")
    
    # Regime-Adaptive Strategy Parameters
    regime_gmm_retrain_hours: int = Field(default=1, description="GMM retraining interval in hours")
    regime_hmm_retrain_days: int = Field(default=7, description="HMM retraining interval in days")
    ensemble_retrain_days: int = Field(default=7, description="Ensemble model retraining interval in days")
    min_prediction_confidence: float = Field(default=0.6, description="Minimum prediction confidence threshold")
    regime_calm_position_multiplier: float = Field(default=1.2, description="Position size multiplier in calm regime")
    regime_volatile_position_multiplier: float = Field(default=0.7, description="Position size multiplier in volatile regime")
    regime_bearish_position_multiplier: float = Field(default=0.5, description="Position size multiplier in bearish regime")
    max_simultaneous_positions: int = Field(default=3, description="Maximum number of simultaneous positions")
    max_portfolio_allocation: float = Field(default=0.85, description="Maximum portfolio allocation (15% cash buffer)")
    model_storage_dir: Path = Field(default=Path("models"), description="Directory for storing trained models")
    
    @validator("data_dir", "log_dir", "figures_dir", pre=True)
    def ensure_directory(cls, v):
        """Ensure directory exists."""
        path = Path(v)
        path.mkdir(exist_ok=True)
        return path
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


def load_config() -> Config:
    """Load configuration from environment variables."""
    # Load .env file
    load_dotenv()
    
    # Get Roostoo API credentials (for trading)
    api_key = os.getenv("ROOSTOO_API_KEY")
    api_secret = os.getenv("ROOSTOO_API_SECRET")
    base_url = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")
    
    if not api_key or not api_secret:
        raise ValueError("Missing required Roostoo environment variables: ROOSTOO_API_KEY and ROOSTOO_API_SECRET")
    
    return Config(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url
    )


# Global config instance
config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global config
    if config is None:
        config = load_config()
    return config
