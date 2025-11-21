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
    max_position_pct: float = Field(default=0.25, description="Maximum position size as fraction of equity (reduced for risk control)")
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
    min_entry_quality: float = Field(default=0.55, description="Minimum signal quality to take trade (MODERATE CONSERVATIVE)")
    
    # Hybrid Strategy Parameters
    regime_detection_window: int = Field(default=24, description="Regime detection window (hours)")
    adx_threshold_trending: float = Field(default=25.0, description="ADX threshold for trending market")
    adx_threshold_ranging: float = Field(default=20.0, description="ADX threshold for ranging market")
    momentum_quality_threshold: float = Field(default=0.50, description="Minimum quality for momentum signals (moderate for trades)")
    mean_reversion_quality_threshold: float = Field(default=0.45, description="Minimum quality for mean reversion signals (moderate for trades)")
    max_positions: int = Field(default=10, description="Maximum number of simultaneous positions (increased for 70% capital utilization)")
    min_correlation: float = Field(default=0.7, description="Minimum correlation to avoid correlated positions")
    sortino_stop_pct: float = Field(default=0.02, description="Default stop loss percentage for Sortino optimization (2%)")
    sortino_target_ratio: float = Field(default=2.5, description="Default risk/reward ratio for Sortino optimization (2.5:1)")
    
    # Volatility Expansion Strategy Parameters
    # Crypto-optimized indicator settings
    rsi_period: int = Field(default=14, description="RSI calculation period")
    rsi_overbought: float = Field(default=80.0, description="RSI overbought threshold (crypto-optimized: 80/20)")
    rsi_oversold: float = Field(default=20.0, description="RSI oversold threshold (crypto-optimized: 80/20)")
    macd_fast: int = Field(default=3, description="MACD fast EMA period (crypto-optimized: 3-10-16)")
    macd_slow: int = Field(default=10, description="MACD slow EMA period (crypto-optimized: 3-10-16)")
    macd_signal: int = Field(default=16, description="MACD signal line period (crypto-optimized: 3-10-16)")
    bb_period: int = Field(default=20, description="Bollinger Band period")
    bb_std_dev: float = Field(default=1.5, description="Bollinger Band standard deviation (crypto-optimized: 1.5-2.0)")
    atr_period: int = Field(default=14, description="ATR calculation period")
    volume_ma_period: int = Field(default=20, description="Volume moving average period")
    volume_spike_threshold: float = Field(default=1.2, description="Volume spike multiplier (1.2 = 120% of average, required for entry)")
    squeeze_threshold: float = Field(default=0.02, description="Bollinger squeeze threshold (band width %)")
    
    # Batch entry settings
    batch_entry_count: int = Field(default=4, description="Number of batch entries")
    batch_spacing_pct: float = Field(default=0.10, description="Price spacing between batches (10%)")
    
    # Staggered exit settings - Optimized for 14-day competition
    use_scaled_exits: bool = Field(default=True, description="Use scaled exits (8%, 15%, 25%) achievable targets")
    exit_1_pct: float = Field(default=0.33, description="First exit percentage (33% for scaled, 50% for original)")
    exit_2_pct: float = Field(default=0.33, description="Second exit percentage (33% for scaled, 30% for original)")
    exit_3_pct: float = Field(default=0.34, description="Third exit percentage (34% for scaled, 20% for original)")
    exit_1_level: float = Field(default=5.0, description="First exit profit level (%): 5% achievable target (lowered from 8%)")
    exit_2_level: float = Field(default=10.0, description="Second exit profit level (%): 10% achievable target (lowered from 15%)")
    exit_3_level: float = Field(default=15.0, description="Third exit profit level (%): 15% achievable target (lowered from 25%)")
    
    # ATR-based stop loss - Balanced stops
    atr_stop_multiplier: float = Field(default=2.0, description="ATR multiplier for stop loss (2.0x ATR - balanced, was 2.5x)")
    risk_per_trade_pct: float = Field(default=0.04, description="Base risk per trade as percentage of equity (4% base for dynamic scaling)")
    max_position_pct: float = Field(default=0.09, description="Base maximum position size per trade (9% base, dynamically adjusted based on portfolio state, positions count, utilization, and cash availability - actual cap ranges from 3% to 20%)")
    
    # Trend filter parameters
    ma_trend_filter_period: int = Field(default=50, description="Moving average period for trend filter (50-period EMA on daily)")
    cooldown_after_stopout_hours: float = Field(default=1.5, description="Hours to wait before re-entering same pair after stop-out")
    
    # Expert optimizations
    confirmation_candle_required: bool = Field(default=False, description="Require confirmation candle (bullish engulfing or hammer) before entry - STEP 1: DISABLED for baseline")
    trailing_stop_distance_pct: float = Field(default=0.02, description="Trailing stop distance as percentage (2% below highest price)")
    weak_signal_mode_enabled: bool = Field(default=True, description="Enable weak signal mode for low-volatility periods")
    weak_signal_atr_threshold: float = Field(default=0.2, description="ATR threshold for weak signal mode (0.2 = 20% of median ATR)")
    breakout_position_pct: float = Field(default=0.6, description="Maximum percentage of positions that can be breakout trades (60%)")
    
    # Utilization-aware dynamic sizing
    target_utilization: float = Field(default=0.90, description="Target portfolio utilization (fraction of equity deployed)")
    utilization_low_threshold: float = Field(default=0.60, description="Below this utilization, use scout mode and increase risk")
    utilization_high_threshold: float = Field(default=0.90, description="Above this utilization, reduce risk to avoid overexposure")
    
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
