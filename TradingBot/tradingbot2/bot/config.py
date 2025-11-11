"""Configuration management for MACD trading bot."""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator


class MACDConfig(BaseModel):
    """Configuration settings for the MACD trading bot."""
    
    # Roostoo API Settings (Trading)
    api_key: str = Field(..., description="Roostoo API key")
    api_secret: str = Field(..., description="Roostoo API secret")
    base_url: str = Field(default="https://mock-api.roostoo.com", description="Roostoo API base URL")
    
    # MACD Parameters (adaptive - will scale with available data)
    macd_fast_period: int = Field(default=12, description="Fast EMA period (adaptive, default 12)")
    macd_slow_period: int = Field(default=26, description="Slow EMA period (adaptive, default 26)")
    macd_signal_period: int = Field(default=9, description="Signal line EMA period (adaptive, default 9)")
    ma_trend_period: int = Field(default=50, description="Trend filter MA period (adaptive, default 50)")
    
    # Trading Constraints
    max_trades_per_minute: int = Field(default=5, description="Maximum trades per minute")
    transaction_fee_pct: float = Field(default=0.001, description="Transaction fee (0.1%)")
    
    # Risk Management (volatility-normalized)
    risk_per_trade_pct: float = Field(default=0.005, description="Risk per trade as % of equity (0.5% default for BATS v1)")
    k_atr_stop: float = Field(default=3.0, description="ATR multiplier for initial stop (3.0 default for BATS v1)")
    max_pair_risk_pct: float = Field(default=0.012, description="Max gross risk per pair as % of equity (1.2% default)")
    max_portfolio_risk_pct: float = Field(default=0.04, description="Max portfolio open risk as % of equity (4% default)")
    max_drawdown_pct: float = Field(default=0.15, description="Maximum drawdown per trade as % of entry (15% hard limit)")
    
    # Exit Logic
    breakeven_at_r: float = Field(default=1.0, description="Move to breakeven at this R multiple (1.0R default)")
    chandelier_at_r: float = Field(default=2.0, description="Activate chandelier trailing stop at this R multiple (2.0R default)")
    chandelier_c: float = Field(default=2.0, description="Chandelier stop: HH/LL ± c*ATR (2.0 default for BATS v1)")
    chandelier_n: int = Field(default=22, description="Chandelier lookback period (22 bars default)")
    time_stop_bars: int = Field(default=30, description="Time stop after N bars if profit between -0.5R and +0.5R (30 default for BATS v1)")
    partial_profit_r: float = Field(default=2.0, description="Partial profit target in R multiples (2.0R default)")
    partial_profit_pct: float = Field(default=0.30, description="Percentage to take at partial profit (30% default)")
    partial_stop_r: float = Field(default=0.5, description="Stop after partial profit at this R multiple (0.5R default)")
    
    # Position Management
    min_holding_bars: int = Field(default=5, description="Minimum holding period in bars (5 default)")
    cooldown_bars: int = Field(default=8, description="Cooldown period after trade in bars (8 default)")
    max_simultaneous_positions: int = Field(default=1, description="Maximum simultaneous positions per asset (1 default)")
    
    # Pyramiding (R-based laddering) - DISABLED for BATS v1 baseline
    enable_pyramiding: bool = Field(default=False, description="Enable pyramiding on winning trades (disabled for baseline)")
    pyramid_spacing_r: float = Field(default=1.0, description="Add every +1.0R from last add (1.0R default)")
    pyramid_sizes: List[float] = Field(default=[1.0, 0.7, 0.5], description="Pyramid sizes as % of initial risk [100%, 70%, 50%]")
    max_pyramid_adds: int = Field(default=0, description="Maximum number of pyramid additions (0 default - disabled for baseline)")
    pyramid_min_adx: float = Field(default=20.0, description="Minimum ADX to allow pyramid (20 default)")
    
    # Entry Gates (tuned)
    min_adx: float = Field(default=20.0, description="Minimum ADX for entry (20 default for BATS v1)")
    hist_z_score_min: float = Field(default=0.7, description="Minimum histogram z-score for entry (0.7 default, loosened from 1.0)")
    atr_band_min: float = Field(default=0.008, description="Minimum ATR/Price ratio (0.8% default for BATS v1)")
    atr_band_max: float = Field(default=0.04, description="Maximum ATR/Price ratio (4.0% default for BATS v1)")
    
    # Risk Controls
    daily_loss_limit_R: float = Field(default=2.0, description="Daily loss limit in R multiples (-2R default for BATS v1)")
    weekly_loss_limit_R: float = Field(default=5.0, description="Weekly loss limit in R multiples (-5R default for BATS v1)")
    daily_loss_limit_pct: float = Field(default=0.02, description="Daily loss limit as % of equity (2% default, deprecated - use daily_loss_limit_R)")
    pair_loss_streak_limit: int = Field(default=3, description="Consecutive losses before pair cooldown (3 default)")
    pair_cooldown_atr: float = Field(default=2.0, description="Pair cooldown period in ATR windows (2.0 default)")
    max_open_positions: int = Field(default=1, description="Maximum open positions (1 default for BATS v1)")
    slippage_bps_major: float = Field(default=2.5, description="Slippage in basis points for majors (BTC/ETH) - 2.5 bps default")
    slippage_bps_alt: float = Field(default=10.0, description="Slippage in basis points for alts - 10 bps default")
    
    # Data Storage
    data_dir: Path = Field(default=Path(__file__).parent.parent.parent.parent / "data2" / "data", description="Directory for market data")
    figures_dir: Path = Field(default=Path(__file__).parent.parent / "figures", description="Directory for plots")
    
    @validator("data_dir", "figures_dir", pre=True)
    def ensure_directory(cls, v):
        """Ensure directory exists."""
        if isinstance(v, str):
            path = Path(v)
        else:
            path = v
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


def load_config() -> MACDConfig:
    """Load configuration from environment variables."""
    # Load .env file from parent directory
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
    
    # Get Roostoo API credentials
    api_key = os.getenv("ROOSTOO_API_KEY")
    api_secret = os.getenv("ROOSTOO_API_SECRET")
    base_url = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")
    
    if not api_key or not api_secret:
        raise ValueError("Missing required Roostoo environment variables: ROOSTOO_API_KEY and ROOSTOO_API_SECRET")
    
    return MACDConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url
    )


# Global config instance
_config: Optional[MACDConfig] = None


def get_config() -> MACDConfig:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

