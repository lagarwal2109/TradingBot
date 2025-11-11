"""Backtesting engine for MACD strategy."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import sys
from pathlib import Path

# Add parent TradingBot to path
parent_bot_path = Path(__file__).parent.parent.parent
if str(parent_bot_path) not in sys.path:
    sys.path.insert(0, str(parent_bot_path))

from bot.datastore import DataStore
from .macd_strategy import MACDStrategy
from .config import MACDConfig, get_config


@dataclass
class Trade:
    """Trade record."""
    pair: str
    side: str  # "long" or "short"
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    amount: float = 0.0
    entry_value: float = 0.0
    exit_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    fees: float = 0.0
    realized_r: float = 0.0  # Realized P&L in R multiples (dollar-based)
    pnl_dollars: float = 0.0  # P&L in dollars (same as pnl, for clarity)
    risk_dollars: float = 0.0  # 1R in dollars (risk amount at entry)
    one_r_dollars: float = 0.0  # 1R in dollars (for diagnostics)


@dataclass
class Position:
    """Open position."""
    pair: str
    side: str
    entry_time: int  # Period number (row number)
    entry_price: float
    amount: float
    entry_value: float
    fees_paid: float
    initial_stop: float  # ATR-based initial stop
    stop_distance: float  # Stop distance in price terms (R = initial risk)
    bars_held: int = 0  # Number of bars position has been held
    scaled_out: bool = False  # Whether partial profit was taken
    scaled_out_pct: float = 0.0  # Percentage scaled out
    moved_to_breakeven: bool = False  # Whether stop moved to breakeven
    highest_price: float = 0.0  # For trailing stop (long)
    lowest_price: float = float('inf')  # For trailing stop (short)
    highest_high_22: float = 0.0  # For chandelier stop (long)
    lowest_low_22: float = float('inf')  # For chandelier stop (short)
    pyramid_count: int = 0  # Number of times position was pyramided
    pyramid_additions: List[Tuple[float, float, float]] = field(default_factory=list)  # (R_level, amount, entry_price) for each pyramid
    last_pyramid_r: float = 0.0  # R level of last pyramid addition
    # R accounting (dollar-based, consistent everywhere)
    risk_dollars: float = 0.0  # 1R in dollars (risk_amount at entry)
    r_per_unit: float = 0.0  # 1R in price units per unit (stop_distance)
    mfe_dollars: float = 0.0  # Maximum favorable excursion in dollars
    mae_dollars: float = 0.0  # Maximum adverse excursion in dollars


class MACDBacktester:
    """Backtesting engine for MACD strategy."""
    
    def __init__(
        self,
        config: Optional[MACDConfig] = None,
        initial_capital: float = 10000.0
    ):
        """Initialize backtester.
        
        Args:
            config: MACD configuration
            initial_capital: Starting capital
        """
        self.config = config or get_config()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Strategy (will use adaptive periods per pair)
        self.strategy = MACDStrategy(
            fast_period=self.config.macd_fast_period,
            slow_period=self.config.macd_slow_period,
            signal_period=self.config.macd_signal_period,
            trend_period=self.config.ma_trend_period
        )
        
        # Pass config values to strategy for stop calculations and gates
        self.strategy.k_atr_stop = self.config.k_atr_stop
        self.strategy.max_drawdown_pct = self.config.max_drawdown_pct
        self.strategy.min_adx = self.config.min_adx
        self.strategy.hist_z_score_min = self.config.hist_z_score_min
        self.strategy.atr_band_min = self.config.atr_band_min
        self.strategy.atr_band_max = self.config.atr_band_max
        
        # Data store
        self.datastore = DataStore(data_dir=self.config.data_dir)
        
        # State
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        
        # Trade rate limiting (5 trades per minute)
        self.trades_by_minute: Dict[str, int] = defaultdict(int)  # minute_key -> count
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        
        # Risk controls
        self.daily_equity_start: Optional[float] = None
        self.pair_loss_streaks: Dict[str, int] = defaultdict(int)  # pair -> consecutive losses
        self.pair_cooldown_until: Dict[str, int] = {}  # pair -> period when cooldown ends
        
        # Gate counter for diagnostics (track pass/fail counts per pair)
        self.gate_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "b_cross": 0,
            "b_regime": 0,
            "b_adx": 0,
            "b_histmag": 0,
            "b_slope": 0,
            "b_volwindow": 0,
            "b_warmup": 0
        })
    
    def _get_minute_key(self, timestamp) -> str:
        """Get minute key for rate limiting."""
        # With sequential periods, use period number as key
        if isinstance(timestamp, (int, np.integer)):
            return f"period_{timestamp}"
        else:
            return timestamp.strftime("%Y-%m-%d %H:%M")
    
    def _can_trade(self, timestamp) -> bool:
        """Check if we can trade (rate limit: 5 trades per minute)."""
        minute_key = self._get_minute_key(timestamp)
        return self.trades_by_minute[minute_key] < self.config.max_trades_per_minute
    
    def _record_trade(self, timestamp):
        """Record a trade for rate limiting."""
        minute_key = self._get_minute_key(timestamp)
        self.trades_by_minute[minute_key] += 1
    
    def _calculate_fee(self, value: float, pair: str = "") -> float:
        """Calculate transaction fee + slippage."""
        fee = value * self.config.transaction_fee_pct
        
        # Add slippage (majors: BTC/ETH, alts: others)
        is_major = pair.endswith("BTCUSD") or pair.endswith("ETHUSD") or "BTC" in pair.upper() or "ETH" in pair.upper()
        slippage_bps = self.config.slippage_bps_major if is_major else self.config.slippage_bps_alt
        slippage = value * (slippage_bps / 10000.0)  # Convert bps to decimal
        
        return fee + slippage
    
    def _open_position(
        self,
        pair: str,
        side: str,
        price: float,
        timestamp,  # Can be int (period) or pd.Timestamp
        stop_distance: float,
        timestamp_index: int
    ) -> bool:
        """Open a new position using risk-based sizing.
        
        Args:
            pair: Trading pair
            side: "long" or "short"
            price: Entry price
            timestamp: Entry timestamp
            stop_distance: ATR-based stop distance
            timestamp_index: Current timestamp index for cool-down tracking
            
        Returns:
            True if position opened successfully
        """
        # Check rate limit
        if not self._can_trade(timestamp):
            return False
        
        symbol = pair.replace("USD", "")
        
        # Check if we already have a position in this asset
        if symbol in self.positions:
            return False
        
        # BATS v1: Max 1 open position total
        if len(self.positions) >= self.config.max_open_positions:
            print(f"[VETO] {pair}: max open positions reached ({len(self.positions)} >= {self.config.max_open_positions})")
            return False
        
        # Check cool-down period
        cooldown_ok = True
        if pair in self.strategy.last_trade_time:
            bars_since_last = timestamp_index - self.strategy.last_trade_time[pair]
            if bars_since_last < self.config.cooldown_bars:
                print(f"[VETO] {pair}: cooldown (bars_since_last={bars_since_last} < {self.config.cooldown_bars})")
                return False
            cooldown_ok = bars_since_last >= self.config.cooldown_bars
        
        # Volatility-normalized position sizing (FIXED: no /100, already decimal)
        total_equity = self._calculate_equity({pair: price})
        
        # Ensure risk_per_trade_pct is valid
        risk_pct_per_trade = self.config.risk_per_trade_pct
        if risk_pct_per_trade <= 0 or not np.isfinite(risk_pct_per_trade):
            print(f"[VETO] {pair}: risk_pct_per_trade={risk_pct_per_trade} (invalid)")
            return False
        
        risk_dollars = total_equity * risk_pct_per_trade  # 0.004 = 0.4% = $40 on $10k
        
        if risk_dollars <= 0 or not np.isfinite(risk_dollars):
            print(f"[VETO] {pair}: risk_dollars={risk_dollars:.2f} (invalid, equity=${total_equity:.2f}, risk_pct={risk_pct_per_trade})")
            return False
        
        # Cap stop distance to max drawdown limit (15% hard limit)
        min_stop = price * 0.001  # 0.1% minimum
        max_stop = price * self.config.max_drawdown_pct  # 15% maximum (hard limit)
        stop_distance = max(min_stop, min(stop_distance, max_stop))
        
        # Additional safety: ensure stop distance never exceeds max_drawdown_pct
        if stop_distance > price * self.config.max_drawdown_pct:
            stop_distance = price * self.config.max_drawdown_pct
        
        # Validate stop distance
        if stop_distance <= 0 or not np.isfinite(stop_distance):
            print(f"[VETO] {pair}: stop_distance={stop_distance:.6f} (invalid, price=${price:.2f}, k_atr={self.config.k_atr_stop})")
            return False
        
        # Position size = Risk $ / Stop distance with safety checks
        amount = risk_dollars / stop_distance
        # Ensure minimum quantity (prevent zero-size positions)
        min_tick_size = 0.00000001  # Very small minimum
        amount = max(amount, min_tick_size)
        
        if amount <= 0 or not np.isfinite(amount):
            print(f"[VETO] {pair}: amount={amount:.6f} (invalid, risk_dollars=${risk_dollars:.2f}, stop_dist=${stop_distance:.6f})")
            return False
        
        # R accounting: store consistent R values
        r_per_unit = stop_distance  # 1R in price units per unit
        
        # Calculate position value (notional)
        position_value = amount * price
        
        # POSITION SIZING CAPS: Avoid 90% notional on 0.5% risk
        max_notional = total_equity * 0.25  # ≤25% of equity per position
        max_leverage = 1.0  # Spot = 1× leverage
        
        # Cap by notional
        if position_value > max_notional:
            amount = max_notional / price
        
        # Cap by leverage
        if (position_value / total_equity) > max_leverage:
            amount = (total_equity * max_leverage) / price
        
        # Recalculate position value after capping
        position_value = amount * price
        
        # PORTFOLIO-WIDE CAPS (dollar-based R)
        # Calculate aggregate risk across all open positions (use risk_dollars)
        portfolio_open_risk = sum(pos.risk_dollars for pos in self.positions.values())
        
        # Per-pair risk (dollar-based)
        per_pair_open_risk = sum(pos.risk_dollars for pos in self.positions.values() if pos.pair == pair)
        
        # New position risk (dollar-based)
        new_risk = risk_dollars
        
        # Calculate ATR for logging (from stop_distance)
        atr = stop_distance / self.config.k_atr_stop if self.config.k_atr_stop > 0 else 0.0
        
        # Daily loss check (in R terms) - sum of realized_r from trades closed today
        # Convert timestamp to date for comparison
        if isinstance(timestamp, pd.Timestamp):
            current_date = timestamp.date()
        elif isinstance(timestamp, int):
            # For sequential periods, use a simple day counter (approximate: 24 periods per day)
            # This is a fallback - ideally we'd have actual dates
            current_date = None
        else:
            current_date = None
        
        # Calculate realized_today_R as sum of realized_r from trades closed today
        realized_today_R = 0.0
        daily_loss_limit_R = self.config.daily_loss_limit_R  # -2R default for BATS v1
        
        if current_date is not None:
            # Sum realized_r from trades closed today
            for t in self.trades:
                if t.exit_time is not None:
                    if isinstance(t.exit_time, pd.Timestamp):
                        trade_date = t.exit_time.date()
                    else:
                        trade_date = None
                    
                    if trade_date == current_date:
                        realized_today_R += t.realized_r
        else:
            # Fallback: if we can't determine date, use last 24 periods (approximate 1 day)
            # Only count trades from recent periods
            recent_trades = [t for t in self.trades if t.exit_time is not None]
            if recent_trades:
                # Use last 24 trades as approximation (assuming ~1 trade per period)
                recent_trades = recent_trades[-24:]
                realized_today_R = sum(t.realized_r for t in recent_trades)
        
        # Clamp to reasonable range
        if not np.isfinite(realized_today_R) or abs(realized_today_R) > 50:
            realized_today_R = np.sign(realized_today_R) * 50 if np.isfinite(realized_today_R) else 0.0
        
        # Correlation check (placeholder - always pass for now)
        corr_ok = True
        
        # Comprehensive entry check logging
        print(f"[ENTRY_CHECK] {pair}: risk_pct={risk_pct_per_trade}, R$={risk_dollars:.2f}, "
              f"ATR=${atr:.6f}, k_atr={self.config.k_atr_stop}, stop_dist=${stop_distance:.6f}, "
              f"per_pair_open_risk=${per_pair_open_risk:.2f}, per_pair_cap=${total_equity * self.config.max_pair_risk_pct:.2f}, "
              f"portfolio_open_risk=${portfolio_open_risk:.2f}, portfolio_cap=${total_equity * self.config.max_portfolio_risk_pct:.2f}, "
              f"realized_today_R={realized_today_R:.2f}, daily_limit_R={daily_loss_limit_R:.2f}, "
              f"cooldown_ok={cooldown_ok}, corr_ok={corr_ok}, qty={amount:.6f}")
        
        # Daily loss limit check
        if realized_today_R <= -daily_loss_limit_R:
            print(f"[VETO] {pair}: daily loss limit (realized_today_R={realized_today_R:.2f} <= -{daily_loss_limit_R:.2f})")
            return False
        
        # Cap total open risk at portfolio limit (4% default)
        max_portfolio_risk = total_equity * self.config.max_portfolio_risk_pct
        if (portfolio_open_risk + new_risk) > max_portfolio_risk:
            print(f"[VETO] {pair}: portfolio cap (open=${portfolio_open_risk:.2f} + new=${new_risk:.2f} > cap=${max_portfolio_risk:.2f})")
            return False
        
        # Cap per-pair risk (1.2% default)
        max_pair_risk = total_equity * self.config.max_pair_risk_pct
        if (per_pair_open_risk + new_risk) > max_pair_risk:
            print(f"[VETO] {pair}: per-pair cap (open=${per_pair_open_risk:.2f} + new=${new_risk:.2f} > cap=${max_pair_risk:.2f})")
            return False
        
        # Cap gross exposure at 60% of equity
        total_notional = sum(pos.amount * pos.entry_price for pos in self.positions.values())
        max_total_notional = total_equity * 0.6
        if (total_notional + position_value) > max_total_notional:
            print(f"[VETO] {pair}: gross exposure cap (total=${total_notional:.2f} + new=${position_value:.2f} > cap=${max_total_notional:.2f})")
            return False
        
        # Final validation after capping
        if amount <= 0 or not np.isfinite(amount) or position_value <= 0:
            print(f"[VETO] {pair}: invalid amount/value after capping (amount={amount:.6f}, value=${position_value:.2f})")
            return False
        
        # Calculate fees (with slippage)
        fees = self._calculate_fee(position_value, pair)
        
        # Check if we have enough capital (including fees)
        if self.capital < position_value + fees:
            print(f"[VETO] {pair}: insufficient capital (capital=${self.capital:.2f} < needed=${position_value + fees:.2f})")
            return False
        
        # Calculate initial stop price
        if side == "long":
            initial_stop = price - stop_distance
        else:  # short
            initial_stop = price + stop_distance
        
        # Sanity check: print first 5 orders for debugging
        if self.total_trades < 5:
            print(f"[SIZE_CHECK] Trade #{self.total_trades + 1} {pair}: equity=${total_equity:.2f}, "
                  f"risk_dollars=${risk_dollars:.2f}, ATR=${stop_distance/self.config.k_atr_stop:.4f}, "
                  f"stop_dist=${stop_distance:.4f}, qty={amount:.6f}, 1R=${risk_dollars:.2f}, R_per_unit=${r_per_unit:.4f}")
        
        # Open position - use timestamp directly (it's the period number/row number)
        self.positions[symbol] = Position(
            pair=pair,
            side=side,
            entry_time=timestamp,  # Period number (row number)
            entry_price=price,
            amount=amount,
            entry_value=position_value,
            fees_paid=fees,
            initial_stop=initial_stop,
            stop_distance=stop_distance,
            bars_held=0,
            scaled_out=False,
            scaled_out_pct=0.0,
            moved_to_breakeven=False,
            highest_price=price if side == "long" else 0.0,
            lowest_price=price if side == "short" else float('inf'),
            highest_high_22=price if side == "long" else 0.0,
            lowest_low_22=price if side == "short" else float('inf'),
            risk_dollars=risk_dollars,  # Store 1R in dollars
            r_per_unit=r_per_unit,  # Store 1R in price units per unit
            mfe_dollars=0.0,  # Track maximum favorable excursion
            mae_dollars=0.0  # Track maximum adverse excursion
        )
        
        # Deduct capital
        self.capital -= (position_value + fees)
        self.total_fees += fees
        self._record_trade(timestamp)
        
        # Update cool-down tracking
        self.strategy.last_trade_time[pair] = timestamp_index
        self.strategy.last_trade_direction[pair] = side
        
        return True
    
    def _close_position(
        self,
        symbol: str,
        price: float,
        timestamp,  # Can be int (period) or pd.Timestamp
        reason: str,
        partial: bool = False
    ) -> Optional[Trade]:
        """Close an existing position (full or partial).
        
        Args:
            symbol: Symbol to close
            price: Exit price
            timestamp: Exit timestamp
            reason: Exit reason
            partial: If True, close 50% of position (scale-out)
            
        Returns:
            Trade record if position was closed (or partially closed)
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate exit amount
        if partial:
            # Partial profit: take configured percentage (30% default)
            exit_pct = self.config.partial_profit_pct  # 0.30 = 30%
            exit_amount = position.amount * exit_pct
            remaining_amount = position.amount * (1 - exit_pct)
            
            # After partial, tighten stop to +0.5R
            if position.side == "long":
                risk = position.entry_price - position.initial_stop
                new_stop = position.entry_price + (risk * self.config.partial_stop_r)
                position.initial_stop = max(position.initial_stop, new_stop)  # Never lower than original
            else:  # short
                risk = position.initial_stop - position.entry_price
                new_stop = position.entry_price - (risk * self.config.partial_stop_r)
                position.initial_stop = min(position.initial_stop, new_stop)  # Never higher than original
        else:
            exit_amount = position.amount
            remaining_amount = 0.0
        
        # Calculate exit value
        exit_value = exit_amount * price
        
        # Calculate fees
        fees = self._calculate_fee(exit_value)
        
        # Calculate P&L for closed portion (weighted average entry for pyramided positions)
        # For pyramided positions, use weighted average entry price
        if position.pyramid_count > 0:
            # Weighted average entry = total_cost / total_amount
            weighted_entry = position.entry_value / position.amount if position.amount > 0 else position.entry_price
        else:
            weighted_entry = position.entry_price
        
        exit_entry_value = weighted_entry * exit_amount
        exit_fees_proportion = position.fees_paid * (exit_amount / position.amount) if position.amount > 0 else 0
        
        if position.side == "long":
            pnl = exit_value - exit_entry_value - fees - exit_fees_proportion
        else:  # short
            pnl = exit_entry_value - exit_value - fees - exit_fees_proportion
        
        pnl_pct = (pnl / exit_entry_value) * 100 if exit_entry_value > 0 else 0
        
        # Calculate realized R (dollar-based)
        # For pyramided positions, use total risk_dollars (sum of all legs)
        realized_r = 0.0
        if position.risk_dollars > 0:
            realized_r = pnl / position.risk_dollars
        elif position.r_per_unit > 0 and exit_amount > 0:
            # Fallback: calculate from price-based R if risk_dollars not set
            risk_per_unit = position.r_per_unit * exit_amount
            if risk_per_unit > 0:
                realized_r = pnl / (risk_per_unit * position.entry_price)
        
        # Clamp absurd values (guardrails)
        if not np.isfinite(realized_r) or abs(realized_r) > 20:
            realized_r = np.sign(realized_r) * 20 if np.isfinite(realized_r) else 0.0
        
        # Store 1R in dollars for diagnostics
        one_r_dollars = position.risk_dollars if position.risk_dollars > 0 else 0.0
        
        # Add capital back
        proceeds = exit_value - fees
        self.capital += proceeds
        self.total_fees += fees
        
        # Create trade record
        trade = Trade(
            pair=position.pair,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            amount=exit_amount,
            entry_value=exit_entry_value,
            exit_value=exit_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            fees=fees + (position.fees_paid * (exit_amount / position.amount)),
            realized_r=realized_r,
            pnl_dollars=pnl,
            risk_dollars=position.risk_dollars,
            one_r_dollars=one_r_dollars
        )
        
        # Update metrics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update or remove position
        if partial:
            # Update position for remaining 50%
            position.amount = remaining_amount
            position.entry_value = position.entry_value * 0.5
            position.fees_paid = position.fees_paid * 0.5
            position.scaled_out = True
            # Move stop to breakeven
            position.initial_stop = position.entry_price
        else:
            # Remove position completely
            del self.positions[symbol]
        
        self._record_trade(timestamp)
        
        return trade
    
    def _calculate_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity (cash + positions) using valid prices only.
        
        For shorts: PnL = (entry_price - current_price) * amount
        Equity = cash + ΣPnL for all positions
        """
        equity = self.capital  # Start with cash
        
        for symbol, position in self.positions.items():
            pair = position.pair
            if pair in prices:
                current_price = prices[pair]
                # Use valid price only (mark-to-market safety)
                if current_price > 0 and np.isfinite(current_price):
                    if position.side == "long":
                        # Long PnL = (current_price - entry_price) * amount
                        pnl = (current_price - position.entry_price) * position.amount
                        equity += pnl
                    else:  # short
                        # Short PnL = (entry_price - current_price) * amount
                        pnl = (position.entry_price - current_price) * position.amount
                        equity += pnl
                # If price invalid, don't add PnL (prevents equity from going to zero due to bad data)
        
        return equity
    
    def run_backtest(
        self,
        pairs: List[str],
        days: int = 15,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Dict:
        """Run backtest on historical data.
        
        Args:
            pairs: List of trading pairs to backtest
            days: Number of days to backtest
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Dictionary with backtest results
        """
        # Load data for all pairs with data quality filters
        all_data: Dict[str, pd.DataFrame] = {}  # Period-indexed (for backtest loop)
        all_data_datetime: Dict[str, pd.DataFrame] = {}  # Datetime-indexed (for HTF regime)
        timestamps = set()
        valid_pairs = []
        
        for pair in pairs:
            df = self.datastore.read_minute_bars(pair)
            if len(df) == 0:
                continue
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            
            # Store DatetimeIndex version BEFORE converting to periods (for HTF regime)
            if isinstance(df.index, pd.DatetimeIndex):
                all_data_datetime[pair] = df.copy()
            
            # Convert DatetimeIndex to sequential periods AFTER storing datetime version
            # This allows HTF regime computation to work properly
            if isinstance(df.index, pd.DatetimeIndex):
                # Store original datetime for reference, but convert to periods
                df = df.reset_index(drop=True)
                df.index = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
                df.index.name = 'period'
            
            # Data quality filter: skip assets with invalid data
            if "price" not in df.columns:
                print(f"[SKIP] {pair}: No price column")
                continue
            
            # Check last 200 bars for data quality (or all if less than 200)
            check_bars = min(200, len(df))
            recent_data = df.tail(check_bars)
            
            # Filter: price must be > 0.05 and finite (skip penny stocks)
            valid_prices = (recent_data["price"] > 0.05) & np.isfinite(recent_data["price"])
            if not valid_prices.all():
                print(f"[SKIP] {pair}: Invalid prices (zero, non-finite, or < $0.05)")
                continue
            
            # Filter: price std must be > 0 (not flat)
            if len(recent_data) >= 100:
                price_std = recent_data["price"].std()
                if price_std <= 0 or not np.isfinite(price_std):
                    print(f"[SKIP] {pair}: Price std is zero (flat price)")
                    continue
            
            # Filter: ATR must be > 0 (calculate ATR for validation)
            if len(recent_data) >= 14:
                high_series = recent_data["high"] if "high" in recent_data.columns else recent_data["price"] * 1.001
                low_series = recent_data["low"] if "low" in recent_data.columns else recent_data["price"] * 0.999
                atr_series = self.strategy.calculate_atr(high_series, low_series, recent_data["price"], period=14)
                if len(atr_series) > 0:
                    last_atr = atr_series.iloc[-1]
                    if pd.isna(last_atr) or last_atr <= 0 or not np.isfinite(last_atr):
                        print(f"[SKIP] {pair}: Invalid ATR (zero or non-finite)")
                        continue
            
            # All checks passed
            all_data[pair] = df
            timestamps.update(df.index)
            valid_pairs.append(pair)
        
        # Update pairs list to only valid pairs
        pairs = valid_pairs
        
        if len(timestamps) == 0:
            raise ValueError("No data available for backtesting")
        
        # Sort timestamps (now they are sequential periods: 1, 2, 3, ...)
        timestamps = sorted(timestamps)
        
        # Filter by date range (if provided, convert to period numbers)
        # Note: With sequential periods, date filtering is less meaningful
        # but we keep it for compatibility
        if start_date:
            # Convert start_date to period if needed (for compatibility)
            timestamps = [t for t in timestamps if isinstance(t, int) or (isinstance(t, pd.Timestamp) and t >= start_date)]
        if end_date:
            timestamps = [t for t in timestamps if isinstance(t, int) or (isinstance(t, pd.Timestamp) and t <= end_date)]
        
        # If days specified, use last N periods (approximate: assume 24 periods per day for 1H bars)
        if days and not start_date:
            periods_per_day = 24  # 1H bars = 24 per day
            if len(timestamps) > days * periods_per_day:
                timestamps = timestamps[-(days * periods_per_day):]
        
        print(f"Running backtest on {len(pairs)} pairs from period {timestamps[0]} to {timestamps[-1]}")
        print(f"Total periods: {len(timestamps)}")
        
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.trades_by_minute = defaultdict(int)
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        self.daily_equity_start = None  # Will be set at start of each day
        self.last_trading_day = None  # Track current trading day for daily loss reset
        
        # Run backtest
        total_timestamps = len(timestamps)
        progress_interval = max(100, total_timestamps // 20)  # Show progress every 5%
        
        for i, period_num in enumerate(timestamps):
            # Use period number directly as timestamp (row number)
            timestamp = period_num
            
            # Initialize daily equity at start of backtest
            if self.daily_equity_start is None:
                self.daily_equity_start = self._calculate_equity({})
            
            # Reset daily tracking at start of new trading day
            # For sequential periods, approximate: reset every 24 periods (1 day for 1H bars)
            if isinstance(timestamp, int):
                current_day = timestamp // 24  # Approximate day number
                if self.last_trading_day is None:
                    self.last_trading_day = current_day
                elif current_day > self.last_trading_day:
                    # New day: reset daily equity start
                    self.daily_equity_start = self._calculate_equity({})
                    self.last_trading_day = current_day
            elif isinstance(timestamp, pd.Timestamp):
                current_date = timestamp.date()
                if self.last_trading_day is None:
                    self.last_trading_day = current_date
                elif current_date > self.last_trading_day:
                    # New day: reset daily equity start
                    self.daily_equity_start = self._calculate_equity({})
                    self.last_trading_day = current_date
            
            # Progress indicator
            if i % progress_interval == 0 or i == total_timestamps - 1:
                progress_pct = (i + 1) / total_timestamps * 100
                equity = self._calculate_equity({})
                print(f"Progress: {i+1}/{total_timestamps} ({progress_pct:.1f}%) | Equity: ${equity:.2f} | Open Positions: {len(self.positions)} | Trades: {self.total_trades}")
            
            # Get current prices (timestamp is now a period number: 1, 2, 3, ...)
            current_prices: Dict[str, float] = {}
            for pair, df in all_data.items():
                if timestamp in df.index:
                    current_prices[pair] = df.loc[timestamp, "price"]
                elif len(df) > 0:
                    # Use last known price (period <= current period)
                    mask = df.index <= timestamp
                    if mask.any():
                        current_prices[pair] = df[mask]["price"].iloc[-1]
            
            # Check existing positions for exits
            positions_to_close = []
            for symbol, position in list(self.positions.items()):
                pair = position.pair
                if pair in current_prices:
                    price = current_prices[pair]
                    
                    # Update bars held
                    position.bars_held += 1
                    
                    # Check minimum holding period
                    if position.bars_held < self.config.min_holding_bars:
                        continue  # Skip exit checks if below min holding period
                    
                    # Get historical prices for ATR calculation (use full DataFrame with OHLCV if available)
                    pair_df = all_data[pair]
                    mask = pair_df.index <= timestamp
                    historical_data = pair_df[mask]
                    historical_prices = historical_data["price"]
                    
                    # Calculate current ATR (use real high/low if available, otherwise close-only)
                    if len(historical_prices) >= 14:
                        if "high" in historical_data.columns and "low" in historical_data.columns:
                            # Real OHLCV data available
                            high_series = historical_data["high"]
                            low_series = historical_data["low"]
                            atr_series = self.strategy.calculate_atr(high_series, low_series, historical_prices, period=14)
                            current_atr = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else position.stop_distance / 3.0
                        else:
                            # Price-only data: use close-only ATR calculation
                            atr_series = self.strategy.calculate_atr_from_close(historical_prices, period=14, min_ratio=0.001)
                            current_atr = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else position.stop_distance / 3.0
                            # For trailing stop calculation, we still need high/low approximations
                            high_series = historical_prices * 1.002
                            low_series = historical_prices * 0.998
                    else:
                        current_atr = position.stop_distance / 3.0
                    
                    # Calculate unrealized P&L in dollars (consistent R accounting)
                    if position.side == "long":
                        unrealized_pnl_dollars = (price - position.entry_price) * position.amount
                    else:  # short
                        unrealized_pnl_dollars = (position.entry_price - price) * position.amount
                    
                    # Update MFE/MAE tracking
                    if unrealized_pnl_dollars > position.mfe_dollars:
                        position.mfe_dollars = unrealized_pnl_dollars
                    if unrealized_pnl_dollars < position.mae_dollars:
                        position.mae_dollars = unrealized_pnl_dollars
                    
                    # Calculate R multiples (dollar-based)
                    profit_r = unrealized_pnl_dollars / position.risk_dollars if position.risk_dollars > 0 else 0.0
                    
                    # Get current bar's high/low for hard stop check (gap-aware)
                    current_low = price
                    current_high = price
                    if "low" in historical_data.columns and "high" in historical_data.columns:
                        if timestamp in historical_data.index:
                            current_low = historical_data.loc[timestamp, "low"]
                            current_high = historical_data.loc[timestamp, "high"]
                    
                    # HARD STOP ENFORCEMENT (before other exit logic) - gap-aware
                    # Determine effective stop price (breakeven if moved, otherwise initial stop)
                    effective_stop = position.initial_stop
                    if position.moved_to_breakeven:
                        effective_stop = position.entry_price
                    
                    # Check if stop was hit this bar
                    stop_hit = False
                    stop_exit_price = price  # Default to current price
                    
                    if position.side == "long":
                        # Long: stop hit if low <= stop_price
                        if current_low <= effective_stop:
                            stop_hit = True
                            # Gap-aware: use max of stop_price and next open (or current price if gap down)
                            stop_exit_price = max(effective_stop, price)
                    else:  # short
                        # Short: stop hit if high >= stop_price
                        if current_high >= effective_stop:
                            stop_hit = True
                            # Gap-aware: use min of stop_price and next open (or current price if gap up)
                            stop_exit_price = min(effective_stop, price)
                    
                    # If hard stop hit, exit immediately (before other exit logic)
                    if stop_hit:
                        stop_reason = "initial_stop" if not position.moved_to_breakeven else "breakeven_stop"
                        positions_to_close.append((symbol, stop_exit_price, stop_reason, False))  # is_partial=False for stops
                        print(f"[STOP_HIT] {symbol} @ ${stop_exit_price:.2f} (stop=${effective_stop:.2f}, low=${current_low:.2f}, high=${current_high:.2f})")
                        continue  # Skip other exit logic for this position
                    
                    # Update price extremes for chandelier stop
                    if position.side == "long":
                        position.highest_price = max(position.highest_price, price)
                        # Update 22-bar highest high for chandelier
                        if len(historical_prices) >= 22:
                            position.highest_high_22 = max(historical_prices.iloc[-22:].max(), position.highest_high_22)
                        else:
                            position.highest_high_22 = max(historical_prices.max(), position.highest_high_22)
                    else:  # short
                        position.lowest_price = min(position.lowest_price, price)
                        # Update 22-bar lowest low for chandelier
                        if len(historical_prices) >= 22:
                            position.lowest_low_22 = min(historical_prices.iloc[-22:].min(), position.lowest_low_22)
                        else:
                            position.lowest_low_22 = min(historical_prices.min(), position.lowest_low_22)
                    
                    # Exit logic: Breakeven at +1R (dollar-based), Chandelier at +2R, Time stop at 15 bars
                    breakeven_stop = None
                    trailing_stop = None
                    
                    # Move to breakeven at +1R (dollar-based, once)
                    if unrealized_pnl_dollars >= position.risk_dollars and not position.moved_to_breakeven:
                        breakeven_stop = position.entry_price
                        position.moved_to_breakeven = True
                    elif position.moved_to_breakeven:
                        breakeven_stop = position.entry_price
                    
                    # Chandelier trailing stop at +2R (dollar-based)
                    if profit_r >= self.config.chandelier_at_r:
                        if position.side == "long":
                            chandelier_stop = position.highest_high_22 - (self.config.chandelier_c * current_atr)
                            trailing_stop = max(position.initial_stop, chandelier_stop)  # Never below initial stop
                        else:  # short
                            chandelier_stop = position.lowest_low_22 + (self.config.chandelier_c * current_atr)
                            trailing_stop = min(position.initial_stop, chandelier_stop)  # Never above initial stop
                    else:
                        # Before +2R, no trailing stop (only initial stop and breakeven)
                        trailing_stop = None
                    
                    # Pyramiding: Add to winning positions (dollar-based R)
                    if self.config.enable_pyramiding and position.pyramid_count < self.config.max_pyramid_adds:
                        # Use dollar-based R for pyramiding triggers
                        last_pyramid_r = position.last_pyramid_r if position.pyramid_count > 0 else 0.0
                        r_since_last_add = profit_r - last_pyramid_r
                        
                        # Add at +1R, +2R, +3R (dollar-based)
                        target_r = (position.pyramid_count + 1) * self.config.pyramid_spacing_r  # +1R, +2R, +3R
                        
                        # Add when profit reaches target R from initial entry
                        if profit_r >= target_r and r_since_last_add >= self.config.pyramid_spacing_r:
                                # Check if we already pyramided at this R level (prevent duplicates)
                                already_pyramided_at_r = any(
                                    abs(pyr_r - profit_r) < 0.05  # Within 0.05R
                                    for pyr_r, _, _ in position.pyramid_additions
                                )
                                
                                if not already_pyramided_at_r:
                                    # R-based laddering with diminishing sizes
                                    # Get initial risk amount (R = stop_distance)
                                    initial_risk = position.stop_distance
                                    
                                    # Get pyramid size multiplier from config (diminishing: [1.0, 0.7, 0.5])
                                    pyramid_idx = min(position.pyramid_count, len(self.config.pyramid_sizes) - 1)
                                    pyramid_size_mult = self.config.pyramid_sizes[pyramid_idx]
                                    
                                    # Calculate pyramid risk amount (as % of initial risk)
                                    pyramid_risk_amount = initial_risk * pyramid_size_mult
                                    
                                    # Calculate pyramid position size
                                    # Use same stop distance as initial position
                                    pyramid_amount = pyramid_risk_amount / position.stop_distance
                                    pyramid_value = pyramid_amount * price
                                    pyramid_fees = self._calculate_fee(pyramid_value)
                                    
                                    # Check momentum conditions (ADX >= 20, dMACD > 0 for longs)
                                    # Get current ADX and MACD slope from signal (would need to pass these)
                                    # For now, skip momentum check (can add later)
                                    
                                    # Check if prior unit is locked at breakeven
                                    if not position.moved_to_breakeven:
                                        continue  # Don't add until prior unit is at breakeven
                                    
                                    # Check per-pair and portfolio risk caps
                                    total_equity = self._calculate_equity({position.pair: price})
                                    
                                    # Per-pair risk check
                                    pair_risk = 0.0
                                    for pos in self.positions.values():
                                        if pos.pair == position.pair:
                                            if pos.side == "long":
                                                pair_risk += abs(pos.entry_price - pos.initial_stop) * pos.amount
                                            else:
                                                pair_risk += abs(pos.initial_stop - pos.entry_price) * pos.amount
                                    
                                    new_pyramid_risk = pyramid_risk_amount * pyramid_amount  # Approximate
                                    if (pair_risk + new_pyramid_risk) > (total_equity * self.config.max_pair_risk_pct):
                                        continue  # Would exceed per-pair risk
                                    
                                    # Portfolio risk check
                                    portfolio_risk = 0.0
                                    for pos in self.positions.values():
                                        if pos.side == "long":
                                            portfolio_risk += abs(pos.entry_price - pos.initial_stop) * pos.amount
                                        else:
                                            portfolio_risk += abs(pos.initial_stop - pos.entry_price) * pos.amount
                                    
                                    if (portfolio_risk + new_pyramid_risk) > (total_equity * self.config.max_portfolio_risk_pct):
                                        continue  # Would exceed portfolio risk
                                    
                                    # Check if we have enough capital
                                    if self.capital >= pyramid_value + pyramid_fees:
                                        # Add to position
                                        position.amount += pyramid_amount
                                        position.entry_value += pyramid_value
                                        position.pyramid_count += 1
                                        position.pyramid_additions.append((profit_r, pyramid_amount, price))
                                        position.last_pyramid_r = profit_r
                                        
                                        # Deduct capital
                                        self.capital -= (pyramid_value + pyramid_fees)
                                        self.total_fees += pyramid_fees
                                        
                                        print(f"[PYRAMID] {symbol} @ ${price:.2f}: Added {pyramid_amount:.6f} units at {profit_r:.2f}R profit (size={pyramid_size_mult*100:.0f}% of initial, total: {position.amount:.6f} units, {position.pyramid_count}/{self.config.max_pyramid_adds} pyramids)")
                    
                    # Hard max drawdown check (15% from entry) - prevents horrible trades
                    max_drawdown_stop_hit = False
                    if position.side == "long":
                        max_drawdown_stop = position.entry_price * (1 - self.config.max_drawdown_pct)
                        if price <= max_drawdown_stop:
                            max_drawdown_stop_hit = True
                            should_exit, reason, is_partial = True, "max_drawdown_stop", False
                    else:  # short
                        max_drawdown_stop = position.entry_price * (1 + self.config.max_drawdown_pct)
                        if price >= max_drawdown_stop:
                            max_drawdown_stop_hit = True
                            should_exit, reason, is_partial = True, "max_drawdown_stop", False
                    
                    # Check exit conditions using new ATR-based logic (if max drawdown not hit)
                    if not max_drawdown_stop_hit:
                        # Calculate R for exit logic
                        if position.side == "long":
                            risk = position.entry_price - position.initial_stop
                            profit_r = (price - position.entry_price) / risk if risk > 0 else 0.0
                        else:  # short
                            risk = position.initial_stop - position.entry_price
                            profit_r = (position.entry_price - price) / risk if risk > 0 else 0.0
                        
                        # Time stop: exit after 15 bars if profit between -0.5R and +0.5R (kill dead trades)
                        if position.bars_held >= self.config.time_stop_bars and -0.5 <= profit_r <= 0.5:
                            should_exit, reason, is_partial = True, "time_stop", False
                        # Partial profit at +2R (take 30%, tighten stop to +0.5R)
                        elif profit_r >= self.config.partial_profit_r and not position.scaled_out:
                            should_exit, reason, is_partial = True, "take_profit_partial", True
                        # Check stops (chandelier/trailing, then breakeven, then initial)
                        elif trailing_stop is not None:
                            if position.side == "long" and price <= trailing_stop:
                                should_exit, reason, is_partial = True, "trailing_stop", False
                            elif position.side == "short" and price >= trailing_stop:
                                should_exit, reason, is_partial = True, "trailing_stop", False
                            elif breakeven_stop is not None:
                                if position.side == "long" and price <= breakeven_stop:
                                    should_exit, reason, is_partial = True, "breakeven_stop", False
                                elif position.side == "short" and price >= breakeven_stop:
                                    should_exit, reason, is_partial = True, "breakeven_stop", False
                                elif position.side == "long" and price <= position.initial_stop:
                                    should_exit, reason, is_partial = True, "initial_stop", False
                                elif position.side == "short" and price >= position.initial_stop:
                                    should_exit, reason, is_partial = True, "initial_stop", False
                                else:
                                    should_exit, reason, is_partial = False, "", False
                            elif position.side == "long" and price <= position.initial_stop:
                                should_exit, reason, is_partial = True, "initial_stop", False
                            elif position.side == "short" and price >= position.initial_stop:
                                should_exit, reason, is_partial = True, "initial_stop", False
                            else:
                                should_exit, reason, is_partial = False, "", False
                        elif breakeven_stop is not None:
                            if position.side == "long" and price <= breakeven_stop:
                                should_exit, reason, is_partial = True, "breakeven_stop", False
                            elif position.side == "short" and price >= breakeven_stop:
                                should_exit, reason, is_partial = True, "breakeven_stop", False
                            elif position.side == "long" and price <= position.initial_stop:
                                should_exit, reason, is_partial = True, "initial_stop", False
                            elif position.side == "short" and price >= position.initial_stop:
                                should_exit, reason, is_partial = True, "initial_stop", False
                            else:
                                should_exit, reason, is_partial = False, "", False
                        elif position.side == "long" and price <= position.initial_stop:
                            should_exit, reason, is_partial = True, "initial_stop", False
                        elif position.side == "short" and price >= position.initial_stop:
                            should_exit, reason, is_partial = True, "initial_stop", False
                        else:
                            should_exit, reason, is_partial = False, "", False
                    
                    if should_exit:
                        positions_to_close.append((symbol, price, reason, is_partial))
            
            # Close positions
            for symbol, price, reason, is_partial in positions_to_close:
                trade = self._close_position(symbol, price, timestamp, reason, partial=is_partial)
                if trade:
                    self.trades.append(trade)
            
            # Generate signals for pairs we don't have positions in
            for pair in pairs:
                if pair not in current_prices:
                    continue
                
                symbol = pair.replace("USD", "")
                if symbol in self.positions:
                    continue  # Already have position
                
                # Get historical data up to current timestamp
                if pair not in all_data:
                    continue
                
                pair_df = all_data[pair]
                mask = pair_df.index <= timestamp
                historical_data = pair_df[mask]
                historical_prices = historical_data["price"]  # Use price column for signal generation
                
                # For BATS v1 Donchian, we use the same data source for prices and high/low
                # Don't switch to datetime version as it causes index misalignment
                # (Donchian doesn't need HTF regime computation like MACD did)
                
                # BATS v1: Donchian breakout requires fixed periods
                # Need: 200 for EMA200, 50 for EMA50, 20 for Donchian, 14 for ATR/ADX
                data_points = len(historical_prices)
                min_required = 200  # Minimum for EMA200
                
                if data_points < min_required:
                    continue
                
                # Generate signal using Donchian breakout (BATS v1)
                try:
                    # Get high/low if available for Donchian calculation
                    # Since we're using the same data source, they should already be aligned
                    historical_high = None
                    historical_low = None
                    if "high" in historical_data.columns and "low" in historical_data.columns:
                        historical_high = historical_data["high"]
                        historical_low = historical_data["low"]
                        # Ensure they're Series with same index as prices
                        if not isinstance(historical_high, pd.Series):
                            historical_high = pd.Series(historical_high, index=historical_prices.index)
                        if not isinstance(historical_low, pd.Series):
                            historical_low = pd.Series(historical_low, index=historical_prices.index)
                        # Align indices just to be safe
                        historical_high = historical_high.reindex(historical_prices.index, method='ffill')
                        historical_low = historical_low.reindex(historical_prices.index, method='ffill')
                    
                    # Use Donchian breakout method (BATS v1)
                    signal = self.strategy.generate_donchian_signal(
                        pair, 
                        historical_prices,
                        high=historical_high,
                        low=historical_low
                    )
                    
                    # Track gate pass counts for diagnostics (BATS v1: Donchian breakout)
                    if signal.b_cross:
                        self.gate_counts[pair]["b_cross"] += 1
                    if signal.b_regime:
                        self.gate_counts[pair]["b_regime"] += 1
                    if signal.b_adx:
                        self.gate_counts[pair]["b_adx"] += 1
                    if signal.b_volwindow:
                        self.gate_counts[pair]["b_volwindow"] += 1
                    if signal.b_warmup:
                        self.gate_counts[pair]["b_warmup"] += 1
                    
                    # Print gate summary only when a candidate passes breakout or is rejected by last failing gate
                    # Only log every 100 periods to reduce noise
                    if i % 100 == 0 and i > 0:
                        for p, counts in self.gate_counts.items():
                            print(f"[GATES] {p}: breakout={counts['b_cross']}, regime={counts['b_regime']}, "
                                  f"adx={counts['b_adx']}, volwindow={counts['b_volwindow']}, "
                                  f"warmup={counts['b_warmup']}")
                    
                    # Enhanced diagnostics: log detailed values only when signal generated or rejected
                    if signal.signal_type != "none" or (signal.b_cross and not signal.b_warmup):
                        if signal.signal_type != "none":
                            print(f"[DEBUG] {timestamp} {pair}: Signal={signal.signal_type} | "
                                  f"price={signal.price:.2f}, ATR={signal.atr:.4f}, "
                                  f"ADX={signal.adx:.1f}, Donchian_H={signal.donchian_high_20:.2f}, "
                                  f"Donchian_L={signal.donchian_low_20:.2f}, EMA50={signal.ema_50:.2f}, "
                                  f"EMA200={signal.ema_200:.2f}")
                        # Also log when gates fail (to identify bottlenecks)
                        elif signal.b_cross:  # Breakout detected but no signal
                            print(f"[GATE_FAIL] {timestamp} {pair}: Breakout=✓ but signal=none | "
                                  f"price={signal.price:.2f}, ATR={signal.atr:.4f}, "
                                  f"warmup={signal.b_warmup}, regime={signal.b_regime}, adx={signal.b_adx}, "
                                  f"volwindow={signal.b_volwindow} | "
                                  f"Donchian_H={signal.donchian_high_20:.2f}, Donchian_L={signal.donchian_low_20:.2f}")
                        # Log data quality issues - but ATR should never be 0 due to floor, so this is a bug if it happens
                        if signal.price <= 0 or (signal.atr <= 0 and signal.signal_type == "none"):
                            print(f"[DATA_ERROR] {timestamp} {pair}: Invalid data | price={signal.price:.2f}, ATR={signal.atr:.4f}")
                    
                    # Open position if signal
                    if signal.signal_type == "long" or signal.signal_type == "short":
                        opened = self._open_position(
                            pair, 
                            signal.signal_type, 
                            signal.price, 
                            timestamp,
                            signal.stop_distance,
                            i
                        )
                        if opened:
                            position = self.positions[symbol]
                            
                            # Calculate targets for logging (correct risk calculation)
                            if signal.signal_type == "long":
                                initial_stop_price = signal.price - signal.stop_distance
                                risk = signal.price - initial_stop_price
                                target_partial = signal.price + (risk * 0.6)  # 0.6R
                                target_full = signal.price + (risk * 1.5)
                            else:  # short
                                initial_stop_price = signal.price + signal.stop_distance
                                risk = initial_stop_price - signal.price
                                target_partial = signal.price - (risk * 0.6)  # 0.6R
                                target_full = signal.price - (risk * 1.5)
                            
                            atr_ratio = signal.atr / signal.price  # Decimal format
                            notional = position.amount * signal.price
                            
                            print(f"[SIGNAL] {timestamp} {pair}: {signal.signal_type.upper()} @ ${signal.price:.2f} | "
                                  f"stop_init=${initial_stop_price:.2f}, 0.6R=${target_partial:.2f}, 1.5R=${target_full:.2f} | "
                                  f"qty={position.amount:.6f}, notional=${notional:.2f}, risk$=${position.risk_dollars:.2f} | "
                                  f"ATR=${signal.atr:.4f}, ATR/price={atr_ratio:.4f} ({atr_ratio*100:.2f}%), ADX={signal.adx:.1f}, "
                                  f"Donchian_H={signal.donchian_high_20:.2f}, Donchian_L={signal.donchian_low_20:.2f}, "
                                  f"EMA50={signal.ema_50:.2f}, EMA200={signal.ema_200:.2f}")
                        elif i % 1000 == 0:
                            print(f"[SKIP] {timestamp} {pair}: Signal generated but position not opened (rate limit/max positions/cool-down)")
                except Exception as e:
                    # Skip if error generating signal
                    if i % 1000 == 0:
                        print(f"  [ERROR] {pair}: {e}")
                    continue
            
            # Record equity
            equity = self._calculate_equity(current_prices)
            self.equity_curve.append((timestamp, equity))
        
        # End-of-test flatten: close all open positions at last valid price
        print("\n[FLATTEN] Closing all open positions at end of backtest...")
        for symbol, position in list(self.positions.items()):
            pair = position.pair
            # Get last valid price
            if pair in all_data:
                pair_df = all_data[pair]
                if len(pair_df) > 0:
                    last_price = pair_df["price"].iloc[-1]
                    if last_price > 0 and np.isfinite(last_price):
                        trade = self._close_position(symbol, last_price, timestamps[-1], "end_of_test", partial=False)
                        if trade:
                            self.trades.append(trade)
                            print(f"  Closed {symbol} @ ${last_price:.2f}")
                    else:
                        # Use entry price if last price invalid
                        trade = self._close_position(symbol, position.entry_price, timestamps[-1], "end_of_test_invalid_price", partial=False)
                        if trade:
                            self.trades.append(trade)
                            print(f"  Closed {symbol} @ entry price ${position.entry_price:.2f} (invalid last price)")
        
        # Clear positions after flattening
        self.positions = {}
        
        # Calculate final metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest metrics."""
        if len(self.equity_curve) == 0:
            return {}
        
        equity_series = pd.Series([e[1] for e in self.equity_curve])
        final_equity = equity_series.iloc[-1]
        
        # Total return
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # Returns - use log returns from equity curve
        equity_array = np.array([e[1] for e in self.equity_curve])
        returns = np.diff(np.log(equity_array))  # Log returns
        
        # Sharpe ratio (annualized) - use equity curve log returns
        if len(returns) > 0 and returns.std(ddof=1) > 0:
            # Annualization factor: assume minute data, 252 trading days, 1440 minutes/day
            annualization_factor = np.sqrt(252 * 1440)
            sharpe = (returns.mean() / returns.std(ddof=1)) * annualization_factor
        else:
            sharpe = 0.0
        
        # Sortino ratio (annualized) - only considers downside deviation
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std(ddof=1) > 0:
                downside_std = downside_returns.std(ddof=1)
                annualization_factor = np.sqrt(252 * 1440)
                sortino = (returns.mean() / downside_std) * annualization_factor
            else:
                sortino = sharpe if sharpe > 0 else 0.0  # If no downside, use Sharpe
        else:
            sortino = 0.0
        
        # Max drawdown
        cumulative = equity_series / equity_series.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calmar ratio (annualized return / max drawdown)
        # Annualize return based on data period
        if len(self.equity_curve) > 0:
            # Estimate period in years (assuming minute data)
            period_years = len(self.equity_curve) / (252 * 1440)  # Convert minutes to trading years
            if period_years > 0:
                annualized_return = ((final_equity / self.initial_capital) ** (1 / period_years) - 1) * 100
            else:
                annualized_return = total_return
            
            if abs(max_drawdown) > 0:
                calmar = annualized_return / abs(max_drawdown)
            else:
                calmar = 0.0 if annualized_return == 0 else float('inf')
        else:
            calmar = 0.0
        
        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        # Average win/loss
        winning_pnl = [t.pnl for t in self.trades if t.pnl > 0]
        losing_pnl = [t.pnl for t in self.trades if t.pnl < 0]
        avg_win = np.mean(winning_pnl) if winning_pnl else 0.0
        avg_loss = np.mean(losing_pnl) if losing_pnl else 0.0
        
        # Size-weighted average win/loss
        winning_trades_with_size = [(t.pnl, t.amount * t.entry_price) for t in self.trades if t.pnl > 0]
        losing_trades_with_size = [(t.pnl, t.amount * t.entry_price) for t in self.trades if t.pnl < 0]
        
        if winning_trades_with_size:
            total_win_notional = sum(notional for _, notional in winning_trades_with_size)
            size_weighted_avg_win = sum(pnl * notional for pnl, notional in winning_trades_with_size) / total_win_notional if total_win_notional > 0 else avg_win
        else:
            size_weighted_avg_win = 0.0
        
        if losing_trades_with_size:
            total_loss_notional = sum(notional for _, notional in losing_trades_with_size)
            size_weighted_avg_loss = sum(pnl * notional for pnl, notional in losing_trades_with_size) / total_loss_notional if total_loss_notional > 0 else avg_loss
        else:
            size_weighted_avg_loss = 0.0
        
        # 1R in dollars (mean & median) - use stored one_r_dollars directly
        one_r_values = [t.one_r_dollars for t in self.trades if t.one_r_dollars > 0]
        
        mean_1r = np.mean(one_r_values) if one_r_values else 0.0
        median_1r = np.median(one_r_values) if one_r_values else 0.0
        
        # Funnel metrics: % hitting +1R and +2R
        trades_hit_1r = 0
        trades_hit_2r = 0
        for t in self.trades:
            if t.side == "long":
                risk = t.entry_price - (t.entry_price - t.exit_price + t.pnl / t.amount) if t.amount > 0 else 0
                profit_r = (t.exit_price - t.entry_price) / risk if risk > 0 else 0
            else:
                risk = (t.entry_price + t.exit_price - t.pnl / t.amount) - t.entry_price if t.amount > 0 else 0
                profit_r = (t.entry_price - t.exit_price) / risk if risk > 0 else 0
            
            if profit_r >= 1.0:
                trades_hit_1r += 1
            if profit_r >= 2.0:
                trades_hit_2r += 1
        
        pct_hit_1r = (trades_hit_1r / self.total_trades * 100) if self.total_trades > 0 else 0.0
        pct_hit_2r = (trades_hit_2r / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        # Flip rate: % of positions that reverse within 10 bars
        flip_count = 0
        for i, t in enumerate(self.trades):
            if i < len(self.trades) - 1:
                next_t = self.trades[i + 1]
                if next_t.pair == t.pair and next_t.side != t.side:
                    # Check if next trade started within 10 bars
                    bars_between = next_t.entry_time - t.exit_time
                    if bars_between <= 10:
                        flip_count += 1
        
        flip_rate = (flip_count / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        # Holding-time histogram (winners vs losers)
        winner_holding_times = [t.exit_time - t.entry_time for t in self.trades if t.pnl > 0]
        loser_holding_times = [t.exit_time - t.entry_time for t in self.trades if t.pnl < 0]
        avg_winner_holding = np.mean(winner_holding_times) if winner_holding_times else 0.0
        avg_loser_holding = np.mean(loser_holding_times) if loser_holding_times else 0.0
        
        # Worst 3 trades contribution to total P/L
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl)
        worst_3 = sorted_trades[:3] if len(sorted_trades) >= 3 else sorted_trades
        worst_3_pnl = sum(t.pnl for t in worst_3)
        total_pnl = sum(t.pnl for t in self.trades)
        worst_3_contribution = (worst_3_pnl / total_pnl * 100) if total_pnl != 0 else 0.0
        
        # R-distribution (histogram buckets) - use realized_r
        r_buckets = {"<0": 0, "0-0.5": 0, "0.5-1": 0, "1-1.5": 0, "1.5-2": 0, "2-3": 0, ">3": 0}
        realized_r_values = []
        for t in self.trades:
            realized_r = t.realized_r
            realized_r_values.append(realized_r)
            
            if realized_r < 0:
                r_buckets["<0"] += 1
            elif realized_r < 0.5:
                r_buckets["0-0.5"] += 1
            elif realized_r < 1.0:
                r_buckets["0.5-1"] += 1
            elif realized_r < 1.5:
                r_buckets["1-1.5"] += 1
            elif realized_r < 2.0:
                r_buckets["1.5-2"] += 1
            elif realized_r < 3.0:
                r_buckets["2-3"] += 1
            else:
                r_buckets[">3"] += 1
        
        median_realized_r = np.median(realized_r_values) if realized_r_values else 0.0
        
        # Win rate by ADX bucket (would need ADX at entry time - approximate for now)
        # This would require storing ADX in Trade record
        
        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": win_rate,
            "total_fees": self.total_fees,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown_pct": max_drawdown,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "size_weighted_avg_win": size_weighted_avg_win,
            "size_weighted_avg_loss": size_weighted_avg_loss,
            "mean_1r_dollars": mean_1r,
            "median_1r_dollars": median_1r,
            "pct_hit_1r": pct_hit_1r,
            "pct_hit_2r": pct_hit_2r,
            "flip_rate": flip_rate,
            "avg_winner_holding_bars": avg_winner_holding,
            "avg_loser_holding_bars": avg_loser_holding,
            "worst_3_contribution_pct": worst_3_contribution,
            "r_distribution": r_buckets,
            "median_realized_r": median_realized_r,
            "trades": self.trades,
            "equity_curve": self.equity_curve
        }

