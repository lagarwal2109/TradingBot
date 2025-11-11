"""Backtester: main backtesting loop."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta

from .data_loader import DataLoader
from .features import FeatureEngine
from .signals import SignalEngine
from .risk import RiskEngine, Position
from .broker import BrokerSim
from .portfolio import Portfolio


class Backtester:
    """Main backtesting engine."""
    
    def __init__(
        self,
        data_dir: Path,
        params: Dict,
        initial_capital: float = 10000.0
    ):
        """Initialize backtester.
        
        Args:
            data_dir: Directory with CSV data files
            params: Configuration parameters
            initial_capital: Starting capital
        """
        self.data_dir = Path(data_dir)
        self.params = params
        self.initial_capital = initial_capital
        
        self.data_loader = DataLoader(data_dir)
        self.risk_engine = RiskEngine(params)
        self.broker = BrokerSim(params)
        self.portfolio = Portfolio(params, initial_capital)
        self.signal_engine = SignalEngine(params)
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: Optional[int] = None
    ) -> Dict:
        """Run backtest.
        
        Args:
            symbols: List of symbols to trade
            start_date: Start date (optional)
            end_date: End date (optional)
            days: Number of days to backtest (optional)
            
        Returns:
            Dictionary with results
        """
        # Load data
        print(f"Loading data for {len(symbols)} symbols...")
        all_data = self.data_loader.load_multiple(symbols)
        
        if not all_data:
            raise ValueError("No data loaded")
        
        # Compute features for each symbol
        print("Computing features...")
        features_data = {}
        for symbol, df in all_data.items():
            fe = FeatureEngine(df, self.params)
            features_data[symbol] = fe.compute_all()
        
        # Create unified timeline (all unique datetimes across symbols)
        all_timestamps = set()
        for df in features_data.values():
            # Convert to pandas Timestamps
            timestamps = [pd.Timestamp(ts) for ts in df['dt'].values]
            all_timestamps.update(timestamps)
        
        timeline = sorted(all_timestamps)
        
        # Filter by date range
        if start_date:
            timeline = [t for t in timeline if t >= start_date]
        if end_date:
            timeline = [t for t in timeline if t <= end_date]
        if days and not start_date:
            # Use last N days
            if len(timeline) > days * 24 * 60:  # Approximate: days * hours * minutes
                timeline = timeline[-(days * 24 * 60):]
        
        print(f"Backtesting {len(timeline)} bars from {timeline[0]} to {timeline[-1]}")
        
        # Reset portfolio
        self.portfolio = Portfolio(self.params, self.initial_capital)
        self.signal_engine = SignalEngine(self.params)
        
        # Main loop
        current_date = None
        progress_interval = max(100, len(timeline) // 20)
        
        for i, dt in enumerate(timeline):
            # Ensure dt is pandas Timestamp
            if not isinstance(dt, pd.Timestamp):
                dt = pd.Timestamp(dt)
            
            # Update daily state
            dt_date = dt.date()
            if current_date != dt_date:
                current_date = dt_date
                equity = self.portfolio.equity({})
                self.portfolio.update_daily_state(current_date, equity)
                self.signal_engine.reset_daily_counters(current_date)
            
            # Check kill switch
            equity = self.portfolio.equity({})
            if self.portfolio.check_kill_switch(current_date, equity):
                # Kill switch active - skip new entries but manage exits
                pass
            
            # Get current bar data for each symbol
            bar_data = {}
            current_prices = {}
            
            for symbol, df in features_data.items():
                mask = df['dt'] == dt
                if mask.any():
                    bar = df[mask].iloc[0]
                    bar_data[symbol] = {
                        'close': bar['close'],
                        'atr': bar.get('atr14', 0.0),
                        'rv': bar.get('rv30', None)
                    }
                    current_prices[symbol] = bar['close']
            
            # Update trailing stops
            for pos_id, position in list(self.portfolio.positions.items()):
                symbol = position.symbol
                if symbol in bar_data:
                    bar = bar_data[symbol]
                    self.risk_engine.update_trailing(
                        position,
                        bar['close'],
                        bar['atr'],
                        position.signal_type
                    )
                    
                    # Check timeout
                    if position.signal_type == "ORB":
                        timeout_hours = self.params.get('orb', {}).get('timeout_hours', 6)
                        if position.timeout_dt is None:
                            position.timeout_dt = position.entry_dt + timedelta(hours=timeout_hours)
                        
                        if self.risk_engine.check_timeout(position, dt):
                            # Timeout exit
                            fill = self.broker.exit(
                                symbol=position.symbol,
                                side=position.side,
                                exit_price=bar['close'],
                                quantity=position.quantity,
                                dt=dt,
                                exit_type="exit_timeout",
                                rv=bar.get('rv')
                            )
                            self.portfolio.close_position(pos_id, fill, "TIMEOUT")
            
            # Check exits
            exit_fills = self.broker.step_exits(
                self.portfolio.positions,
                bar_data,
                dt
            )
            
            for pos_id, fill in exit_fills:
                exit_reason = fill.fill_type.replace("exit_", "").upper()
                self.portfolio.close_position(pos_id, fill, exit_reason)
            
            # Generate signals for current bar
            signals = []
            
            for symbol, df in features_data.items():
                # Get current bar data
                mask = df['dt'] == dt
                if not mask.any():
                    continue
                
                current_bar = df[mask].iloc[0]
                
                # Check if we already have a position in this symbol
                has_position = any(
                    pos.symbol == symbol for pos in self.portfolio.positions.values()
                )
                
                if has_position:
                    continue  # Skip if already have position
                
                # Get historical data up to current bar for indicators
                mask_hist = df['dt'] <= dt
                df_hist = df[mask_hist]
                
                if len(df_hist) < 200:  # Need enough data for indicators
                    continue
                
                # Generate ORB signals
                orb_signals = self.signal_engine.orb_signals(df_hist, symbol)
                if len(orb_signals) > 0:
                    # Filter to current bar
                    orb_mask = orb_signals['dt'] == dt
                    if orb_mask.any():
                        signals.extend(orb_signals[orb_mask].to_dict('records'))
                
                # Generate Scalp signals
                scalp_signals = self.signal_engine.scalp_signals(df_hist, symbol)
                if len(scalp_signals) > 0:
                    # Filter to current bar
                    scalp_mask = scalp_signals['dt'] == dt
                    if scalp_mask.any():
                        signals.extend(scalp_signals[scalp_mask].to_dict('records'))
            
            # Process signals (take positions)
            for signal in signals:
                symbol = signal['symbol']
                side = signal['side']
                signal_type = signal['signal_type']
                price = signal['price']
                stop_distance = signal['stop_distance']
                
                # Calculate position size
                quantity = self.risk_engine.size_trade(
                    equity,
                    stop_distance,
                    signal_type
                )
                
                if quantity <= 0:
                    continue
                
                # Check if we can take position
                can_take, reason = self.portfolio.can_take_position(
                    symbol, side, quantity, price
                )
                
                if not can_take:
                    continue  # Skip if can't take
                
                # Calculate stops and target
                stop_init, target = self.risk_engine.calculate_stops(
                    price, side, stop_distance, signal_type
                )
                
                # Create position
                position = Position(
                    symbol=symbol,
                    side=side,
                    signal_type=signal_type,
                    entry_dt=dt,
                    entry_price=price,
                    quantity=quantity,
                    stop_init=stop_init,
                    stop_current=stop_init,
                    target=target,
                    risk_dollars=equity * (
                        self.params.get('orb', {}).get('risk_frac', 0.0075) if signal_type == "ORB"
                        else self.params.get('scalp', {}).get('risk_frac', 0.005)
                    )
                )
                
                # Set timeout for ORB
                if signal_type == "ORB":
                    timeout_hours = self.params.get('orb', {}).get('timeout_hours', 6)
                    position.timeout_dt = dt + timedelta(hours=timeout_hours)
                
                # Get entry fill
                bar = bar_data.get(symbol, {})
                fill = self.broker.enter(
                    symbol=symbol,
                    side=side,
                    signal_price=price,
                    quantity=quantity,
                    dt=dt,
                    rv=bar.get('rv')
                )
                
                # Add position
                self.portfolio.add_position(position, fill)
            
            # Record equity
            equity = self.portfolio.equity(current_prices)
            self.portfolio.record_equity(dt, equity)
            
            # Progress
            if i % progress_interval == 0 or i == len(timeline) - 1:
                progress = (i + 1) / len(timeline) * 100
                print(f"Progress: {i+1}/{len(timeline)} ({progress:.1f}%) | "
                      f"Equity: ${equity:.2f} | Positions: {len(self.portfolio.positions)} | "
                      f"Trades: {len(self.portfolio.trades)}")
        
        # Close all remaining positions at end
        print("\nClosing remaining positions...")
        final_dt = timeline[-1]
        for pos_id, position in list(self.portfolio.positions.items()):
            symbol = position.symbol
            if symbol in current_prices:
                exit_price = current_prices[symbol]
                bar = bar_data.get(symbol, {})
                fill = self.broker.exit(
                    symbol=symbol,
                    side=position.side,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    dt=final_dt,
                    exit_type="exit_manual",
                    rv=bar.get('rv')
                )
                self.portfolio.close_position(pos_id, fill, "END_OF_TEST")
        
        # Return results
        return {
            'trades': self.portfolio.trades,
            'equity_curve': self.portfolio.equity_curve,
            'final_equity': self.portfolio.equity({}),
            'initial_capital': self.initial_capital
        }

