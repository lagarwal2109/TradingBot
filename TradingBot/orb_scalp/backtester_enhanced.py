"""Enhanced Backtester with diagnostic mode and gate tracking."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta, date
from collections import defaultdict

from .data_loader import DataLoader
from .features_enhanced import FeatureEngineEnhanced
from .signals_enhanced import SignalEngineEnhanced
from .risk import RiskEngine, Position
from .broker import BrokerSim
from .portfolio import Portfolio


class BacktesterEnhanced:
    """Enhanced backtesting engine with diagnostics."""
    
    def __init__(
        self,
        data_dir: Path,
        params: Dict,
        initial_capital: float = 10000.0
    ):
        """Initialize enhanced backtester.
        
        Args:
            data_dir: Directory with CSV data files
            params: Configuration parameters
            initial_capital: Starting capital
        """
        self.data_dir = Path(data_dir)
        self.params = params
        self.initial_capital = initial_capital
        self.diagnostic_mode = params.get('diagnostic_mode', False)
        
        self.data_loader = DataLoader(data_dir)
        self.risk_engine = RiskEngine(params)
        self.broker = BrokerSim(params)
        self.portfolio = Portfolio(params, initial_capital)
        self.signal_engine = SignalEngineEnhanced(params)
        
        # Diagnostic tracking
        self.portfolio_blocks: Dict[str, int] = defaultdict(int)
        self.all_gate_stats: List[pd.DataFrame] = []
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: Optional[int] = None,
        max_bars: Optional[int] = None
    ) -> Dict:
        """Run backtest with enhanced diagnostics.
        
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
        
        # Compute features with enhanced engine
        print("Computing features (enhanced)...")
        features_data = {}
        for symbol, df in all_data.items():
            fe = FeatureEngineEnhanced(df, self.params)
            features_df = fe.compute_all()
            
            # Set warmup_ok based on history length
            warmup_bars = self.params.get('warmup_bars', 200)
            if self.diagnostic_mode:
                warmup_bars = self.params.get('warmup_bars', 50)
            
            for idx in features_df.index:
                hist_len = len(features_df.loc[:idx])
                features_df.loc[idx, 'warmup_ok'] = hist_len >= warmup_bars
            
            features_data[symbol] = features_df
        
        # Create unified timeline
        all_periods = set()
        for df in features_data.values():
            periods = df['dt'].values  # dt now stores period index (integer)
            all_periods.update(periods)
        
        timeline = sorted(all_periods)
        
        # Filter by date range
        if start_date:
            timeline = [t for t in timeline if t >= start_date]
        if end_date:
            timeline = [t for t in timeline if t <= end_date]
        if days and not start_date:
            if len(timeline) > days * 24 * 60:
                timeline = timeline[-(days * 24 * 60):]
        
        # Limit total bars if max_bars specified (for quick testing)
        if max_bars and len(timeline) > max_bars:
            timeline = timeline[:max_bars]
            print(f"⚠️  Limited to {max_bars} bars for quick testing")
        
        print(f"Backtesting {len(timeline)} bars from {timeline[0]} to {timeline[-1]}")
        if self.diagnostic_mode:
            print("DIAGNOSTIC MODE: Relaxed filters enabled")
        
        # Reset portfolio and signal engine
        self.portfolio = Portfolio(self.params, self.initial_capital)
        self.signal_engine = SignalEngineEnhanced(self.params)
        self.portfolio_blocks = defaultdict(int)
        self.all_gate_stats = []
        self.prev_closes = {}  # Track previous closes for proxy high/low
        
        # Main loop
        current_date = None
        progress_interval = max(100, len(timeline) // 20)
        
        for i, dt in enumerate(timeline):
            # Ensure dt is pandas Timestamp
            # Update daily state using day index (period // 1440)
            day_index = dt // 1440
            if current_date != day_index:
                current_date = day_index
                equity = self.portfolio.equity({})
                self.portfolio.update_daily_state(current_date, equity)
                self.signal_engine.reset_daily_counters(current_date)
            
            # Check kill switch
            equity = self.portfolio.equity({})
            if self.portfolio.check_kill_switch(current_date, equity):
                pass  # Kill switch active
            
            # Get current bar data
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
            
            # Update trailing stops and check timeouts
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
                    
                    # Check timeout for ORB
                    if position.signal_type == "ORB":
                        timeout_hours = self.params.get('orb', {}).get('timeout_hours', 6)
                        if position.timeout_dt is None:
                            position.timeout_dt = position.entry_dt + timedelta(hours=timeout_hours)
                        
                        if self.risk_engine.check_timeout(position, dt):
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
            
            # Check exits (with proxy high/low for close-only data)
            exit_fills = self.broker.step_exits(
                self.portfolio.positions,
                bar_data,
                dt,
                prev_close=self.prev_closes
            )
            
            for pos_id, fill in exit_fills:
                exit_reason = fill.fill_type.replace("exit_", "").upper()
                self.portfolio.close_position(pos_id, fill, exit_reason)
            
            # Update previous closes for next iteration
            self.prev_closes = current_prices.copy()
            
            # Generate signals with gate tracking
            signals = []
            all_gate_stats = []
            
            for symbol, df in features_data.items():
                # Get current bar
                mask = df['dt'] == dt
                if not mask.any():
                    continue
                
                # Check if already have position
                has_position = any(
                    pos.symbol == symbol for pos in self.portfolio.positions.values()
                )
                
                if has_position:
                    continue
                
                # Get historical data
                mask_hist = df['dt'] <= dt
                df_hist = df[mask_hist]
                
                warmup_bars = self.params.get('warmup_bars', 200)
                if self.diagnostic_mode:
                    warmup_bars = self.params.get('warmup_bars', 50)
                
                if len(df_hist) < warmup_bars:
                    continue
                
                # Generate ORB signals with gate tracking
                # Always track in diagnostic mode, otherwise track every 100 bars
                track_now = self.diagnostic_mode or (i % 100 == 0)
                orb_signals, orb_gates = self.signal_engine.orb_signals(
                    df_hist, symbol, track_gates=track_now
                )
                
                # Store gate stats if we tracked
                if track_now and len(orb_gates) > 0:
                    orb_mask = orb_gates['dt'] == dt
                    if orb_mask.any():
                        all_gate_stats.append(orb_gates[orb_mask])
                    elif len(orb_gates) > 0:
                        # If current bar not in gates, take the last one (should be current)
                        all_gate_stats.append(orb_gates.tail(1))
                
                if len(orb_signals) > 0:
                    orb_mask = orb_signals['dt'] == dt
                    if orb_mask.any():
                        signals.extend(orb_signals[orb_mask].to_dict('records'))
                
                # Generate Scalp signals with gate tracking
                scalp_signals, scalp_gates = self.signal_engine.scalp_signals(
                    df_hist, symbol, track_gates=track_now
                )
                
                # Store gate stats if we tracked
                if track_now and len(scalp_gates) > 0:
                    scalp_mask = scalp_gates['dt'] == dt
                    if scalp_mask.any():
                        all_gate_stats.append(scalp_gates[scalp_mask])
                    elif len(scalp_gates) > 0:
                        # If current bar not in gates, take the last one (should be current)
                        all_gate_stats.append(scalp_gates.tail(1))
                
                if len(scalp_signals) > 0:
                    scalp_mask = scalp_signals['dt'] == dt
                    if scalp_mask.any():
                        signals.extend(scalp_signals[scalp_mask].to_dict('records'))
            
            # Store gate stats
            if all_gate_stats:
                self.all_gate_stats.extend(all_gate_stats)
            
            # Process signals
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
                    self.portfolio_blocks['QUANTITY_ZERO'] += 1
                    continue
                
                # Check if we can take position
                can_take, reason = self.portfolio.can_take_position(
                    symbol, side, quantity, price
                )
                
                if not can_take:
                    self.portfolio_blocks[reason] += 1
                    if self.diagnostic_mode:
                        print(f"[BLOCK] {symbol} {side} {signal_type}: {reason}")
                    continue
                
                # Check daily scalp limit (moved from signal stage to portfolio layer)
                if signal_type == "SCALP":
                    # Convert dt to date for daily tracking
                    if isinstance(dt, pd.Timestamp):
                        trade_day = dt.date()
                    else:
                        # If dt is period index, approximate day
                        trade_day = date.fromordinal(int(dt // 1440) + 1)
                    
                    can_scalp, scalp_reason = self.portfolio.check_scalp_daily_limit(trade_day)
                    if not can_scalp:
                        self.portfolio_blocks[scalp_reason] += 1
                        if self.diagnostic_mode:
                            print(f"[BLOCK] {symbol} {side} {signal_type}: {scalp_reason}")
                        continue
                
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
                    stop_distance=stop_distance,
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
                pos_id = self.portfolio.add_position(position, fill)
                if self.diagnostic_mode:
                    print(f"[ENTRY] {symbol} {side} {signal_type} @ ${price:.2f} (qty={quantity:.6f}, stop=${stop_init:.2f})")
            
            # Record equity
            equity = self.portfolio.equity(current_prices)
            self.portfolio.record_equity(dt, equity)
            
            # Progress
            if i % progress_interval == 0 or i == len(timeline) - 1:
                progress = (i + 1) / len(timeline) * 100
                print(f"Progress: {i+1}/{len(timeline)} ({progress:.1f}%) | "
                      f"Equity: ${equity:.2f} | Positions: {len(self.portfolio.positions)} | "
                      f"Trades: {len(self.portfolio.trades)}")
        
        # Close remaining positions
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
        
        # Combine gate stats
        combined_gate_stats = pd.DataFrame()
        if self.all_gate_stats:
            combined_gate_stats = pd.concat(self.all_gate_stats, ignore_index=True)
        
        # Return results
        return {
            'trades': self.portfolio.trades,
            'equity_curve': self.portfolio.equity_curve,
            'final_equity': self.portfolio.equity({}),
            'initial_capital': self.initial_capital,
            'gate_stats': self.signal_engine.get_gate_summary(),
            'gate_stats_df': combined_gate_stats,
            'portfolio_blocks': dict(self.portfolio_blocks)
        }

