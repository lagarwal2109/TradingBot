"""Enhanced trading engine with breakout strategy and advanced risk management."""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from bot.config import get_config
from bot.roostoo_v3 import RoostooV3Client, ExchangeInfo
from bot.datastore import DataStore
from bot.signals_enhanced import EnhancedSignalGenerator
from bot.risk import RiskManager


logger = logging.getLogger(__name__)


@dataclass 
class Position:
    """Track open position details."""
    symbol: str
    entry_price: float
    amount: float
    side: str  # "long" or "short"
    stop_loss: float
    take_profit: float
    trailing_stop_active: bool = False
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float('inf')  # For short trailing stop
    entry_time: int = 0
    

class EnhancedTradingEngine:
    """Enhanced trading engine with multi-timeframe breakout strategy."""
    
    def __init__(
        self,
        trading_client: RoostooV3Client,
        market_data_client: Optional[Any],  # Not used anymore, kept for compatibility
        datastore: DataStore,
        risk_manager: RiskManager
    ):
        """Initialize enhanced trading engine.
        
        Args:
            trading_client: Roostoo v3 client for trading execution
            market_data_client: Not used (kept for compatibility)
            datastore: Data storage instance
            risk_manager: Risk management instance
        """
        self.trading_client = trading_client  # Roostoo for trading and market data
        self.market_client = None  # Not using Horus anymore
        self.datastore = datastore
        self.risk_manager = risk_manager
        self.config = get_config()
        
        # Initialize enhanced signal generator
        self.signal_generator = EnhancedSignalGenerator(
            trend_window_long=self.config.trend_window_long,
            trend_window_short=self.config.trend_window_short,
            entry_window=self.config.entry_window,
            volume_window=self.config.volume_window,
            support_resistance_days=self.config.support_resistance_days,
            breakout_threshold=self.config.breakout_threshold,
            volume_surge_multiplier=self.config.volume_surge_multiplier
        )
        
        # Cache for exchange info
        self.exchange_info_map: Dict[str, ExchangeInfo] = {}
        
        # Track open positions with details
        self.positions: Dict[str, Position] = {}
        
        # Alias for compatibility with safe_startup and trading_loop
        self.client = self.trading_client
        self.mode = "enhanced"  # For compatibility with logging
    
    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            open_orders = self.trading_client.get_open_orders()
            for order in open_orders:
                try:
                    self.trading_client.cancel_order(order.order_id)
                    logger.info(f"Cancelled order {order.order_id}")
                except Exception as e:
                    logger.error(f"Failed to cancel order {order.order_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
        
    def update_exchange_info(self) -> None:
        """Update cached exchange information from trading API."""
        try:
            # Get exchange info from Roostoo (trading API)
            exchange_info_list = self.trading_client.exchange_info()
            self.exchange_info_map = {info.pair: info for info in exchange_info_list}
            self.datastore.mark_exchange_info_updated()
            logger.info(f"Updated exchange info for {len(self.exchange_info_map)} pairs")
        except Exception as e:
            logger.error(f"Failed to update exchange info: {e}")
            raise
    
    def get_total_equity(self, balance: Dict[str, float], prices: Dict[str, float]) -> float:
        """Calculate total equity in USD."""
        total = balance.get("USD", 0.0)
        
        for asset, amount in balance.items():
            if asset != "USD" and amount > 0:
                pair = f"{asset}USD"
                if pair in prices:
                    total += amount * prices[pair]
        
        return total
    
    def update_trailing_stops(self, prices: Dict[str, float]) -> List[str]:
        """Update trailing stops and return list of positions to close.
        
        Args:
            prices: Current prices
            
        Returns:
            List of symbols to close due to stop hit
        """
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            pair = f"{symbol}USD"
            if pair not in prices:
                continue
                
            current_price = prices[pair]
            
            if position.side == "long":
                # Update highest price
                if current_price > position.highest_price:
                    position.highest_price = current_price
                    
                    # Activate trailing stop once we're in profit
                    if self.config.use_trailing_stop and current_price > position.entry_price * 1.01:
                        position.trailing_stop_active = True
                        
                # Check stops
                if position.trailing_stop_active:
                    trailing_stop = position.highest_price * (1 - self.config.trailing_stop_pct)
                    if current_price <= trailing_stop:
                        logger.info(f"Trailing stop hit for {symbol} at {current_price:.2f}")
                        positions_to_close.append(symbol)
                        continue
                        
                # Fixed stop loss
                if current_price <= position.stop_loss:
                    logger.info(f"Stop loss hit for {symbol} at {current_price:.2f}")
                    positions_to_close.append(symbol)
                    continue
                    
                # Take profit
                if current_price >= position.take_profit:
                    logger.info(f"Take profit hit for {symbol} at {current_price:.2f}")
                    positions_to_close.append(symbol)
                    
            elif position.side == "short":
                # Update lowest price
                if current_price < position.lowest_price:
                    position.lowest_price = current_price
                    
                    # Activate trailing stop once we're in profit
                    if self.config.use_trailing_stop and current_price < position.entry_price * 0.99:
                        position.trailing_stop_active = True
                        
                # Check stops  
                if position.trailing_stop_active:
                    trailing_stop = position.lowest_price * (1 + self.config.trailing_stop_pct)
                    if current_price >= trailing_stop:
                        logger.info(f"Trailing stop hit for {symbol} at {current_price:.2f}")
                        positions_to_close.append(symbol)
                        continue
                        
                # Fixed stop loss
                if current_price >= position.stop_loss:
                    logger.info(f"Stop loss hit for {symbol} at {current_price:.2f}")
                    positions_to_close.append(symbol)
                    continue
                    
                # Take profit
                if current_price <= position.take_profit:
                    logger.info(f"Take profit hit for {symbol} at {current_price:.2f}")
                    positions_to_close.append(symbol)
                    
        return positions_to_close
    
    def analyze_market(self, ticker_data: List[Dict[str, any]]) -> Dict[str, Dict[str, any]]:
        """Analyze all pairs and generate trading signals.
        
        Args:
            ticker_data: List of ticker data
            
        Returns:
            Dictionary of pair -> signal
        """
        all_signals = {}
        
        # Create ticker lookup with volume data
        ticker_lookup = {t["pair"]: t for t in ticker_data}
        
        # Get all pairs from ticker data (not just those with historical data files)
        # This ensures we analyze all available pairs
        all_pairs = set(ticker_lookup.keys())
        pairs_with_data = set(self.datastore.get_all_pairs_with_data())
        
        # Analyze pairs - prioritize those with data, but also try those without
        pairs_to_analyze = list(all_pairs | pairs_with_data)
        
        logger.debug(f"Analyzing {len(pairs_to_analyze)} pairs ({len(pairs_with_data)} with historical data)")
        
        # Analyze each pair
        for pair in pairs_to_analyze:
            if pair in ticker_lookup:
                signal = self.signal_generator.compute_trading_signal(
                    pair, self.datastore, ticker_lookup[pair]
                )
                all_signals[pair] = signal
                
        return all_signals
    
    def execute_trade_signal(self, pair: str, signal: Dict[str, any], balance: Dict[str, float]) -> bool:
        """Execute a trading signal.
        
        Args:
            pair: Trading pair (e.g. "BTCUSD")
            signal: Signal dictionary
            balance: Current balances
            
        Returns:
            True if trade executed successfully
        """
        try:
            from bot.utils import to_roostoo_pair
            base_currency = pair.replace("USD", "")
            roostoo_pair = to_roostoo_pair(pair)
            
            if signal["signal"] == "buy":
                # Calculate position size
                total_equity = self.get_total_equity(balance, {pair: signal["current_price"]})
                exchange_info = self.exchange_info_map.get(roostoo_pair)
                
                if not exchange_info:
                    logger.error(f"No exchange info for {roostoo_pair}")
                    return False
                    
                # Use entry quality to scale position size
                position_scale = 0.5 + (signal["entry_quality"] - 0.5) * 0.5  # 0.5-1.0 scale
                max_position = self.config.max_position_pct * position_scale
                
                amount = self.risk_manager.calculate_position_size(
                    total_equity,
                    0.02,  # Default volatility, could be calculated from data
                    signal["current_price"],
                    exchange_info,
                    volatility_scalar=position_scale
                )
                
                # Validate order
                is_valid, error_msg = self.risk_manager.validate_order(
                    amount, signal["current_price"], "buy", balance, exchange_info
                )
                
                if not is_valid:
                    logger.warning(f"Order validation failed: {error_msg}")
                    return False
                    
                # Place order via Roostoo (trading client)
                order = self.trading_client.place_order(
                    pair=roostoo_pair,
                    side="buy",
                    type="market",
                    quantity=amount
                )
                
                # Track position - update if exists, create if new
                if base_currency in self.positions:
                    # Update existing position
                    existing_pos = self.positions[base_currency]
                    # Weighted average entry price
                    total_amount = existing_pos.amount + amount
                    existing_pos.entry_price = (
                        (existing_pos.entry_price * existing_pos.amount + signal["current_price"] * amount) / total_amount
                    )
                    existing_pos.amount = total_amount
                    # Update stops to be more conservative (use tighter of the two)
                    existing_pos.stop_loss = max(existing_pos.stop_loss, signal["stop_loss"])
                    existing_pos.take_profit = min(existing_pos.take_profit, signal["take_profit"])
                else:
                    # Create new position
                    self.positions[base_currency] = Position(
                        symbol=base_currency,
                        entry_price=signal["current_price"],
                        amount=amount,
                        side="long",
                        stop_loss=signal["stop_loss"],
                        take_profit=signal["take_profit"],
                        highest_price=signal["current_price"],
                        entry_time=int(time.time() * 1000)
                    )
                
                logger.info(
                    f"Opened long position: {amount:.6f} {base_currency} at {signal['current_price']:.2f}, "
                    f"SL: {signal['stop_loss']:.2f}, TP: {signal['take_profit']:.2f}"
                )
                return True
                
            elif signal["signal"] == "sell":
                # Check if we have a position to sell
                if base_currency in balance and balance[base_currency] > 0:
                    amount = balance[base_currency]
                    
                    # Round to precision
                    exchange_info = self.exchange_info_map.get(roostoo_pair)
                    if exchange_info:
                        amount = self.risk_manager.round_to_precision(
                            amount, exchange_info.amount_precision
                        )
                        
                    # Place sell order via Roostoo (trading client)
                    order = self.trading_client.place_order(
                        pair=roostoo_pair,
                        side="sell",
                        type="market",
                        quantity=amount
                    )
                    
                    # Remove from tracked positions
                    if base_currency in self.positions:
                        del self.positions[base_currency]
                        
                    logger.info(f"Closed position: sold {amount:.6f} {base_currency}")
                    return True
                else:
                    logger.warning(f"No position to sell for {base_currency}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            self.datastore.increment_error_count()
            return False
    
    def close_position(self, symbol: str) -> bool:
        """Close a specific position.
        
        Args:
            symbol: Asset symbol (e.g. "BTC")
            
        Returns:
            True if closed successfully
        """
        try:
            # Get balance from Roostoo (trading client)
            balance = self.trading_client.balance()
            if symbol in balance and balance[symbol] > 0:
                pair = f"{symbol}USD"
                return self.execute_trade_signal(
                    pair,
                    {"signal": "sell", "current_price": 0},  # Price not needed for market order
                    balance
                )
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            
        return False
    
    def rebalance(self) -> bool:
        """Execute rebalancing based on signals and risk management.
        
        Returns:
            True if rebalancing successful
        """
        # Check if we can trade (1 minute rule)
        if not self.datastore.check_trade_allowed():
            seconds_to_wait = 60 - ((time.time() * 1000 - self.datastore.state.last_trade_ts) / 1000)
            logger.info(f"Trade cooldown: {seconds_to_wait:.1f}s remaining")
            return True
            
        try:
            # Get market data from Roostoo
            from bot.utils import from_roostoo_pair
            roostoo_tickers = self.trading_client.ticker()
            ticker_list = [
                {
                    "pair": from_roostoo_pair(t.pair),
                    "price": t.last_price,
                    "volume_24h": t.coin_trade_value,
                    "bid": t.max_bid,
                    "ask": t.min_ask
                }
                for t in roostoo_tickers
            ]
            prices = {from_roostoo_pair(t.pair): t.last_price for t in roostoo_tickers}
            
            # Get balance from Roostoo (trading client)
            balance = self.trading_client.balance()
            
            # Update trailing stops and close positions if needed
            positions_to_close = self.update_trailing_stops(prices)
            for symbol in positions_to_close:
                if self.close_position(symbol):
                    self.datastore.record_trade(None)
                    time.sleep(1)  # Brief pause between trades
                    
            # Generate signals
            all_signals = self.analyze_market(ticker_list)
            
            # Debug: log signal stats
            total_signals = len(all_signals)
            buy_signals = sum(1 for s in all_signals.values() if s.get("signal") == "buy")
            sell_signals = sum(1 for s in all_signals.values() if s.get("signal") == "sell")
            neutral_signals = sum(1 for s in all_signals.values() if s.get("signal") == "neutral")
            high_quality = sum(1 for s in all_signals.values() if s.get("entry_quality", 0) > 0.5)
            
            logger.info(f"Signal analysis: {total_signals} pairs analyzed, {buy_signals} buy, {sell_signals} sell, {neutral_signals} neutral, {high_quality} high-quality")
            
            # Show all buy/sell signals found
            if buy_signals > 0 or sell_signals > 0:
                logger.info("All trading signals found:")
                for pair, sig in all_signals.items():
                    if sig.get("signal") != "neutral":
                        logger.info(f"  {pair}: {sig['signal']} (quality={sig.get('entry_quality', 0):.2f}, strength={sig.get('strength', 0):.2f}) - {sig.get('reason', 'N/A')}")
            
            # PRIORITY 1: Check ALL existing positions for sell signals and risk conditions
            # This ensures we close risky positions immediately, regardless of new opportunities
            positions_closed_this_cycle = False
            for symbol, pos in list(self.positions.items()):
                pos_pair = f"{symbol}USD"
                
                # Check for sell signal on this position
                if pos_pair in all_signals:
                    pos_signal = all_signals[pos_pair]
                    if pos_signal["signal"] == "sell":
                        logger.info(
                            f"[RISK] Sell signal detected for {symbol} "
                            f"(quality: {pos_signal.get('entry_quality', 0):.2f}, "
                            f"reason: {pos_signal.get('reason', 'N/A')}) - closing immediately"
                        )
                        if self.close_position(symbol):
                            self.datastore.record_trade(None)
                            positions_closed_this_cycle = True
                            time.sleep(0.5)  # Brief pause
                            # Update balance and prices after closing
                            balance = self.trading_client.balance()
                            roostoo_tickers = self.trading_client.ticker()
                            from bot.utils import from_roostoo_pair
                            prices = {from_roostoo_pair(t.pair): t.last_price for t in roostoo_tickers}
                            continue
                
                # Check for deteriorating signal quality (risk indicator)
                if pos_pair in all_signals:
                    pos_signal = all_signals[pos_pair]
                    current_quality = pos_signal.get("entry_quality", 0)
                    current_strength = pos_signal.get("strength", 0)
                    
                    # Close if signal quality dropped significantly or turned bearish
                    if (pos_signal.get("signal") == "neutral" and 
                        current_quality < 0.3 and 
                        current_strength < 0.2):
                        logger.info(
                            f"[RISK] {symbol} signal quality deteriorated "
                            f"(quality: {current_quality:.2f}, strength: {current_strength:.2f}) - closing position"
                        )
                        if self.close_position(symbol):
                            self.datastore.record_trade(None)
                            positions_closed_this_cycle = True
                            time.sleep(0.5)
                            balance = self.trading_client.balance()
                            roostoo_tickers = self.trading_client.ticker()
                            from bot.utils import from_roostoo_pair
                            prices = {from_roostoo_pair(t.pair): t.last_price for t in roostoo_tickers}
                            continue
                    
                    # Close if trend reversed to bearish while we're holding
                    if (pos_signal.get("signal") == "neutral" and
                        pos_pair in all_signals):
                        # Check if we can extract trend info from signal reason or check price action
                        # If price dropped significantly from entry, consider closing
                        if pos_pair in prices:
                            current_price = prices[pos_pair]
                            price_change_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                            
                            # Close if down more than 3% and signal is weak
                            if price_change_pct < -3.0 and current_quality < 0.4:
                                logger.info(
                                    f"[RISK] {symbol} down {price_change_pct:.1f}% with weak signal "
                                    f"(quality: {current_quality:.2f}) - closing to limit losses"
                                )
                                if self.close_position(symbol):
                                    self.datastore.record_trade(None)
                                    positions_closed_this_cycle = True
                                    time.sleep(0.5)
                                    balance = self.trading_client.balance()
                                    roostoo_tickers = self.trading_client.ticker()
                                    from bot.utils import from_roostoo_pair
                                    prices = {from_roostoo_pair(t.pair): t.last_price for t in roostoo_tickers}
                                    continue
            
            # If we closed positions, update balance and prices for buy logic
            if positions_closed_this_cycle:
                balance = self.trading_client.balance()
                roostoo_tickers = self.trading_client.ticker()
                from bot.utils import from_roostoo_pair
                prices = {from_roostoo_pair(t.pair): t.last_price for t in roostoo_tickers}
            
            # Show top 3 signals for debugging
            if all_signals:
                sorted_by_quality = sorted(
                    [(p, s) for p, s in all_signals.items() if s.get("signal") != "neutral"],
                    key=lambda x: x[1].get("entry_quality", 0),
                    reverse=True
                )[:3]
                if sorted_by_quality:
                    logger.info("Top signals:")
                    for pair, sig in sorted_by_quality:
                        logger.info(f"  {pair}: {sig['signal']} (quality={sig.get('entry_quality', 0):.2f}, strength={sig.get('strength', 0):.2f}, reason={sig.get('reason', 'N/A')})")
            
            # Rank opportunities (only buy signals now, since we've handled all sells)
            ranked_signals = self.signal_generator.rank_trading_opportunities(all_signals)
            
            if not ranked_signals:
                logger.info("No trading opportunities found (all signals below quality threshold or insufficient data)")
                return True
                
            # Execute best signal (should be buy now, since we closed all sells)
            best_pair, best_signal = ranked_signals[0]
            base_currency = best_pair.replace("USD", "")  # Define base_currency for all cases
            
            # Double-check it's not a sell (shouldn't happen after filtering, but safety check)
            if best_signal["signal"] == "sell":
                if base_currency in self.positions:
                    logger.info(f"[RISK] Best signal is sell for {base_currency} - closing position")
                    if self.execute_trade_signal(best_pair, best_signal, balance):
                        self.datastore.record_trade(None)
                        self.datastore.reset_error_count()
                        return True
                return True
            
            # Handle buy signals - dynamic position management based on capital and signal quality
            elif best_signal["signal"] == "buy":
                from bot.utils import to_roostoo_pair
                total_equity = self.get_total_equity(balance, {best_pair: best_signal["current_price"]})
                
                # Calculate current capital allocation across all positions
                total_allocated = 0.0
                position_values = {}
                for symbol, pos in self.positions.items():
                    pos_pair = f"{symbol}USD"
                    if pos_pair in prices:
                        pos_value = pos.amount * prices[pos_pair]
                        total_allocated += pos_value
                        position_values[symbol] = pos_value
                
                allocated_pct = total_allocated / total_equity if total_equity > 0 else 0
                max_allocated_pct = 0.85  # Keep 15% cash buffer for opportunities and risk management
                available_capital_pct = max_allocated_pct - allocated_pct
                
                if base_currency in self.positions:
                    # Already have this position - check if we should add more or hold
                    current_pos = self.positions[base_currency]
                    current_value = current_pos.amount * best_signal["current_price"]
                    current_pct = current_value / total_equity if total_equity > 0 else 0
                    
                    # Add more if:
                    # 1. Position is below target size (80% of max per position)
                    # 2. Signal is still strong (quality > 0.6)
                    # 3. We have available capital
                    target_pct = self.config.max_position_pct * 0.8
                    if current_pct < target_pct and best_signal["entry_quality"] > 0.6 and available_capital_pct > 0.05:
                        logger.info(
                            f"Adding to {base_currency} position "
                            f"(current: {current_pct:.1%}, target: {target_pct:.1%}, "
                            f"available capital: {available_capital_pct:.1%})"
                        )
                        if self.execute_trade_signal(best_pair, best_signal, balance):
                            self.datastore.reset_error_count()
                            return True
                    else:
                        logger.info(
                            f"Holding {base_currency} position "
                            f"(size: {current_pct:.1%}, quality: {best_signal['entry_quality']:.2f}, "
                            f"allocated: {allocated_pct:.1%})"
                        )
                        return True
                        
                else:
                    # New position - check if we have capital and it's a good signal
                    # Only open if:
                    # 1. We have available capital (at least 5% free)
                    # 2. Signal quality is good enough (>= 0.6)
                    # 3. We're not over-allocated
                    
                    if available_capital_pct < 0.05:
                        # Low on capital - check if we should switch from a weaker position
                        worst_position = None
                        worst_score = float('inf')
                        
                        for symbol, pos in self.positions.items():
                            pos_pair = f"{symbol}USD"
                            if pos_pair in all_signals:
                                pos_signal = all_signals[pos_pair]
                                pos_score = pos_signal.get("entry_quality", 0) * pos_signal.get("strength", 0)
                            else:
                                pos_score = 0.3  # Default low score if no signal
                            
                            if pos_score < worst_score:
                                worst_score = pos_score
                                worst_position = symbol
                        
                        # Switch if new signal is significantly better (at least 25% better)
                        best_score = best_signal["entry_quality"] * best_signal["strength"]
                        improvement_ratio = best_score / worst_score if worst_score > 0 else float('inf')
                        
                        if improvement_ratio > 1.25 and worst_position and best_signal["entry_quality"] >= 0.6:
                            logger.info(
                                f"Switching from {worst_position} (score: {worst_score:.2f}) "
                                f"to {base_currency} (score: {best_score:.2f}, improvement: {improvement_ratio:.1%}) "
                                f"due to capital constraints (allocated: {allocated_pct:.1%})"
                            )
                            # Close worst position
                            if self.close_position(worst_position):
                                time.sleep(0.5)  # Brief pause
                                # Open new position
                                if self.execute_trade_signal(best_pair, best_signal, balance):
                                    self.datastore.record_trade(base_currency)
                                    self.datastore.reset_error_count()
                                    return True
                        else:
                            logger.info(
                                f"Skipping {base_currency} - low capital (allocated: {allocated_pct:.1%}), "
                                f"and not significantly better than worst position {worst_position} "
                                f"(improvement: {improvement_ratio:.1%} if {worst_position})"
                            )
                            return True
                    
                    elif best_signal["entry_quality"] >= 0.6:
                        # Good signal and we have capital - open new position
                        current_position_count = len(self.positions)
                        logger.info(
                            f"Opening new position in {base_currency} "
                            f"(#{current_position_count + 1}, quality: {best_signal['entry_quality']:.2f}, "
                            f"allocated: {allocated_pct:.1%}, available: {available_capital_pct:.1%})"
                        )
                        if self.execute_trade_signal(best_pair, best_signal, balance):
                            self.datastore.record_trade(base_currency)
                            self.datastore.reset_error_count()
                            return True
                    else:
                        logger.info(
                            f"Skipping {base_currency} - signal quality {best_signal['entry_quality']:.2f} "
                            f"below threshold 0.60"
                        )
                        return True
                
        except Exception as e:
            logger.error(f"Rebalancing failed: {e}")
            self.datastore.increment_error_count()
            return False
            
        return True
    
    def check_kill_switch(self) -> bool:
        """Check if kill switch should be activated."""
        if self.datastore.state.consecutive_errors >= self.config.max_consecutive_errors:
            logger.critical(f"Kill switch activated: {self.datastore.state.consecutive_errors} consecutive errors")
            
            # Close all positions
            try:
                # Get balance from Roostoo
                balance = self.trading_client.balance()
                for asset, amount in balance.items():
                    if asset != "USD" and amount > 0:
                        self.close_position(asset)
            except Exception as e:
                logger.error(f"Failed to close positions during kill switch: {e}")
                
            return True
            
        return False
    
    def run_cycle(self) -> Dict[str, Any]:
        """Run a complete trading cycle.
        
        Returns:
            Dictionary with cycle results
        """
        cycle_start = time.time()
        results = {
            "timestamp": int(cycle_start * 1000),
            "success": False,
            "positions": list(self.positions.keys()),
            "error": None
        }
        
        try:
            # Check kill switch
            if self.check_kill_switch():
                results["error"] = "Kill switch activated"
                return results
                
            # Update exchange info if needed
            if self.datastore.should_update_exchange_info() or not self.exchange_info_map:
                self.update_exchange_info()
                
            # Get ticker data from Roostoo
            from bot.utils import from_roostoo_pair
            roostoo_tickers = self.trading_client.ticker()
            ticker_list = [
                {
                    "pair": from_roostoo_pair(t.pair),
                    "price": t.last_price,
                    "volume_24h": t.coin_trade_value,
                    "bid": t.max_bid,
                    "ask": t.min_ask
                }
                for t in roostoo_tickers
            ]
            prices = {from_roostoo_pair(t.pair): t.last_price for t in roostoo_tickers}
            
            # Collect minute bars with volume
            self.datastore.collect_minute_bars(ticker_list)
            
            # Rebalance portfolio
            if self.rebalance():
                results["success"] = True
                
            # Get final state
            from bot.utils import filter_zero_balances
            balance = self.trading_client.balance()  # Roostoo for balance
            # prices already set above
            results["total_equity"] = self.get_total_equity(balance, prices)
            results["balances"] = filter_zero_balances(balance)  # Only show non-zero balances
            results["open_positions"] = len(self.positions)
            # Get current position from state or positions
            if self.positions:
                results["current_position"] = list(self.positions.keys())[0]  # First position
            else:
                results["current_position"] = self.datastore.state.current_position
            results["target_position"] = None  # Enhanced mode doesn't use target_position
            
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            results["error"] = str(e)
            self.datastore.increment_error_count()
            
        # Log cycle time
        cycle_time = time.time() - cycle_start
        results["cycle_time"] = cycle_time
        logger.info(f"Cycle completed in {cycle_time:.2f}s")
        
        return results
