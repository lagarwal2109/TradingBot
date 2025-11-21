"""Trading engine for volatility expansion strategy with batch entries and staggered exits."""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from bot.config import get_config
from bot.roostoo_v3 import RoostooV3Client, ExchangeInfo
from bot.datastore import DataStore
from bot.signals_volatility_expansion import VolatilityExpansionSignalGenerator, VolatilityExpansionSignal
from bot.position_manager import PositionManager, ManagedPosition
from bot.risk import RiskManager
from bot.indicators import calculate_rsi, calculate_macd
from bot.utils import to_roostoo_pair, from_roostoo_pair


logger = logging.getLogger(__name__)


class VolatilityExpansionEngine:
    """Trading engine for volatility expansion strategy."""
    
    def __init__(
        self,
        client: RoostooV3Client,
        datastore: DataStore,
        signal_generator: VolatilityExpansionSignalGenerator,
        risk_manager: RiskManager,
        position_manager: PositionManager
    ):
        """Initialize volatility expansion engine.
        
        Args:
            client: Roostoo API client
            datastore: DataStore instance
            signal_generator: Volatility expansion signal generator
            risk_manager: Risk management instance
            position_manager: Position manager instance
        """
        self.client = client
        self.datastore = datastore
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.config = get_config()
        
        # Cache for exchange info
        self.exchange_info_map: Dict[str, ExchangeInfo] = {}
    
    def update_exchange_info(self) -> None:
        """Update cached exchange information."""
        try:
            exchange_info_list = self.client.exchange_info()
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
    
    def check_signal_reversal(
        self,
        pair: str,
        position: ManagedPosition,
        datastore: DataStore,
        current_price: float
    ) -> bool:
        """Check if MACD/RSI shows reversal signal (exit condition).
        
        Args:
            pair: Trading pair
            position: Current position
            current_price: Current price
            
        Returns:
            True if reversal signal detected
        """
        try:
            # Read recent data
            df = datastore.read_minute_bars(pair, limit=200)
            if len(df) < 50:
                return False
            
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            prices = df['price'] if 'price' in df.columns else df['close']
            
            # Calculate MACD
            macd_line, signal_line, histogram = calculate_macd(
                prices, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
            )
            
            if len(macd_line) < 2:
                return False
            
            current_macd = macd_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            current_signal = signal_line.iloc[-1]
            prev_signal = signal_line.iloc[-2]
            
            # Check for MACD reversal
            # For longs: exit if MACD crosses below signal
            if position.macd_signal == "bullish":
                if prev_macd > prev_signal and current_macd < current_signal:
                    logger.info(f"MACD reversal detected for {pair}: MACD crossed below signal")
                    return True
            
            # For shorts: exit if MACD crosses above signal
            elif position.macd_signal == "bearish":
                if prev_macd < prev_signal and current_macd > current_signal:
                    logger.info(f"MACD reversal detected for {pair}: MACD crossed above signal")
                    return True
            
            # Calculate RSI
            rsi = calculate_rsi(prices, self.config.rsi_period)
            if len(rsi) < 2:
                return False
            
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            
            # Check for RSI breakdown
            # For longs: exit if RSI breaks below 50 after being above
            if position.macd_signal == "bullish":
                if prev_rsi > 50 and current_rsi < 50:
                    logger.info(f"RSI breakdown detected for {pair}: RSI crossed below 50")
                    return True
            
            # For shorts: exit if RSI breaks above 50 after being below
            elif position.macd_signal == "bearish":
                if prev_rsi < 50 and current_rsi > 50:
                    logger.info(f"RSI breakdown detected for {pair}: RSI crossed above 50")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking signal reversal for {pair}: {e}")
            return False
    
    def execute_batch_entry(
        self,
        pair: str,
        signal: VolatilityExpansionSignal,
        total_equity: float,
        balance: Dict[str, float]
    ) -> bool:
        """Execute batch entry orders (4 batches with 10% spacing).
        
        Args:
            pair: Trading pair
            signal: Volatility expansion signal
            total_equity: Total account equity
            balance: Current account balances
            
        Returns:
            True if at least one batch entry successful
        """
        if not signal.batch_entries:
            return False
        
        # Calculate position size per batch using utilization-aware dynamic risk
        # Estimate current utilization from position manager and latest prices
        try:
            positions_now = self.position_manager.get_positions()
            # Build latest prices map from balance symbols (approx) and ticker list gathered in run_cycle
            # Here, we approximate utilization from total_equity and USD balance
            usd_cash = balance.get("USD", 0.0)
            invested_estimate = max(0.0, total_equity - usd_cash)
            current_utilization = (invested_estimate / total_equity) if total_equity > 0 else 0.0
        except Exception:
            current_utilization = 0.0
        
        # Base risk and dynamic adjustment toward target utilization
        base_position_pct = self.config.risk_per_trade_pct
        target_u = getattr(self.config, "target_utilization", 0.90)
        low_u = getattr(self.config, "utilization_low_threshold", 0.60)
        high_u = getattr(self.config, "utilization_high_threshold", 0.90)
        dynamic_risk = base_position_pct * (1.0 + max(0.0, (target_u - current_utilization)) / max(target_u, 1e-6))
        if current_utilization < low_u:
            dynamic_risk = max(dynamic_risk, 0.030)  # at least 3.0% when under-utilized
        elif current_utilization > high_u:
            dynamic_risk = min(dynamic_risk, 0.008)  # at most 0.8% when over-utilized
        
        quality_multiplier = 0.7 + (signal.quality * 0.6)  # 0.7x to 1.3x
        position_pct = min(dynamic_risk * quality_multiplier, self.config.max_position_pct)  # Cap by config
        total_position_value = total_equity * position_pct
        position_size_per_batch = total_position_value / len(signal.batch_entries)
        
        # Get exchange info
        roostoo_pair = to_roostoo_pair(pair)
        if roostoo_pair not in self.exchange_info_map:
            logger.error(f"No exchange info for {roostoo_pair}")
            return False
        
        exchange_info = self.exchange_info_map[roostoo_pair]
        
        # Calculate amount per batch
        amounts_per_batch = []
        for entry_price in signal.batch_entries:
            amount = position_size_per_batch / entry_price
            amount = self.risk_manager.round_to_precision(amount, exchange_info.amount_precision)
            amounts_per_batch.append(amount)
        
        # Execute batch entries (limit orders)
        filled_count = 0
        for i, (entry_price, amount) in enumerate(zip(signal.batch_entries, amounts_per_batch)):
            if amount <= 0:
                continue
            
            # Check if we can trade (1 minute rule)
            if not self.datastore.check_trade_allowed():
                logger.info(f"Trade cooldown, skipping batch {i+1}")
                continue
            
            try:
                # Place limit order
                order = self.client.place_order(
                    pair=roostoo_pair,
                    side="buy" if signal.signal == "buy" else "sell",
                    type="limit",
                    quantity=amount,
                    price=entry_price
                )
                
                logger.info(f"Placed batch {i+1} order {order.order_id} for {amount} {pair} at {entry_price}")
                
                # Wait and check if filled
                time.sleep(2)
                order_status_list = self.client.query_order(order.order_id)
                order_status = order_status_list[0] if order_status_list else order
                
                if order_status.status in ["FILLED", "PARTIALLY_FILLED", "filled", "partially_filled"]:
                    filled_amount = getattr(order_status, 'filled_quantity', amount)
                    self.position_manager.fill_batch_entry(pair, i, filled_amount)
                    filled_count += 1
                    logger.info(f"Batch {i+1} filled: {filled_amount}")
                else:
                    # Cancel unfilled limit order
                    try:
                        self.client.cancel_order(order.order_id)
                        logger.info(f"Cancelled unfilled batch {i+1} order")
                    except:
                        pass
                
                # Update last trade timestamp
                self.datastore.update_state(last_trade_ts=int(time.time() * 1000))
                
            except Exception as e:
                logger.error(f"Failed to execute batch {i+1} entry: {e}")
                continue
        
        return filled_count > 0
    
    def execute_staggered_exit(
        self,
        pair: str,
        exit_amount: float,
        reason: str
    ) -> bool:
        """Execute staggered exit (partial position close).
        
        Args:
            pair: Trading pair
            exit_amount: Amount to exit
            reason: Exit reason
            
        Returns:
            True if successful
        """
        if exit_amount <= 0:
            return False
        
        # Check if we can trade
        if not self.datastore.check_trade_allowed():
            logger.info(f"Trade cooldown, cannot exit {pair}")
            return False
        
        try:
            roostoo_pair = to_roostoo_pair(pair)
            
            # Place market sell order
            order = self.client.place_order(
                pair=roostoo_pair,
                side="sell",
                type="market",
                quantity=exit_amount
            )
            
            logger.info(f"Executed staggered exit for {pair}: {exit_amount} ({reason})")
            
            # Wait and verify
            time.sleep(2)
            order_status_list = self.client.query_order(order.order_id)
            order_status = order_status_list[0] if order_status_list else order
            
            if order_status.status in ["FILLED", "PARTIALLY_FILLED", "filled", "partially_filled"]:
                # Update position amount
                position = self.position_manager.get_position(pair)
                if position:
                    # Reduce total amount
                    position.total_amount -= exit_amount
                    if position.total_amount <= 0:
                        # Close entire position
                        self.position_manager.remove_position(pair)
                
                self.datastore.update_state(last_trade_ts=int(time.time() * 1000))
                return True
            else:
                logger.warning(f"Staggered exit order {order.order_id} not filled: {order_status.status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute staggered exit for {pair}: {e}")
            return False
    
    def check_exits(self, ticker_data: List[Dict[str, any]]) -> None:
        """Check all positions for exit conditions.
        
        Args:
            ticker_data: Current ticker data
        """
        # Get current prices
        prices = {from_roostoo_pair(t["pair"]): t["price"] for t in ticker_data}
        
        # Update position prices
        self.position_manager.update_prices(prices)
        
        # Check for exits
        exits = self.position_manager.check_exits(prices, self.config.trailing_stop_pct)
        
        for pair, exit_amount, reason in exits:
            self.execute_staggered_exit(pair, exit_amount, reason)
        
        # Check for signal reversals
        positions = self.position_manager.get_positions()
        for pair, position in positions.items():
            if pair in prices:
                if self.check_signal_reversal(pair, position, self.datastore, prices[pair]):
                    # Exit entire position on reversal
                    if position.total_amount > 0:
                        self.execute_staggered_exit(pair, position.total_amount, "Signal reversal")
    
    def run_cycle(self) -> Dict[str, Any]:
        """Run a complete trading cycle.
        
        Returns:
            Dictionary with cycle results
        """
        cycle_start = time.time()
        results = {
            "timestamp": int(cycle_start * 1000),
            "success": False,
            "signals_generated": 0,
            "positions_opened": 0,
            "exits_executed": 0,
            "error": None
        }
        
        try:
            # Update exchange info if needed
            if self.datastore.should_update_exchange_info() or not self.exchange_info_map:
                self.update_exchange_info()
                time.sleep(0.5)
            
            # Get ticker data
            ticker_data = self.client.ticker()
            ticker_list = [{"pair": from_roostoo_pair(t.pair), "price": t.last_price,
                           "volume_24h": t.coin_trade_value,
                           "bid": t.max_bid, "ask": t.min_ask} for t in ticker_data]
            # Zero/invalid price guard
            prices = {from_roostoo_pair(t.pair): t.last_price for t in ticker_data if (t.last_price is not None and t.last_price > 0)}
            # Filter ticker_list to valid prices
            ticker_list = [t for t in ticker_list if t["price"] is not None and t["price"] > 0]
            
            # Collect minute bars
            self.datastore.collect_minute_bars(ticker_list)
            
            # Check exits first
            self.check_exits(ticker_list)
            
            # Generate signals
            all_signals = {}
            # Compute current utilization for adaptive thresholds
            try:
                positions_now = self.position_manager.get_positions()
                invested_value = 0.0
                for p, pos in positions_now.items():
                    if p in prices:
                        invested_value += pos.total_amount * prices[p]
                balance = self.client.balance()
                total_equity = self.get_total_equity(balance, prices)
                current_utilization = (invested_value / total_equity) if total_equity > 0 else 0.0
            except Exception:
                balance = {"USD": 0.0}
                total_equity = 0.0
                current_utilization = 0.0
            
            # Dynamic quality threshold: looser when under-utilized, stricter when saturated
            if current_utilization < 0.85:
                min_quality_threshold = 0.55
            elif current_utilization < self.config.utilization_high_threshold:
                min_quality_threshold = 0.60
            else:
                min_quality_threshold = 0.75
            
            # Diagnostic tracking
            pairs_checked = 0
            pairs_with_data = 0
            signals_found = 0
            signals_below_quality = 0
            error_count = 0
            
            for ticker in ticker_list:
                pair = ticker["pair"]
                pairs_checked += 1
                
                # Skip if we already have a position
                if self.position_manager.get_position(pair):
                    continue
                
                try:
                    signal = self.signal_generator.compute_signal(
                        pair, self.datastore, ticker, utilization=current_utilization
                    )
                    
                    # Check if we have data (signal generator returns None if insufficient data)
                    if signal is None:
                        continue
                    
                    pairs_with_data += 1
                    
                    if signal.signal != "neutral":
                        signals_found += 1
                        if signal.quality >= min_quality_threshold:
                            all_signals[pair] = signal
                            logger.info(f"Signal found: {pair} - {signal.strategy_type}, quality={signal.quality:.2f}, RSI={getattr(signal, 'rsi_value', 'N/A')}")
                        else:
                            signals_below_quality += 1
                            logger.debug(f"Signal below quality threshold: {pair} - quality={signal.quality:.2f} (min={min_quality_threshold:.2f})")
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Failed to generate signal for {pair}: {e}")
                    continue
            
            # Log diagnostic summary
            if pairs_checked > 0:
                logger.info(f"Signal generation summary: {pairs_checked} pairs checked, {pairs_with_data} with sufficient data, "
                          f"{signals_found} signals found, {signals_below_quality} below quality threshold, {error_count} errors")
            
            # If no signals but we have data, log diagnostic info from signal generator
            if len(all_signals) == 0 and pairs_with_data > 0:
                diag = getattr(self.signal_generator, '_diagnostic_counter', {})
                early = getattr(self.signal_generator, '_early_return_counter', {})
                if diag or early:
                    logger.info(f"Diagnostics - RSI<30: {diag.get('rsi_oversold', 0)}, PriceNearBB: {diag.get('price_near_lower', 0)}, "
                              f"Both: {diag.get('both_met', 0)}, Early returns: {sum(early.values())}")
            
            results["signals_generated"] = len(all_signals)
            
            # Rank signals
            ranked_signals = self.signal_generator.rank_signals(all_signals)
            
            # Filter out short positions - only allow long positions (compliance: no shorting)
            ranked_signals = [(pair, sig) for pair, sig in ranked_signals if sig.signal == "buy"]
            
            # Get balance and equity
            balance = self.client.balance()
            total_equity = self.get_total_equity(balance, prices)
            
            # Execute top signals (respect max positions and per-cycle cap)
            positions_opened = 0
            max_new_per_cycle = 4 if current_utilization < 0.85 else 2
            for pair, signal in ranked_signals:
                if len(self.position_manager.get_positions()) >= self.config.max_positions:
                    break
                if positions_opened >= max_new_per_cycle:
                    break
                
                # Add position to manager
                success = self.position_manager.add_position(
                    pair=pair,
                    strategy_type=signal.strategy_type,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    quality=signal.quality,
                    atr_value=signal.atr_value,
                    rsi_value=signal.rsi_value,
                    macd_signal=signal.macd_signal,
                    batch_entry_prices=signal.batch_entries,
                    batch_spacing_pct=self.config.batch_spacing_pct,
                    use_scaled_exits=self.config.use_scaled_exits
                )
                
                if success:
                    # Execute batch entries
                    if self.execute_batch_entry(pair, signal, total_equity, balance):
                        positions_opened += 1
                        logger.info(f"Opened position in {pair} with batch entries")
            
            results["positions_opened"] = positions_opened
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            results["error"] = str(e)
            self.datastore.increment_error_count()
        
        cycle_time = time.time() - cycle_start
        results["cycle_time"] = cycle_time
        logger.info(f"Cycle completed in {cycle_time:.2f}s")
        
        return results

