"""Trading engine with decision logic and rebalancing."""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from bot.config import get_config
from bot.roostoo_v3 import RoostooV3Client, ExchangeInfo
from bot.datastore import DataStore
from bot.signals import SignalGenerator
from bot.risk import RiskManager
from bot.utils import to_roostoo_pair, from_roostoo_pair


logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine that coordinates signal generation and execution."""
    
    def __init__(
        self,
        client: RoostooV3Client,
        datastore: DataStore,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        mode: str = "sharpe"  # "sharpe" or "tangency"
    ):
        """Initialize trading engine.
        
        Args:
            client: Roostoo API client
            datastore: Data storage instance
            signal_generator: Signal generator instance
            risk_manager: Risk management instance
            mode: Trading mode ("sharpe" or "tangency")
        """
        self.client = client
        self.datastore = datastore
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.mode = mode
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
        """Calculate total equity in USD.
        
        Args:
            balance: Account balances
            prices: Current prices for assets
        
        Returns:
            Total equity in USD
        """
        total = balance.get("USD", 0.0)
        
        for asset, amount in balance.items():
            if asset != "USD" and amount > 0:
                pair = f"{asset}USD"
                if pair in prices:
                    total += amount * prices[pair]
        
        return total
    
    def decide_target(self, features: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Decide target position based on signals.
        
        Args:
            features: Feature dictionary for all pairs
        
        Returns:
            Target coin symbol or None for USD (flat)
        """
        # Filter for liquid pairs with sufficient data
        # Note: liquidity_score can be 0 if bid/ask missing, but volume_24h is still valid
        eligible = {
            pair: feat for pair, feat in features.items()
            if feat["has_sufficient_data"] and (feat["liquidity_score"] > 0 or feat.get("volume_24h", 0) > 10000)
        }
        
        if not eligible:
            logger.info("No eligible pairs found, staying flat")
            return None
        
        if self.mode == "sharpe":
            # Sharpe ratio ranking strategy
            best_pair = None
            best_sharpe = 0.0
            
            for pair, feat in eligible.items():
                if (feat["sharpe_ratio"] > self.config.min_sharpe and 
                    feat["momentum"] > 0):
                    if feat["sharpe_ratio"] > best_sharpe:
                        best_sharpe = feat["sharpe_ratio"]
                        best_pair = pair
            
            if best_pair:
                # Extract base currency from pair (e.g., "BTCUSD" -> "BTC")
                base_currency = best_pair.replace("USD", "")
                logger.info(f"Selected {base_currency} with Sharpe {best_sharpe:.4f}")
                return base_currency
            
        elif self.mode == "tangency":
            # Tangency portfolio mode (simplified to single best asset)
            # Full implementation would use portfolio weights
            weights = self.signal_generator.compute_tangency_weights(eligible, top_n=10)
            if weights:
                # Pick highest weight asset
                best_pair = max(weights.items(), key=lambda x: x[1])[0]
                base_currency = best_pair.replace("USD", "")
                logger.info(f"Selected {base_currency} from tangency portfolio")
                return base_currency
        
        logger.info("No suitable position found, staying flat")
        return None
    
    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            open_orders = self.client.get_open_orders()
            for order in open_orders:
                try:
                    self.client.cancel_order(order.order_id)
                    logger.info(f"Cancelled order {order.order_id}")
                except Exception as e:
                    logger.error(f"Failed to cancel order {order.order_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
    
    def execute_trade(self, trade: Dict[str, Any]) -> bool:
        """Execute a single trade.
        
        Args:
            trade: Trade dictionary with pair, side, amount, type
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert pair format for Roostoo API
            roostoo_pair = to_roostoo_pair(trade["pair"])
            
            # Place order
            order = self.client.place_order(
                pair=roostoo_pair,
                side=trade["side"],
                type=trade["type"],
                quantity=trade["amount"]
            )
            
            logger.info(f"Placed {trade['side']} order {order.order_id} for {trade['amount']} {trade['pair']}")
            
            # For market orders, wait briefly and check status
            time.sleep(2)
            
            # Query order status
            order_status_list = self.client.query_order(order.order_id)
            order_status = order_status_list[0] if order_status_list else order
            
            if order_status.status in ["FILLED", "PARTIALLY_FILLED", "filled", "partially_filled"]:
                logger.info(f"Order {order.order_id} executed successfully")
                return True
            else:
                logger.warning(f"Order {order.order_id} status: {order_status.status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            self.datastore.increment_error_count()
            return False
    
    def rebalance(self, target_position: Optional[str], cached_balance: Optional[Dict] = None, cached_prices: Optional[Dict] = None) -> bool:
        """Rebalance portfolio to target position.
        
        Args:
            target_position: Target coin symbol or None for USD
            cached_balance: Pre-fetched balance (optional)
            cached_prices: Pre-fetched prices (optional)
        
        Returns:
            True if rebalancing successful or not needed
        """
        # Check if we can trade (1 minute rule)
        if not self.datastore.check_trade_allowed():
            seconds_to_wait = 60 - ((time.time() * 1000 - self.datastore.state.last_trade_ts) / 1000)
            logger.info(f"Trade cooldown: {seconds_to_wait:.1f}s remaining")
            return True
        
        current_position = self.datastore.state.current_position
        
        # No change needed
        if target_position == current_position:
            return True
        
        try:
            # Use cached data if provided, otherwise fetch
            balance = cached_balance if cached_balance else self.client.balance()
            
            if cached_prices is None:
                ticker_data = self.client.ticker()
                prices = {from_roostoo_pair(t.pair): t.last_price for t in ticker_data}
            else:
                prices = cached_prices
            
            # Calculate total equity
            total_equity = self.get_total_equity(balance, prices)
            
            # Get features for target position
            features = None
            if target_position:
                pair = f"{target_position}USD"
                # Get ticker data for the target pair to compute features
                if cached_prices and pair in cached_prices:
                    # Reconstruct ticker info from cached prices
                    ticker_info = {
                        "pair": pair,
                        "price": cached_prices[pair],
                        "volume_24h": 0,  # Not critical for rebalancing
                        "bid": cached_prices[pair] * 0.999,  # Estimate
                        "ask": cached_prices[pair] * 1.001  # Estimate
                    }
                    try:
                        features = self.signal_generator.compute_features(
                            pair, self.datastore, ticker_info
                        )
                        logger.debug(f"Computed features for {pair}: price={features.get('price', 0):.4f}, std={features.get('rolling_std', 0):.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to compute features for {pair}: {e}")
                        # Create minimal features dict with just price
                        features = {
                            "pair": pair,
                            "price": cached_prices[pair],
                            "rolling_std": 0.02,  # Default 2% volatility
                            "volume_24h": 0,
                            "liquidity_score": 0
                        }
                else:
                    logger.warning(f"Pair {pair} not in cached_prices, fetching ticker data")
                    # Fallback: fetch ticker data for this pair
                    try:
                        ticker_data = self.client.ticker(pair=to_roostoo_pair(pair))
                        if ticker_data:
                            ticker_info = {
                                "pair": pair,
                                "price": ticker_data.last_price if hasattr(ticker_data, 'last_price') else cached_prices.get(pair, 0),
                                "volume_24h": getattr(ticker_data, 'coin_trade_value', 0),
                                "bid": getattr(ticker_data, 'max_bid', cached_prices.get(pair, 0) * 0.999),
                                "ask": getattr(ticker_data, 'min_ask', cached_prices.get(pair, 0) * 1.001)
                            }
                            features = self.signal_generator.compute_features(
                                pair, self.datastore, ticker_info
                            )
                    except Exception as e:
                        logger.error(f"Failed to fetch ticker for {pair}: {e}")
            
            # Determine rebalancing trade
            logger.info(f"Rebalancing: current={current_position}, target={target_position}, equity=${total_equity:.2f}")
            trade = self.risk_manager.rebalancing_trades(
                target_position=target_position,
                current_position=current_position,
                total_equity=total_equity,
                features=features,
                exchange_info_map=self.exchange_info_map,
                balance=balance
            )
            
            if trade:
                logger.info(f"Trade generated: {trade}")
                # Validate order
                exchange_info = self.exchange_info_map.get(trade["pair"])
                if not exchange_info:
                    logger.error(f"No exchange info for {trade['pair']}")
                    return False
                
                # Convert pair format for price lookup
                internal_pair = from_roostoo_pair(trade["pair"])
                price = prices.get(internal_pair, 0)
                if price == 0:
                    # Fallback: get price from exchange info or fetch it
                    logger.warning(f"Price not found for {internal_pair}, fetching from API")
                    try:
                        ticker_data = self.client.ticker(pair=trade["pair"])
                        if ticker_data:
                            price = ticker_data.last_price if hasattr(ticker_data, 'last_price') else 0
                    except Exception as e:
                        logger.error(f"Failed to fetch price for {trade['pair']}: {e}")
                
                is_valid, error_msg = self.risk_manager.validate_order(
                    amount=trade["amount"],
                    price=price,
                    side=trade["side"],
                    balance=balance,
                    exchange_info=exchange_info
                )
                
                if not is_valid:
                    logger.warning(f"Order validation failed: {error_msg}")
                    return False
                
                # Execute trade
                if self.execute_trade(trade):
                    # Update state
                    new_position = target_position if trade["action"] == "open_position" else None
                    self.datastore.record_trade(new_position)
                    self.datastore.reset_error_count()
                    return True
                else:
                    return False
            else:
                logger.info("No trade needed for rebalancing")
                return True
                
        except Exception as e:
            logger.error(f"Rebalancing failed: {e}")
            self.datastore.increment_error_count()
            return False
    
    def check_kill_switch(self) -> bool:
        """Check if kill switch should be activated.
        
        Returns:
            True if kill switch activated
        """
        if self.datastore.state.consecutive_errors >= self.config.max_consecutive_errors:
            logger.critical(f"Kill switch activated: {self.datastore.state.consecutive_errors} consecutive errors")
            
            # Cancel all orders
            self.cancel_all_orders()
            
            # Go flat if we have a position
            if self.datastore.state.current_position:
                logger.info("Closing position due to kill switch")
                self.rebalance(None)
            
            return True
        
        # Check for stuck orders
        try:
            open_orders = self.client.get_open_orders()
            if open_orders:
                logger.warning(f"Found {len(open_orders)} pending orders, cancelling...")
                for order in open_orders:
                    try:
                        self.client.cancel_order(order.order_id)
                        logger.info(f"Cancelled pending order {order.order_id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order.order_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to check open orders: {e}")
        
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
            "target_position": None,
            "current_position": self.datastore.state.current_position,
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
                time.sleep(0.5)  # Pause after exchange info
            
            # Get ticker data ONCE per cycle
            ticker_data = self.client.ticker()
            ticker_list = [{"pair": from_roostoo_pair(t.pair), "price": t.last_price, 
                           "volume_24h": t.coin_trade_value, 
                           "bid": t.max_bid, "ask": t.min_ask} for t in ticker_data]
            prices = {from_roostoo_pair(t.pair): t.last_price for t in ticker_data}
            
            # Collect minute bars
            self.datastore.collect_minute_bars(ticker_list)
            
            # Compute features
            all_features = self.signal_generator.compute_all_features(self.datastore, ticker_list)
            
            # Debug: log feature stats
            eligible_count = sum(1 for f in all_features.values() 
                               if f.get("has_sufficient_data", False))
            logger.info(f"Computed features for {len(all_features)} pairs, {eligible_count} with sufficient data")
            
            # Show top 5 by Sharpe for debugging
            sorted_by_sharpe = sorted(
                [(p, f) for p, f in all_features.items() if f.get("has_sufficient_data", False)],
                key=lambda x: x[1].get("sharpe_ratio", -999),
                reverse=True
            )[:5]
            if sorted_by_sharpe:
                logger.info(f"Top pairs by Sharpe: {[(p, f['sharpe_ratio']) for p, f in sorted_by_sharpe]}")
            else:
                # Show why pairs don't have sufficient data
                sample_pairs = list(all_features.items())[:3]
                for pair, feat in sample_pairs:
                    df = self.datastore.read_minute_bars(pair, limit=100)
                    logger.info(f"  {pair}: {len(df)} rows, need {self.signal_generator.window_size + 1}, has_data={feat.get('has_sufficient_data', False)}")
            
            # Decide target position
            target = self.decide_target(all_features)
            results["target_position"] = target
            
            # Get balance
            balance = self.client.balance()
            
            # Rebalance with cached data to avoid re-fetching
            if self.rebalance(target, cached_balance=balance, cached_prices=prices):
                results["success"] = True
                results["current_position"] = self.datastore.state.current_position
                # Refresh balance after trade
                balance = self.client.balance()
            
            results["total_equity"] = self.get_total_equity(balance, prices)
            results["balances"] = balance
            
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            results["error"] = str(e)
            self.datastore.increment_error_count()
        
        # Log cycle time
        cycle_time = time.time() - cycle_start
        results["cycle_time"] = cycle_time
        logger.info(f"Cycle completed in {cycle_time:.2f}s")
        
        return results
