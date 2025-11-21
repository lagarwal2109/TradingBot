"""Risk management and position sizing for the trading bot."""

import math
from typing import Dict, Optional, Tuple
from bot.config import get_config
from bot.roostoo import ExchangeInfo


class RiskManager:
    """Manages position sizing and risk parameters."""
    
    def __init__(self, max_position_pct: float = 0.4):
        """Initialize risk manager.
        
        Args:
            max_position_pct: Maximum position size as fraction of equity
        """
        self.max_position_pct = max_position_pct
    
    def calculate_position_size(
        self,
        total_equity: float,
        volatility: float,
        price: float,
        exchange_info: ExchangeInfo,
        volatility_scalar: float = 1.0
    ) -> float:
        """Calculate position size based on volatility-adjusted sizing.
        
        Position size is inversely proportional to volatility to maintain
        consistent risk across different market conditions.
        
        Args:
            total_equity: Total account equity in quote currency (USD)
            volatility: Recent volatility (standard deviation of returns)
            price: Current price of the asset
            exchange_info: Exchange information for the pair
            volatility_scalar: Scaling factor for volatility adjustment
        
        Returns:
            Position size in base currency units
        """
        # Base position value as fraction of equity
        base_position_value = total_equity * self.max_position_pct
        
        # Adjust for volatility (inverse relationship)
        # Higher volatility = smaller position
        if volatility > 0:
            volatility_adjusted_value = base_position_value * (volatility_scalar / (volatility * 100))
        else:
            # Use base position if volatility is zero
            volatility_adjusted_value = base_position_value
        
        # Ensure we don't exceed max position
        position_value = min(volatility_adjusted_value, base_position_value)
        
        # Convert to position size in base currency
        if price > 0:
            position_size = position_value / price
        else:
            return 0.0
        
        # Round to exchange precision
        position_size = self.round_to_precision(position_size, exchange_info.amount_precision)
        
        # Check minimum order constraint
        position_size = self.apply_minimum_order_constraint(
            position_size, price, exchange_info
        )
        
        return position_size
    
    def round_to_precision(self, value: float, precision: int) -> float:
        """Round value to specified decimal precision.
        
        Args:
            value: Value to round
            precision: Number of decimal places
        
        Returns:
            Rounded value
        """
        if precision >= 0:
            return round(value, precision)
        else:
            # Handle negative precision (rounding to powers of 10)
            factor = 10 ** (-precision)
            return round(value / factor) * factor
    
    def apply_minimum_order_constraint(
        self,
        amount: float,
        price: float,
        exchange_info: ExchangeInfo
    ) -> float:
        """Apply minimum order constraint.
        
        Args:
            amount: Proposed order amount
            price: Current price
            exchange_info: Exchange information with minimum order value
        
        Returns:
            Adjusted amount that satisfies minimum order constraint
        """
        notional_value = amount * price
        
        if notional_value < exchange_info.mini_order:
            # Order is too small
            if price > 0:
                # Calculate minimum amount needed
                min_amount = exchange_info.mini_order / price
                # Round up to ensure we meet minimum
                min_amount = math.ceil(min_amount * (10 ** exchange_info.amount_precision)) / (10 ** exchange_info.amount_precision)
                return min_amount
            else:
                return 0.0
        
        return amount
    
    def validate_order(
        self,
        amount: float,
        price: float,
        side: str,
        balance: Dict[str, float],
        exchange_info: ExchangeInfo
    ) -> Tuple[bool, Optional[str]]:
        """Validate if an order can be placed.
        
        Args:
            amount: Order amount in base currency
            price: Order price
            side: "buy" or "sell"
            balance: Current account balances
            exchange_info: Exchange information for the pair
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check minimum order size
        notional_value = amount * price
        if notional_value < exchange_info.mini_order:
            return False, f"Order value {notional_value:.2f} below minimum {exchange_info.mini_order}"
        
        # Check balance sufficiency
        if side.lower() == "buy":
            # Need quote currency (USD) for buying
            required_balance = notional_value
            # Roostoo v3 uses 'unit' instead of 'quote_currency'
            quote_currency = getattr(exchange_info, 'quote_currency', None) or getattr(exchange_info, 'unit', 'USD')
            
            if quote_currency not in balance or balance[quote_currency] < required_balance:
                available = balance.get(quote_currency, 0)
                return False, f"Insufficient {quote_currency} balance: {available:.2f} < {required_balance:.2f}"
        
        elif side.lower() == "sell":
            # Need base currency for selling
            # Roostoo v3 uses 'coin' instead of 'base_currency'
            base_currency = getattr(exchange_info, 'base_currency', None) or getattr(exchange_info, 'coin', '')
            
            if base_currency not in balance or balance[base_currency] < amount:
                available = balance.get(base_currency, 0)
                return False, f"Insufficient {base_currency} balance: {available:.8f} < {amount:.8f}"
        
        return True, None
    
    def calculate_stop_loss(self, entry_price: float, volatility: float, multiplier: float = 2.0) -> float:
        """Calculate stop loss price based on volatility.
        
        Args:
            entry_price: Entry price for the position
            volatility: Recent volatility (standard deviation)
            multiplier: Number of standard deviations for stop loss
        
        Returns:
            Stop loss price
        """
        stop_distance = entry_price * volatility * multiplier
        return entry_price - stop_distance
    
    def calculate_atr_stop_loss(
        self,
        entry_price: float,
        atr_value: float,
        multiplier: float = 1.5,
        is_long: bool = True
    ) -> float:
        """Calculate stop loss price based on ATR (Average True Range).
        
        Uses ATR multiplied by a factor (default 1.5x) for dynamic stop-loss distance.
        This is more adaptive to market volatility than fixed percentage stops.
        
        Args:
            entry_price: Entry price for the position
            atr_value: Current ATR value
            multiplier: ATR multiplier (default 1.5 for volatility expansion strategy)
            is_long: True for long positions, False for short positions
        
        Returns:
            Stop loss price
        """
        stop_distance = atr_value * multiplier
        
        if is_long:
            # For longs: stop loss is below entry (entry_price - 1.5x ATR)
            stop_loss = entry_price - stop_distance
        else:
            # For shorts: stop loss is above entry (entry_price + 1.5x ATR)
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_position_size_with_atr(
        self,
        total_equity: float,
        entry_price: float,
        atr_value: float,
        risk_per_trade_pct: float = 0.015,  # 1.5% risk per trade
        atr_multiplier: float = 1.5,
        exchange_info: Optional[ExchangeInfo] = None
    ) -> float:
        """Calculate position size using ATR-based risk management.
        
        Position size = (Account Risk × Account Size) / Stop Loss Distance
        Where Stop Loss Distance = ATR × multiplier
        
        Args:
            total_equity: Total account equity in USD
            entry_price: Entry price for the position
            atr_value: Current ATR value
            risk_per_trade_pct: Risk per trade as percentage of equity (default 1.5%)
            atr_multiplier: ATR multiplier for stop loss (default 1.5)
            exchange_info: Exchange information for precision rounding (optional)
        
        Returns:
            Position size in base currency units
        """
        # Calculate account risk amount
        account_risk = total_equity * risk_per_trade_pct
        
        # Calculate stop loss distance (ATR × multiplier)
        stop_loss_distance = atr_value * atr_multiplier
        
        if stop_loss_distance <= 0 or entry_price <= 0:
            return 0.0
        
        # Calculate position size: P = (Account Risk × Account Size) / Stop Loss Distance
        # But we need to account for price, so: amount = risk_amount / stop_loss_distance
        # However, we need to convert to base currency: amount = risk_amount / (stop_loss_distance / entry_price)
        # Simplified: amount = (risk_amount * entry_price) / stop_loss_distance
        
        position_value = (account_risk * entry_price) / stop_loss_distance
        position_size = position_value / entry_price
        
        # Round to precision if exchange info provided
        if exchange_info:
            position_size = self.round_to_precision(position_size, exchange_info.amount_precision)
            position_size = self.apply_minimum_order_constraint(
                position_size, entry_price, exchange_info
            )
        
        return position_size
    
    def calculate_portfolio_weights(
        self,
        target_weights: Dict[str, float],
        current_balances: Dict[str, float],
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate actual portfolio weights from balances.
        
        Args:
            target_weights: Target portfolio weights
            current_balances: Current asset balances
            prices: Current prices for assets
        
        Returns:
            Dictionary of actual weights
        """
        # Calculate total portfolio value
        total_value = 0.0
        for asset, balance in current_balances.items():
            if asset == "USD":
                total_value += balance
            elif asset in prices:
                total_value += balance * prices[asset]
        
        if total_value <= 0:
            return {}
        
        # Calculate actual weights
        weights = {}
        for asset, balance in current_balances.items():
            if balance > 0:
                if asset == "USD":
                    weights[asset] = balance / total_value
                elif asset in prices:
                    weights[asset] = (balance * prices[asset]) / total_value
        
        return weights
    
    def rebalancing_trades(
        self,
        target_position: Optional[str],
        current_position: Optional[str],
        total_equity: float,
        features: Dict[str, float],
        exchange_info_map: Dict[str, ExchangeInfo],
        balance: Dict[str, float]
    ) -> Optional[Dict[str, any]]:
        """Determine rebalancing trade needed.
        
        Args:
            target_position: Target position (coin symbol or None for USD)
            current_position: Current position (coin symbol or None for USD)
            total_equity: Total account equity in USD
            features: Features for the target coin (if any)
            exchange_info_map: Map of pair to exchange info
            balance: Current account balances
        
        Returns:
            Trade dictionary with 'action', 'pair', 'side', 'amount' or None
        """
        # No change needed
        if target_position == current_position:
            return None
        
        # Need to close current position first
        if current_position is not None:
            pair = f"{current_position}USD"
            # Convert to Roostoo format (with slash) for lookup
            from bot.utils import to_roostoo_pair
            roostoo_pair = to_roostoo_pair(pair)
            
            if roostoo_pair not in exchange_info_map:
                return None
            
            # Sell entire balance of current position
            base_currency = current_position
            if base_currency in balance and balance[base_currency] > 0:
                amount = balance[base_currency]
                # Round to precision
                exchange_info = exchange_info_map[roostoo_pair]
                amount = self.round_to_precision(amount, exchange_info.amount_precision)
                
                return {
                    "action": "close_position",
                    "pair": roostoo_pair,  # Use Roostoo format for trading
                    "side": "sell",
                    "amount": amount,
                    "type": "market"
                }
        
        # Open new position
        if target_position is not None:
            pair = f"{target_position}USD"
            # Convert to Roostoo format (with slash) for lookup
            from bot.utils import to_roostoo_pair
            roostoo_pair = to_roostoo_pair(pair)
            
            if roostoo_pair not in exchange_info_map:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Pair {roostoo_pair} (from {pair}) not in exchange_info_map. Available: {list(exchange_info_map.keys())[:5]}...")
                return None
            
            exchange_info = exchange_info_map[roostoo_pair]
            
            # Calculate position size
            # Handle None features gracefully
            if features is None:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"No features provided for {pair}, using defaults")
                # Try to get price from exchange info or use a default
                # We need price to calculate position size, so we'll need to fetch it
                return None  # Can't calculate without price
            
            volatility = features.get("rolling_std", 0.01)  # Default 1% if not available
            price = features.get("price", 0)
            
            if price <= 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Invalid price {price} for {pair}")
                return None
            
            amount = self.calculate_position_size(
                total_equity,
                volatility,
                price,
                exchange_info
            )
            
            if amount > 0:
                return {
                    "action": "open_position",
                    "pair": roostoo_pair,  # Use Roostoo format for trading
                    "side": "buy",
                    "amount": amount,
                    "type": "market"
                }
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Calculated amount {amount} is <= 0 for {pair}")
        
        return None
