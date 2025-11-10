#!/usr/bin/env python3
"""Script to close a position and reset bot state."""

import sys
import logging
from bot.config import get_config
from bot.roostoo_v3 import RoostooV3Client
from bot.datastore import DataStore
from bot.risk import RiskManager
from bot.engine_enhanced import EnhancedTradingEngine
from bot.utils import to_roostoo_pair

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def close_position_and_reset(symbol: str = "UNI"):
    """Close a position and reset bot state."""
    try:
        # Initialize components
        config = get_config()
        client = RoostooV3Client()
        datastore = DataStore()
        risk_manager = RiskManager(max_position_pct=config.max_position_pct)
        engine = EnhancedTradingEngine(
            trading_client=client,
            market_data_client=None,
            datastore=datastore,
            risk_manager=risk_manager
        )
        
        # Get current balance
        from bot.utils import filter_zero_balances
        balance = client.balance()
        non_zero_balance = filter_zero_balances(balance)
        logger.info(f"Current balances: {non_zero_balance}")
        
        # Check if we have the position
        if symbol not in balance or balance[symbol] <= 0:
            logger.info(f"No {symbol} position found. Current position: {datastore.state.current_position}")
            if datastore.state.current_position:
                logger.info("Resetting state to None")
                datastore.update_state(current_position=None)
            return
        
        # Close the position
        logger.info(f"Closing {symbol} position: {balance[symbol]:.6f} {symbol}")
        
        # Update exchange info first
        engine.update_exchange_info()
        
        # Update state to reflect current position before closing
        datastore.update_state(current_position=symbol)
        
        # Close position using engine
        if engine.close_position(symbol):
            logger.info(f"Successfully closed {symbol} position")
            # Reset state
            datastore.update_state(current_position=None)
            logger.info("Bot state reset to None (flat position)")
        else:
            logger.error(f"Failed to close {symbol} position")
            return
        
        # Verify final balance
        from bot.utils import filter_zero_balances
        final_balance = client.balance()
        non_zero_final = filter_zero_balances(final_balance)
        logger.info(f"Final balances: {non_zero_final}")
        
        # Calculate equity
        if "USD" in final_balance:
            logger.info(f"Total USD: ${final_balance['USD']:,.2f}")
        
    except Exception as e:
        logger.error(f"Error closing position: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Close a position and reset bot state")
    parser.add_argument("--symbol", default="UNI", help="Symbol to close (default: UNI)")
    args = parser.parse_args()
    
    close_position_and_reset(args.symbol)

