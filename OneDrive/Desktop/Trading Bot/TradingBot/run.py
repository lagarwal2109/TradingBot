#!/usr/bin/env python3
"""Main execution loop for the Roostoo trading bot."""

import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse

from bot.config import get_config, Config
from bot.roostoo_v3 import RoostooV3Client
from bot.datastore import DataStore
from bot.signals import SignalGenerator, TangencyPortfolioSignals
from bot.risk import RiskManager
from bot.engine import TradingEngine
from bot.engine_enhanced import EnhancedTradingEngine
from bot.engine_regime_ensemble import RegimeEnsembleTradingEngine


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
    
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler - JSON lines format
    log_file = log_dir / "bot.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Custom JSON formatter
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "module": record.name,
                "message": record.getMessage()
            }
            if hasattr(record, "extra_data"):
                log_obj.update(record.extra_data)
            return json.dumps(log_obj)
    
    file_handler.setFormatter(JsonFormatter())
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def safe_startup(engine: TradingEngine, logger: logging.Logger) -> None:
    """Safely start the bot, handling any existing positions.
    
    Args:
        engine: Trading engine instance
        logger: Logger instance
    """
    logger.info("Starting bot safely...")
    
    try:
        # Cancel any open orders (if any)
        try:
            engine.cancel_all_orders()
        except Exception as e:
            # Expected if no orders - not an error
            if "no order matched" not in str(e).lower():
                logger.warning(f"Error cancelling orders: {e}")
        
        # Check current balances
        from bot.utils import filter_zero_balances
        balance = engine.client.balance()
        non_zero_balance = filter_zero_balances(balance)
        logger.info(f"Current balances: {non_zero_balance}")
        
        # Check if we have any non-USD positions
        current_positions = [
            asset for asset, amount in balance.items()
            if asset != "USD" and amount > 0.0001  # Small threshold for dust
        ]
        
        if current_positions:
            logger.warning(f"Found existing positions: {current_positions}")
            
            # If state says we should be flat but we have positions, close them
            if engine.datastore.state.current_position is None and current_positions:
                logger.info("State indicates flat position but found holdings, closing positions")
                for position in current_positions:
                    try:
                        engine.datastore.state.current_position = position
                        engine.rebalance(None)  # Go flat
                    except Exception as e:
                        logger.error(f"Failed to close position {position}: {e}")
            
            # If state has a position but it doesn't match holdings, sync state
            elif engine.datastore.state.current_position not in current_positions:
                if len(current_positions) == 1:
                    logger.info(f"Syncing state to current position: {current_positions[0]}")
                    engine.datastore.update_state(current_position=current_positions[0])
                else:
                    logger.warning("Multiple positions found, going flat")
                    engine.datastore.update_state(current_position=None)
                    engine.rebalance(None)
        else:
            # No positions, ensure state reflects this
            if engine.datastore.state.current_position is not None:
                logger.info("No positions found, updating state to flat")
                engine.datastore.update_state(current_position=None)
        
        # Reset error count for fresh start
        engine.datastore.reset_error_count()
        
    except Exception as e:
        logger.error(f"Error during safe startup: {e}")
        raise


def collector_loop(trading_client: RoostooV3Client, datastore: DataStore, logger: logging.Logger) -> None:
    """Run the collector loop to gather minute bars.
    
    Args:
        trading_client: Roostoo client for market data
        datastore: DataStore instance
        logger: Logger instance
    """
    logger.info("Running in collector mode - using Roostoo API for market data")
    
    while True:
        try:
            # Get ticker data from Roostoo
            roostoo_tickers = trading_client.ticker()
            ticker_list = [
                {
                    "pair": t.pair.replace("/", ""),  # Convert "BTC/USD" to "BTCUSD"
                    "price": t.last_price,
                    "volume_24h": t.coin_trade_value,
                    "bid": t.max_bid,
                    "ask": t.min_ask
                }
                for t in roostoo_tickers
            ]
            logger.info(f"Collected data for {len(ticker_list)} pairs from Roostoo API")
            
            # Collect minute bars
            datastore.collect_minute_bars(ticker_list)
            
            # Sleep until next minute
            now = time.time()
            sleep_time = 60 - (now % 60)
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Collector stopped by user")
            break
        except Exception as e:
            logger.error(f"Collector error: {e}")
            time.sleep(10)  # Brief pause before retry


def trading_loop(engine: TradingEngine, logger: logging.Logger, run_once: bool = False) -> None:
    """Run the main trading loop.
    
    Args:
        engine: Trading engine instance
        logger: Logger instance
        run_once: If True, run only one cycle and exit
    """
    # Safe startup
    safe_startup(engine, logger)
    
    logger.info(f"Starting trading loop (mode: {engine.mode})")
    
    while True:
        try:
            # Sync with server time
            server_time = engine.client.server_time()
            local_time = int(time.time() * 1000)
            time_diff = server_time - local_time
            logger.info(f"Server time sync: diff={time_diff}ms")
            
            # Run trading cycle
            results = engine.run_cycle()
            
            # Log results
            logger.info("Cycle results", extra={"extra_data": results})
            
            if results.get("error"):
                logger.error(f"Cycle error: {results['error']}")
            else:
                from bot.utils import filter_zero_balances
                balances = results.get('balances', {})
                non_zero_balances = filter_zero_balances(balances) if balances else {}
                logger.info(
                    f"Position: {results['current_position']} "
                    f"(target: {results['target_position']}), "
                    f"Equity: ${results.get('total_equity', 0):.2f}, "
                    f"Balances: {non_zero_balances}"
                )
            
            if run_once:
                break
            
            # Sleep until next minute
            now = time.time()
            sleep_time = 60 - (now % 60)
            logger.info(f"Sleeping for {sleep_time:.1f}s until next cycle")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
            break
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            if not run_once:
                time.sleep(10)  # Brief pause before retry
            else:
                raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Roostoo Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["trade", "collect", "sharpe", "tangency", "enhanced", "regime_ensemble"],
        default="regime_ensemble",
        help="Operating mode: regime_ensemble (recommended), enhanced, sharpe, tangency, or collect data"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,  # Use config default if not provided
        help="Rolling window size for signals (minutes, default: use config)"
    )
    parser.add_argument(
        "--momentum",
        type=int,
        default=None,
        help="Momentum lookback period (minutes, default: use config)"
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=None,
        help="Maximum position size as fraction of equity (default: use config)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run only one cycle and exit"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = get_config()
        
        # Override config with command line args (only if provided)
        if args.window is not None:
            config.window_size = args.window
        if args.momentum is not None:
            config.momentum_lookback = args.momentum
        if args.max_position is not None:
            config.max_position_pct = args.max_position
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure ROOSTOO_API_KEY and ROOSTOO_API_SECRET are set in .env file")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(config.log_dir)
    logger.info(f"Starting Roostoo Trading Bot v0.1.0 (mode: {args.mode})")
    
    # Initialize components
    try:
        # Initialize API clients
        trading_client = RoostooV3Client()  # For trading execution and market data
        logger.info("Initialized Roostoo client (for trading and market data)")
        
        datastore = DataStore()
        
        # Choose engine and signal generator based on mode
        if args.mode == "regime_ensemble":
            # Use regime-adaptive ensemble trading engine
            risk_manager = RiskManager(max_position_pct=config.max_position_pct)
            engine = RegimeEnsembleTradingEngine(
                trading_client=trading_client,
                datastore=datastore,
                risk_manager=risk_manager
            )
        elif args.mode == "enhanced":
            # Use enhanced trading engine with breakout strategy
            risk_manager = RiskManager(max_position_pct=config.max_position_pct)
            engine = EnhancedTradingEngine(
                trading_client=trading_client,
                market_data_client=None,  # Not using Horus anymore
                datastore=datastore,
                risk_manager=risk_manager
            )
        else:
            # Use legacy engine
            if args.mode == "tangency":
                # window_size is already in minutes (despite config description)
                signal_generator = TangencyPortfolioSignals(
                    window_size=config.window_size,  # Already in minutes
                    momentum_lookback=config.momentum_lookback  # Already in minutes
                )
                trading_mode = "tangency"
            else:
                # window_size is already in minutes (despite config description)
                signal_generator = SignalGenerator(
                    window_size=config.window_size,  # Already in minutes
                    momentum_lookback=config.momentum_lookback  # Already in minutes
                )
                trading_mode = "sharpe"
            
            risk_manager = RiskManager(max_position_pct=config.max_position_pct)
            
            engine = TradingEngine(
                client=trading_client,
                datastore=datastore,
                signal_generator=signal_generator,
                risk_manager=risk_manager,
                mode=trading_mode
            )
        
        # Run appropriate mode
        if args.mode == "collect":
            collector_loop(trading_client, datastore, logger)
        else:
            trading_loop(engine, logger, run_once=args.once)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()
