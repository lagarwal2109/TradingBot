#!/usr/bin/env python3
"""Live trading launcher for volatility expansion strategy."""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import get_config
from bot.roostoo_v3 import RoostooV3Client
from bot.datastore import DataStore
from bot.signals_volatility_expansion import VolatilityExpansionSignalGenerator
from bot.position_manager import PositionManager
from bot.risk import RiskManager
from bot.engine_volatility import VolatilityExpansionEngine


def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / "volatility_expansion_live.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger


def safe_startup(engine: VolatilityExpansionEngine, logger: logging.Logger):
    """Safe startup sequence."""
    logger.info("="*80)
    logger.info("VOLATILITY EXPANSION STRATEGY - LIVE TRADING")
    logger.info("="*80)
    
    try:
        # Preload minute bars from existing CSVs to eliminate warm-up
        logger.info("Preloading historical minute bars from data/ ...")
        normalized = engine.datastore.preload_from_data_folder(max_rows=6000)
        logger.info(f"Preload complete. Normalized files: {normalized}")
        
        # Update exchange info
        logger.info("Updating exchange information...")
        engine.update_exchange_info()
        logger.info(f"Exchange info updated: {len(engine.exchange_info_map)} pairs available")
        
        # Get initial balance
        balance = engine.client.balance()
        ticker_data = engine.client.ticker()
        prices = {t.pair.replace("/", ""): t.last_price for t in ticker_data if t.last_price and t.last_price > 0}
        total_equity = engine.get_total_equity(balance, prices)
        
        logger.info(f"Initial Balance: {balance}")
        logger.info(f"Initial Total Equity: ${total_equity:,.2f}")
        logger.info(f"Max Positions: {engine.config.max_positions}")
        logger.info(f"Target Utilization: {engine.config.target_utilization*100:.0f}%")
        logger.info(f"Risk per Trade: {engine.config.risk_per_trade_pct*100:.1f}%")
        logger.info(f"Max Position Size: {engine.config.max_position_pct*100:.1f}%")
        logger.info("="*80)
        logger.info("Starting live trading cycles...")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise


def trading_loop(engine: VolatilityExpansionEngine, logger: logging.Logger):
    """Run the main trading loop."""
    cycle_num = 0
    
    while True:
        try:
            cycle_num += 1
            cycle_start = time.time()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"CYCLE #{cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            # Run trading cycle
            results = engine.run_cycle()
            
            # Log results
            logger.info(f"Cycle Results:")
            logger.info(f"  Signals Generated: {results.get('signals_generated', 0)}")
            logger.info(f"  Positions Opened: {results.get('positions_opened', 0)}")
            logger.info(f"  Exits Executed: {results.get('exits_executed', 0)}")
            logger.info(f"  Cycle Time: {results.get('cycle_time', 0):.2f}s")
            
            if results.get("error"):
                logger.error(f"Cycle Error: {results['error']}")
            
            # Get current status
            balance = engine.client.balance()
            ticker_data = engine.client.ticker()
            prices = {t.pair.replace("/", ""): t.last_price for t in ticker_data if t.last_price and t.last_price > 0}
            total_equity = engine.get_total_equity(balance, prices)
            
            positions = engine.position_manager.get_positions()
            position_value = sum(
                pos.total_amount * prices.get(pair, 0)
                for pair, pos in positions.items()
                if pair in prices
            )
            utilization = (position_value / total_equity * 100) if total_equity > 0 else 0.0
            
            logger.info(f"\nPortfolio Status:")
            logger.info(f"  Total Equity: ${total_equity:,.2f}")
            logger.info(f"  Cash: ${balance.get('USD', 0):,.2f}")
            logger.info(f"  Positions: {len(positions)}/{engine.config.max_positions}")
            logger.info(f"  Utilization: {utilization:.1f}%")
            
            if positions:
                logger.info(f"\nActive Positions:")
                active_count = 0
                for pair, pos in positions.items():
                    if pair in prices:
                        current_price = prices[pair]
                        pos.update_pnl(current_price)
                        # Only show positions with actual holdings (total_amount > 0)
                        if pos.total_amount > 0:
                            active_count += 1
                            logger.info(
                                f"  {pair}: {pos.total_amount:.6f} @ ${pos.average_entry_price:.4f} avg | "
                                f"Current: ${current_price:.4f} | "
                                f"P&L: ${pos.unrealized_pnl:+.2f} ({pos.unrealized_pnl_pct:+.2f}%)"
                            )
                        else:
                            # Show pending positions (waiting for batch fills) with batch info
                            filled_batches = sum(1 for b in pos.batch_entries if b.filled)
                            logger.debug(
                                f"  {pair}: Pending ({filled_batches}/{len(pos.batch_entries)} batches filled) | "
                                f"Current: ${current_price:.4f}"
                            )
                if active_count == 0 and len(positions) > 0:
                    logger.info(f"  (All {len(positions)} positions pending batch fills)")
            
            logger.info(f"{'='*80}\n")
            
            # Sleep until next minute
            now = time.time()
            sleep_time = 60 - (now % 60)
            if sleep_time < 0:
                sleep_time = 0
            
            logger.info(f"Sleeping for {sleep_time:.1f}s until next cycle...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("\nTrading stopped by user")
            break
        except Exception as e:
            logger.error(f"Trading loop error: {e}", exc_info=True)
            time.sleep(10)  # Brief pause before retry


def main():
    """Main entry point."""
    logger = setup_logging()
    
    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize components
        logger.info("Initializing components...")
        client = RoostooV3Client()
        datastore = DataStore()
        signal_generator = VolatilityExpansionSignalGenerator(
            rsi_period=config.rsi_period,
            rsi_overbought=config.rsi_overbought,
            rsi_oversold=config.rsi_oversold,
            macd_fast=config.macd_fast,
            macd_slow=config.macd_slow,
            macd_signal=config.macd_signal,
            bb_period=config.bb_period,
            bb_std_dev=config.bb_std_dev,
            atr_period=config.atr_period,
            volume_ma_period=config.volume_ma_period,
            volume_spike_threshold=config.volume_spike_threshold,
            squeeze_threshold=config.squeeze_threshold,
            batch_count=config.batch_entry_count,
            batch_spacing_pct=config.batch_spacing_pct,
            stop_loss_atr_multiplier=config.atr_stop_multiplier
        )
        risk_manager = RiskManager(max_position_pct=config.max_position_pct)
        # Instantiate PositionManager with explicit parameters from config (not the config object itself)
        position_manager = PositionManager(
            max_positions=config.max_positions,
            max_position_pct=config.max_position_pct,
            min_correlation=config.min_correlation,
            use_scaled_exits=config.use_scaled_exits
        )
        
        # Create engine
        engine = VolatilityExpansionEngine(
            client=client,
            datastore=datastore,
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_manager=position_manager
        )
        
        # Safe startup
        safe_startup(engine, logger)
        
        # Run trading loop
        trading_loop(engine, logger)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()

