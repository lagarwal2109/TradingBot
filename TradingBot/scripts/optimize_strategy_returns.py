#!/usr/bin/env python3
"""Optimize strategy parameters for returns rather than pure Sharpe."""

import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.diagnostics import DiagnosticsSuite
from bot.diagnostics.tuner import NestedWFOTuner
from bot.datastore import DataStore
from bot.signals_enhanced import EnhancedSignalGenerator
from bot.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> tuple:
    """Load historical data."""
    data_dict = {}
    prices_dict = {}
    all_timestamps = set()
    
    for csv_file in data_dir.glob("*.csv"):
        if csv_file.stem == "state":
            continue
        
        pair = csv_file.stem.replace("_", "")
        try:
            df = pd.read_csv(csv_file)
            if "timestamp" not in df.columns or "price" not in df.columns:
                continue
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp").sort_index()
            
            if len(df) > 100:
                data_dict[pair] = df
                prices_dict[pair] = df["price"]
                all_timestamps.update(df.index)
        except Exception as e:
            logger.error(f"Error loading {csv_file.stem}: {e}")
            continue
    
    timestamps = pd.DatetimeIndex(sorted(all_timestamps))
    return data_dict, prices_dict, timestamps


def create_signal_generator_factory(datastore, base_config):
    """Create signal generator factory function."""
    def factory(
        min_entry_quality: float = 0.4,
        trend_window_long: int = None,
        trend_window_short: int = None,
        breakout_threshold: float = None,
        **kwargs
    ):
        """Create signal generator with given parameters."""
        return EnhancedSignalGenerator(
            trend_window_long=trend_window_long or base_config.trend_window_long,
            trend_window_short=trend_window_short or base_config.trend_window_short,
            entry_window=base_config.entry_window,
            volume_window=base_config.volume_window,
            support_resistance_days=base_config.support_resistance_days,
            breakout_threshold=breakout_threshold or base_config.breakout_threshold,
            volume_surge_multiplier=base_config.volume_surge_multiplier
        )
    
    return factory


def create_signal_generator_wrapper(signal_gen, datastore):
    """Create signal generator function for backtesting."""
    def signal_generator(timestamp, current_prices):
        signals = {}
        for pair, price in current_prices.items():
            ticker_data = {
                "pair": pair,
                "price": price,
                "volume_24h": 1000000.0,
                "bid": price * 0.9999,
                "ask": price * 1.0001
            }
            try:
                signal = signal_gen.compute_trading_signal(pair, datastore, ticker_data)
                signals[pair] = signal
            except Exception:
                continue
        return signals
    return signal_generator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimize strategy for returns")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with historical data"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("figures/optimized_params.json"),
        help="Output file for optimized parameters"
    )
    parser.add_argument(
        "--objective",
        choices=["return", "sharpe", "sortino", "calmar"],
        default="return",
        help="Optimization objective"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    data_dict, prices_dict, timestamps = load_data(args.data_dir)
    
    # Create datastore
    datastore = DataStore(args.data_dir)
    
    # Get base config
    base_config = get_config()
    
    # Initialize diagnostics
    diagnostics = DiagnosticsSuite(
        initial_capital=args.initial_capital,
        commission_bps=10.0,
        slippage_bps=5.0
    )
    
    # Create tuner
    tuner = NestedWFOTuner(
        backtester=diagnostics.backtester,
        splitter=diagnostics.splitter,
        objective=args.objective
    )
    
    # Define parameter grid
    param_grid = {
        "min_entry_quality": [0.3, 0.4, 0.5, 0.6],
        "trend_window_long": [48, 72, 144],
        "trend_window_short": [12, 24, 48],
        "breakout_threshold": [0.015, 0.02, 0.025]
    }
    
    # Create signal generator factory
    signal_gen_factory = create_signal_generator_factory(datastore, base_config)
    
    # Run tuning
    logger.info(f"Running nested WFO tuning (objective: {args.objective})...")
    result = tuner.tune(
        timestamps=timestamps,
        prices=prices_dict,
        param_grid=param_grid,
        signal_generator_factory=signal_gen_factory,
        n_outer_splits=3,
        n_inner_folds=5
    )
    
    # Save results
    output_data = {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "objective": args.objective,
        "validation_scores": result.validation_scores,
        "top_configs": [
            {"params": params, "score": score}
            for params, score in result.param_scores[:10]
        ]
    }
    
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("STRATEGY OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Objective: {args.objective}")
    print(f"Best Score: {result.best_score:.3f}")
    print("\nBest Parameters:")
    for param, value in result.best_params.items():
        print(f"  {param}: {value}")
    print("\nValidation Scores (OOS):")
    for split_name, score in result.validation_scores.items():
        print(f"  {split_name}: {score:.3f}")
    print(f"\nResults saved to: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

