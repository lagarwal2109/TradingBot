#!/usr/bin/env python3
"""Run comprehensive overfitting diagnostics on trading strategy."""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.diagnostics import DiagnosticsSuite
from bot.diagnostics.risk_scorer import RiskScorer
from bot.diagnostics.reporter import DiagnosticsReporter
from bot.datastore import DataStore
from bot.signals_enhanced import EnhancedSignalGenerator
from bot.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> tuple:
    """Load historical data for diagnostics.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Tuple of (data_dict, prices_dict, timestamps)
    """
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
            
            # Remove duplicate timestamps (keep last)
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]
            
            if len(df) > 100:  # Need minimum data
                data_dict[pair] = df
                prices_dict[pair] = df["price"]
                all_timestamps.update(df.index)
                logger.info(f"Loaded {pair}: {len(df)} records")
        except Exception as e:
            logger.error(f"Error loading {csv_file.stem}: {e}")
            continue
    
    if len(data_dict) == 0:
        raise ValueError("No data files found")
    
    timestamps = pd.DatetimeIndex(sorted(all_timestamps))
    logger.info(f"Total timestamps: {len(timestamps)}")
    logger.info(f"Date range: {timestamps.min()} to {timestamps.max()}")
    
    return data_dict, prices_dict, timestamps


def create_signal_generator(config) -> EnhancedSignalGenerator:
    """Create signal generator from config.
    
    Args:
        config: Config object
        
    Returns:
        EnhancedSignalGenerator
    """
    return EnhancedSignalGenerator(
        trend_window_long=config.trend_window_long,
        trend_window_short=config.trend_window_short,
        entry_window=config.entry_window,
        volume_window=config.volume_window,
        support_resistance_days=config.support_resistance_days,
        breakout_threshold=config.breakout_threshold,
        volume_surge_multiplier=config.volume_surge_multiplier
    )


def create_signal_generator_wrapper(signal_gen, datastore, data_dict):
    """Create signal generator function for backtesting.
    
    Args:
        signal_gen: EnhancedSignalGenerator instance
        datastore: DataStore instance
        data_dict: Dictionary of loaded dataframes by pair
        
    Returns:
        Signal generator function
    """
    # Create a mock datastore that uses pre-loaded data
    class MockDataStore:
        def __init__(self, data_dict):
            self.data_dict = data_dict
        
        def read_minute_bars(self, pair, limit=None):
            if pair in self.data_dict:
                df = self.data_dict[pair].copy()
                if limit:
                    return df.tail(limit)
                return df
            return pd.DataFrame(columns=["timestamp", "price", "volume"])
    
    mock_datastore = MockDataStore(data_dict)
    
    def signal_generator(timestamp, current_prices):
        """Generate signals at a given timestamp."""
        signals = {}
        
        for pair, price in current_prices.items():
            ticker_data = {
                "pair": pair,
                "price": price,
                "volume_24h": 1000000.0,  # Placeholder
                "bid": price * 0.9999,
                "ask": price * 1.0001
            }
            
            try:
                # Use mock datastore that filters data up to current timestamp
                current_data = {}
                for p, df in data_dict.items():
                    current_data[p] = df[df.index <= timestamp]
                mock_datastore.data_dict = current_data
                
                signal = signal_gen.compute_trading_signal(pair, mock_datastore, ticker_data)
                signals[pair] = signal
            except Exception as e:
                logger.debug(f"Error generating signal for {pair}: {e}")
                continue
        
        return signals
    
    return signal_generator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run overfitting diagnostics")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with historical data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for output reports"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital for backtesting"
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=10.0,
        help="Commission in basis points"
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    data_dict, prices_dict, timestamps = load_data(args.data_dir)
    
    # Create signal generator - use default config values instead of loading from env
    # This avoids requiring API keys for diagnostics
    from bot.config import Config as BotConfig
    
    # Create a minimal config with defaults
    try:
        config = get_config()
    except ValueError:
        # If API keys are missing, create a minimal config with defaults
        logger.warning("API keys not found, using default config values for diagnostics")
        config = BotConfig(
            api_key="dummy",
            api_secret="dummy",
            base_url="https://mock-api.roostoo.com"
        )
    
    signal_gen = create_signal_generator(config)
    signal_generator_func = create_signal_generator_wrapper(signal_gen, None, data_dict)
    
    # Initialize diagnostics suite
    logger.info("Initializing diagnostics suite...")
    diagnostics = DiagnosticsSuite(
        initial_capital=args.initial_capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        label_horizon_bars=180,
        embargo_bars=180
    )
    
    # Prepare data DataFrame - use union of all timestamps to avoid duplicates
    # For diagnostics, we don't actually need the full DataFrame, just the timestamps
    # But we'll create a minimal one for integrity checks
    unique_timestamps = pd.DatetimeIndex(sorted(set(timestamps)))
    main_df = pd.DataFrame(index=unique_timestamps)
    # Only add a few key pairs to avoid reindexing issues
    for pair in list(data_dict.keys())[:5]:  # Just use first 5 pairs for main_df
        df = data_dict[pair]
        if len(df) > 0:
            # Reindex to main_df index, forward fill missing values
            try:
                main_df[f"{pair}_price"] = df["price"].reindex(main_df.index, method='ffill')
            except Exception as e:
                logger.debug(f"Could not add {pair} to main_df: {e}")
                continue
    
    # Run diagnostics
    logger.info("Running full diagnostics suite...")
    results = diagnostics.run_full_diagnostics(
        data=main_df,
        prices=prices_dict,
        signal_generator=signal_generator_func
    )
    
    # Calculate risk score
    logger.info("Calculating overfit risk score...")
    risk_scorer = RiskScorer()
    diagnostic_results = results.get("diagnostics", {})
    risk_score = risk_scorer.calculate_risk_score(diagnostic_results)
    
    # Generate reports
    logger.info("Generating reports...")
    reporter = DiagnosticsReporter(args.output_dir)
    html_path = reporter.generate_html_report(results, risk_score)
    json_path = reporter.save_json_report(results, risk_score)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OVERFITTING DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"Overfit Risk Score: {risk_score.total_score:.1f}/100")
    print(f"Risk Level: {risk_score.risk_level.upper()}")
    print("\nComponent Scores:")
    for component, score in risk_score.component_scores.items():
        print(f"  {component}: {score:.1f}")
    print("\nDiagnostic Test Results:")
    for test_name, result in diagnostic_results.items():
        if hasattr(result, 'passed'):
            status = "PASS" if result.passed else "FAIL"
            print(f"  {test_name}: {status} (score: {result.score:.3f})")
    print(f"\nReports saved to:")
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

