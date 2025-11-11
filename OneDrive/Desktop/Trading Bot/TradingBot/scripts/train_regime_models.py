#!/usr/bin/env python3
"""Train GMM and HMM regime detection models on historical data."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime, timedelta
from bot.datastore import DataStore
from bot.models.regime_detection import GMMRegimeDetector, HMMTrendDetector
from bot.models.model_storage import ModelStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_gmm_models(datastore: DataStore, days: int = 30):
    """Train GMM models for all pairs.
    
    Args:
        datastore: DataStore instance
        days: Number of days of data to use
    """
    logger.info(f"Training GMM regime detectors on {days} days of data...")
    
    pairs = datastore.get_all_pairs_with_data()
    logger.info(f"Found {len(pairs)} pairs with data")
    
    model_storage = ModelStorage()
    gmm_detector = GMMRegimeDetector(n_components=2, model_storage=model_storage)
    
    trained_count = 0
    
    for pair in pairs:
        try:
            # Load historical data
            df = datastore.read_minute_bars(pair, limit=None)
            
            if len(df) < 1440:  # Need at least 24 hours of minute data
                logger.debug(f"Skipping {pair}: insufficient data ({len(df)} rows)")
                continue
            
            # Filter to last N days
            end_date = df.index.max()
            start_date = end_date - timedelta(days=days)
            df_filtered = df[df.index >= start_date]
            
            if len(df_filtered) < 1440:
                logger.debug(f"Skipping {pair}: insufficient filtered data")
                continue
            
            # Train GMM
            gmm_detector.fit(df_filtered, window_hours=24)
            
            if gmm_detector.is_fitted:
                # Save model
                version = gmm_detector.save(name=f"gmm_regime_{pair.replace('USD', '')}")
                logger.info(f"Trained and saved GMM for {pair} (version {version})")
                trained_count += 1
            else:
                logger.warning(f"GMM training failed for {pair}")
                
        except Exception as e:
            logger.error(f"Error training GMM for {pair}: {e}")
            continue
    
    logger.info(f"Trained GMM models for {trained_count} pairs")


def train_hmm_models(datastore: DataStore, days: int = 30):
    """Train HMM models for all pairs.
    
    Args:
        datastore: DataStore instance
        days: Number of days of data to use
    """
    logger.info(f"Training HMM trend detectors on {days} days of data...")
    
    pairs = datastore.get_all_pairs_with_data()
    logger.info(f"Found {len(pairs)} pairs with data")
    
    model_storage = ModelStorage()
    hmm_detector = HMMTrendDetector(n_states=3, model_storage=model_storage)
    
    trained_count = 0
    
    for pair in pairs:
        try:
            # Load aggregated 4-hour data
            df_4h = datastore.read_aggregated_bars(pair, interval="4h", limit=None)
            
            if len(df_4h) < 30:  # Need at least 30 4h bars (5 days)
                logger.debug(f"Skipping {pair}: insufficient 4h data ({len(df_4h)} bars)")
                continue
            
            # Filter to last N days
            if df_4h.empty or "timestamp" not in df_4h.columns:
                logger.debug(f"Skipping {pair}: invalid 4h data format")
                continue
            
            # Convert timestamp to datetime if needed
            if "timestamp" in df_4h.columns:
                df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"])
                df_4h = df_4h.set_index("timestamp")
            elif not isinstance(df_4h.index, pd.DatetimeIndex):
                # If no timestamp column and index is not datetime, try to convert index
                try:
                    df_4h.index = pd.to_datetime(df_4h.index)
                except:
                    logger.debug(f"Skipping {pair}: cannot convert index to datetime")
                    continue
            
            if df_4h.empty:
                logger.debug(f"Skipping {pair}: empty 4h data after processing")
                continue
            
            end_date = df_4h.index.max()
            start_date = end_date - timedelta(days=days)
            df_filtered = df_4h[df_4h.index >= start_date].copy()
            
            # Ensure we have price column
            if "price" not in df_filtered.columns:
                logger.debug(f"Skipping {pair}: no price column in 4h data")
                continue
            
            if len(df_filtered) < 30:
                logger.debug(f"Skipping {pair}: insufficient filtered 4h data")
                continue
            
            # Train HMM
            try:
                hmm_detector.fit(df_filtered, window_days=days)
            except Exception as e:
                logger.error(f"Error training HMM for {pair}: {e}", exc_info=True)
                continue
            
            if hmm_detector.is_fitted:
                # Save model
                version = hmm_detector.save(name=f"hmm_trend_{pair.replace('USD', '')}")
                logger.info(f"Trained and saved HMM for {pair} (version {version})")
                trained_count += 1
            else:
                logger.warning(f"HMM training failed for {pair}")
                
        except Exception as e:
            logger.error(f"Error training HMM for {pair}: {e}")
            continue
    
    logger.info(f"Trained HMM models for {trained_count} pairs")


def validate_models(datastore: DataStore):
    """Validate trained regime models.
    
    Args:
        datastore: DataStore instance
    """
    logger.info("Validating regime models...")
    
    model_storage = ModelStorage()
    pairs = datastore.get_all_pairs_with_data()[:5]  # Test on first 5 pairs
    
    for pair in pairs:
        try:
            # Test GMM
            gmm_detector = GMMRegimeDetector(model_storage=model_storage)
            try:
                gmm_detector.load(name=f"gmm_regime_{pair.replace('USD', '')}")
                df = datastore.read_minute_bars(pair, limit=1440)
                gmm_proba = gmm_detector.predict_proba(df)
                logger.info(f"{pair} GMM: {gmm_proba}")
            except Exception as e:
                logger.warning(f"{pair} GMM validation failed: {e}")
            
            # Test HMM
            hmm_detector = HMMTrendDetector(model_storage=model_storage)
            try:
                hmm_detector.load(name=f"hmm_trend_{pair.replace('USD', '')}")
                df_4h = datastore.read_aggregated_bars(pair, interval="4h", limit=100)
                if not df_4h.empty:
                    hmm_proba = hmm_detector.predict_proba(df_4h)
                    logger.info(f"{pair} HMM: {hmm_proba}")
            except Exception as e:
                logger.warning(f"{pair} HMM validation failed: {e}")
                
        except Exception as e:
            logger.error(f"Error validating {pair}: {e}")
            continue


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train regime detection models")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data to use for training"
    )
    parser.add_argument(
        "--mode",
        choices=["gmm", "hmm", "both", "validate"],
        default="both",
        help="Which models to train"
    )
    
    args = parser.parse_args()
    
    # Initialize datastore
    datastore = DataStore()
    
    if args.mode in ["gmm", "both"]:
        train_gmm_models(datastore, days=args.days)
    
    if args.mode in ["hmm", "both"]:
        train_hmm_models(datastore, days=args.days)
    
    if args.mode == "validate":
        validate_models(datastore)


if __name__ == "__main__":
    import pandas as pd
    main()

