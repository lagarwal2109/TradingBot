#!/usr/bin/env python3
"""Train stacked ensemble model on historical data."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from bot.datastore import DataStore
from bot.models.feature_engineering import FeatureEngineer
from bot.models.regime_detection import GMMRegimeDetector, HMMTrendDetector, RegimeFusion
from bot.models.ensemble_model import (
    XGBoostPredictor, LightGBMPredictor, RandomForestPredictor,
    LSTMPredictor, StackedEnsemble
)
from bot.models.model_storage import ModelStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_target_labels(df: pd.DataFrame, forward_window_minutes: int = 60) -> pd.Series:
    """Create target labels: 1 if price goes up, 0 if down.
    
    Args:
        df: DataFrame with price column
        forward_window_minutes: Look-ahead window in minutes
    
    Returns:
        Binary labels (1 = price up, 0 = price down)
    """
    prices = df["price"].values
    
    # Calculate future returns
    future_prices = np.roll(prices, -forward_window_minutes)
    future_returns = (future_prices / prices) - 1
    
    # Create binary labels: 1 if return > 0, 0 otherwise
    # Use float dtype to allow NaN values
    labels = (future_returns > 0).astype(float)
    
    # Remove last N labels (no future data) - set to NaN
    labels[-forward_window_minutes:] = np.nan
    
    return pd.Series(labels, index=df.index, dtype=float)


def prepare_training_data(
    datastore: DataStore,
    days: int = 30,
    forward_window_minutes: int = 180
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Prepare training data with features and labels.
    
    Args:
        datastore: DataStore instance
        days: Number of days of historical data
        forward_window_minutes: Forward window for target labels
    
    Returns:
        Tuple of (feature_matrix, labels, feature_names)
    """
    logger.info(f"Preparing training data from last {days} days...")
    
    pairs = datastore.get_all_pairs_with_data()
    logger.info(f"Found {len(pairs)} pairs with data")
    
    feature_engineer = FeatureEngineer()
    
    # Load regime detectors (if available)
    model_storage = ModelStorage()
    regime_detector = None
    
    try:
        # Try to load regime detectors - try multiple pairs until we find one with trained models
        if pairs:
            gmm = GMMRegimeDetector(model_storage=model_storage)
            hmm = HMMTrendDetector(model_storage=model_storage)
            
            # Try to find a pair with trained models
            loaded = False
            for pair in pairs[:10]:  # Try first 10 pairs
                try:
                    pair_name = pair.replace("USD", "")
                    gmm.load(name=f"gmm_regime_{pair_name}")
                    hmm.load(name=f"hmm_trend_{pair_name}")
                    
                    # Verify models are actually fitted
                    if gmm.is_fitted and hmm.is_fitted:
                        regime_fusion = RegimeFusion()
                        regime_detector = (gmm, hmm, regime_fusion)
                        logger.info(f"Loaded regime detectors from {pair} for feature engineering")
                        loaded = True
                        break
                    else:
                        logger.debug(f"Models for {pair} exist but are not fitted")
                except Exception as e:
                    logger.debug(f"Could not load models for {pair}: {e}")
                    continue
            
            if not loaded:
                logger.warning("Could not load regime detectors, proceeding without regime features. "
                             "Run 'python scripts/train_regime_models.py --days 30' to train them first.")
    except Exception as e:
        logger.warning(f"Error loading regime detectors: {e}", exc_info=True)
    
    all_features = []
    all_labels = []
    
    # First, determine the expected feature set by computing features on a sample pair
    # This ensures all pairs have the same feature columns
    expected_feature_names = None
    sample_pair = None
    
    for pair in pairs:
        try:
            df = datastore.read_minute_bars(pair, limit=None)
            if df.empty or len(df) < 200:
                continue
            
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                else:
                    continue
            
            # Compute features on a sample to get feature names
            df_sample = df.tail(200).copy()
            df_features_sample = feature_engineer.compute_technical_indicators(df_sample)
            df_features_sample = feature_engineer.compute_volume_features(df_features_sample)
            df_features_sample = feature_engineer.compute_volatility_features(df_features_sample)
            df_features_sample = feature_engineer.compute_multi_timeframe_returns(df_features_sample)
            
            # Add regime features (use default if regime detector not available)
            df_features_sample = feature_engineer.compute_regime_features(df_features_sample)
            
            # Get numeric columns (these are our features)
            numeric_cols = df_features_sample.select_dtypes(include=[np.number]).columns.tolist()
            expected_feature_names = numeric_cols
            sample_pair = pair
            logger.info(f"Determined feature set from {pair}: {len(expected_feature_names)} features")
            break
        except Exception as e:
            logger.debug(f"Error determining feature set from {pair}: {e}")
            continue
    
    if expected_feature_names is None:
        raise ValueError("Could not determine feature set from any pair")
    
    for pair in pairs:
        try:
            # Load historical data
            df = datastore.read_minute_bars(pair, limit=None)
            
            if df.empty:
                logger.debug(f"Skipping {pair}: empty data")
                continue
            
            # Check if index is datetime, if not try to convert
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                else:
                    logger.debug(f"Skipping {pair}: no datetime index")
                    continue
            
            # Need at least some data (relaxed requirement)
            min_samples = max(200, forward_window_minutes + 100)  # At least 200 samples or enough for forward window + history
            if len(df) < min_samples:
                logger.debug(f"Skipping {pair}: insufficient data ({len(df)} samples, need {min_samples})")
                continue
            
            # Filter to last N days
            end_date = df.index.max()
            start_date = end_date - timedelta(days=days)
            df_filtered = df[df.index >= start_date].copy()
            
            if len(df_filtered) < min_samples:
                logger.debug(f"Skipping {pair}: insufficient filtered data ({len(df_filtered)} samples)")
                continue
            
            # Create target labels
            labels = create_target_labels(df_filtered, forward_window_minutes)
            
            # Remove rows with NaN labels
            valid_mask = ~labels.isna()
            df_filtered = df_filtered[valid_mask]
            labels = labels[valid_mask]
            
            if len(df_filtered) < 100:  # Need minimum samples
                logger.debug(f"Skipping {pair}: insufficient valid samples after label filtering ({len(df_filtered)} samples)")
                continue
            
            # Get regime probabilities if available
            regime_probs = None
            if regime_detector:
                try:
                    gmm, hmm, fusion = regime_detector
                    df_4h = datastore.read_aggregated_bars(pair, interval="4h", limit=100)
                    
                    gmm_proba = gmm.predict_proba(df_filtered)
                    hmm_proba = hmm.predict_proba(df_4h) if not df_4h.empty else {"bearish": 0.33, "neutral": 0.33, "bullish": 0.34}
                    regime_probs = fusion.fuse_regimes(gmm_proba, hmm_proba)
                except:
                    pass
            
            # Compute features for each time step
            feature_list = []
            label_list = []
            
            for i in range(100, len(df_filtered)):  # Start from index 100 to have enough history
                df_window = df_filtered.iloc[:i+1]
                
                # Compute features
                df_features = feature_engineer.compute_technical_indicators(df_window)
                df_features = feature_engineer.compute_volume_features(df_features)
                df_features = feature_engineer.compute_volatility_features(df_features)
                df_features = feature_engineer.compute_multi_timeframe_returns(df_features)
                
                if regime_probs:
                    df_features = feature_engineer.compute_regime_features(df_features.iloc[-1:], regime_probs)
                
                # Get latest row features
                latest_features = df_features.iloc[-1:].copy()
                
                # Select numeric columns only
                numeric_cols = latest_features.select_dtypes(include=[np.number]).columns.tolist()
                
                # Build feature vector with consistent feature order
                feature_vector = []
                for feat_name in expected_feature_names:
                    if feat_name in numeric_cols:
                        feature_vector.append(float(latest_features[feat_name].iloc[0]))
                    else:
                        # Feature missing, fill with 0
                        feature_vector.append(0.0)
                
                feature_vector = np.array(feature_vector, dtype=float)
                
                feature_list.append(feature_vector)
                label_list.append(labels.iloc[i])
            
            if feature_list:
                all_features.extend(feature_list)
                all_labels.extend(label_list)
                logger.info(f"Added {len(feature_list)} samples from {pair}")
        
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}", exc_info=True)
            continue
    
    if not all_features:
        logger.error(f"No training data prepared from {len(pairs)} pairs. Check data availability and requirements.")
        raise ValueError("No training data prepared")
    
    # Convert to arrays - now all should have the same length
    try:
        feature_matrix = np.array(all_features, dtype=float)
        labels_array = np.array(all_labels, dtype=float)
    except ValueError as e:
        # If still having issues, check feature lengths
        feature_lengths = [len(f) for f in all_features]
        logger.error(f"Feature length mismatch. Lengths: min={min(feature_lengths)}, max={max(feature_lengths)}, unique={set(feature_lengths)}")
        raise ValueError(f"Cannot create feature matrix: {e}")
    
    # Use the expected feature names
    feature_names = expected_feature_names
    
    logger.info(f"Prepared {len(feature_matrix)} samples with {feature_matrix.shape[1]} features")
    logger.info(f"Label distribution: {np.bincount(labels_array.astype(int))}")
    
    return pd.DataFrame(feature_matrix, columns=feature_names), labels_array, feature_names


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_lstm: bool = False
) -> StackedEnsemble:
    """Train stacked ensemble model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        use_lstm: Whether to include LSTM (requires sequential data)
    
    Returns:
        Trained StackedEnsemble
    """
    logger.info("Initializing base models...")
    
    model_storage = ModelStorage()
    
    # Create base models
    base_models = [
        XGBoostPredictor(model_storage=model_storage),
        LightGBMPredictor(model_storage=model_storage),
        RandomForestPredictor(model_storage=model_storage)
    ]
    
    # Add LSTM if requested (requires special data preparation)
    if use_lstm:
        logger.warning("LSTM requires sequential data preparation, skipping for now")
        # TODO: Implement sequential data preparation for LSTM
    
    # Create meta-model
    meta_model = XGBoostPredictor(model_storage=model_storage)
    
    # Create ensemble
    ensemble = StackedEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        cv_folds=5,
        model_storage=model_storage
    )
    
    # Train ensemble
    logger.info("Training stacked ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_proba = ensemble.predict_proba(X_val)
    val_pred = (val_proba[:, 1] > 0.5).astype(int)
    
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred, zero_division=0)
    recall = recall_score(y_val, val_pred, zero_division=0)
    f1 = f1_score(y_val, val_pred, zero_division=0)
    
    logger.info(f"Validation Metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred))
    
    return ensemble


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train stacked ensemble model")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data to use"
    )
    parser.add_argument(
        "--forward-window",
        type=int,
        default=180,
        help="Forward window in minutes for target labels"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Test set split ratio"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation set split ratio"
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Include LSTM in ensemble (experimental)"
    )
    
    args = parser.parse_args()
    
    # Prepare data
    datastore = DataStore()
    feature_df, labels, feature_names = prepare_training_data(
        datastore,
        days=args.days,
        forward_window_minutes=args.forward_window
    )
    
    # Split data
    X = feature_df.values
    y = labels
    
    # Train/Val/Test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )
    
    val_size = args.val_split / (1 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Train ensemble
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, use_lstm=args.use_lstm)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_proba = ensemble.predict_proba(X_test)
    test_pred = (test_proba[:, 1] > 0.5).astype(int)
    
    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    
    logger.info(f"Test Metrics:")
    logger.info(f"  Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall: {test_recall:.4f}")
    logger.info(f"  F1 Score: {test_f1:.4f}")
    
    # Save model
    model_storage = ModelStorage()
    try:
        version = ensemble.save(name="stacked_ensemble")
        logger.info(f"Ensemble saved with version {version}")
        logger.info("Training complete!")
        print(f"\n{'='*60}")
        print(f"✓ Ensemble model saved successfully!")
        print(f"  Version: {version}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1 Score: {test_f1:.4f}")
        print(f"{'='*60}\n")
    except Exception as e:
        logger.error(f"Error saving ensemble: {e}", exc_info=True)
        print(f"\n✗ Error saving ensemble: {e}\n")
        raise


if __name__ == "__main__":
    main()

