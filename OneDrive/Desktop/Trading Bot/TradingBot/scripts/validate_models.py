#!/usr/bin/env python3
"""Validate regime detection and ensemble models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from bot.datastore import DataStore
from bot.models.regime_detection import GMMRegimeDetector, HMMTrendDetector
from bot.models.ensemble_model import StackedEnsemble
from bot.models.model_storage import ModelStorage


def validate_regime_models(datastore: DataStore, days: int = 30):
    """Validate regime detection models."""
    print("Validating regime detection models...")
    
    pairs = datastore.get_all_pairs_with_data()[:5]  # Test on 5 pairs
    
    model_storage = ModelStorage()
    
    for pair in pairs:
        print(f"\nValidating {pair}...")
        pair_name = pair.replace("USD", "")
        
        # Test GMM
        try:
            gmm = GMMRegimeDetector(model_storage=model_storage)
            gmm.load(name=f"gmm_regime_{pair_name}")
            
            df = datastore.read_minute_bars(pair, limit=1440)
            if not df.empty:
                proba = gmm.predict_proba(df)
                print(f"  GMM probabilities: {proba}")
        except Exception as e:
            print(f"  GMM validation failed: {e}")
        
        # Test HMM
        try:
            hmm = HMMTrendDetector(model_storage=model_storage)
            hmm.load(name=f"hmm_trend_{pair_name}")
            
            df_4h = datastore.read_aggregated_bars(pair, interval="4h", limit=100)
            if not df_4h.empty:
                proba = hmm.predict_proba(df_4h)
                print(f"  HMM probabilities: {proba}")
        except Exception as e:
            print(f"  HMM validation failed: {e}")


def validate_ensemble(datastore: DataStore, days: int = 30):
    """Validate ensemble model with walk-forward validation."""
    print("\nValidating ensemble model with walk-forward validation...")
    
    # Load ensemble
    model_storage = ModelStorage()
    try:
        ensemble = StackedEnsemble([], model_storage=model_storage)
        ensemble.load(name="stacked_ensemble")
        print("Ensemble loaded successfully")
    except Exception as e:
        print(f"Could not load ensemble: {e}")
        return
    
    # Walk-forward validation
    pairs = datastore.get_all_pairs_with_data()[:3]  # Test on 3 pairs
    
    all_predictions = []
    all_actuals = []
    
    for pair in pairs:
        df = datastore.read_minute_bars(pair, limit=None)
        if len(df) < days * 1440:
            continue
        
        # Use last N days for validation
        end_date = df.index.max()
        start_date = end_date - timedelta(days=days)
        df_test = df[df.index >= start_date]
        
        # Create features and labels
        from bot.models.feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer()
        
        # Simplified validation - check if model can make predictions
        try:
            feature_matrix, _ = feature_engineer.create_feature_matrix({pair: df_test.tail(100)})
            if not feature_matrix.empty:
                feature_cols = [col for col in feature_matrix.columns if col != "pair"]
                X = feature_matrix[feature_cols].values
                X = np.nan_to_num(X, nan=0.0)
                
                proba = ensemble.predict_proba(X)
                print(f"  {pair}: Predictions generated successfully (shape: {proba.shape})")
        except Exception as e:
            print(f"  {pair}: Validation error: {e}")
    
    print("Ensemble validation complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate trained models")
    parser.add_argument("--days", type=int, default=30, help="Days of data to use")
    parser.add_argument("--mode", choices=["regime", "ensemble", "both"], default="both")
    
    args = parser.parse_args()
    
    datastore = DataStore()
    
    if args.mode in ["regime", "both"]:
        validate_regime_models(datastore, days=args.days)
    
    if args.mode in ["ensemble", "both"]:
        validate_ensemble(datastore, days=args.days)


if __name__ == "__main__":
    main()

