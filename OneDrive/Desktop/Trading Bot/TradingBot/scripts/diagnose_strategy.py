#!/usr/bin/env python3
"""Diagnose why the strategy isn't generating enough trades."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bot.datastore import DataStore
from bot.models.model_storage import ModelStorage
from bot.models.feature_engineering import FeatureEngineer
from bot.models.regime_detection import GMMRegimeDetector, HMMTrendDetector, RegimeFusion
from bot.models.ensemble_model import StackedEnsemble

print("="*70)
print("STRATEGY DIAGNOSTIC")
print("="*70)

# Load models
datastore = DataStore()
model_storage = ModelStorage()
feature_engineer = FeatureEngineer()

# Check if ensemble exists
try:
    ensemble = StackedEnsemble([], model_storage=model_storage)
    ensemble.load('stacked_ensemble')
    print(f"\n[OK] Ensemble model loaded")
    print(f"  Base models: {len(ensemble.base_models)}")
    print(f"  Is fitted: {ensemble.is_fitted}")
except Exception as e:
    print(f"\n[ERROR] Failed to load ensemble: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get some data
pairs = datastore.get_all_pairs_with_data()[:5]  # Test first 5 pairs
print(f"\nTesting with {len(pairs)} pairs: {pairs}")

# Load regime detectors
gmm = GMMRegimeDetector(model_storage=model_storage)
hmm = HMMTrendDetector(model_storage=model_storage)
regime_fusion = RegimeFusion()

loaded_regime = False
for pair in pairs:
    try:
        pair_name = pair.replace("USD", "")
        gmm.load(name=f"gmm_regime_{pair_name}")
        hmm.load(name=f"hmm_trend_{pair_name}")
        if gmm.is_fitted and hmm.is_fitted:
            loaded_regime = True
            print(f"\n[OK] Loaded regime detectors from {pair}")
            break
    except:
        continue

if not loaded_regime:
    print("\n[WARNING] No regime detectors loaded (will use defaults)")

# Test signal generation
print("\n" + "="*70)
print("TESTING SIGNAL GENERATION")
print("="*70)

min_confidence = 0.35
long_threshold = 0.45
short_threshold = 0.45
momentum_threshold = 0.001

total_signals = 0
long_signals = 0
short_signals = 0
rejected_confidence = 0
rejected_prob = 0
rejected_momentum = 0

for pair in pairs:
    try:
        df = datastore.read_minute_bars(pair, limit=500)
        if df.empty or len(df) < 100:
            continue
        
        # Get latest timestamp
        latest_timestamp = df.index[-1]
        df_hist = df.tail(500)
        
        # Calculate momentum
        if len(df_hist) >= 20:
            prices = df_hist["price"].values
            momentum_20 = (prices[-1] / prices[-20] - 1) if prices[-20] > 0 else 0.0
        else:
            momentum_20 = 0.0
        
        # Detect regime
        try:
            df_4h = datastore.read_aggregated_bars(pair, interval="4h", limit=100)
            gmm_proba = gmm.predict_proba(df_hist) if gmm.is_fitted else {"calm": 0.5, "volatile": 0.5}
            hmm_proba = hmm.predict_proba(df_4h) if (hmm.is_fitted and not df_4h.empty) else {"bearish": 0.33, "neutral": 0.33, "bullish": 0.34}
            regime_proba, regime = regime_fusion.fuse_regimes(gmm_proba, hmm_proba), "calm_bullish"
        except:
            regime = "calm_bullish"
        
        # Create features
        feature_matrix, _ = feature_engineer.create_feature_matrix({pair: df_hist}, regime_detector=None)
        if feature_matrix.empty:
            continue
        
        # Get prediction
        feature_cols = [col for col in feature_matrix.columns if col != "pair"]
        X = feature_matrix[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        proba = ensemble.predict_proba(X)
        confidence = ensemble.get_confidence_score(proba)[0]
        prob_up = proba[0, 1] if proba.shape[1] == 2 else proba[0, 0]
        
        # Check conditions
        print(f"\n{pair}:")
        print(f"  Confidence: {confidence:.3f}, Prob_up: {prob_up:.3f}, Momentum: {momentum_20:.4f}")
        
        if confidence < min_confidence:
            rejected_confidence += 1
            print(f"  [REJECT] Confidence {confidence:.3f} < {min_confidence}")
            continue
        
        # Check long
        long_ok = False
        if prob_up > 0.55 or (confidence >= long_threshold and prob_up > 0.52):
            if momentum_20 > momentum_threshold:
                long_ok = True
                long_signals += 1
                total_signals += 1
                print(f"  [LONG] Signal generated")
            else:
                rejected_momentum += 1
                print(f"  [REJECT] Long: momentum {momentum_20:.4f} <= {momentum_threshold}")
        else:
            rejected_prob += 1
            print(f"  [REJECT] Long: prob_up {prob_up:.3f} not high enough")
        
        # Check short
        short_ok = False
        if prob_up < 0.45 or (confidence >= short_threshold and prob_up < 0.48):
            if momentum_20 < -momentum_threshold:
                short_ok = True
                short_signals += 1
                if not long_ok:
                    total_signals += 1
                print(f"  [SHORT] Signal generated")
            else:
                if not long_ok:
                    rejected_momentum += 1
                print(f"  [REJECT] Short: momentum {momentum_20:.4f} >= {-momentum_threshold}")
        else:
            if not long_ok:
                rejected_prob += 1
            print(f"  [REJECT] Short: prob_up {prob_up:.3f} not low enough")
        
    except Exception as e:
        print(f"\n{pair}: Error - {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total signals generated: {total_signals}")
print(f"  Long signals: {long_signals}")
print(f"  Short signals: {short_signals}")
print(f"\nRejections:")
print(f"  Low confidence: {rejected_confidence}")
print(f"  Wrong probability: {rejected_prob}")
print(f"  Wrong momentum: {rejected_momentum}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if total_signals == 0:
    print("[CRITICAL] NO SIGNALS GENERATED!")
    print("\nPossible fixes:")
    print("1. Lower min_confidence from 0.35 to 0.25")
    print("2. Lower momentum_threshold from 0.001 to 0.0005")
    print("3. Relax probability thresholds:")
    print("   - Long: prob_up > 0.52 OR (confidence > 0.40 AND prob_up > 0.50)")
    print("   - Short: prob_up < 0.48 OR (confidence > 0.40 AND prob_up < 0.50)")
    print("4. Check if model predictions are reasonable")
elif total_signals < len(pairs) * 0.2:
    print("[WARNING] TOO FEW SIGNALS!")
    print(f"Only {total_signals} signals from {len(pairs)} pairs")
    print("Consider lowering thresholds")
else:
    print("[OK] Signal generation looks reasonable")

