#!/usr/bin/env python3
"""Hyperparameter tuning using Optuna."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    sys.exit(1)

import argparse
from bot.datastore import DataStore
from bot.models.ensemble_model import XGBoostPredictor, LightGBMPredictor, RandomForestPredictor, StackedEnsemble
from bot.models.model_storage import ModelStorage
from scripts.train_ensemble import prepare_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function."""
    
    # Suggest hyperparameters for base models
    xgb_params = {
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3),
        "n_estimators": trial.suggest_int("xgb_n_est", 50, 200),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
    }
    
    lgb_params = {
        "num_leaves": trial.suggest_int("lgb_num_leaves", 20, 50),
        "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.3),
        "feature_fraction": trial.suggest_float("lgb_feat_frac", 0.6, 1.0),
    }
    
    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_est", 50, 200),
        "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
    }
    
    # Create models
    model_storage = ModelStorage()
    base_models = [
        XGBoostPredictor(model_storage=model_storage, **xgb_params),
        LightGBMPredictor(model_storage=model_storage, **lgb_params),
        RandomForestPredictor(model_storage=model_storage, **rf_params)
    ]
    
    # Create ensemble
    ensemble = StackedEnsemble(base_models=base_models, cv_folds=5, model_storage=model_storage)
    
    # Train
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    val_proba = ensemble.predict_proba(X_val)
    val_pred = (val_proba[:, 1] > 0.5).astype(int)
    
    score = f1_score(y_val, val_pred, zero_division=0)
    return score


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters")
    parser.add_argument("--days", type=int, default=30, help="Days of training data")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    
    args = parser.parse_args()
    
    print(f"Starting hyperparameter tuning with {args.trials} trials...")
    
    # Prepare data
    datastore = DataStore()
    feature_df, labels, _ = prepare_training_data(datastore, days=args.days)
    
    X = feature_df.values
    y = labels
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Create study
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=args.trials
    )
    
    print("\nBest hyperparameters:")
    print(study.best_params)
    print(f"Best F1 score: {study.best_value:.4f}")
    
    # Save results
    import json
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value
    }
    
    results_file = Path("figures") / "hyperparameter_tuning.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

