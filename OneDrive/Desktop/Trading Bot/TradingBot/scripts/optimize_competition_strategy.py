#!/usr/bin/env python3
"""Optimize strategy parameters to maximize competition score for 15-day competition."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    sys.exit(1)

import argparse
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from bot.datastore import DataStore
from bot.config import get_config, Config
from bot.risk import RiskManager
from bot.models.model_storage import ModelStorage
from bot.models.feature_engineering import FeatureEngineer
from bot.models.regime_detection import GMMRegimeDetector, HMMTrendDetector, RegimeFusion
from bot.models.ensemble_model import StackedEnsemble
from backtest_regime_ensemble import RegimeEnsembleBacktest

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


class CompetitionOptimizer:
    """Optimize strategy parameters for competition score."""
    
    def __init__(self, datastore: DataStore, days: int = 15):
        self.datastore = datastore
        self.days = days
        self.best_score = float('-inf')
        self.best_params = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function - maximize competition score."""
        
        # Strategy parameters to optimize (adjusted for more aggressive trading)
        params = {
            # Confidence thresholds (lower = more trades) - make more aggressive
            "min_confidence": trial.suggest_float("min_confidence", 0.40, 0.65, step=0.05),
            "high_confidence_threshold": trial.suggest_float("high_confidence_threshold", 0.60, 0.85, step=0.05),
            
            # Position sizing multipliers (more aggressive)
            "base_position_pct": trial.suggest_float("base_position_pct", 0.20, 0.40, step=0.05),
            "calm_multiplier": trial.suggest_float("calm_multiplier", 1.1, 1.6, step=0.1),
            "volatile_multiplier": trial.suggest_float("volatile_multiplier", 0.6, 1.0, step=0.1),
            "bullish_multiplier": trial.suggest_float("bullish_multiplier", 1.2, 1.8, step=0.1),
            "bearish_multiplier": trial.suggest_float("bearish_multiplier", 0.3, 0.7, step=0.1),
            
            # Stop loss / Take profit (tighter stops, wider targets)
            "base_stop_loss_pct": trial.suggest_float("base_stop_loss_pct", 0.015, 0.04, step=0.005),
            "base_take_profit_pct": trial.suggest_float("base_take_profit_pct", 0.03, 0.08, step=0.01),
            "stop_loss_tightening": trial.suggest_float("stop_loss_tightening", 0.6, 1.2, step=0.1),
            "take_profit_expansion": trial.suggest_float("take_profit_expansion", 1.1, 2.0, step=0.1),
            
            # Risk management (more positions allowed)
            "max_simultaneous_positions": trial.suggest_int("max_simultaneous_positions", 3, 6),
            "max_portfolio_allocation": trial.suggest_float("max_portfolio_allocation", 0.75, 0.90, step=0.05),
            
            # Trading frequency (more frequent trading)
            "trade_interval_minutes": trial.suggest_int("trade_interval_minutes", 30, 90, step=15),
        }
        
        try:
            # Run backtest with these parameters
            score = self.run_backtest_with_params(params)
            
            # Track best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print(f"\n[New Best] Score: {score:.4f}")
                print(f"  Confidence: {params['min_confidence']:.2f}, "
                      f"Base Position: {params['base_position_pct']:.2f}, "
                      f"Max Positions: {params['max_simultaneous_positions']}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial: {e}", exc_info=True)
            return float('-inf')
    
    def run_backtest_with_params(self, params: Dict[str, Any]) -> float:
        """Run backtest with given parameters and return competition score."""
        
        # Create modified config
        config = get_config()
        
        # Temporarily modify config values
        original_values = {}
        for key, value in params.items():
            if hasattr(config, key):
                original_values[key] = getattr(config, key)
                setattr(config, key, value)
        
        try:
            # Create backtest instance with parameter overrides
            backtest = RegimeEnsembleBacktest(
                config=config,
                data_dir=Path("data"),
                initial_capital=50000.0,
                param_overrides=params
            )
            
            # Run backtest
            backtest.run_backtest(self.datastore, days=self.days)
            
            # Extract competition score
            metrics = backtest.calculate_metrics()
            score = metrics.get("competition_score", float('-inf'))
            
            if np.isnan(score) or np.isinf(score):
                score = float('-inf')
            
            # Also consider total return and other metrics
            total_return = metrics.get("total_return_pct", 0)
            total_trades = metrics.get("total_trades", 0)
            max_drawdown = metrics.get("max_drawdown_pct", 100)
            
            # Heavy penalty for too few trades (need activity for 15-day competition)
            if total_trades < 10:
                penalty = max(0.1, total_trades / 10.0)  # Scale penalty by number of trades
                score = score * penalty
            
            # Heavy penalty for excessive drawdown
            if max_drawdown > 20:
                drawdown_penalty = max(0.1, 1.0 - (max_drawdown - 20) / 50.0)  # Penalize drawdowns > 20%
                score = score * drawdown_penalty
            
            # Bonus for high returns (primary goal)
            if total_return > 15:
                score = score * 1.5  # Big bonus for >15% returns
            elif total_return > 10:
                score = score * 1.3
            elif total_return > 5:
                score = score * 1.2
            elif total_return > 2:
                score = score * 1.1
            
            # Penalize negative returns
            if total_return < 0:
                score = score * 0.3  # Heavy penalty for losses
            
            return score
            
        finally:
            # Restore original config values
            for key, value in original_values.items():
                setattr(config, key, value)
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run optimization."""
        print(f"\n{'='*70}")
        print(f"Optimizing Strategy for 15-Day Competition")
        print(f"{'='*70}\n")
        print(f"Running {n_trials} trials...")
        print(f"Target: Maximize Competition Score (0.4*Sortino + 0.3*Sharpe + 0.3*Calmar)\n")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name="competition_strategy_optimization"
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}\n")
        print(f"Best Competition Score: {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        return {
            "best_score": study.best_value,
            "best_params": study.best_params,
            "n_trials": n_trials,
            "optimization_date": datetime.now().isoformat()
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimize strategy for competition")
    parser.add_argument(
        "--days",
        type=int,
        default=15,
        help="Competition duration in days (default: 15)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50, recommend 100+ for best results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/optimized_strategy_params.json",
        help="Output file for best parameters"
    )
    
    args = parser.parse_args()
    
    # Initialize
    datastore = DataStore()
    optimizer = CompetitionOptimizer(datastore, days=args.days)
    
    # Run optimization
    results = optimizer.optimize(n_trials=args.trials)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\nTo use these parameters, update your config.py or pass them to the trading engine.")
    
    # Run final backtest with best params to show detailed results
    print("\n" + "="*70)
    print("Running final backtest with optimized parameters...")
    print("="*70 + "\n")
    
    final_score = optimizer.run_backtest_with_params(results["best_params"])
    print(f"\nFinal validation score: {final_score:.4f}")


if __name__ == "__main__":
    main()

