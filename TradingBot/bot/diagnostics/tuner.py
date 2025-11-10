"""Nested walk-forward hyperparameter tuning."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging

from .time_series_splitter import TimeSeriesSplitter, SplitWindow
from .backtester import CostAwareBacktester, BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""
    best_params: Dict[str, Any]
    best_score: float
    param_scores: List[Tuple[Dict[str, Any], float]]
    validation_scores: Dict[str, float]


class NestedWFOTuner:
    """Nested walk-forward optimization for hyperparameter tuning."""
    
    def __init__(
        self,
        backtester: CostAwareBacktester,
        splitter: TimeSeriesSplitter,
        objective: str = "sharpe"  # "sharpe", "sortino", "calmar", "return"
    ):
        """Initialize nested WFO tuner.
        
        Args:
            backtester: Cost-aware backtester instance
            splitter: Time series splitter
            objective: Optimization objective metric
        """
        self.backtester = backtester
        self.splitter = splitter
        self.objective = objective
    
    def _extract_metric(self, metrics: BacktestMetrics) -> float:
        """Extract objective metric from backtest metrics.
        
        Args:
            metrics: BacktestMetrics object
            
        Returns:
            Metric value
        """
        if self.objective == "sharpe":
            return metrics.sharpe_ratio
        elif self.objective == "sortino":
            return metrics.sortino_ratio
        elif self.objective == "calmar":
            return metrics.calmar_ratio
        elif self.objective == "return":
            return metrics.annualized_return
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
    
    def tune(
        self,
        timestamps: pd.DatetimeIndex,
        prices: Dict[str, pd.Series],
        param_grid: Dict[str, List[Any]],
        signal_generator_factory: Callable,
        n_outer_splits: int = 3,
        n_inner_folds: int = 5,
        min_train_bars: int = 1000
    ) -> TuningResult:
        """Run nested WFO tuning.
        
        Args:
            timestamps: Timestamps for backtest
            prices: Price series
            param_grid: Parameter grid to search
            signal_generator_factory: Function that creates signal generator from params
            n_outer_splits: Number of outer WFO splits
            n_inner_folds: Number of inner PKFE folds
            min_train_bars: Minimum training bars required
            
        Returns:
            TuningResult with best parameters
        """
        # Generate outer WFO splits
        outer_splits = self.splitter.walk_forward_split(
            timestamps,
            train_years=0.5,
            valid_years=0.1,
            test_years=0.1,
            step_years=0.2
        )[:n_outer_splits]
        
        if len(outer_splits) == 0:
            raise ValueError("No valid outer splits generated")
        
        logger.info(f"Using {len(outer_splits)} outer splits for tuning")
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_configs = list(product(*param_values))
        
        logger.info(f"Testing {len(all_configs)} parameter configurations")
        
        # Store scores for each config across outer splits
        config_scores = {config: [] for config in all_configs}
        
        # Outer loop: WFO splits
        for outer_idx, outer_split in enumerate(outer_splits):
            logger.info(f"Outer split {outer_idx + 1}/{len(outer_splits)}")
            
            # Inner loop: PKFE on training data
            train_timestamps = timestamps[outer_split.train_indices]
            
            # Generate inner PKFE splits from training data
            inner_splits = self.splitter.pkfe_split(train_timestamps, k=n_inner_folds)
            
            if len(inner_splits) == 0:
                logger.warning(f"No inner splits for outer split {outer_idx + 1}, skipping")
                continue
            
            # Test each parameter configuration
            for config in all_configs:
                params = dict(zip(param_names, config))
                
                # Inner validation scores
                inner_scores = []
                
                for inner_split in inner_splits:
                    # Train on inner train, validate on inner valid
                    train_indices = inner_split.train_indices
                    valid_indices = inner_split.valid_indices
                    
                    if len(train_indices) < min_train_bars:
                        continue
                    
                    # Create signal generator with these params
                    signal_gen = signal_generator_factory(**params)
                    
                    # Backtest on validation set
                    try:
                        valid_timestamps = train_timestamps[valid_indices]
                        valid_prices = {
                            pair: series.loc[valid_timestamps]
                            for pair, series in prices.items()
                        }
                        
                        equity = self.backtester.run_backtest(
                            valid_timestamps, valid_prices, signal_gen
                        )
                        metrics = self.backtester.calculate_metrics(equity)
                        score = self._extract_metric(metrics)
                        inner_scores.append(score)
                    except Exception as e:
                        logger.warning(f"Error in inner validation: {e}")
                        continue
                
                if len(inner_scores) > 0:
                    # Use median validation score (robust to outliers)
                    median_score = np.median(inner_scores)
                    config_scores[config].append(median_score)
        
        # Find best configuration (highest median score across outer splits)
        best_config = None
        best_score = float('-inf')
        param_scores = []
        
        for config, scores in config_scores.items():
            if len(scores) > 0:
                median_score = np.median(scores)
                params = dict(zip(param_names, config))
                param_scores.append((params, median_score))
                
                if median_score > best_score:
                    best_score = median_score
                    best_config = config
        
        if best_config is None:
            raise ValueError("No valid configurations found")
        
        best_params = dict(zip(param_names, best_config))
        
        # Calculate validation scores on outer test sets
        validation_scores = {}
        for outer_idx, outer_split in enumerate(outer_splits):
            test_timestamps = timestamps[outer_split.test_indices]
            test_prices = {
                pair: series.loc[test_timestamps]
                for pair, series in prices.items()
            }
            
            signal_gen = signal_generator_factory(**best_params)
            equity = self.backtester.run_backtest(test_timestamps, test_prices, signal_gen)
            metrics = self.backtester.calculate_metrics(equity)
            validation_scores[f"outer_split_{outer_idx + 1}"] = self._extract_metric(metrics)
        
        # Sort param_scores by score
        param_scores.sort(key=lambda x: x[1], reverse=True)
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            param_scores=param_scores,
            validation_scores=validation_scores
        )

