"""Main diagnostics orchestrator for overfitting detection."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging

from .data_integrity import DataIntegrityChecker
from .time_series_splitter import TimeSeriesSplitter
from .backtester import CostAwareBacktester
from .overfitting_tests import OverfittingDiagnostics, DiagnosticResult

logger = logging.getLogger(__name__)


class DiagnosticsSuite:
    """Orchestrates all overfitting diagnostic tests."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        label_horizon_bars: int = 180,
        embargo_bars: int = 180
    ):
        """Initialize diagnostics suite.
        
        Args:
            initial_capital: Starting capital for backtests
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
            label_horizon_bars: Label horizon in bars
            embargo_bars: Embargo period in bars
        """
        self.backtester = CostAwareBacktester(
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps
        )
        self.diagnostics = OverfittingDiagnostics(
            backtester=self.backtester,
            label_horizon_bars=label_horizon_bars
        )
        self.integrity_checker = DataIntegrityChecker(label_horizon_bars=label_horizon_bars)
        self.splitter = TimeSeriesSplitter(embargo_bars=embargo_bars)
    
    def run_full_diagnostics(
        self,
        data: pd.DataFrame,
        prices: Dict[str, pd.Series],
        signal_generator: Callable,
        features: Optional[pd.DataFrame] = None,
        param_grid: Optional[Dict] = None,
        signal_generator_factory: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run complete diagnostic suite.
        
        Args:
            data: Main DataFrame with timestamps
            prices: Dictionary of price series by pair
            signal_generator: Signal generation function
            features: Optional feature DataFrame
            param_grid: Optional parameter grid for stability test
            signal_generator_factory: Optional factory function for parameter stability
            
        Returns:
            Dictionary with all diagnostic results
        """
        results = {}
        
        # 1. Data integrity checks
        logger.info("Running data integrity checks...")
        integrity_results = self.integrity_checker.run_all_checks(
            data, 
            prices=list(prices.values())[0] if prices else None,
            features=features
        )
        results["integrity"] = integrity_results
        
        # 2. Time series splits
        logger.info("Generating time series splits...")
        timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data["timestamp"])
        wfo_splits = self.splitter.walk_forward_split(
            timestamps,
            train_years=0.5,
            valid_years=0.1,
            test_years=0.1
        )
        results["splits"] = {
            "wfo": len(wfo_splits),
            "split_info": self.splitter.get_split_info(wfo_splits).to_dict() if wfo_splits else {}
        }
        
        # 3. Run diagnostic tests
        diagnostic_results = {}
        
        # Random label test
        logger.info("Running random label test...")
        try:
            random_label_result = self.diagnostics.random_label_test(
                timestamps, prices, signal_generator, n_trials=10
            )
            diagnostic_results["random_label"] = random_label_result
        except Exception as e:
            logger.error(f"Random label test failed: {e}")
            diagnostic_results["random_label"] = DiagnosticResult(
                "random_label", False, 0, f"Test failed: {e}"
            )
        
        # Future shift test
        logger.info("Running future shift test...")
        try:
            future_shift_result = self.diagnostics.future_shift_test(
                timestamps, prices, signal_generator
            )
            diagnostic_results["future_shift"] = future_shift_result
        except Exception as e:
            logger.error(f"Future shift test failed: {e}")
            diagnostic_results["future_shift"] = DiagnosticResult(
                "future_shift", False, 0, f"Test failed: {e}"
            )
        
        # Cost sensitivity
        logger.info("Running cost sensitivity test...")
        try:
            cost_sensitivity_result = self.diagnostics.cost_sensitivity_test(
                timestamps, prices, signal_generator
            )
            diagnostic_results["cost_sensitivity"] = cost_sensitivity_result
        except Exception as e:
            logger.error(f"Cost sensitivity test failed: {e}")
            diagnostic_results["cost_sensitivity"] = DiagnosticResult(
                "cost_sensitivity", False, 0, f"Test failed: {e}"
            )
        
        # Block bootstrap
        logger.info("Running block bootstrap test...")
        try:
            equity = self.backtester.run_backtest(timestamps, prices, signal_generator)
            returns = equity["equity"].pct_change().dropna()
            bootstrap_result = self.diagnostics.block_bootstrap(returns)
            diagnostic_results["block_bootstrap"] = bootstrap_result
        except Exception as e:
            logger.error(f"Block bootstrap test failed: {e}")
            diagnostic_results["block_bootstrap"] = DiagnosticResult(
                "block_bootstrap", False, 0, f"Test failed: {e}"
            )
        
        # PBO and DSR (if we have multiple splits)
        if len(wfo_splits) > 1:
            logger.info("Calculating PBO and DSR...")
            try:
                is_scores = []
                oos_scores = []
                
                for split in wfo_splits[:5]:  # Use first 5 splits
                    # Train on train, test on valid (IS)
                    train_timestamps = timestamps[split.train_indices]
                    train_prices = {
                        pair: series.loc[train_timestamps]
                        for pair, series in prices.items()
                    }
                    train_equity = self.backtester.run_backtest(
                        train_timestamps, train_prices, signal_generator
                    )
                    train_metrics = self.backtester.calculate_metrics(train_equity)
                    is_scores.append(train_metrics.sharpe_ratio)
                    
                    # Test on test (OOS)
                    test_timestamps = timestamps[split.test_indices]
                    test_prices = {
                        pair: series.loc[test_timestamps]
                        for pair, series in prices.items()
                    }
                    test_equity = self.backtester.run_backtest(
                        test_timestamps, test_prices, signal_generator
                    )
                    test_metrics = self.backtester.calculate_metrics(test_equity)
                    oos_scores.append(test_metrics.sharpe_ratio)
                
                pbo_result = self.diagnostics.calculate_pbo(is_scores, oos_scores)
                diagnostic_results["pbo"] = pbo_result
                
                # DSR on OOS returns
                all_oos_returns = []
                for split in wfo_splits[:5]:
                    test_timestamps = timestamps[split.test_indices]
                    test_prices = {
                        pair: series.loc[test_timestamps]
                        for pair, series in prices.items()
                    }
                    test_equity = self.backtester.run_backtest(
                        test_timestamps, test_prices, signal_generator
                    )
                    returns = test_equity["equity"].pct_change().dropna()
                    all_oos_returns.extend(returns.tolist())
                
                if all_oos_returns:
                    dsr_result = self.diagnostics.calculate_dsr(
                        pd.Series(all_oos_returns), n_trials=len(wfo_splits)
                    )
                    diagnostic_results["dsr"] = dsr_result
            except Exception as e:
                logger.error(f"PBO/DSR calculation failed: {e}")
        
        # Parameter stability (if provided)
        if param_grid and signal_generator_factory:
            logger.info("Running parameter stability test...")
            try:
                stability_result = self.diagnostics.parameter_stability_test(
                    param_grid, timestamps, prices, signal_generator_factory
                )
                diagnostic_results["parameter_stability"] = stability_result
            except Exception as e:
                logger.error(f"Parameter stability test failed: {e}")
        
        # Regime robustness
        logger.info("Running regime robustness test...")
        try:
            regime_tags = self.integrity_checker.tag_regime_breaks(timestamps)
            regime_result = self.diagnostics.regime_robustness_test(
                timestamps, prices, signal_generator, regime_tags
            )
            diagnostic_results["regime_robustness"] = regime_result
        except Exception as e:
            logger.error(f"Regime robustness test failed: {e}")
            diagnostic_results["regime_robustness"] = DiagnosticResult(
                "regime_robustness", False, 0, f"Test failed: {e}"
            )
        
        # Reality check
        logger.info("Running reality check...")
        try:
            reality_result = self.diagnostics.reality_check(
                timestamps, prices, signal_generator
            )
            diagnostic_results["reality_check"] = reality_result
        except Exception as e:
            logger.error(f"Reality check failed: {e}")
            diagnostic_results["reality_check"] = DiagnosticResult(
                "reality_check", False, 0, f"Test failed: {e}"
            )
        
        results["diagnostics"] = diagnostic_results
        
        return results

