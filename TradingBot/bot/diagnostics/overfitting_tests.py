"""Overfitting diagnostic tests for trading strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from .backtester import CostAwareBacktester, BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test."""
    test_name: str
    passed: bool
    score: float
    message: str
    details: Optional[Dict[str, Any]] = None


class OverfittingDiagnostics:
    """Battery of overfitting diagnostic tests."""
    
    def __init__(
        self,
        backtester: CostAwareBacktester,
        label_horizon_bars: int = 180
    ):
        """Initialize diagnostics.
        
        Args:
            backtester: Cost-aware backtester instance
            label_horizon_bars: Label horizon in bars
        """
        self.backtester = backtester
        self.label_horizon_bars = label_horizon_bars
    
    def random_label_test(
        self,
        timestamps: pd.DatetimeIndex,
        prices: Dict[str, pd.Series],
        signal_generator: Callable,
        n_trials: int = 10,
        block_size: int = 100
    ) -> DiagnosticResult:
        """Random-label (permutation) test with time-aware blocking.
        
        Args:
            timestamps: Timestamps for backtest
            prices: Price series
            signal_generator: Signal generation function
            n_trials: Number of permutation trials
            block_size: Size of time blocks for permutation
            
        Returns:
            DiagnosticResult
        """
        # Run normal backtest
        normal_equity = self.backtester.run_backtest(timestamps, prices, signal_generator)
        normal_metrics = self.backtester.calculate_metrics(normal_equity)
        normal_sharpe = normal_metrics.sharpe_ratio
        
        # Run permuted label tests
        permuted_sharpes = []
        
        for trial in range(n_trials):
            # Create permuted signal generator
            def permuted_signal_gen(timestamp, current_prices):
                signals = signal_generator(timestamp, current_prices)
                # Permute signal types within time blocks
                block_idx = len([t for t in timestamps if t <= timestamp]) // block_size
                np.random.seed(trial * 1000 + block_idx)
                
                permuted_signals = {}
                signal_list = list(signals.values())
                np.random.shuffle(signal_list)
                
                for i, (pair, _) in enumerate(signals.items()):
                    if i < len(signal_list):
                        permuted_signals[pair] = signal_list[i].copy()
                        permuted_signals[pair]["pair"] = pair  # Keep original pair
                
                return permuted_signals
            
            permuted_equity = self.backtester.run_backtest(timestamps, prices, permuted_signal_gen)
            permuted_metrics = self.backtester.calculate_metrics(permuted_equity)
            permuted_sharpes.append(permuted_metrics.sharpe_ratio)
        
        avg_permuted_sharpe = np.mean(permuted_sharpes)
        sharpe_delta = normal_sharpe - avg_permuted_sharpe
        
        # Test passes if permuted Sharpe is near zero
        passed = abs(avg_permuted_sharpe) < 0.5 and sharpe_delta > 1.0
        
        return DiagnosticResult(
            test_name="random_label",
            passed=passed,
            score=sharpe_delta,
            message=f"Normal Sharpe: {normal_sharpe:.3f}, Permuted Sharpe: {avg_permuted_sharpe:.3f}, Delta: {sharpe_delta:.3f}",
            details={
                "normal_sharpe": normal_sharpe,
                "permuted_sharpe": avg_permuted_sharpe,
                "sharpe_delta": sharpe_delta,
                "n_trials": n_trials
            }
        )
    
    def future_shift_test(
        self,
        timestamps: pd.DatetimeIndex,
        prices: Dict[str, pd.Series],
        signal_generator: Callable,
        shift_bars: Optional[int] = None
    ) -> DiagnosticResult:
        """Future-shift test: shift features forward by H bars.
        
        Args:
            timestamps: Timestamps for backtest
            prices: Price series
            signal_generator: Signal generation function
            shift_bars: Number of bars to shift (default: label_horizon)
            
        Returns:
            DiagnosticResult
        """
        if shift_bars is None:
            shift_bars = self.label_horizon_bars
        
        # Run normal backtest
        normal_equity = self.backtester.run_backtest(timestamps, prices, signal_generator)
        normal_metrics = self.backtester.calculate_metrics(normal_equity)
        normal_sharpe = normal_metrics.sharpe_ratio
        
        # Create shifted signal generator (uses future information)
        def shifted_signal_gen(timestamp, current_prices):
            # Shift timestamp forward
            shifted_idx = timestamps.get_loc(timestamp) + shift_bars
            if shifted_idx < len(timestamps):
                shifted_timestamp = timestamps[shifted_idx]
                # Use future prices
                future_prices = {pair: series.iloc[min(shifted_idx, len(series)-1)] 
                               for pair, series in prices.items()}
                return signal_generator(shifted_timestamp, future_prices)
            return {}
        
        shifted_equity = self.backtester.run_backtest(timestamps, prices, shifted_signal_gen)
        shifted_metrics = self.backtester.calculate_metrics(shifted_equity)
        shifted_sharpe = shifted_metrics.sharpe_ratio
        
        # Performance should collapse with future-shifted features
        sharpe_drop = normal_sharpe - shifted_sharpe
        passed = abs(shifted_sharpe) < 0.5 or sharpe_drop > 1.0
        
        return DiagnosticResult(
            test_name="future_shift",
            passed=passed,
            score=sharpe_drop,
            message=f"Normal Sharpe: {normal_sharpe:.3f}, Shifted Sharpe: {shifted_sharpe:.3f}, Drop: {sharpe_drop:.3f}",
            details={
                "normal_sharpe": normal_sharpe,
                "shifted_sharpe": shifted_sharpe,
                "sharpe_drop": sharpe_drop,
                "shift_bars": shift_bars
            }
        )
    
    def adversarial_validation(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        n_samples: int = 1000
    ) -> DiagnosticResult:
        """Adversarial validation: can we distinguish train from test?
        
        Args:
            train_features: Training feature DataFrame
            test_features: Test feature DataFrame
            n_samples: Number of samples to use
            
        Returns:
            DiagnosticResult
        """
        # Sample equal amounts from train and test
        n_train = min(n_samples // 2, len(train_features))
        n_test = min(n_samples // 2, len(test_features))
        
        train_sample = train_features.sample(n_train, random_state=42)
        test_sample = test_features.sample(n_test, random_state=42)
        
        # Create labels: 0 for train, 1 for test
        X = pd.concat([train_sample, test_sample])
        y = np.array([0] * n_train + [1] * n_test)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        
        # Predict
        y_pred_proba = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        
        # High AUC means we can distinguish train from test (distribution shift)
        # Pass if AUC <= 0.65 (can't easily distinguish)
        passed = auc <= 0.65
        
        return DiagnosticResult(
            test_name="adversarial_validation",
            passed=passed,
            score=auc,
            message=f"Adversarial AUC: {auc:.3f} (lower is better, <0.65 passes)",
            details={
                "auc": auc,
                "n_train": n_train,
                "n_test": n_test
            }
        )
    
    def cost_sensitivity_test(
        self,
        timestamps: pd.DatetimeIndex,
        prices: Dict[str, pd.Series],
        signal_generator: Callable,
        cost_sweep: List[float] = [5, 10, 20, 35, 50]
    ) -> DiagnosticResult:
        """Cost sensitivity: how does performance degrade with increasing costs?
        
        Args:
            timestamps: Timestamps for backtest
            prices: Price series
            signal_generator: Signal generation function
            cost_sweep: List of commission costs in bps
            
        Returns:
            DiagnosticResult
        """
        sharpes = []
        
        for cost_bps in cost_sweep:
            # Create backtester with this cost
            test_backtester = CostAwareBacktester(
                initial_capital=self.backtester.initial_capital,
                commission_bps=cost_bps,
                slippage_bps=self.backtester.slippage_bps * 10000,
                execution_model=self.backtester.execution_model
            )
            
            equity = test_backtester.run_backtest(timestamps, prices, signal_generator)
            metrics = test_backtester.calculate_metrics(equity)
            sharpes.append(metrics.sharpe_ratio)
        
        # Calculate sensitivity slope
        if len(sharpes) > 1:
            cost_array = np.array(cost_sweep)
            sharpe_array = np.array(sharpes)
            slope = np.polyfit(cost_array, sharpe_array, 1)[0]
            
            # Check if Sharpe degrades smoothly (not instantly destroyed)
            # Pass if slope is negative but not too steep
            passed = slope < 0 and abs(slope) < 0.1  # Not too sensitive
        else:
            slope = 0
            passed = True
        
        return DiagnosticResult(
            test_name="cost_sensitivity",
            passed=passed,
            score=slope,
            message=f"Cost sensitivity slope: {slope:.4f} (should be negative but not too steep)",
            details={
                "cost_sweep": cost_sweep,
                "sharpes": sharpes,
                "slope": slope
            }
        )
    
    def block_bootstrap(
        self,
        returns: pd.Series,
        block_length: int = 100,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> DiagnosticResult:
        """Moving block bootstrap for confidence intervals.
        
        Args:
            returns: Return series
            block_length: Length of blocks
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            DiagnosticResult
        """
        n = len(returns)
        if n < block_length * 2:
            return DiagnosticResult(
                test_name="block_bootstrap",
                passed=False,
                score=0,
                message="Insufficient data for bootstrap",
                details={}
            )
        
        # Calculate observed Sharpe
        observed_sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        # Bootstrap samples
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            # Sample blocks
            blocks = []
            remaining = n
            
            while remaining > 0:
                block_size = min(block_length, remaining)
                start_idx = np.random.randint(0, n - block_size + 1)
                block = returns.iloc[start_idx:start_idx + block_size]
                blocks.append(block)
                remaining -= block_size
            
            # Concatenate blocks
            bootstrap_returns = pd.concat(blocks)[:n]  # Trim to original length
            bootstrap_sharpe = bootstrap_returns.mean() / (bootstrap_returns.std() + 1e-10) * np.sqrt(252)
            bootstrap_sharpes.append(bootstrap_sharpe)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_sharpes, (alpha / 2) * 100)
        upper = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)
        
        # Check if observed Sharpe is within CI
        passed = lower <= observed_sharpe <= upper
        
        return DiagnosticResult(
            test_name="block_bootstrap",
            passed=passed,
            score=observed_sharpe,
            message=f"Observed Sharpe: {observed_sharpe:.3f}, CI: [{lower:.3f}, {upper:.3f}]",
            details={
                "observed_sharpe": observed_sharpe,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence": confidence
            }
        )
    
    def calculate_pbo(
        self,
        in_sample_scores: List[float],
        out_of_sample_scores: List[float]
    ) -> DiagnosticResult:
        """Calculate Probability of Backtest Overfitting (PBO).
        
        Args:
            in_sample_scores: IS performance scores (e.g., Sharpe ratios)
            out_of_sample_scores: OOS performance scores
            
        Returns:
            DiagnosticResult with PBO
        """
        if len(in_sample_scores) != len(out_of_sample_scores):
            raise ValueError("IS and OOS scores must have same length")
        
        n = len(in_sample_scores)
        
        # Rank strategies by IS performance
        is_ranks = pd.Series(in_sample_scores).rank(ascending=False)
        
        # Calculate logit transform
        logits = []
        for rank in is_ranks:
            # Convert rank to probability (lower rank = better)
            prob = (n - rank + 1) / (n + 1)
            logit = np.log(prob / (1 - prob + 1e-10))
            logits.append(logit)
        
        # Fit logistic regression: logit(IS) -> OOS
        from sklearn.linear_model import LogisticRegression
        
        X = np.array(logits).reshape(-1, 1)
        # Convert OOS to binary: 1 if positive, 0 if negative
        y = (np.array(out_of_sample_scores) > 0).astype(int)
        
        if len(set(y)) < 2:  # Need both classes
            pbo = 0.5  # Neutral
        else:
            clf = LogisticRegression(random_state=42)
            clf.fit(X, y)
            
            # PBO is probability that best IS strategy has negative OOS
            best_is_idx = np.argmax(in_sample_scores)
            best_is_logit = logits[best_is_idx]
            pbo = 1 - clf.predict_proba([[best_is_logit]])[0][1]
        
        # Pass if PBO < 0.2
        passed = pbo < 0.2
        
        return DiagnosticResult(
            test_name="pbo",
            passed=passed,
            score=pbo,
            message=f"PBO: {pbo:.3f} (should be <0.2)",
            details={
                "pbo": pbo,
                "n_strategies": n
            }
        )
    
    def calculate_dsr(
        self,
        returns: pd.Series,
        n_trials: int = 1
    ) -> DiagnosticResult:
        """Calculate Deflated Sharpe Ratio (DSR).
        
        Args:
            returns: Return series
            n_trials: Number of trials/strategies tested
            
        Returns:
            DiagnosticResult with DSR
        """
        n = len(returns)
        if n < 2:
            return DiagnosticResult(
                test_name="dsr",
                passed=False,
                score=0,
                message="Insufficient data",
                details={}
            )
        
        # Calculate observed Sharpe
        mean_return = returns.mean()
        std_return = returns.std()
        observed_sr = mean_return / (std_return + 1e-10) * np.sqrt(252)
        
        # Expected maximum Sharpe under null (multiple testing correction)
        # Using approximation from Bailey & López de Prado (2014)
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) if n_trials > 1 else 0
        
        # Variance of Sharpe ratio estimator
        var_sr = (1 + 0.5 * observed_sr**2) / n
        
        # Deflated Sharpe Ratio
        dsr = (observed_sr - expected_max_sr) / np.sqrt(var_sr + 1e-10)
        
        # Significance test (one-sided)
        from scipy import stats
        p_value = 1 - stats.norm.cdf(dsr)
        
        # Pass if significant at 5% level
        passed = p_value < 0.05 and dsr > 0
        
        return DiagnosticResult(
            test_name="dsr",
            passed=passed,
            score=dsr,
            message=f"DSR: {dsr:.3f}, p-value: {p_value:.4f} (should be significant at 5%)",
            details={
                "observed_sr": observed_sr,
                "expected_max_sr": expected_max_sr,
                "dsr": dsr,
                "p_value": p_value,
                "n_trials": n_trials
            }
        )
    
    def parameter_stability_test(
        self,
        param_grid: Dict[str, List[Any]],
        timestamps: pd.DatetimeIndex,
        prices: Dict[str, pd.Series],
        signal_generator_factory: Callable,
        max_configs: int = 50
    ) -> DiagnosticResult:
        """Parameter stability: check for flat maxima vs needle peaks.
        
        Args:
            param_grid: Parameter grid to test
            timestamps: Timestamps for backtest
            prices: Price series
            signal_generator_factory: Function that creates signal generator from params
            max_configs: Maximum number of configs to test
            
        Returns:
            DiagnosticResult
        """
        from itertools import product
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        configs = list(product(*param_values))[:max_configs]
        
        sharpes = []
        
        for config in configs:
            params = dict(zip(param_names, config))
            signal_gen = signal_generator_factory(**params)
            
            equity = self.backtester.run_backtest(timestamps, prices, signal_gen)
            metrics = self.backtester.calculate_metrics(equity)
            sharpes.append(metrics.sharpe_ratio)
        
        if len(sharpes) == 0:
            return DiagnosticResult(
                test_name="parameter_stability",
                passed=False,
                score=0,
                message="No configurations tested",
                details={}
            )
        
        sharpes = np.array(sharpes)
        best_sharpe = sharpes.max()
        
        # Calculate flatness: what % of configs are within 10-20% of best?
        threshold_10pct = best_sharpe * 0.9
        threshold_20pct = best_sharpe * 0.8
        
        within_10pct = (sharpes >= threshold_10pct).sum()
        within_20pct = (sharpes >= threshold_20pct).sum()
        
        flatness_10pct = within_10pct / len(sharpes)
        flatness_20pct = within_20pct / len(sharpes)
        
        # Pass if top 10% of configs are within 10-20% of best (flat maximum)
        passed = flatness_10pct >= 0.1 or flatness_20pct >= 0.2
        
        return DiagnosticResult(
            test_name="parameter_stability",
            passed=passed,
            score=flatness_10pct,
            message=f"Flatness (10%): {flatness_10pct:.2%}, (20%): {flatness_20pct:.2%}",
            details={
                "best_sharpe": best_sharpe,
                "flatness_10pct": flatness_10pct,
                "flatness_20pct": flatness_20pct,
                "n_configs": len(configs)
            }
        )
    
    def regime_robustness_test(
        self,
        timestamps: pd.DatetimeIndex,
        prices: Dict[str, pd.Series],
        signal_generator: Callable,
        regime_tags: pd.Series
    ) -> DiagnosticResult:
        """Regime robustness: check performance across different market regimes.
        
        Args:
            timestamps: Timestamps for backtest
            prices: Price series
            signal_generator: Signal generation function
            regime_tags: Series with regime labels for each timestamp
            
        Returns:
            DiagnosticResult
        """
        if len(regime_tags) != len(timestamps):
            raise ValueError("Regime tags must match timestamps")
        
        regime_performance = {}
        
        # Run backtest for each regime
        unique_regimes = regime_tags.unique()
        
        for regime in unique_regimes:
            regime_mask = regime_tags == regime
            regime_timestamps = timestamps[regime_mask]
            
            if len(regime_timestamps) < 10:
                continue
            
            # Filter prices for this regime
            regime_prices = {}
            for pair, price_series in prices.items():
                regime_price = price_series.loc[regime_timestamps]
                if len(regime_price) > 0:
                    regime_prices[pair] = regime_price
            
            if len(regime_prices) == 0:
                continue
            
            # Run backtest
            equity = self.backtester.run_backtest(regime_timestamps, regime_prices, signal_generator)
            metrics = self.backtester.calculate_metrics(equity)
            
            regime_performance[regime] = {
                "sharpe": metrics.sharpe_ratio,
                "return": metrics.total_return,
                "max_dd": metrics.max_drawdown
            }
        
        if len(regime_performance) == 0:
            return DiagnosticResult(
                test_name="regime_robustness",
                passed=False,
                score=0,
                message="No regimes tested",
                details={}
            )
        
        # Check consistency: same sign of edge across regimes
        sharpes = [perf["sharpe"] for perf in regime_performance.values()]
        positive_sharpes = [s for s in sharpes if s > 0]
        negative_sharpes = [s for s in sharpes if s < 0]
        
        consistency = len(positive_sharpes) / len(sharpes) if len(sharpes) > 0 else 0
        
        # Pass if ≥70% of regimes have same-sign edge
        passed = consistency >= 0.7 or consistency <= 0.3  # Either mostly positive or mostly negative
        
        return DiagnosticResult(
            test_name="regime_robustness",
            passed=passed,
            score=consistency,
            message=f"Consistency: {consistency:.2%} regimes with positive Sharpe",
            details={
                "regime_performance": regime_performance,
                "consistency": consistency
            }
        )
    
    def reality_check(
        self,
        timestamps: pd.DatetimeIndex,
        prices: Dict[str, pd.Series],
        signal_generator: Callable,
        baseline_methods: List[str] = ["buy_hold", "momentum", "equal_weight"]
    ) -> DiagnosticResult:
        """Reality check: compare against naive baselines.
        
        Args:
            timestamps: Timestamps for backtest
            prices: Price series
            signal_generator: Signal generation function
            baseline_methods: List of baseline methods to test
            
        Returns:
            DiagnosticResult
        """
        # Run strategy backtest
        strategy_equity = self.backtester.run_backtest(timestamps, prices, signal_generator)
        strategy_metrics = self.backtester.calculate_metrics(strategy_equity)
        strategy_sharpe = strategy_metrics.sharpe_ratio
        
        baseline_sharpes = {}
        
        # Buy-and-hold baseline
        if "buy_hold" in baseline_methods and len(prices) > 0:
            # Use first available pair
            first_pair = list(prices.keys())[0]
            price_series = prices[first_pair]
            
            if len(price_series) > 0:
                returns = price_series.pct_change().dropna()
                if len(returns) > 0:
                    bh_sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
                    baseline_sharpes["buy_hold"] = bh_sharpe
        
        # Momentum baseline (12-1 momentum)
        if "momentum" in baseline_methods and len(prices) > 0:
            first_pair = list(prices.keys())[0]
            price_series = prices[first_pair]
            
            if len(price_series) > 12:
                momentum_returns = []
                for i in range(12, len(price_series)):
                    ret_12 = (price_series.iloc[i] / price_series.iloc[i-12]) - 1
                    ret_1 = (price_series.iloc[i] / price_series.iloc[i-1]) - 1
                    if ret_12 > 0:  # Buy if 12-period return positive
                        momentum_returns.append(ret_1)
                    else:  # Hold cash
                        momentum_returns.append(0)
                
                if len(momentum_returns) > 0:
                    mom_returns = pd.Series(momentum_returns)
                    mom_sharpe = mom_returns.mean() / (mom_returns.std() + 1e-10) * np.sqrt(252)
                    baseline_sharpes["momentum"] = mom_sharpe
        
        # Strategy should beat at least some baselines
        if baseline_sharpes:
            best_baseline = max(baseline_sharpes.values())
            passed = strategy_sharpe > best_baseline * 0.8  # At least 80% of best baseline
        else:
            passed = True  # No baselines to compare
        
        return DiagnosticResult(
            test_name="reality_check",
            passed=passed,
            score=strategy_sharpe,
            message=f"Strategy Sharpe: {strategy_sharpe:.3f}, Baselines: {baseline_sharpes}",
            details={
                "strategy_sharpe": strategy_sharpe,
                "baseline_sharpes": baseline_sharpes
            }
        )

