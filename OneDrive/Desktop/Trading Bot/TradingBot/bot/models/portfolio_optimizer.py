"""Portfolio optimization using minimum variance frontier for maximum Sharpe/Sortino/Calmar."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Optimize portfolio weights to maximize Sharpe, Sortino, and Calmar ratios."""
    
    def __init__(self, risk_free_rate: float = 0.0):
        """Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe/Sortino calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_expected_returns_from_signals(
        self,
        signals: Dict[str, Dict],
        base_return_estimate: float = 0.02
    ) -> pd.Series:
        """Convert ML signals to expected returns.
        
        Args:
            signals: Dictionary mapping pair to signal dict with 'confidence', 'probability', etc.
            base_return_estimate: Base expected return for high-confidence signals (2% default)
        
        Returns:
            Series of expected returns for each pair
        """
        expected_returns = {}
        
        for pair, signal in signals.items():
            confidence = signal.get("confidence", 0.5)
            prob_up = signal.get("probability", 0.5)
            side = signal.get("side", "long")
            
            # Expected return = (prob_up - 0.5) * confidence * base_return
            # For long: positive if prob_up > 0.5
            # For short: negative if prob_up < 0.5 (we profit when price goes down)
            if side == "long":
                expected_return = (prob_up - 0.5) * confidence * base_return_estimate * 2
            else:  # short
                expected_return = (0.5 - prob_up) * confidence * base_return_estimate * 2
            
            expected_returns[pair] = expected_return
        
        return pd.Series(expected_returns)
    
    def calculate_covariance_matrix(
        self,
        historical_returns: Dict[str, pd.Series],
        min_periods: int = 20
    ) -> pd.DataFrame:
        """Calculate covariance matrix from historical returns.
        
        Args:
            historical_returns: Dictionary mapping pair to Series of returns
            min_periods: Minimum periods required for calculation
        
        Returns:
            Covariance matrix DataFrame
        """
        # Align all returns to common index
        returns_df = pd.DataFrame(historical_returns)
        
        # Forward fill missing values (assume no change if no data)
        returns_df = returns_df.fillna(0.0)
        
        # Calculate covariance
        cov_matrix = returns_df.cov()
        
        # Ensure positive semi-definite (add small regularization if needed)
        eigenvals = np.linalg.eigvals(cov_matrix.values)
        if np.any(eigenvals < 0):
            # Add small regularization to diagonal
            regularization = np.eye(len(cov_matrix)) * 1e-6
            cov_matrix = pd.DataFrame(
                cov_matrix.values + regularization,
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )
        
        return cov_matrix
    
    def calculate_downside_covariance(
        self,
        historical_returns: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Calculate downside-only covariance for Sortino ratio optimization.
        
        Args:
            historical_returns: Dictionary mapping pair to Series of returns
        
        Returns:
            Downside covariance matrix (only negative returns contribute)
        """
        # Align all returns
        returns_df = pd.DataFrame(historical_returns).fillna(0.0)
        
        # Keep only negative returns (set positive to 0)
        downside_returns = returns_df.copy()
        downside_returns[downside_returns > 0] = 0
        
        # Calculate downside covariance
        downside_cov = downside_returns.cov()
        
        # Regularize if needed
        eigenvals = np.linalg.eigvals(downside_cov.values)
        if np.any(eigenvals < 0):
            regularization = np.eye(len(downside_cov)) * 1e-6
            downside_cov = pd.DataFrame(
                downside_cov.values + regularization,
                index=downside_cov.index,
                columns=downside_cov.columns
            )
        
        return downside_cov
    
    def optimize_tangency_portfolio(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        long_only: bool = True,
        max_weight: float = 0.3
    ) -> Tuple[pd.Series, float]:
        """Find tangency portfolio (maximum Sharpe ratio).
        
        Args:
            expected_returns: Expected returns for each asset
            covariance: Covariance matrix
            long_only: If True, weights must be non-negative
            max_weight: Maximum weight per asset (for diversification)
        
        Returns:
            Tuple of (optimal weights, Sharpe ratio)
        """
        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()
        
        # Ensure covariance matches expected returns
        cov_matrix = covariance.loc[assets, assets].values
        mu = expected_returns.values
        
        # Objective: maximize Sharpe ratio = (w'*mu - rf) / sqrt(w'*Sigma*w)
        # Equivalently: minimize -Sharpe = -((w'*mu - rf) / sqrt(w'*Sigma*w))
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mu) - self.risk_free_rate
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std < 1e-8:
                return 1e6  # Penalty for zero variance
            
            return -portfolio_return / portfolio_std
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]  # Weights sum to 1
        
        # Bounds
        if long_only:
            bounds = [(0.0, max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=assets)
                sharpe = -result.fun
                return weights, sharpe
            else:
                logger.warning(f"Optimization failed: {result.message}, using equal weights")
                weights = pd.Series(np.ones(n_assets) / n_assets, index=assets)
                return weights, 0.0
        except Exception as e:
            logger.error(f"Error in optimization: {e}", exc_info=True)
            weights = pd.Series(np.ones(n_assets) / n_assets, index=assets)
            return weights, 0.0
    
    def optimize_sortino_portfolio(
        self,
        expected_returns: pd.Series,
        downside_covariance: pd.DataFrame,
        long_only: bool = True,
        max_weight: float = 0.3
    ) -> Tuple[pd.Series, float]:
        """Find portfolio with maximum Sortino ratio.
        
        Args:
            expected_returns: Expected returns for each asset
            downside_covariance: Downside-only covariance matrix
            long_only: If True, weights must be non-negative
            max_weight: Maximum weight per asset
        
        Returns:
            Tuple of (optimal weights, Sortino ratio)
        """
        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()
        
        cov_matrix = downside_covariance.loc[assets, assets].values
        mu = expected_returns.values
        
        def negative_sortino(weights):
            portfolio_return = np.dot(weights, mu) - self.risk_free_rate
            portfolio_downside_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_downside_std = np.sqrt(portfolio_downside_variance)
            
            if portfolio_downside_std < 1e-8:
                return 1e6
            
            return -portfolio_return / portfolio_downside_std
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        if long_only:
            bounds = [(0.0, max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                negative_sortino,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=assets)
                sortino = -result.fun
                return weights, sortino
            else:
                logger.warning(f"Sortino optimization failed: {result.message}")
                weights = pd.Series(np.ones(n_assets) / n_assets, index=assets)
                return weights, 0.0
        except Exception as e:
            logger.error(f"Error in Sortino optimization: {e}", exc_info=True)
            weights = pd.Series(np.ones(n_assets) / n_assets, index=assets)
            return weights, 0.0
    
    def optimize_combined_portfolio(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        downside_covariance: pd.DataFrame,
        historical_returns: Dict[str, pd.Series],
        long_only: bool = True,
        max_weight: float = 0.3,
        sharpe_weight: float = 0.3,
        sortino_weight: float = 0.4,
        calmar_weight: float = 0.3
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Optimize portfolio to maximize combined competition score.
        
        Competition score = 0.4×Sortino + 0.3×Sharpe + 0.3×Calmar
        
        Args:
            expected_returns: Expected returns
            covariance: Full covariance matrix
            downside_covariance: Downside covariance matrix
            historical_returns: Historical returns for drawdown calculation
            long_only: If True, weights must be non-negative
            max_weight: Maximum weight per asset
            sharpe_weight: Weight for Sharpe in combined objective
            sortino_weight: Weight for Sortino in combined objective
            calmar_weight: Weight for Calmar in combined objective
        
        Returns:
            Tuple of (optimal weights, metrics dict)
        """
        n_assets = len(expected_returns)
        assets = expected_returns.index.tolist()
        
        cov_matrix = covariance.loc[assets, assets].values
        downside_cov_matrix = downside_covariance.loc[assets, assets].values
        mu = expected_returns.values
        
        # Get historical returns for drawdown calculation
        returns_df = pd.DataFrame({k: v for k, v in historical_returns.items() if k in assets})
        returns_df = returns_df.fillna(0.0)
        
        def negative_combined_score(weights):
            # Portfolio metrics
            portfolio_return = np.dot(weights, mu) - self.risk_free_rate
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            portfolio_downside_variance = np.dot(weights, np.dot(downside_cov_matrix, weights))
            portfolio_downside_std = np.sqrt(portfolio_downside_variance)
            
            # Sharpe
            if portfolio_std < 1e-8:
                sharpe = 0.0
            else:
                sharpe = portfolio_return / portfolio_std
            
            # Sortino
            if portfolio_downside_std < 1e-8:
                sortino = 0.0
            else:
                sortino = portfolio_return / portfolio_downside_std
            
            # Calmar (approximate using historical drawdown)
            # Calculate portfolio returns from historical data
            if len(returns_df) > 0:
                # Ensure weights is a Series with matching index
                weights_series = pd.Series(weights, index=returns_df.columns) if isinstance(weights, (list, np.ndarray)) else weights
                portfolio_returns_hist = (returns_df * weights_series).sum(axis=1)
                cumulative = (1 + portfolio_returns_hist).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.01
                
                if max_dd < 1e-8:
                    calmar = 0.0
                else:
                    # Annualized return / max drawdown
                    annual_return = portfolio_return * np.sqrt(252 * 24)  # Approximate
                    calmar = annual_return / max_dd
            else:
                calmar = 0.0
            
            # Combined score (negative because we're minimizing)
            combined = -(sortino_weight * sortino + sharpe_weight * sharpe + calmar_weight * calmar)
            
            return combined
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        if long_only:
            bounds = [(0.0, max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        
        x0 = np.ones(n_assets) / n_assets
        
        try:
            # Try multiple optimization methods for robustness
            methods = ['SLSQP', 'L-BFGS-B', 'TNC']
            result = None
            
            for method in methods:
                try:
                    if method == 'L-BFGS-B' or method == 'TNC':
                        # These methods don't support equality constraints, use penalty instead
                        def objective_with_penalty(w):
                            penalty = 1000 * (np.sum(w) - 1.0) ** 2
                            return negative_combined_score(w) + penalty
                        
                        result = minimize(
                            objective_with_penalty,
                            x0,
                            method=method,
                            bounds=bounds,
                            options={'maxiter': 500}
                        )
                    else:
                        result = minimize(
                            negative_combined_score,
                            x0,
                            method=method,
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-6}
                        )
                    
                    if result.success:
                        break
                except Exception as e:
                    logger.debug(f"Optimization method {method} failed: {e}")
                    continue
            
            if result is None:
                result = type('obj', (object,), {'success': False, 'message': 'All methods failed', 'x': x0})()
            
            if result.success:
                weights = pd.Series(result.x, index=assets)
                
                # Calculate final metrics
                portfolio_return = np.dot(weights.values, mu) - self.risk_free_rate
                portfolio_variance = np.dot(weights.values, np.dot(cov_matrix, weights.values))
                portfolio_std = np.sqrt(portfolio_variance)
                portfolio_downside_variance = np.dot(weights.values, np.dot(downside_cov_matrix, weights.values))
                portfolio_downside_std = np.sqrt(portfolio_downside_variance)
                
                sharpe = portfolio_return / portfolio_std if portfolio_std > 1e-8 else 0.0
                sortino = portfolio_return / portfolio_downside_std if portfolio_downside_std > 1e-8 else 0.0
                
                # Calmar
                if len(returns_df) > 0:
                    # Ensure weights is a Series with matching index
                    weights_series = pd.Series(weights, index=returns_df.columns) if isinstance(weights, (list, np.ndarray)) else weights
                    portfolio_returns_hist = (returns_df * weights_series).sum(axis=1)
                    cumulative = (1 + portfolio_returns_hist).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.01
                    annual_return = portfolio_return * np.sqrt(252 * 24)
                    calmar = annual_return / max_dd if max_dd > 1e-8 else 0.0
                else:
                    calmar = 0.0
                
                metrics = {
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "calmar": calmar,
                    "competition_score": 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar
                }
                
                return weights, metrics
            else:
                logger.warning(f"Combined optimization failed: {result.message}")
                weights = pd.Series(np.ones(n_assets) / n_assets, index=assets)
                return weights, {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "competition_score": 0.0}
        except Exception as e:
            logger.error(f"Error in combined optimization: {e}", exc_info=True)
            weights = pd.Series(np.ones(n_assets) / n_assets, index=assets)
            return weights, {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "competition_score": 0.0}

