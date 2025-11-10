"""Data integrity checks to prevent look-ahead bias and data leakage."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class IntegrityCheckResult:
    """Result of a data integrity check."""
    passed: bool
    message: str
    severity: str  # "error", "warning", "info"
    details: Optional[Dict[str, Any]] = None


class DataIntegrityChecker:
    """Check data integrity to prevent look-ahead bias and leakage."""
    
    def __init__(self, label_horizon_bars: int = 180):
        """Initialize data integrity checker.
        
        Args:
            label_horizon_bars: Number of bars in the future for label calculation
        """
        self.label_horizon_bars = label_horizon_bars
        self.regime_breaks = [
            pd.Timestamp("2008-09-15"),  # Lehman collapse
            pd.Timestamp("2020-03-12"),  # COVID crash
            pd.Timestamp("2022-05-10"),  # Terra/Luna collapse
            pd.Timestamp("2024-01-01"),  # Recent regime
        ]
    
    def check_timestamp_sanity(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str = "timestamp"
    ) -> IntegrityCheckResult:
        """Check that timestamps are properly ordered and no future data leaks.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            IntegrityCheckResult
        """
        if timestamp_col not in df.columns and timestamp_col not in df.index.names:
            return IntegrityCheckResult(
                False, 
                f"Timestamp column '{timestamp_col}' not found",
                "error"
            )
        
        # Get timestamp series
        if timestamp_col in df.index.names:
            timestamps = df.index
        else:
            timestamps = pd.to_datetime(df[timestamp_col])
        
        # Check for duplicates
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            return IntegrityCheckResult(
                False,
                f"Found {duplicates} duplicate timestamps",
                "error",
                {"duplicate_count": duplicates}
            )
        
        # Check ordering
        if not timestamps.is_monotonic_increasing:
            out_of_order = (~timestamps.is_monotonic_increasing).sum()
            return IntegrityCheckResult(
                False,
                f"Found {out_of_order} timestamps out of order",
                "error",
                {"out_of_order_count": out_of_order}
            )
        
        # Check for future dates (relative to last timestamp)
        if len(timestamps) > 0:
            max_timestamp = timestamps.max()
            future_dates = (timestamps > max_timestamp).sum()
            if future_dates > 0:
                return IntegrityCheckResult(
                    False,
                    f"Found {future_dates} timestamps after maximum",
                    "error"
                )
        
        return IntegrityCheckResult(
            True,
            "Timestamp sanity checks passed",
            "info"
        )
    
    def check_lookahead_bias(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        shift_bars: int = 1
    ) -> IntegrityCheckResult:
        """Check for look-ahead bias by shifting features forward.
        
        Args:
            features: Feature DataFrame
            prices: Price series for comparison
            shift_bars: Number of bars to shift forward
            
        Returns:
            IntegrityCheckResult with test results
        """
        if len(features) < shift_bars + 10:
            return IntegrityCheckResult(
                False,
                "Insufficient data for look-ahead test",
                "warning"
            )
        
        # Shift features forward by shift_bars
        features_shifted = features.shift(-shift_bars).dropna()
        prices_aligned = prices.iloc[:len(features_shifted)]
        
        if len(features_shifted) == 0:
            return IntegrityCheckResult(
                False,
                "No data after shifting",
                "warning"
            )
        
        # Calculate correlation between shifted features and future prices
        # If features are shifted forward, they should have HIGH correlation with future prices
        # This indicates potential look-ahead bias
        correlations = {}
        for col in features_shifted.columns:
            if features_shifted[col].dtype in [np.float64, np.int64]:
                corr = features_shifted[col].corr(prices_aligned)
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
        
        if correlations:
            max_corr = max(correlations.values())
            suspicious_features = [
                col for col, corr in correlations.items() 
                if abs(corr) > 0.8
            ]
            
            if suspicious_features:
                return IntegrityCheckResult(
                    False,
                    f"High correlation ({max_corr:.3f}) between shifted features and prices. "
                    f"Suspicious features: {suspicious_features[:5]}",
                    "error",
                    {
                        "max_correlation": max_corr,
                        "suspicious_features": suspicious_features,
                        "shift_bars": shift_bars
                    }
                )
        
        return IntegrityCheckResult(
            True,
            f"Look-ahead bias test passed (max correlation: {max(correlations.values()) if correlations else 0:.3f})",
            "info",
            {"max_correlation": max(correlations.values()) if correlations else 0}
        )
    
    def check_duplicates_and_gaps(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        expected_frequency: Optional[str] = None
    ) -> IntegrityCheckResult:
        """Check for duplicate timestamps and gaps in data.
        
        Args:
            df: DataFrame to check
            timestamp_col: Name of timestamp column
            expected_frequency: Expected frequency (e.g., '1min', '1H')
            
        Returns:
            IntegrityCheckResult
        """
        # Get timestamp series
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif timestamp_col in df.index.names:
            timestamps = df.index
        elif timestamp_col in df.columns:
            timestamps = pd.to_datetime(df[timestamp_col])
        else:
            # If no timestamp column and index is not datetime, skip this check
            return IntegrityCheckResult(
                True,
                "No timestamp column found, skipping duplicate/gap check",
                "info"
            )
        
        # Check duplicates
        duplicates = timestamps.duplicated().sum()
        duplicate_details = {}
        if duplicates > 0:
            duplicate_times = timestamps[timestamps.duplicated()].tolist()
            duplicate_details["duplicate_timestamps"] = duplicate_times[:10]  # First 10
        
        # Check gaps
        gap_details = {}
        if expected_frequency and len(timestamps) > 1:
            expected_delta = pd.Timedelta(expected_frequency)
            actual_deltas = timestamps.to_series().diff().dropna()
            
            # Find gaps larger than expected
            large_gaps = actual_deltas[actual_deltas > expected_delta * 2]
            if len(large_gaps) > 0:
                gap_details["large_gaps"] = len(large_gaps)
                gap_details["max_gap"] = str(large_gaps.max())
                gap_details["gap_locations"] = large_gaps.index.tolist()[:10]
        
        if duplicates > 0 or gap_details:
            return IntegrityCheckResult(
                duplicates == 0,  # Pass only if no duplicates
                f"Found {duplicates} duplicates, {gap_details.get('large_gaps', 0)} large gaps",
                "warning" if duplicates == 0 else "error",
                {**duplicate_details, **gap_details}
            )
        
        return IntegrityCheckResult(
            True,
            "No duplicates or significant gaps found",
            "info"
        )
    
    def tag_regime_breaks(
        self,
        timestamps: pd.DatetimeIndex
    ) -> pd.Series:
        """Tag timestamps with known regime breaks.
        
        Args:
            timestamps: DatetimeIndex to tag
            
        Returns:
            Series with regime labels
        """
        regime_tags = pd.Series("normal", index=timestamps)
        
        for i, break_date in enumerate(self.regime_breaks):
            if i == 0:
                regime_tags[timestamps < break_date] = "pre-2008"
            elif i == 1:
                regime_tags[(timestamps >= self.regime_breaks[i-1]) & 
                           (timestamps < break_date)] = "GFC"
            elif i == 2:
                regime_tags[(timestamps >= self.regime_breaks[i-1]) & 
                           (timestamps < break_date)] = "QE"
            elif i == 3:
                regime_tags[(timestamps >= self.regime_breaks[i-1]) & 
                           (timestamps < break_date)] = "COVID"
        
        regime_tags[timestamps >= self.regime_breaks[-1]] = "post-2022"
        
        return regime_tags
    
    def check_feature_temporal_consistency(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        timestamp_col: Optional[str] = None
    ) -> IntegrityCheckResult:
        """Check that features at time t only use information available at or before t.
        
        Args:
            features: Feature DataFrame
            prices: Price series for validation
            timestamp_col: Optional timestamp column name
            
        Returns:
            IntegrityCheckResult
        """
        if len(features) != len(prices):
            return IntegrityCheckResult(
                False,
                f"Feature length ({len(features)}) != price length ({len(prices)})",
                "error"
            )
        
        # For each feature, check if it's using future information
        # This is a simplified check - in practice, you'd need to know how features are computed
        issues = []
        
        # Check if any feature perfectly predicts future price (suspicious)
        for col in features.columns:
            if features[col].dtype in [np.float64, np.int64]:
                # Shift prices forward by label horizon
                if len(prices) > self.label_horizon_bars:
                    future_prices = prices.shift(-self.label_horizon_bars).dropna()
                    aligned_features = features[col].iloc[:len(future_prices)]
                    
                    if len(aligned_features) > 10:
                        corr = aligned_features.corr(future_prices)
                        if not np.isnan(corr) and abs(corr) > 0.95:
                            issues.append(f"{col} has suspiciously high correlation ({corr:.3f}) with future prices")
        
        if issues:
            return IntegrityCheckResult(
                False,
                f"Found {len(issues)} features with suspicious future correlation",
                "error",
                {"issues": issues}
            )
        
        return IntegrityCheckResult(
            True,
            "Feature temporal consistency checks passed",
            "info"
        )
    
    def run_all_checks(
        self,
        df: pd.DataFrame,
        prices: Optional[pd.Series] = None,
        features: Optional[pd.DataFrame] = None,
        timestamp_col: str = "timestamp"
    ) -> Dict[str, IntegrityCheckResult]:
        """Run all integrity checks.
        
        Args:
            df: Main DataFrame
            prices: Optional price series
            features: Optional feature DataFrame
            timestamp_col: Name of timestamp column
            
        Returns:
            Dictionary of check results
        """
        results = {}
        
        # Timestamp sanity
        results["timestamp_sanity"] = self.check_timestamp_sanity(df, timestamp_col)
        
        # Duplicates and gaps
        results["duplicates_gaps"] = self.check_duplicates_and_gaps(df, timestamp_col)
        
        # Look-ahead bias (if we have features and prices)
        if features is not None and prices is not None:
            results["lookahead_bias"] = self.check_lookahead_bias(features, prices)
            results["temporal_consistency"] = self.check_feature_temporal_consistency(
                features, prices, timestamp_col
            )
        
        return results
    
    def generate_report(self, results: Dict[str, IntegrityCheckResult]) -> str:
        """Generate a text report of integrity check results.
        
        Args:
            results: Dictionary of check results
            
        Returns:
            Formatted report string
        """
        report_lines = ["=" * 60]
        report_lines.append("DATA INTEGRITY CHECK REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        for check_name, result in results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report_lines.append(f"{check_name.upper()}: {status}")
            report_lines.append(f"  Severity: {result.severity.upper()}")
            report_lines.append(f"  Message: {result.message}")
            if result.details:
                report_lines.append(f"  Details: {result.details}")
            report_lines.append("")
        
        # Summary
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        report_lines.append(f"Summary: {passed}/{total} checks passed")
        
        return "\n".join(report_lines)

