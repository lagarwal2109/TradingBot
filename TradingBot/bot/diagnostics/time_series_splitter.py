"""Time-aware data splitting for time series to prevent leakage."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SplitWindow:
    """Represents a single train/valid/test split window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: List[int]
    valid_indices: List[int]
    test_indices: List[int]


class TimeSeriesSplitter:
    """Time-aware data splitting to prevent time leakage."""
    
    def __init__(
        self,
        embargo_bars: int = 180,
        min_train_bars: int = 1000
    ):
        """Initialize time series splitter.
        
        Args:
            embargo_bars: Number of bars to embargo after validation/test sets
            min_train_bars: Minimum number of bars required for training
        """
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars
    
    def walk_forward_split(
        self,
        timestamps: pd.DatetimeIndex,
        train_years: float = 0.5,
        valid_years: float = 0.1,
        test_years: float = 0.1,
        step_years: float = 0.1
    ) -> List[SplitWindow]:
        """Generate walk-forward optimization splits.
        
        Args:
            timestamps: Sorted DatetimeIndex
            train_years: Training period in years
            valid_years: Validation period in years
            test_years: Test period in years
            step_years: Step size for rolling forward in years
            
        Returns:
            List of SplitWindow objects
        """
        if len(timestamps) == 0:
            return []
        
        total_span = (timestamps.max() - timestamps.min()).total_seconds() / (365.25 * 24 * 3600)
        
        if total_span < train_years + valid_years + test_years:
            logger.warning(
                f"Total data span ({total_span:.2f} years) is less than required "
                f"({train_years + valid_years + test_years:.2f} years)"
            )
            return []
        
        windows = []
        start_date = timestamps.min()
        end_date = timestamps.max()
        
        current_start = start_date
        
        while True:
            # Calculate window boundaries
            train_start = current_start
            train_end = train_start + pd.Timedelta(days=train_years * 365.25)
            valid_start = train_end
            valid_end = valid_start + pd.Timedelta(days=valid_years * 365.25)
            test_start = valid_end
            test_end = test_start + pd.Timedelta(days=test_years * 365.25)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            # Get indices for each period
            train_mask = (timestamps >= train_start) & (timestamps < train_end)
            valid_mask = (timestamps >= valid_start) & (timestamps < valid_end)
            test_mask = (timestamps >= test_start) & (timestamps < test_end)
            
            train_indices = np.where(train_mask)[0].tolist()
            valid_indices = np.where(valid_mask)[0].tolist()
            test_indices = np.where(test_mask)[0].tolist()
            
            # Apply embargo: remove embargo_bars from train after valid/test
            if len(valid_indices) > 0:
                valid_start_idx = min(valid_indices)
                valid_end_idx = max(valid_indices)
                embargo_end_idx = valid_end_idx + self.embargo_bars
                # Remove indices in embargo period from train
                train_indices = [idx for idx in train_indices if idx < valid_start_idx]
            
            if len(test_indices) > 0:
                test_start_idx = min(test_indices)
                test_end_idx = max(test_indices)
                embargo_end_idx = test_end_idx + self.embargo_bars
                # Remove indices in embargo period from train
                if len(valid_indices) > 0:
                    valid_start_idx = min(valid_indices)
                    train_indices = [idx for idx in train_indices if idx < valid_start_idx]
            
            # Check minimum training size
            if len(train_indices) < self.min_train_bars:
                current_start += pd.Timedelta(days=step_years * 365.25)
                continue
            
            windows.append(SplitWindow(
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
                train_indices=train_indices,
                valid_indices=valid_indices,
                test_indices=test_indices
            ))
            
            # Move forward
            current_start += pd.Timedelta(days=step_years * 365.25)
        
        logger.info(f"Generated {len(windows)} walk-forward splits")
        return windows
    
    def pkfe_split(
        self,
        timestamps: pd.DatetimeIndex,
        k: int = 5
    ) -> List[SplitWindow]:
        """Generate Purged K-Fold with Embargo splits.
        
        Args:
            timestamps: Sorted DatetimeIndex
            k: Number of folds
            
        Returns:
            List of SplitWindow objects
        """
        if len(timestamps) == 0:
            return []
        
        total_bars = len(timestamps)
        bars_per_fold = total_bars // k
        
        windows = []
        
        for i in range(k):
            # Calculate fold boundaries
            fold_start_idx = i * bars_per_fold
            fold_end_idx = (i + 1) * bars_per_fold if i < k - 1 else total_bars
            
            # Validation fold
            valid_start_idx = fold_start_idx
            valid_end_idx = fold_end_idx
            
            # Training: before validation (with embargo)
            train_end_idx = valid_start_idx - self.embargo_bars
            train_start_idx = 0
            
            # Test: after validation (with embargo)
            test_start_idx = valid_end_idx + self.embargo_bars
            test_end_idx = (i + 2) * bars_per_fold if i < k - 1 else total_bars
            
            # Ensure valid indices
            if train_end_idx <= train_start_idx:
                continue
            if test_start_idx >= total_bars:
                test_start_idx = total_bars
                test_end_idx = total_bars
            
            train_indices = list(range(train_start_idx, train_end_idx))
            valid_indices = list(range(valid_start_idx, valid_end_idx))
            test_indices = list(range(test_start_idx, min(test_end_idx, total_bars)))
            
            # Check minimum training size
            if len(train_indices) < self.min_train_bars:
                continue
            
            windows.append(SplitWindow(
                train_start=timestamps[train_start_idx],
                train_end=timestamps[min(train_end_idx - 1, total_bars - 1)],
                valid_start=timestamps[valid_start_idx],
                valid_end=timestamps[min(valid_end_idx - 1, total_bars - 1)],
                test_start=timestamps[min(test_start_idx, total_bars - 1)],
                test_end=timestamps[min(test_end_idx - 1, total_bars - 1)],
                train_indices=train_indices,
                valid_indices=valid_indices,
                test_indices=test_indices
            ))
        
        logger.info(f"Generated {len(windows)} PKFE splits")
        return windows
    
    def split_dataframe(
        self,
        df: pd.DataFrame,
        method: str = "wfo",
        **kwargs
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Split DataFrame using specified method.
        
        Args:
            df: DataFrame with datetime index
            method: "wfo" or "pkfe"
            **kwargs: Additional arguments for split method
            
        Returns:
            List of (train_df, valid_df, test_df) tuples
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        timestamps = df.index
        
        if method == "wfo":
            windows = self.walk_forward_split(timestamps, **kwargs)
        elif method == "pkfe":
            windows = self.pkfe_split(timestamps, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        splits = []
        for window in windows:
            train_df = df.iloc[window.train_indices].copy()
            valid_df = df.iloc[window.valid_indices].copy()
            test_df = df.iloc[window.test_indices].copy()
            
            splits.append((train_df, valid_df, test_df))
        
        return splits
    
    def get_split_info(self, windows: List[SplitWindow]) -> pd.DataFrame:
        """Get summary information about splits.
        
        Args:
            windows: List of SplitWindow objects
            
        Returns:
            DataFrame with split information
        """
        info = []
        for i, window in enumerate(windows):
            info.append({
                "split": i + 1,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "train_size": len(window.train_indices),
                "valid_start": window.valid_start,
                "valid_end": window.valid_end,
                "valid_size": len(window.valid_indices),
                "test_start": window.test_start,
                "test_end": window.test_end,
                "test_size": len(window.test_indices),
            })
        
        return pd.DataFrame(info)

