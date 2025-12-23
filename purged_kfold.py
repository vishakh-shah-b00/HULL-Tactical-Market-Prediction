"""
Purged K-Fold Cross Validation
Custom CV splitter to prevent data leakage in time-series financial data.
"""
import numpy as np
from typing import Generator, Tuple

class PurgedKFold:
    """
    K-Fold Cross Validation with 'Purging' and 'Embargo'.
    
    Standard K-Fold is dangerous for financial time-series because of two leakages:
    1. Look-ahead bias: Training on future data.
    2. Short-term momentum: Training on data immediately before/after test set if they are correlated.
    
    This implementation:
    - Splits data into N folds.
    - Ensures Training Data does not overlap with Test Data.
    - Adds an 'Embargo' period (gap) after the Test set to prevent leakage from short-term momentum.
    """
    
    def __init__(self, n_splits: int = 5, embargo_days: int = 20):
        """
        Args:
            n_splits (int): Number of folds.
            embargo_days (int): Number of days to skip AFTER the test set before resuming training data.
                                Default 20 (approx 1 trading month) is industry standard.
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Logic:
        1. Sequentially split data into n_splits (standard KFold).
        2. Purge: Remove training samples that overlap with the test set time range.
        3. Embargo: Remove training samples immediately following the test set (to prevent leakage from short-term momentum).
        
        Args:
            X: Training data (must be pandas DataFrame/Series with index/date info if needed, or treated sequentially).
               Here we assume standard RangeIndex implies time order.
            y: Target variable (unused).
            groups: Group labels (unused).
            
        Yields:
            train_indices, test_indices
        """
        indices = np.arange(X.shape[0])
        
        # Standard KFold calculation of fold sizes
        fold_size = len(X) // self.n_splits
        
        for i in range(self.n_splits):
            # Define Test Range
            start = i * fold_size
            stop = start + fold_size if i < self.n_splits - 1 else len(X)
            test_indices = indices[start:stop]
            
            # Define Embargo (Safety Gap)
            # We discard the 'embargo_days' immediately AFTER the test set.
            # This prevents the model from learning from the immediate future of the test set 
            # (which would track the test set via momentum).
            embargo_end = min(len(X), stop + self.embargo_days)
            embargo_indices = indices[stop:embargo_end]
            
            # Define Train Indices
            # Train = All indices - (Test + Embargo)
            # We explicitly exclude the Test block AND the Embargo block.
            exclude_indices = np.concatenate([test_indices, embargo_indices])
            train_indices = np.setdiff1d(indices, exclude_indices)
            
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
