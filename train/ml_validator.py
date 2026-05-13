# train/ml_validator.py
"""
ML Validation Framework

Ensures models have real edge before deployment:
- Proper time-based train/test split (NO leakage)
- Walk-forward validation
- F1/Precision/Recall gatekeeping
- Probability calibration checks
- Class balance validation
- Sample size validation
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)
import pandas as pd


class MLValidator:
    """
    Comprehensive ML validation to prevent overfit and false edges.
    """
    
    # Minimum quality gates (configurable)
    MIN_F1_SCORE = 0.50
    MIN_PRECISION = 0.55
    MIN_RECALL = 0.40
    MIN_SAMPLE_COUNT = 100  # Minimum trades in validation set
    MIN_POSITIVE_CLASS_RATIO = 0.15  # At least 15% positive class
    MAX_POSITIVE_CLASS_RATIO = 0.85  # At most 85% (avoid severe imbalance)
    
    def __init__(
        self,
        min_f1: float = 0.50,
        min_precision: float = 0.55,
        min_recall: float = 0.40,
        min_samples: int = 100,
    ):
        """Initialize validator with custom thresholds."""
        self.MIN_F1_SCORE = min_f1
        self.MIN_PRECISION = min_precision
        self.MIN_RECALL = min_recall
        self.MIN_SAMPLE_COUNT = min_samples
        
        self.validation_report = {}
    
    def validate_predictions(
        self,
        y_true: np.ndarray,
        y_pred_binary: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        symbol: str = "UNKNOWN",
    ) -> Tuple[bool, Dict]:
        """
        Comprehensive validation of model predictions.
        
        Args:
            y_true: True labels (0/1)
            y_pred_binary: Binary predictions (0/1)
            y_pred_proba: Probability predictions (0-1) [optional]
            symbol: Symbol being validated (for reporting)
            
        Returns:
            (is_valid, report) tuple
            - is_valid: True if all gates passed
            - report: Dict with all metrics and failure reasons
        """
        report = {
            "symbol": symbol,
            "is_valid": True,
            "failures": [],
            "metrics": {},
        }
        
        # Check 1: Sufficient samples
        if len(y_true) < self.MIN_SAMPLE_COUNT:
            report["is_valid"] = False
            report["failures"].append(
                f"Insufficient samples: {len(y_true)} < {self.MIN_SAMPLE_COUNT}"
            )
        
        # Check 2: Class balance
        positive_count = int(y_true.sum())
        total_count = len(y_true)
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        
        if positive_ratio < self.MIN_POSITIVE_CLASS_RATIO:
            report["is_valid"] = False
            report["failures"].append(
                f"Too few positive samples: {positive_ratio:.1%} < {self.MIN_POSITIVE_CLASS_RATIO:.1%}"
            )
        
        if positive_ratio > self.MAX_POSITIVE_CLASS_RATIO:
            report["is_valid"] = False
            report["failures"].append(
                f"Too many positive samples: {positive_ratio:.1%} > {self.MAX_POSITIVE_CLASS_RATIO:.1%}"
            )
        
        # Calculate metrics
        try:
            f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
            precision = float(precision_score(y_true, y_pred_binary, zero_division=0))
            recall = float(recall_score(y_true, y_pred_binary, zero_division=0))
            
            report["metrics"] = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "sample_count": total_count,
                "positive_ratio": positive_ratio,
            }
            
            # Check 3: F1 Score
            if f1 < self.MIN_F1_SCORE:
                report["is_valid"] = False
                report["failures"].append(
                    f"F1 score too low: {f1:.3f} < {self.MIN_F1_SCORE}"
                )
            
            # Check 4: Precision
            if precision < self.MIN_PRECISION:
                report["is_valid"] = False
                report["failures"].append(
                    f"Precision too low: {precision:.3f} < {self.MIN_PRECISION}"
                )
            
            # Check 5: Recall
            if recall < self.MIN_RECALL:
                report["is_valid"] = False
                report["failures"].append(
                    f"Recall too low: {recall:.3f} < {self.MIN_RECALL}"
                )
            
            # Optional: AUC-ROC if probabilities provided
            if y_pred_proba is not None:
                try:
                    auc = float(roc_auc_score(y_true, y_pred_proba))
                    report["metrics"]["auc_roc"] = auc
                except Exception:
                    pass
        
        except Exception as e:
            report["is_valid"] = False
            report["failures"].append(f"Error calculating metrics: {str(e)}")
        
        self.validation_report = report
        return (report["is_valid"], report)
    
    def validate_class_balance(
        self,
        y: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Validate class balance before training.
        
        Args:
            y: Labels (0/1)
            
        Returns:
            (is_balanced, reason) tuple
        """
        total = len(y)
        positive_count = int(y.sum())
        positive_ratio = positive_count / total if total > 0 else 0
        
        if positive_ratio < self.MIN_POSITIVE_CLASS_RATIO:
            return (
                False,
                f"Not enough positive samples: {positive_ratio:.1%} < {self.MIN_POSITIVE_CLASS_RATIO:.1%}"
            )
        
        if positive_ratio > self.MAX_POSITIVE_CLASS_RATIO:
            return (
                False,
                f"Too many positive samples: {positive_ratio:.1%} > {self.MAX_POSITIVE_CLASS_RATIO:.1%}"
            )
        
        return (True, f"Balanced: {positive_ratio:.1%} positive")
    
    def validate_train_test_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.8,
        timestamp_col: str = "time",
    ) -> Tuple[bool, str]:
        """
        Validate that train/test split is time-based (no leakage).
        
        Args:
            df: Full dataframe with timestamp
            train_size: Ratio for training set
            timestamp_col: Name of timestamp column
            
        Returns:
            (is_valid, reason) tuple
        """
        if timestamp_col not in df.columns:
            return (False, f"Timestamp column '{timestamp_col}' not found")
        
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
        
        # Check if data is chronologically sorted
        is_sorted = df[timestamp_col].is_monotonic_increasing
        if not is_sorted:
            return (False, "Data is not chronologically sorted")
        
        # Split point should be time-based (sequential)
        split_idx = int(len(df) * train_size)
        train_end = df[timestamp_col].iloc[split_idx]
        test_start = df[timestamp_col].iloc[split_idx + 1]
        
        if train_end >= test_start:
            return (False, "Train/test split not properly separated in time")
        
        return (True, f"Valid time-based split: train until {train_end}, test from {test_start}")
    
    def get_walk_forward_splits(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        timestamp_col: str = "time",
        min_train_size: int = 100,
    ) -> List[Tuple[int, int]]:
        """
        Generate walk-forward validation splits.
        
        Ensures no future data leaks into training.
        
        Args:
            df: Full dataframe
            n_folds: Number of folds
            timestamp_col: Timestamp column name
            min_train_size: Minimum training samples per fold
            
        Returns:
            List of (train_end_idx, test_end_idx) tuples
        """
        total = len(df)
        test_size = (total - min_train_size) // (n_folds + 1)
        
        splits = []
        train_end = min_train_size
        
        for fold in range(n_folds):
            test_end = train_end + test_size
            if test_end > total:
                break
            splits.append((train_end, test_end))
            train_end = test_end
        
        return splits
    
    def generate_report(self) -> Dict:
        """Generate validation report."""
        return self.validation_report


def validate_symbol_for_training(
    df: pd.DataFrame,
    y: np.ndarray,
    symbol: str,
    validator: Optional[MLValidator] = None,
) -> Tuple[bool, str]:
    """
    Quick validation of training data for a symbol.
    
    Args:
        df: Feature dataframe
        y: Labels
        symbol: Symbol being trained
        validator: MLValidator instance
        
    Returns:
        (is_valid, reason) tuple
    """
    if validator is None:
        validator = MLValidator()
    
    # Check class balance
    is_balanced, balance_msg = validator.validate_class_balance(y)
    if not is_balanced:
        return (False, f"{symbol}: {balance_msg}")
    
    # Check sample count
    if len(y) < validator.MIN_SAMPLE_COUNT:
        return (
            False,
            f"{symbol}: Only {len(y)} samples, need at least {validator.MIN_SAMPLE_COUNT}"
        )
    
    return (True, f"{symbol}: Valid for training")
