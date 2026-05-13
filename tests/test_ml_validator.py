"""
Tests for ML Validation Framework

Ensures models meet quality gates before deployment.
"""

import pytest
import numpy as np
import pandas as pd
from train.ml_validator import MLValidator, validate_symbol_for_training


class TestMLValidatorBasics:
    """Test basic validation functionality."""
    
    def test_good_predictions(self):
        """Validate good predictions that meet all gates."""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0] * 20)  # 20 samples, 50% positive
        y_pred = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0] * 20)  # Perfect predictions
        
        validator = MLValidator()
        is_valid, report = validator.validate_predictions(y_true, y_pred, symbol="BTC/USDT")
        
        assert is_valid, f"Should be valid: {report['failures']}"
        assert report["metrics"]["f1"] == pytest.approx(1.0)
        assert report["metrics"]["precision"] == pytest.approx(1.0)
    
    def test_poor_f1_score(self):
        """Reject predictions with F1 < threshold."""
        y_true = np.array([1, 0, 1, 0, 1, 0] * 20)  # 120 samples
        y_pred = np.array([0, 0, 0, 0, 0, 0] * 20)  # All zeros (terrible)
        
        validator = MLValidator(min_f1=0.50)
        is_valid, report = validator.validate_predictions(y_true, y_pred, symbol="BTC/USDT")
        
        assert not is_valid, "Should reject low F1"
        assert any("F1" in str(f) for f in report["failures"])
    
    def test_low_precision(self):
        """Reject predictions with precision < threshold."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 20)  # 160 samples
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1] * 20)  # Always predict 1 (low precision)
        
        validator = MLValidator(min_precision=0.55)
        is_valid, report = validator.validate_predictions(y_true, y_pred, symbol="ETH/USDT")
        
        assert not is_valid
        assert any("Precision" in str(f) for f in report["failures"])
    
    def test_insufficient_samples(self):
        """Reject if not enough validation samples."""
        y_true = np.array([0, 1, 0, 1, 1])  # Only 5 samples
        y_pred = np.array([0, 1, 0, 1, 1])
        
        validator = MLValidator(min_samples=100)
        is_valid, report = validator.validate_predictions(y_true, y_pred, symbol="SOL/USDT")
        
        assert not is_valid
        assert any("samples" in str(f).lower() for f in report["failures"])
    
    def test_severe_class_imbalance_too_few_positive(self):
        """Reject if positive class ratio too low."""
        y_true = np.array([0] * 95 + [1] * 5)  # Only 5% positive
        y_pred = np.array([0] * 95 + [1] * 5)
        
        validator = MLValidator(
            min_samples=50,
            min_precision=0.50,
            min_f1=0.10
        )
        is_valid, report = validator.validate_predictions(y_true, y_pred, symbol="AVAX/USDT")
        
        assert not is_valid
        assert any("positive" in str(f).lower() for f in report["failures"])
    
    def test_severe_class_imbalance_too_many_positive(self):
        """Reject if positive class ratio too high."""
        y_true = np.array([1] * 90 + [0] * 10)  # 90% positive
        y_pred = np.array([1] * 90 + [0] * 10)
        
        validator = MLValidator()
        is_valid, report = validator.validate_predictions(y_true, y_pred, symbol="LINK/USDT")
        
        assert not is_valid
        assert any("positive" in str(f).lower() for f in report["failures"])


class TestClassBalanceValidation:
    """Test class balance validation before training."""
    
    def test_balanced_class_distribution(self):
        """Validate balanced classes."""
        y = np.array([0, 1, 0, 1, 0, 1] * 20)  # 50% positive
        
        validator = MLValidator()
        is_valid, reason = validator.validate_class_balance(y)
        
        assert is_valid
        assert "50.0%" in reason
    
    def test_acceptable_imbalance(self):
        """Validate classes with acceptable imbalance."""
        y = np.array([0] * 70 + [1] * 30)  # 30% positive
        
        validator = MLValidator()
        is_valid, reason = validator.validate_class_balance(y)
        
        assert is_valid
    
    def test_too_few_positive(self):
        """Reject with too few positive samples."""
        y = np.array([0] * 95 + [1] * 5)  # Only 5% positive
        
        validator = MLValidator()
        is_valid, reason = validator.validate_class_balance(y)
        
        assert not is_valid
        assert "positive" in reason.lower()
    
    def test_too_many_positive(self):
        """Reject with too many positive samples."""
        y = np.array([1] * 90 + [0] * 10)  # 90% positive
        
        validator = MLValidator()
        is_valid, reason = validator.validate_class_balance(y)
        
        assert not is_valid
        assert "positive" in reason.lower()


class TestTrainTestSplitValidation:
    """Test time-based train/test split validation."""
    
    def test_valid_time_based_split(self):
        """Validate proper time-based split."""
        timestamps = pd.date_range("2023-01-01", periods=1000, freq="h")
        df = pd.DataFrame({
            "time": timestamps,
            "value": np.random.randn(1000),
        })
        
        validator = MLValidator()
        is_valid, reason = validator.validate_train_test_split(df, train_size=0.8)
        
        assert is_valid
        assert "Valid" in reason
    
    def test_non_chronological_data(self):
        """Reject if data not in chronological order."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "time": np.random.permutation(dates),
            "value": np.random.randn(100),
        })
        
        validator = MLValidator()
        is_valid, reason = validator.validate_train_test_split(df)
        
        assert not is_valid
        assert "sorted" in reason.lower()
    
    def test_missing_timestamp_column(self):
        """Reject if timestamp column missing."""
        df = pd.DataFrame({
            "value": np.random.randn(100),
        })
        
        validator = MLValidator()
        is_valid, reason = validator.validate_train_test_split(
            df,
            timestamp_col="nonexistent"
        )
        
        assert not is_valid
        assert "not found" in reason.lower()


class TestWalkForwardSplits:
    """Test walk-forward validation split generation."""
    
    def test_walk_forward_splits_generation(self):
        """Generate valid walk-forward splits."""
        df = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=1000, freq="h"),
        })
        
        validator = MLValidator()
        splits = validator.get_walk_forward_splits(df, n_folds=5, min_train_size=200)
        
        assert len(splits) == 5
        
        # Verify splits are sequential
        prev_test_end = 200
        for train_end, test_end in splits:
            assert train_end >= prev_test_end
            assert test_end > train_end
            prev_test_end = test_end
    
    def test_walk_forward_no_overlap(self):
        """Ensure walk-forward splits don't overlap."""
        df = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=500, freq="h"),
        })
        
        validator = MLValidator()
        splits = validator.get_walk_forward_splits(df, n_folds=3, min_train_size=100)
        
        # Each test set should only test once
        tested_indices = set()
        for train_end, test_end in splits:
            test_indices = set(range(train_end, test_end))
            # No overlap with previously tested
            assert len(test_indices & tested_indices) == 0
            tested_indices.update(test_indices)


class TestSymbolValidationQuick:
    """Test quick symbol validation for training."""
    
    def test_valid_symbol_data(self):
        """Validate good symbol data."""
        df = pd.DataFrame({
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
        })
        y = np.array([0, 1, 0, 1] * 50)  # 50% positive, 200 samples
        
        is_valid, reason = validate_symbol_for_training(df, y, "BTC/USDT")
        
        assert is_valid
        assert "Valid" in reason
    
    def test_insufficient_samples_for_symbol(self):
        """Reject symbol with insufficient samples."""
        df = pd.DataFrame({
            "feature1": np.random.randn(20),
        })
        y = np.array([0, 1] * 10)
        
        is_valid, reason = validate_symbol_for_training(df, y, "SHIB/USDT")
        
        assert not is_valid
        assert "samples" in reason.lower()
    
    def test_imbalanced_symbol_data(self):
        """Reject symbol with imbalanced classes."""
        df = pd.DataFrame({
            "feature1": np.random.randn(200),
        })
        y = np.array([0] * 190 + [1] * 10)  # Only 5% positive
        
        is_valid, reason = validate_symbol_for_training(df, y, "DOGE/USDT")
        
        assert not is_valid


class TestValidationReport:
    """Test validation report generation."""
    
    def test_report_generation(self):
        """Validate report contains all metrics."""
        y_true = np.array([0, 1, 0, 1, 1] * 30)  # 150 samples
        y_pred = np.array([0, 1, 0, 1, 1] * 30)  # Perfect predictions
        
        validator = MLValidator()
        is_valid, report = validator.validate_predictions(y_true, y_pred, symbol="TEST")
        
        assert "metrics" in report
        assert "f1" in report["metrics"]
        assert "precision" in report["metrics"]
        assert "recall" in report["metrics"]
        assert "sample_count" in report["metrics"]
        assert "positive_ratio" in report["metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
