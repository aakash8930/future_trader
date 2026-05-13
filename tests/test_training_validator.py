"""
Tests for Phase 8: Training Integration - Symbol Validation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

from train.training_validator import (
    TrainingSymbolFilter,
    TrainingPipelineValidator,
)


def create_test_df(
    length: int = 500,
    volatility: float = 0.02,
    close_start: float = 100.0,
) -> pd.DataFrame:
    """Create test dataframe"""
    dates = pd.date_range(start="2024-01-01", periods=length, freq="h")
    closes = [close_start]

    for i in range(1, length):
        change = np.random.randn() * volatility
        closes.append(closes[-1] * (1 + change))

    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000000] * length,
        },
        index=dates,
    )


class TestTrainingSymbolFilter:
    """Test symbol filtering"""

    def test_filter_creation(self):
        f = TrainingSymbolFilter()
        assert f.min_candles == 500
        assert f.min_daily_volume_usd == 50000

    def test_whitelist_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            whitelist_file = Path(tmpdir) / "symbols.json"
            with open(whitelist_file, "w") as f:
                json.dump({"approved_symbols": ["BTC/USDT", "ETH/USDT"]}, f)

            filter = TrainingSymbolFilter(whitelist_file=str(whitelist_file))
            assert "BTC/USDT" in filter.approved_symbols
            assert "ETH/USDT" in filter.approved_symbols

    def test_whitelist_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            whitelist_file = Path(tmpdir) / "symbols.json"
            filter = TrainingSymbolFilter(whitelist_file=str(whitelist_file))
            filter.save_whitelist(["BTC/USDT", "ETH/USDT"])

            assert whitelist_file.exists()
            with open(whitelist_file) as f:
                data = json.load(f)
            assert set(data["approved_symbols"]) == {"BTC/USDT", "ETH/USDT"}

    def test_reject_leveraged_token_bull(self):
        f = TrainingSymbolFilter()
        valid, reason = f.check_symbol("BULLBTC/USDT")
        assert not valid
        assert "Leveraged" in reason

    def test_reject_leveraged_token_bear(self):
        f = TrainingSymbolFilter()
        valid, reason = f.check_symbol("BEARETH/USDT")
        assert not valid
        assert "Leveraged" in reason

    def test_reject_non_usdt(self):
        f = TrainingSymbolFilter()
        valid, reason = f.check_symbol("BTC/USDC")
        assert not valid
        assert "USDT" in reason

    def test_reject_insufficient_candles(self):
        f = TrainingSymbolFilter(min_candles=500)
        df = create_test_df(length=100)  # Only 100 candles
        valid, reason = f.check_symbol("BTC/USDT", df=df)
        assert not valid
        assert "Insufficient" in reason

    def test_accept_sufficient_candles(self):
        f = TrainingSymbolFilter(min_candles=500)
        df = create_test_df(length=600)  # 600 candles
        valid, reason = f.check_symbol("BTC/USDT", df=df)
        # May fail on other criteria, but not candles
        assert "Insufficient" not in reason

    def test_reject_low_volatility(self):
        f = TrainingSymbolFilter(
            min_candles=100, min_volatility_pct=0.005, max_volatility_pct=0.25
        )
        df = create_test_df(length=100, volatility=0.0001)  # Very stable
        valid, reason = f.check_symbol("BTC/USDT", df=df)
        assert not valid
        assert "stable" in reason.lower()

    def test_reject_high_volatility(self):
        f = TrainingSymbolFilter(
            min_candles=100, min_volatility_pct=0.001, max_volatility_pct=0.05
        )
        df = create_test_df(length=100, volatility=0.30)  # Very volatile
        valid, reason = f.check_symbol("BTC/USDT", df=df)
        assert not valid
        assert "volatile" in reason.lower()

    def test_accept_normal_volatility(self):
        f = TrainingSymbolFilter(
            min_candles=100, min_volatility_pct=0.001, max_volatility_pct=0.25
        )
        df = create_test_df(length=100, volatility=0.02)
        valid, reason = f.check_symbol("BTC/USDT", df=df)
        assert valid

    def test_reject_low_volume(self):
        f = TrainingSymbolFilter(min_candles=100, min_daily_volume_usd=50000)
        df = create_test_df(length=100)
        volume_data = {"BTC/USDT": 10000}  # Too low
        valid, reason = f.check_symbol("BTC/USDT", df=df, volume_data=volume_data)
        assert not valid
        assert "volume" in reason.lower()

    def test_accept_sufficient_volume(self):
        f = TrainingSymbolFilter(min_candles=100, min_daily_volume_usd=50000)
        df = create_test_df(length=100)
        volume_data = {"BTC/USDT": 100000}  # Sufficient
        valid, reason = f.check_symbol("BTC/USDT", df=df, volume_data=volume_data)
        assert valid

    def test_whitelist_enforcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            whitelist_file = Path(tmpdir) / "symbols.json"
            with open(whitelist_file, "w") as f:
                json.dump({"approved_symbols": ["BTC/USDT"]}, f)

            f = TrainingSymbolFilter(
                min_candles=100,
                whitelist_file=str(whitelist_file)
            )

            # BTC in whitelist (with minimal data requirement)
            df = create_test_df(length=100)
            valid, reason = f.check_symbol("BTC/USDT", df=df)
            assert valid

            # ETH not in whitelist
            valid, reason = f.check_symbol("ETH/USDT", df=df)
            assert not valid
            assert "whitelist" in reason.lower()

    def test_filter_symbols_list(self):
        f = TrainingSymbolFilter(min_candles=100)
        symbols = ["BTC/USDT", "ETH/USDT", "BULLBTC/USDT"]
        volume_data = {"BTC/USDT": 100000, "ETH/USDT": 50000}

        approved, rejections = f.filter_symbols(symbols, volume_data)

        assert len(approved) == 0 or "BTC/USDT" in approved
        assert "BULLBTC/USDT" in rejections


class TestTrainingPipelineValidator:
    """Test pipeline validation"""

    def test_validator_creation(self):
        validator = TrainingPipelineValidator()
        assert validator.symbol_filter is not None

    def test_validate_training_data_insufficient(self):
        validator = TrainingPipelineValidator()
        df = create_test_df(length=50)

        valid, reason = validator.validate_training_data(
            "BTC/USDT", df, min_samples=100
        )

        assert not valid
        assert "Insufficient" in reason

    def test_validate_training_data_sufficient(self):
        validator = TrainingPipelineValidator()
        df = create_test_df(length=500)

        valid, reason = validator.validate_training_data(
            "BTC/USDT", df, min_samples=100
        )

        # May fail on symbol qualification, but not data
        assert "Insufficient" not in reason or "qualified" in reason.lower()

    def test_validate_training_data_single_class(self):
        validator = TrainingPipelineValidator()
        df = create_test_df(length=500)  # Sufficient data
        y = np.ones(500)  # All class 1

        valid, reason = validator.validate_training_data(
            "BTC/USDT", df, y=y, min_samples=50
        )

        assert not valid
        assert "single class" in reason.lower()

    def test_validate_training_data_imbalanced(self):
        validator = TrainingPipelineValidator()
        df = create_test_df(length=500)  # Sufficient data
        y = np.concatenate([np.zeros(475), np.ones(25)])  # 95/5 split

        valid, reason = validator.validate_training_data(
            "BTC/USDT", df, y=y, min_samples=50
        )

        assert not valid
        assert "imbalance" in reason.lower()

    def test_validate_training_data_balanced(self):
        validator = TrainingPipelineValidator()
        df = create_test_df(length=100)
        y = np.concatenate([np.zeros(40), np.ones(60)])  # 40/60 split

        valid, reason = validator.validate_training_data(
            "BTC/USDT", df, y=y, min_samples=50
        )

        # May still fail on symbol qualification
        assert "imbalance" not in reason.lower()

    def test_pre_training_check(self):
        validator = TrainingPipelineValidator()

        symbols = ["BTC/USDT", "ETH/USDT", "BULLBTC/USDT"]
        data_map = {
            "BTC/USDT": create_test_df(length=500),
            "ETH/USDT": create_test_df(length=500),
            "BULLBTC/USDT": create_test_df(length=500),
        }

        report = validator.pre_training_check(symbols, data_map)

        assert "total_symbols" in report
        assert "approved_symbols" in report
        assert "rejected_symbols" in report
        assert "BULLBTC/USDT" in report["rejected_symbols"]

    def test_pre_training_check_missing_data(self):
        validator = TrainingPipelineValidator()

        symbols = ["BTC/USDT", "UNKNOWN/USDT"]
        data_map = {"BTC/USDT": create_test_df(length=500)}

        report = validator.pre_training_check(symbols, data_map)

        assert "UNKNOWN/USDT" in report["rejected_symbols"]

    def test_approval_summary(self):
        validator = TrainingPipelineValidator()

        symbols = ["BTC/USDT", "ETH/USDT"]
        data_map = {
            "BTC/USDT": create_test_df(length=500),
            "ETH/USDT": create_test_df(length=50),  # Too short
        }

        report = validator.pre_training_check(symbols, data_map)

        # Both should be in report
        assert report["total_symbols"] == 2
        assert len(report["rejected_symbols"]) > 0
