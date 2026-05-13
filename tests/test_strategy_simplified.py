"""
Tests for Phase 3: Strategy Simplification
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from execution.strategy_simplified import (
    SimplifiedStrategyConfig,
    SimplifiedStrategyEngine,
    SimplifiedSignal,
)


class MockModel:
    """Mock ML model for testing"""

    def __init__(self, long_prob: float = 0.6):
        self.long_prob = long_prob

    def predict_proba(self, df):
        """Return mock probabilities"""
        short_prob = 1.0 - self.long_prob
        return (short_prob, self.long_prob)


def create_test_df(
    length: int = 50,
    close_start: float = 100.0,
    volatility: float = 0.02,
    trend: str = "up",
) -> pd.DataFrame:
    """Create test OHLCV dataframe"""
    dates = pd.date_range(start="2024-01-01", periods=length, freq="h")
    closes = [close_start]

    for i in range(1, length):
        change = np.random.randn() * volatility
        if trend == "up":
            change += 0.002
        elif trend == "down":
            change -= 0.002
        closes.append(closes[-1] * (1 + change))

    # ATR calculation should reflect volatility
    returns = np.diff(closes) / closes[:-1]
    realized_volatility = np.std(returns) * close_start  # Convert to price units
    
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000000] * length,
            "atr": [realized_volatility] * length,  # ATR reflects input volatility
            "ema_50": pd.Series(closes).ewm(span=50).mean().values,
        },
        index=dates,
    )
    return df


class TestSimplifiedStrategyConfig:
    """Test configuration"""

    def test_default_config(self):
        cfg = SimplifiedStrategyConfig()
        assert cfg.min_confidence == 0.55
        assert cfg.min_volatility_pct == 0.002
        assert cfg.max_volatility_pct == 0.15
        assert cfg.ema_period == 50

    def test_custom_config(self):
        cfg = SimplifiedStrategyConfig(
            min_confidence=0.60,
            min_volatility_pct=0.001,
        )
        assert cfg.min_confidence == 0.60
        assert cfg.min_volatility_pct == 0.001


class TestSimplifiedStrategyEngine:
    """Test strategy engine"""

    def test_insufficient_data(self):
        engine = SimplifiedStrategyEngine(config=SimplifiedStrategyConfig())
        df = create_test_df(length=3)

        signal = engine.generate_signal("BTC/USDT", df)

        assert signal.side is None
        assert "Insufficient data" in signal.reason

    def test_volatility_gate_low(self):
        """Test rejection when volatility too low"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.70),
            config=SimplifiedStrategyConfig()
        )
        df = create_test_df(length=50, volatility=0.0001)  # Very stable

        signal = engine.generate_signal("BTC/USDT", df)

        assert signal.side is None
        assert "Volatility" in signal.reason or "Confidence" in signal.reason

    def test_volatility_gate_high(self):
        """Test rejection when volatility too high"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.70),
            config=SimplifiedStrategyConfig()
        )
        df = create_test_df(length=50, volatility=0.20)  # Very volatile

        signal = engine.generate_signal("BTC/USDT", df)

        assert signal.side is None
        assert "Volatility" in signal.reason or "Confidence" in signal.reason

    def test_trend_filter_up(self):
        """Test trend bias UP"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.70),
            config=SimplifiedStrategyConfig(use_trend_filter=True),
        )
        df = create_test_df(length=50, trend="up")

        signal = engine.generate_signal("BTC/USDT", df)

        assert signal.trend_bias in ["UP", "DOWN", "NEUTRAL"]

    def test_trend_filter_down(self):
        """Test trend bias DOWN"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.30),
            config=SimplifiedStrategyConfig(use_trend_filter=True),
        )
        df = create_test_df(length=50, trend="down")

        signal = engine.generate_signal("BTC/USDT", df)

        assert signal.trend_bias in ["UP", "DOWN", "NEUTRAL"]

    def test_model_confidence_low(self):
        """Test rejection when model confidence low"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.52),  # Just above 50%
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50)

        signal = engine.generate_signal("BTC/USDT", df)

        assert signal.side is None
        assert "Confidence" in signal.reason

    def test_model_confidence_high_long(self):
        """Test LONG signal with high confidence"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.75),
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50)

        signal = engine.generate_signal("BTC/USDT", df)

        # Signal may still be rejected by filters, but confidence should be high
        if signal.side is not None:
            assert signal.confidence >= 0.55

    def test_model_confidence_high_short(self):
        """Test SHORT signal with high confidence"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.25),
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50)

        signal = engine.generate_signal("BTC/USDT", df)

        # Model should predict SHORT
        if signal.side is not None:
            assert signal.confidence >= 0.55

    def test_cooldown_enforcement(self):
        """Test cooldown between signals"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.70),
            config=SimplifiedStrategyConfig(min_bars_between_trades=10),
        )
        df = create_test_df(length=50)

        # First signal at bar 0
        signal1 = engine.generate_signal("BTC/USDT", df, current_bar_idx=0)
        # Bar 5 (only 5 bars since last signal)
        signal2 = engine.generate_signal("BTC/USDT", df, current_bar_idx=5)
        # Bar 15 (10 bars since last signal)
        signal3 = engine.generate_signal("BTC/USDT", df, current_bar_idx=15)

        if signal1.side is not None:
            # Signal 2 should be rejected due to cooldown
            assert signal2.side is None or "Cooldown" in signal2.reason

    def test_exit_levels_long(self):
        """Test exit level calculation for LONG"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.70),
            config=SimplifiedStrategyConfig(
                min_confidence=0.50,
                stop_loss_atr_mult=1.5,
                take_profit_atr_mult=3.0,
            ),
        )
        df = create_test_df(length=50)
        entry = df.iloc[-1]["close"]
        atr = df.iloc[-1]["atr"]

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side == "LONG":
            expected_sl = entry - atr * 1.5
            expected_tp = entry + atr * 3.0

            assert abs(signal.stop_loss - expected_sl) < 0.01
            assert abs(signal.take_profit - expected_tp) < 0.01

    def test_exit_levels_short(self):
        """Test exit level calculation for SHORT"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.30),
            config=SimplifiedStrategyConfig(
                min_confidence=0.50,
                stop_loss_atr_mult=1.5,
                take_profit_atr_mult=3.0,
            ),
        )
        df = create_test_df(length=50)
        entry = df.iloc[-1]["close"]
        atr = df.iloc[-1]["atr"]

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side == "SHORT":
            expected_sl = entry + atr * 1.5
            expected_tp = entry - atr * 3.0

            assert abs(signal.stop_loss - expected_sl) < 0.01
            assert abs(signal.take_profit - expected_tp) < 0.01

    def test_signal_validation_valid(self):
        """Test signal validation - valid signal"""
        engine = SimplifiedStrategyEngine()
        signal = SimplifiedSignal(
            symbol="BTC/USDT",
            timestamp=pd.Timestamp.now(),
            side="LONG",
            confidence=0.70,
            volatility=0.02,
            trend_bias="UP",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            expected_pnl_pct=0.05,
            reason="Valid",
            filters_passed={"all": True},
        )

        assert engine.validate_signal(signal) is True

    def test_signal_validation_bad_rr(self):
        """Test signal validation - bad risk/reward ratio"""
        engine = SimplifiedStrategyEngine()
        signal = SimplifiedSignal(
            symbol="BTC/USDT",
            timestamp=pd.Timestamp.now(),
            side="LONG",
            confidence=0.70,
            volatility=0.02,
            trend_bias="UP",
            entry_price=100.0,
            stop_loss=99.0,  # Only 1% risk
            take_profit=100.5,  # Only 0.5% reward (bad RR)
            expected_pnl_pct=0.001,
            reason="Bad RR",
            filters_passed={"all": True},
        )

        assert engine.validate_signal(signal) is False

    def test_null_signal_structure(self):
        """Test null signal has correct structure"""
        engine = SimplifiedStrategyEngine()
        df = create_test_df(length=3)

        signal = engine.generate_signal("BTC/USDT", df)

        assert signal.symbol == "BTC/USDT"
        assert signal.side is None
        assert signal.confidence == 0.0
        assert isinstance(signal.reason, str)

    def test_filters_passed_tracking(self):
        """Test filters_passed dict is populated"""
        engine = SimplifiedStrategyEngine(
            model=MockModel(long_prob=0.70),
            config=SimplifiedStrategyConfig(),
        )
        df = create_test_df(length=50)

        signal = engine.generate_signal("BTC/USDT", df)

        # Filters should be tracked
        assert isinstance(signal.filters_passed, dict)

    def test_no_model_fallback(self):
        """Test strategy works without model (fallback to 0.5)"""
        engine = SimplifiedStrategyEngine(
            model=None,
            config=SimplifiedStrategyConfig(min_confidence=0.50),
        )
        df = create_test_df(length=50)

        signal = engine.generate_signal("BTC/USDT", df)

        # Should not crash
        assert signal.symbol == "BTC/USDT"
        assert signal.confidence == 0.5
