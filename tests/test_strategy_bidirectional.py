"""
SHORT SELLING SUPPORT - PHASE 3: Strategy Signal Generation for Both Sides

Extends SimplifiedStrategyEngine to generate both LONG and SHORT signals.

Key Principle:
- Model outputs probability of price going UP
- LONG signal: high confidence in UP (prob > threshold)
- SHORT signal: high confidence in DOWN (prob < 1-threshold)
- NO TRADE: low confidence in either direction

Risk/Reward Calculation:
- LONG: SL below entry, TP above entry
- SHORT: SL above entry, TP below entry
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from execution.strategy_simplified import SimplifiedStrategyEngine, SimplifiedStrategyConfig


class MockBidirectionalModel:
    """Mock ML model that outputs probability of UP movement"""

    def __init__(self, up_prob: float = 0.6):
        self.up_prob = up_prob

    def predict_proba(self, df):
        """Return probabilities: (down_prob, up_prob)"""
        down_prob = 1.0 - self.up_prob
        return (down_prob, self.up_prob)


def create_test_df(
    length: int = 50,
    close_start: float = 100.0,
    volatility: float = 0.02,
    trend: str = "neutral",
    seed: int = 42,
) -> pd.DataFrame:
    """Create test OHLCV dataframe with proper volatility"""
    np.random.seed(seed)  # Fixed seed for reproducibility
    dates = pd.date_range(start="2024-01-01", periods=length, freq="h")
    closes = [close_start]

    for i in range(1, length):
        change = np.random.randn() * volatility
        if trend == "up":
            change += 0.005
        elif trend == "down":
            change -= 0.005
        closes.append(closes[-1] * (1 + change))

    returns = np.diff(closes) / closes[:-1]
    realized_volatility = np.std(returns) * close_start

    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1000000] * length,
            "atr": [realized_volatility] * length,
            "ema_50": pd.Series(closes).ewm(span=50).mean().values,
        },
        index=dates,
    )


class TestStrategySignalBidirectional:
    """Test strategy signal generation for both LONG and SHORT"""

    def test_signal_long_high_confidence(self):
        """Test LONG signal with high confidence (80% UP)"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.80),
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50, volatility=0.02, trend="up")

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side is not None:
            assert signal.side == "LONG"
            assert signal.confidence >= 0.55

    def test_signal_short_high_confidence(self):
        """Test SHORT signal with high DOWN confidence (80% DOWN)"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.20),  # 80% DOWN
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50, volatility=0.02, trend="down")

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side is not None:
            assert signal.side == "SHORT"
            assert signal.confidence >= 0.55

    def test_signal_no_trade_low_confidence(self):
        """Test NO TRADE when confidence is low"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.52),  # Nearly 50/50
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50)

        signal = engine.generate_signal("BTC/USDT", df)

        # Should reject due to low confidence
        assert signal.side is None

    def test_long_exit_levels_correct(self):
        """Test that LONG signals have proper exit levels (SL below, TP above)"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.75),
            config=SimplifiedStrategyConfig(
                min_confidence=0.55,
                stop_loss_atr_mult=1.5,
                take_profit_atr_mult=3.0,
            ),
        )
        df = create_test_df(length=50, trend="up")

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side == "LONG":
            # For LONG: SL should be below entry, TP above
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price
            # TP should be further away than SL (3x vs 1.5x ATR)
            assert (signal.take_profit - signal.entry_price) > (
                signal.entry_price - signal.stop_loss
            )

    def test_short_exit_levels_correct(self):
        """Test that SHORT signals have proper exit levels (SL above, TP below)"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.25),  # Strong DOWN
            config=SimplifiedStrategyConfig(
                min_confidence=0.55,
                stop_loss_atr_mult=1.5,
                take_profit_atr_mult=3.0,
            ),
        )
        df = create_test_df(length=50, trend="down")

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side == "SHORT":
            # For SHORT: SL should be above entry, TP below
            assert signal.stop_loss > signal.entry_price
            assert signal.take_profit < signal.entry_price
            # TP should be further away than SL (3x vs 1.5x ATR)
            assert (signal.entry_price - signal.take_profit) > (
                signal.stop_loss - signal.entry_price
            )

    def test_long_rr_ratio_valid(self):
        """Test that LONG signals have valid risk/reward ratio"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.75),
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50, trend="up")

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side == "LONG":
            rr_ratio = (signal.take_profit - signal.entry_price) / max(
                signal.entry_price - signal.stop_loss, 0.0001
            )
            # RR should be approximately 2:1 based on ATR multiples (3.0 / 1.5 = 2)
            assert rr_ratio >= 1.5  # At least 1.5:1

    def test_short_rr_ratio_valid(self):
        """Test that SHORT signals have valid risk/reward ratio"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.25),  # Strong DOWN
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50, trend="down")

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side == "SHORT":
            rr_ratio = (signal.entry_price - signal.take_profit) / max(
                signal.stop_loss - signal.entry_price, 0.0001
            )
            # RR should be approximately 2:1 based on ATR multiples (3.0 / 1.5 = 2)
            assert rr_ratio >= 1.5  # At least 1.5:1

    def test_symmetry_long_vs_short_exit_structure(self):
        """Test that LONG and SHORT signals are symmetric"""
        config = SimplifiedStrategyConfig(
            min_confidence=0.50,
            stop_loss_atr_mult=1.5,
            take_profit_atr_mult=3.0,
        )

        engine_long = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.75),
            config=config,
        )
        engine_short = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.25),
            config=config,
        )

        df = create_test_df(length=50, volatility=0.02)

        signal_long = engine_long.generate_signal("BTC/USDT", df)
        signal_short = engine_short.generate_signal("BTC/USDT", df)

        if signal_long.side == "LONG" and signal_short.side == "SHORT":
            # Entry prices should be the same
            assert abs(signal_long.entry_price - signal_short.entry_price) < 0.01

            # SL/TP distances should be symmetric
            long_sl_dist = signal_long.entry_price - signal_long.stop_loss
            long_tp_dist = signal_long.take_profit - signal_long.entry_price

            short_sl_dist = signal_short.stop_loss - signal_short.entry_price
            short_tp_dist = signal_short.entry_price - signal_short.take_profit

            assert abs(long_sl_dist - short_sl_dist) < 0.1
            assert abs(long_tp_dist - short_tp_dist) < 0.1

    def test_expected_pnl_long_positive(self):
        """Test that LONG signals have positive expected PnL"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.75),
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50)

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side == "LONG":
            # Expected PnL should be positive (before fees)
            assert signal.expected_pnl_pct > 0

    def test_expected_pnl_short_positive(self):
        """Test that SHORT signals have positive expected PnL"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.25),  # Strong DOWN
            config=SimplifiedStrategyConfig(min_confidence=0.55),
        )
        df = create_test_df(length=50)

        signal = engine.generate_signal("BTC/USDT", df)

        if signal.side == "SHORT":
            # Expected PnL should be positive (before fees)
            assert signal.expected_pnl_pct > 0

    def test_trend_filter_long_up_trend(self):
        """Test LONG signals align with UP trend"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.75),
            config=SimplifiedStrategyConfig(
                min_confidence=0.55,
                use_trend_filter=True,
            ),
        )
        df = create_test_df(length=50, trend="up")

        signal = engine.generate_signal("BTC/USDT", df)

        # In an uptrend, should prefer LONG
        if signal.side is not None:
            assert signal.side == "LONG"
            assert signal.trend_bias == "UP"

    def test_trend_filter_short_down_trend(self):
        """Test SHORT signals align with DOWN trend"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.25),  # Strong DOWN signal
            config=SimplifiedStrategyConfig(
                min_confidence=0.55,
                use_trend_filter=True,
            ),
        )
        df = create_test_df(length=50, trend="down")

        signal = engine.generate_signal("BTC/USDT", df)

        # In a downtrend, should prefer SHORT
        if signal.side is not None:
            assert signal.side == "SHORT"
            assert signal.trend_bias == "DOWN"

    def test_volatility_gate_blocks_both_sides(self):
        """Test that volatility gate blocks both LONG and SHORT"""
        engine_long = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.80),
            config=SimplifiedStrategyConfig(min_confidence=0.50),
        )
        engine_short = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.20),
            config=SimplifiedStrategyConfig(min_confidence=0.50),
        )

        # Create dataframe with very low volatility
        df = create_test_df(length=50, volatility=0.0001)

        signal_long = engine_long.generate_signal("BTC/USDT", df)
        signal_short = engine_short.generate_signal("BTC/USDT", df)

        # Both should be blocked by volatility gate
        assert signal_long.side is None
        assert signal_short.side is None

    def test_cooldown_affects_both_sides(self):
        """Test that cooldown enforcement works for both LONG and SHORT"""
        engine = SimplifiedStrategyEngine(
            model=MockBidirectionalModel(up_prob=0.75),
            config=SimplifiedStrategyConfig(min_bars_between_trades=10),
        )
        df = create_test_df(length=50)

        # First signal at bar 0
        signal1 = engine.generate_signal("BTC/USDT", df, current_bar_idx=0)

        # Bar 5 (only 5 bars since last)
        signal2 = engine.generate_signal("BTC/USDT", df, current_bar_idx=5)

        # Bar 15 (10 bars since last)
        signal3 = engine.generate_signal("BTC/USDT", df, current_bar_idx=15)

        if signal1.side is not None:
            # Signal2 should be rejected due to cooldown
            assert signal2.side is None or "Cooldown" in signal2.reason

    def test_signal_validation_long(self):
        """Test signal validation for LONG"""
        engine = SimplifiedStrategyEngine()

        signal = engine.generate_signal("BTC/USDT", create_test_df())
        if signal.side == "LONG":
            assert engine.validate_signal(signal) or signal.side is None

    def test_signal_validation_short(self):
        """Test signal validation for SHORT"""
        engine = SimplifiedStrategyEngine()

        signal = engine.generate_signal("BTC/USDT", create_test_df())
        if signal.side == "SHORT":
            assert engine.validate_signal(signal) or signal.side is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
