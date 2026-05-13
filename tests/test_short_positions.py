"""Focused tests for short-side trading support."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from execution.broker import PaperBroker
from execution.position import Position, Side
from execution.strategy_simplified import SimplifiedStrategyConfig, SimplifiedStrategyEngine
from risk.liquidation_guard import LiquidationGuard


class MockModel:
    def __init__(self, long_prob: float):
        self.long_prob = long_prob

    def predict_proba(self, df):
        return (1.0 - self.long_prob, self.long_prob)


def make_df(close: float = 100.0, atr: float = 2.0, ema: float = 101.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=10, freq="h")
    closes = np.linspace(close * 1.01, close, 10)
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": 1_000_000,
            "atr": atr,
            "ema_50": ema,
        },
        index=index,
    )
    return frame


def test_short_profit_and_loss():
    pos = Position(side=Side.SHORT, entry_price=100.0, qty=2.0, entry_time=datetime.utcnow(), leverage=3.0)

    assert pos.pnl(90.0) == pytest.approx(60.0)
    assert pos.pnl(110.0) == pytest.approx(-60.0)


def test_short_leverage_scaling():
    pos = Position(side=Side.SHORT, entry_price=100.0, qty=1.5, entry_time=datetime.utcnow(), leverage=4.0)

    assert pos.pnl(95.0) == pytest.approx((100.0 - 95.0) * 1.5 * 4.0)


def test_short_signal_stop_loss_and_take_profit():
    engine = SimplifiedStrategyEngine(
        model=MockModel(long_prob=0.25),
        config=SimplifiedStrategyConfig(min_confidence=0.50),
    )
    df = make_df(close=100.0, atr=2.0, ema=102.0)

    signal = engine.generate_signal("BTC/USDT", df)

    assert signal.side == Side.SHORT
    assert signal.stop_loss == pytest.approx(signal.entry_price + 2.0 * 1.5)
    assert signal.take_profit == pytest.approx(signal.entry_price - 2.0 * 3.0)


def test_short_liquidation_edge_case():
    guard = LiquidationGuard()
    liquidation = guard.calculate_liquidation_price(side="SHORT", entry_price=100.0, leverage=3.0)

    assert liquidation > 100.0
    assert liquidation < 140.0

    distance = guard.calculate_distance_to_liquidation(
        side="SHORT",
        current_price=110.0,
        liquidation_price=liquidation,
    )
    assert distance > 0


def test_short_broker_cycle():
    broker = PaperBroker()
    broker.open_position(side=Side.SHORT, price=100.0, qty=1.0, leverage=2.0)

    pnl = broker.close_position(price=92.0)

    assert pnl == pytest.approx((100.0 - 92.0) * 1.0 * 2.0)
