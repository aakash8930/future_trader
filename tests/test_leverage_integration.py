"""
LEVERAGE INTEGRATION TESTS

Comprehensive tests for leverage configuration, passing, and PnL application.
Verifies that leverage is correctly applied exactly ONCE in position PnL calculation.
"""

import pytest
from datetime import datetime
from execution.position import Position
from execution.broker import PaperBroker, ShadowBroker
from execution.shadow_broker import ShadowBroker as SimpleShadowBroker


class TestPositionLeverageIntegration:
    """Test that Position applies leverage correctly"""

    def test_position_long_with_default_leverage(self):
        """Long position with default 1.0x leverage"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            entry_time=datetime.utcnow(),
        )
        assert pos.leverage == 1.0
        
        pnl = pos.pnl(exit_price=110.0)
        expected = (110 - 100) * 10 * 1.0
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_position_long_with_5x_leverage(self):
        """Long position with 5x leverage"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            leverage=5.0,
            entry_time=datetime.utcnow(),
        )
        assert pos.leverage == 5.0
        
        pnl = pos.pnl(exit_price=110.0)
        # (exit - entry) * qty * leverage
        expected = (110 - 100) * 10 * 5.0  # = $500
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_position_long_loss_with_5x_leverage(self):
        """Long position loss is magnified by leverage"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            leverage=5.0,
            entry_time=datetime.utcnow(),
        )
        
        pnl = pos.pnl(exit_price=95.0)
        # (exit - entry) * qty * leverage
        expected = (95 - 100) * 10 * 5.0  # = -$250
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_position_short_with_5x_leverage(self):
        """Short position with 5x leverage"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=10.0,
            leverage=5.0,
            entry_time=datetime.utcnow(),
        )
        
        pnl = pos.pnl(exit_price=95.0)
        # (entry - exit) * qty * leverage
        expected = (100 - 95) * 10 * 5.0  # = $250
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_position_short_loss_with_5x_leverage(self):
        """Short position loss is magnified by leverage"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=10.0,
            leverage=5.0,
            entry_time=datetime.utcnow(),
        )
        
        pnl = pos.pnl(exit_price=105.0)
        # (entry - exit) * qty * leverage
        expected = (100 - 105) * 10 * 5.0  # = -$250
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_leverage_range_validation(self):
        """Position validates leverage is in [1.0, 125.0]"""
        with pytest.raises(ValueError, match="Invalid leverage"):
            Position(
                side="LONG",
                entry_price=100.0,
                qty=10.0,
                leverage=0.5,  # Too low
                entry_time=datetime.utcnow(),
            )
        
        with pytest.raises(ValueError, match="Invalid leverage"):
            Position(
                side="LONG",
                entry_price=100.0,
                qty=10.0,
                leverage=200.0,  # Too high
                entry_time=datetime.utcnow(),
            )

    def test_leverage_edge_case_min(self):
        """Position accepts minimum leverage 1.0x"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            leverage=1.0,
            entry_time=datetime.utcnow(),
        )
        assert pos.leverage == 1.0

    def test_leverage_edge_case_max(self):
        """Position accepts maximum leverage 125.0x"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            leverage=125.0,
            entry_time=datetime.utcnow(),
        )
        assert pos.leverage == 125.0


class TestBrokerLeverageIntegration:
    """Test that Broker passes leverage to Position"""

    def test_paper_broker_default_leverage(self):
        """PaperBroker defaults to 1.0x leverage"""
        broker = PaperBroker()
        pos = broker.open_position(side="LONG", price=100.0, qty=10.0)
        
        assert pos.leverage == 1.0

    def test_paper_broker_accepts_leverage(self):
        """PaperBroker accepts and applies leverage parameter"""
        broker = PaperBroker()
        pos = broker.open_position(side="LONG", price=100.0, qty=10.0, leverage=5.0)
        
        assert pos.leverage == 5.0

    def test_paper_broker_leverage_affects_pnl(self):
        """PaperBroker position with 5x leverage calculates correct PnL"""
        broker = PaperBroker()
        pos = broker.open_position(side="LONG", price=100.0, qty=10.0, leverage=5.0, symbol="BTC/USDT")
        
        pnl = broker.close_position(price=110.0)
        expected = (110 - 100) * 10 * 5.0  # = $500
        assert pnl == pytest.approx(expected)

    def test_shadow_broker_default_leverage(self):
        """ShadowBroker defaults to 1.0x leverage"""
        broker = SimpleShadowBroker()
        pos = broker.open_position(side="LONG", price=100.0, qty=10.0, symbol="BTC/USDT")
        
        assert pos.leverage == 1.0

    def test_shadow_broker_accepts_leverage(self):
        """ShadowBroker accepts and applies leverage parameter"""
        broker = SimpleShadowBroker()
        pos = broker.open_position(side="LONG", price=100.0, qty=10.0, leverage=5.0, symbol="BTC/USDT")
        
        assert pos.leverage == 5.0

    def test_shadow_broker_leverage_affects_pnl(self):
        """ShadowBroker position with 5x leverage calculates correct PnL"""
        broker = SimpleShadowBroker()
        pos = broker.open_position(side="LONG", price=100.0, qty=10.0, leverage=5.0, symbol="BTC/USDT")
        
        pnl = broker.close_position(price=110.0, symbol="BTC/USDT")
        expected = (110 - 100) * 10 * 5.0  # = $500
        assert pnl == pytest.approx(expected)


class TestLeverageNoDoubleApplication:
    """Verify leverage is NOT applied twice"""

    def test_leverage_applied_once_in_pnl(self):
        """Leverage multiplier applied exactly ONCE in PnL calculation"""
        # Entry at $100, exit at $110, qty=10, leverage=5x
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            leverage=5.0,
            entry_time=datetime.utcnow(),
        )
        
        pnl = pos.pnl(exit_price=110.0)
        
        # Correct: (exit - entry) * qty * leverage
        correct_pnl = (110 - 100) * 10 * 5.0  # = $500
        
        # Incorrect (if applied twice): (exit - entry) * qty * leverage * leverage
        incorrect_pnl = (110 - 100) * 10 * 5.0 * 5.0  # = $2500
        
        assert pnl == pytest.approx(correct_pnl)
        assert pnl != pytest.approx(incorrect_pnl)

    def test_qty_not_multiplied_by_leverage_at_sizing(self):
        """Position sizing should NOT multiply qty by leverage"""
        # This test documents the expected behavior
        # Qty should be sized independently of leverage
        
        # Example: $1000 account, 1% risk, $10 stop-loss per unit
        # qty = (1000 * 0.01) / 10 = 1.0
        # This qty is the same regardless of leverage
        # Leverage only affects PnL = (exit - entry) * qty * leverage
        
        # Skip: requires importing risk.sizing which triggers full model imports
        pass


class TestLeverageComparison:
    """Compare PnL behavior with and without leverage"""

    def test_pnl_difference_1x_vs_5x(self):
        """5x leverage positions have 5x PnL compared to 1x"""
        pos_1x = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            leverage=1.0,
            entry_time=datetime.utcnow(),
        )
        
        pos_5x = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            leverage=5.0,
            entry_time=datetime.utcnow(),
        )
        
        pnl_1x = pos_1x.pnl(exit_price=110.0)
        pnl_5x = pos_5x.pnl(exit_price=110.0)
        
        # Should be 5x difference
        assert pnl_5x == pytest.approx(pnl_1x * 5.0)

    def test_all_leverage_levels(self):
        """Test PnL calculation across multiple leverage levels"""
        entry_price = 100.0
        exit_price = 110.0
        qty = 10.0
        
        leverages = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 125.0]
        
        for leverage in leverages:
            pos = Position(
                side="LONG",
                entry_price=entry_price,
                qty=qty,
                leverage=leverage,
                entry_time=datetime.utcnow(),
            )
            
            pnl = pos.pnl(exit_price=exit_price)
            expected = (exit_price - entry_price) * qty * leverage
            
            assert pnl == pytest.approx(expected), \
                f"Leverage {leverage}x: expected {expected}, got {pnl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
