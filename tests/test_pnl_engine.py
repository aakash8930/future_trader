"""
Comprehensive PnL engine tests.

Validates that all position calculations are mathematically correct
and match exchange behavior.
"""

import pytest
from datetime import datetime
from execution.position import Position


class TestPositionPnL:
    """Test basic position PnL calculations."""

    def test_long_profit_1x_leverage(self):
        """LONG position with profit at 1x leverage."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0,
        )
        pnl = pos.pnl(110.0)
        assert pnl == pytest.approx(10.0), f"Expected 10.0, got {pnl}"

    def test_long_loss_1x_leverage(self):
        """LONG position with loss at 1x leverage."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0,
        )
        pnl = pos.pnl(90.0)
        assert pnl == pytest.approx(-10.0), f"Expected -10.0, got {pnl}"

    def test_short_profit_1x_leverage(self):
        """SHORT position with profit at 1x leverage."""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0,
        )
        pnl = pos.pnl(90.0)
        assert pnl == pytest.approx(10.0), f"Expected 10.0, got {pnl}"

    def test_short_loss_1x_leverage(self):
        """SHORT position with loss at 1x leverage."""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0,
        )
        pnl = pos.pnl(110.0)
        assert pnl == pytest.approx(-10.0), f"Expected -10.0, got {pnl}"

    def test_long_with_5x_leverage_profit(self):
        """LONG position with profit at 5x leverage."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=5.0,
        )
        pnl = pos.pnl(105.0)  # 5% price move
        expected = (105.0 - 100.0) * 1.0 * 5.0  # 5 * 5x = 25
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_short_with_5x_leverage_profit(self):
        """SHORT position with profit at 5x leverage."""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=5.0,
        )
        pnl = pos.pnl(95.0)  # 5% price move down
        expected = (100.0 - 95.0) * 1.0 * 5.0  # 5 * 5x = 25
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_long_with_10x_leverage_rekt(self):
        """LONG position with liquidation at 10x leverage."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=10.0,
        )
        # 10% loss at 10x = 100% loss (liquidation)
        pnl = pos.pnl(90.0)
        expected = (90.0 - 100.0) * 1.0 * 10.0  # -10 * 10x = -100
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_quantity_scaling(self):
        """Verify PnL scales with quantity."""
        pos1 = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        pos2 = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            entry_time=datetime.utcnow(),
        )
        pnl1 = pos1.pnl(110.0)
        pnl2 = pos2.pnl(110.0)
        assert pnl2 == pytest.approx(pnl1 * 10), f"Qty should scale PnL: {pnl2} vs {pnl1 * 10}"

    def test_small_prices(self):
        """Test with small decimal prices (e.g., altcoins)."""
        pos = Position(
            side="LONG",
            entry_price=0.01,
            qty=1000.0,
            entry_time=datetime.utcnow(),
        )
        pnl = pos.pnl(0.011)
        expected = (0.011 - 0.01) * 1000.0  # 0.001 * 1000 = 1.0
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"

    def test_large_prices(self):
        """Test with large prices (e.g., BTC)."""
        pos = Position(
            side="LONG",
            entry_price=50000.0,
            qty=0.01,
            entry_time=datetime.utcnow(),
        )
        pnl = pos.pnl(51000.0)
        expected = (51000.0 - 50000.0) * 0.01  # 1000 * 0.01 = 10
        assert pnl == pytest.approx(expected), f"Expected {expected}, got {pnl}"


class TestPositionAveraging:
    """Test pyramid additions (add_to_position)."""

    def test_simple_pyramid_add(self):
        """Add to position at higher price."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        pos.add_to_position(price=110.0, qty=1.0)
        
        # Avg entry should be 105.0
        assert pos.avg_entry == pytest.approx(105.0)
        assert pos.qty == pytest.approx(2.0)
        assert pos.add_count == 1
        
        # PnL with total qty at average entry
        pnl = pos.pnl(110.0)
        expected = (110.0 - 105.0) * 2.0  # 5 * 2 = 10
        assert pnl == pytest.approx(expected)

    def test_multiple_pyramid_adds(self):
        """Multiple pyramid additions."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        pos.add_to_position(price=105.0, qty=1.0)
        pos.add_to_position(price=110.0, qty=1.0)
        
        # Avg: (100*1 + 105*1 + 110*1) / 3 = 315/3 = 105
        assert pos.avg_entry == pytest.approx(105.0)
        assert pos.qty == pytest.approx(3.0)
        assert pos.add_count == 2

    def test_cannot_change_leverage_on_add(self):
        """Should reject adding with different leverage."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=5.0,
        )
        with pytest.raises(ValueError, match="Cannot change leverage"):
            pos.add_to_position(price=105.0, qty=1.0, leverage=10.0)


class TestUnrealizedPnL:
    """Test unrealized PnL calculations."""

    def test_unrealized_long(self):
        """Unrealized PnL for LONG position."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        unrealized = pos.unrealized_pnl(105.0)
        assert unrealized == pytest.approx(5.0)

    def test_unrealized_short(self):
        """Unrealized PnL for SHORT position."""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        unrealized = pos.unrealized_pnl(95.0)
        assert unrealized == pytest.approx(5.0)


class TestValidation:
    """Test input validation."""

    def test_invalid_leverage_too_low(self):
        """Reject leverage < 1.0."""
        with pytest.raises(ValueError, match="Invalid leverage"):
            Position(
                side="LONG",
                entry_price=100.0,
                qty=1.0,
                entry_time=datetime.utcnow(),
                leverage=0.5,
            )

    def test_invalid_leverage_too_high(self):
        """Reject leverage > 125.0."""
        with pytest.raises(ValueError, match="Invalid leverage"):
            Position(
                side="LONG",
                entry_price=100.0,
                qty=1.0,
                entry_time=datetime.utcnow(),
                leverage=150.0,
            )

    def test_invalid_pnl_zero_price(self):
        """Reject zero exit price."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        with pytest.raises(ValueError, match="Invalid PnL inputs"):
            pos.pnl(0.0)

    def test_invalid_pnl_negative_price(self):
        """Reject negative exit price."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        with pytest.raises(ValueError, match="Invalid PnL inputs"):
            pos.pnl(-10.0)

    def test_invalid_side(self):
        """Reject invalid position side."""
        with pytest.raises(ValueError, match="Invalid side"):
            pos = Position(
                side="INVALID",
                entry_price=100.0,
                qty=1.0,
                entry_time=datetime.utcnow(),
            )
            pos.pnl(110.0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_breakeven(self):
        """Exit at exactly entry price = zero PnL."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        pnl = pos.pnl(100.0)
        assert pnl == pytest.approx(0.0)

    def test_very_small_quantity(self):
        """Handle very small quantities (e.g., 0.001 BTC)."""
        pos = Position(
            side="LONG",
            entry_price=50000.0,
            qty=0.001,
            entry_time=datetime.utcnow(),
        )
        pnl = pos.pnl(51000.0)
        expected = (51000.0 - 50000.0) * 0.001  # 1
        assert pnl == pytest.approx(expected)

    def test_very_large_quantity(self):
        """Handle very large quantities."""
        pos = Position(
            side="LONG",
            entry_price=1.0,
            qty=1_000_000.0,
            entry_time=datetime.utcnow(),
        )
        pnl = pos.pnl(1.01)
        expected = (1.01 - 1.0) * 1_000_000.0  # 10,000
        assert pnl == pytest.approx(expected)

    def test_maximum_leverage(self):
        """Test at maximum allowed leverage (125x)."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=125.0,
        )
        pnl = pos.pnl(101.0)
        expected = (101.0 - 100.0) * 1.0 * 125.0  # 125
        assert pnl == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
