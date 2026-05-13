"""
SHORT SELLING SUPPORT - PHASE 1: Enhanced Position Model & Core Tests

This phase establishes the foundation for bidirectional (long/short) futures trading.

Design Goals:
- Symmetric position handling (LONG == opposite of SHORT)
- Correct PnL calculations for both sides
- Leverage-aware risk for both sides
- Explicit state management (LONG/SHORT/NO_POSITION)
- Zero code duplication between sides
- Full backward compatibility

Position State Machine:
  NO_POSITION → open_long() → LONG_OPEN
  NO_POSITION → open_short() → SHORT_OPEN
  LONG_OPEN → close() → NO_POSITION
  SHORT_OPEN → close() → NO_POSITION
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from execution.position import Position


class TestPositionModelBidirectional:
    """Test Position model with both LONG and SHORT sides"""

    def test_long_position_creation(self):
        """Test creating a LONG position"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        assert pos.side == "LONG"
        assert pos.entry_price == 100.0
        assert pos.qty == 1.0
        assert pos.leverage == 1.0

    def test_short_position_creation(self):
        """Test creating a SHORT position"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        assert pos.side == "SHORT"
        assert pos.entry_price == 100.0
        assert pos.qty == 1.0

    def test_long_pnl_profit(self):
        """Test LONG position profit calculation"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        # Price goes up to 110
        pnl = pos.pnl(110.0)
        assert pnl == 10.0  # (110 - 100) * 1 * 1.0

    def test_long_pnl_loss(self):
        """Test LONG position loss calculation"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        # Price goes down to 90
        pnl = pos.pnl(90.0)
        assert pnl == -10.0  # (90 - 100) * 1 * 1.0

    def test_short_pnl_profit(self):
        """Test SHORT position profit when price falls"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        # Price goes down to 90 (profit for short)
        pnl = pos.pnl(90.0)
        assert pnl == 10.0  # (100 - 90) * 1 * 1.0

    def test_short_pnl_loss(self):
        """Test SHORT position loss when price rises"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        # Price goes up to 110 (loss for short)
        pnl = pos.pnl(110.0)
        assert pnl == -10.0  # (100 - 110) * 1 * 1.0

    def test_long_with_leverage(self):
        """Test LONG position with leverage"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=5.0
        )
        # Price goes up to 110
        pnl = pos.pnl(110.0)
        assert pnl == 50.0  # (110 - 100) * 1 * 5.0

    def test_short_with_leverage(self):
        """Test SHORT position with leverage"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=5.0
        )
        # Price goes down to 90
        pnl = pos.pnl(90.0)
        assert pnl == 50.0  # (100 - 90) * 1 * 5.0

    def test_long_multiple_contracts(self):
        """Test LONG with multiple contracts"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=5.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        pnl = pos.pnl(105.0)
        assert pnl == 25.0  # (105 - 100) * 5 * 1.0

    def test_short_multiple_contracts(self):
        """Test SHORT with multiple contracts"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=5.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        pnl = pos.pnl(95.0)
        assert pnl == 25.0  # (100 - 95) * 5 * 1.0

    def test_symmetry_long_vs_short(self):
        """Test symmetry: long profit = short loss at opposite prices"""
        long_pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        short_pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )

        # Price up to 110
        long_pnl_up = long_pos.pnl(110.0)      # +10
        short_pnl_up = short_pos.pnl(110.0)    # -10

        # Price down to 90
        long_pnl_down = long_pos.pnl(90.0)     # -10
        short_pnl_down = short_pos.pnl(90.0)   # +10

        # Verify symmetry
        assert long_pnl_up == -short_pnl_up
        assert long_pnl_down == -short_pnl_down

    def test_unrealized_pnl_long(self):
        """Test unrealized PnL for LONG"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        upnl = pos.unrealized_pnl(105.0)
        assert upnl == 5.0

    def test_unrealized_pnl_short(self):
        """Test unrealized PnL for SHORT"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        upnl = pos.unrealized_pnl(95.0)
        assert upnl == 5.0

    def test_leverage_validation_minimum(self):
        """Test leverage minimum validation"""
        with pytest.raises(ValueError):
            Position(
                side="LONG",
                entry_price=100.0,
                qty=1.0,
                entry_time=datetime.utcnow(),
                leverage=0.5  # Below 1.0
            )

    def test_leverage_validation_maximum(self):
        """Test leverage maximum validation"""
        with pytest.raises(ValueError):
            Position(
                side="LONG",
                entry_price=100.0,
                qty=1.0,
                entry_time=datetime.utcnow(),
                leverage=200.0  # Above 125.0
            )

    def test_leverage_edge_cases(self):
        """Test leverage edge cases (1x and 125x)"""
        pos_1x = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        pnl_1x = pos_1x.pnl(110.0)
        assert pnl_1x == 10.0

        pos_125x = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=125.0
        )
        pnl_125x = pos_125x.pnl(90.0)
        assert pnl_125x == 1250.0  # (100-90) * 1 * 125

    def test_pyramid_add_long(self):
        """Test adding to LONG position (pyramiding)"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        # Add at price 105
        pos.add_to_position(105.0, 1.0)

        # Verify avg_entry = (100*1 + 105*1) / 2 = 102.5
        assert pos.avg_entry == 102.5
        assert pos.qty == 2.0
        assert pos.add_count == 1

        # PnL at 110: (110 - 102.5) * 2 * 1.0 = 15.0
        pnl = pos.pnl(110.0)
        assert pnl == 15.0

    def test_pyramid_add_short(self):
        """Test adding to SHORT position (pyramiding)"""
        pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        # Add short at price 95
        pos.add_to_position(95.0, 1.0)

        # Verify avg_entry = (100*1 + 95*1) / 2 = 97.5
        assert pos.avg_entry == 97.5
        assert pos.qty == 2.0

        # PnL at 90: (97.5 - 90) * 2 * 1.0 = 15.0
        pnl = pos.pnl(90.0)
        assert pnl == 15.0

    def test_pyramid_add_cannot_change_leverage(self):
        """Test that pyramiding cannot change leverage"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=5.0
        )
        with pytest.raises(ValueError):
            pos.add_to_position(105.0, 1.0, leverage=10.0)

    def test_position_repr_long(self):
        """Test string representation of LONG position"""
        pos = Position(
            side="LONG",
            entry_price=100.5,
            qty=2.5,
            entry_time=datetime.utcnow(),
            leverage=3.0
        )
        repr_str = repr(pos)
        assert "LONG" in repr_str
        assert "2.5" in repr_str or "2.500000" in repr_str
        assert "100.5" in repr_str
        assert "3.0" in repr_str

    def test_position_repr_short(self):
        """Test string representation of SHORT position"""
        pos = Position(
            side="SHORT",
            entry_price=50.25,
            qty=1.5,
            entry_time=datetime.utcnow(),
            leverage=2.0
        )
        repr_str = repr(pos)
        assert "SHORT" in repr_str
        assert "50.25" in repr_str

    def test_edge_case_zero_qty(self):
        """Test that PnL with zero qty raises error"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        pos.qty = 0.0  # Manually set to 0

        with pytest.raises(ValueError):
            pos.pnl(110.0)

    def test_edge_case_negative_price(self):
        """Test that negative prices raise error"""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        with pytest.raises(ValueError):
            pos.pnl(-50.0)

    def test_pnl_precision(self):
        """Test PnL calculation precision"""
        pos = Position(
            side="LONG",
            entry_price=99.99,
            qty=0.001,
            entry_time=datetime.utcnow(),
            leverage=1.0
        )
        pnl = pos.pnl(100.01)
        expected = (100.01 - 99.99) * 0.001
        assert abs(pnl - expected) < 1e-8  # Within floating point precision


# Additional test file to run all bidirectional tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
