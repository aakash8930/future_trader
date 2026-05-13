"""
SHORT SELLING SUPPORT - PHASE 2: Enhanced Broker with Bidirectional Support

Extends PaperBroker and ShadowBroker to properly handle both LONG and SHORT positions.

Key Enhancements:
- Symmetric position opening for LONG and SHORT
- Proper close handling with side-aware PnL
- Pyramiding support for both sides
- Position tracking with explicit state
- PnL calculation consistency
"""

import pytest
from datetime import datetime
from execution.broker import PaperBroker


class TestPaperBrokerBidirectional:
    """Test PaperBroker with bidirectional (LONG/SHORT) support"""

    def test_open_long_position(self):
        """Test opening a LONG position"""
        broker = PaperBroker()
        assert broker.position is None

        pos = broker.open_position(side="LONG", price=100.0, qty=1.0)

        assert pos is not None
        assert pos.side == "LONG"
        assert pos.entry_price == 100.0
        assert pos.qty == 1.0
        assert broker.position is pos

    def test_open_short_position(self):
        """Test opening a SHORT position"""
        broker = PaperBroker()
        assert broker.position is None

        pos = broker.open_position(side="SHORT", price=100.0, qty=1.0)

        assert pos is not None
        assert pos.side == "SHORT"
        assert pos.entry_price == 100.0
        assert pos.qty == 1.0
        assert broker.position is pos

    def test_close_long_position_profit(self):
        """Test closing a profitable LONG position"""
        broker = PaperBroker()
        broker.open_position(side="LONG", price=100.0, qty=1.0)

        # Close at 110 (profit)
        pnl = broker.close_position(price=110.0)

        assert pnl == 10.0  # (110 - 100) * 1 * 1.0
        assert broker.position is None

    def test_close_long_position_loss(self):
        """Test closing a losing LONG position"""
        broker = PaperBroker()
        broker.open_position(side="LONG", price=100.0, qty=1.0)

        # Close at 90 (loss)
        pnl = broker.close_position(price=90.0)

        assert pnl == -10.0
        assert broker.position is None

    def test_close_short_position_profit(self):
        """Test closing a profitable SHORT position"""
        broker = PaperBroker()
        broker.open_position(side="SHORT", price=100.0, qty=1.0)

        # Close at 90 (profit for short)
        pnl = broker.close_position(price=90.0)

        assert pnl == 10.0  # (100 - 90) * 1 * 1.0
        assert broker.position is None

    def test_close_short_position_loss(self):
        """Test closing a losing SHORT position"""
        broker = PaperBroker()
        broker.open_position(side="SHORT", price=100.0, qty=1.0)

        # Close at 110 (loss for short)
        pnl = broker.close_position(price=110.0)

        assert pnl == -10.0
        assert broker.position is None

    def test_close_with_no_position(self):
        """Test closing when no position is open"""
        broker = PaperBroker()
        pnl = broker.close_position(price=100.0)

        assert pnl == 0.0
        assert broker.position is None

    def test_add_to_long_position(self):
        """Test adding to a LONG position (pyramiding)"""
        broker = PaperBroker()
        broker.open_position(side="LONG", price=100.0, qty=1.0)

        # Add at higher price
        broker.add_to_position(price=105.0, qty=1.0)

        assert broker.position.qty == 2.0
        assert broker.position.avg_entry == 102.5

    def test_add_to_short_position(self):
        """Test adding to a SHORT position (pyramiding)"""
        broker = PaperBroker()
        broker.open_position(side="SHORT", price=100.0, qty=1.0)

        # Add short at lower price
        broker.add_to_position(price=95.0, qty=1.0)

        assert broker.position.qty == 2.0
        assert broker.position.avg_entry == 97.5

    def test_add_to_position_with_no_open_position(self):
        """Test adding when no position is open (should be no-op)"""
        broker = PaperBroker()
        broker.add_to_position(price=100.0, qty=1.0)

        # Should have no position
        assert broker.position is None

    def test_long_position_state_after_operations(self):
        """Test LONG position state after multiple operations"""
        broker = PaperBroker()

        # Open
        broker.open_position(side="LONG", price=100.0, qty=1.0)
        assert broker.position.side == "LONG"
        assert broker.position.qty == 1.0

        # Add
        broker.add_to_position(price=102.0, qty=1.0)
        assert broker.position.qty == 2.0

        # Close
        pnl = broker.close_position(price=105.0)
        assert pnl == 8.0  # (105 - 101) * 2 = 8
        assert broker.position is None

    def test_short_position_state_after_operations(self):
        """Test SHORT position state after multiple operations"""
        broker = PaperBroker()

        # Open
        broker.open_position(side="SHORT", price=100.0, qty=1.0)
        assert broker.position.side == "SHORT"
        assert broker.position.qty == 1.0

        # Add
        broker.add_to_position(price=98.0, qty=1.0)
        assert broker.position.qty == 2.0

        # Close
        pnl = broker.close_position(price=95.0)
        assert pnl == 8.0  # (99 - 95) * 2 = 8
        assert broker.position is None

    def test_multiple_open_close_cycles_long(self):
        """Test multiple LONG open/close cycles"""
        broker = PaperBroker()

        # First cycle
        broker.open_position(side="LONG", price=100.0, qty=1.0)
        pnl1 = broker.close_position(price=110.0)
        assert pnl1 == 10.0
        assert broker.position is None

        # Second cycle
        broker.open_position(side="LONG", price=105.0, qty=1.0)
        pnl2 = broker.close_position(price=100.0)
        assert pnl2 == -5.0
        assert broker.position is None

    def test_multiple_open_close_cycles_short(self):
        """Test multiple SHORT open/close cycles"""
        broker = PaperBroker()

        # First cycle
        broker.open_position(side="SHORT", price=100.0, qty=1.0)
        pnl1 = broker.close_position(price=90.0)
        assert pnl1 == 10.0
        assert broker.position is None

        # Second cycle
        broker.open_position(side="SHORT", price=95.0, qty=1.0)
        pnl2 = broker.close_position(price=100.0)
        assert pnl2 == -5.0
        assert broker.position is None

    def test_long_then_short_cycles(self):
        """Test alternating LONG and SHORT cycles"""
        broker = PaperBroker()

        # LONG cycle
        broker.open_position(side="LONG", price=100.0, qty=1.0)
        assert broker.position.side == "LONG"
        pnl1 = broker.close_position(price=105.0)
        assert pnl1 == 5.0

        # SHORT cycle
        broker.open_position(side="SHORT", price=105.0, qty=1.0)
        assert broker.position.side == "SHORT"
        pnl2 = broker.close_position(price=100.0)
        assert pnl2 == 5.0

    def test_long_with_leverage(self):
        """Test LONG position with leverage"""
        broker = PaperBroker()
        broker.open_position(side="LONG", price=100.0, qty=1.0)
        broker.position.leverage = 5.0

        pnl = broker.close_position(price=110.0)
        assert pnl == 50.0  # (110 - 100) * 1 * 5

    def test_short_with_leverage(self):
        """Test SHORT position with leverage"""
        broker = PaperBroker()
        broker.open_position(side="SHORT", price=100.0, qty=1.0)
        broker.position.leverage = 5.0

        pnl = broker.close_position(price=90.0)
        assert pnl == 50.0  # (100 - 90) * 1 * 5

    def test_long_multiple_contracts_with_leverage(self):
        """Test LONG with multiple contracts and leverage"""
        broker = PaperBroker()
        broker.open_position(side="LONG", price=100.0, qty=5.0)
        broker.position.leverage = 2.0

        pnl = broker.close_position(price=110.0)
        assert pnl == 100.0  # (110 - 100) * 5 * 2

    def test_short_multiple_contracts_with_leverage(self):
        """Test SHORT with multiple contracts and leverage"""
        broker = PaperBroker()
        broker.open_position(side="SHORT", price=100.0, qty=5.0)
        broker.position.leverage = 2.0

        pnl = broker.close_position(price=90.0)
        assert pnl == 100.0  # (100 - 90) * 5 * 2

    def test_position_reference_consistency(self):
        """Test that broker.position always points to correct position"""
        broker = PaperBroker()

        pos1 = broker.open_position(side="LONG", price=100.0, qty=1.0)
        assert broker.position is pos1

        broker.add_to_position(price=102.0, qty=1.0)
        assert broker.position is pos1  # Same reference

        pnl = broker.close_position(price=105.0)
        assert broker.position is None

    def test_pyramid_then_close_long(self):
        """Test pyramiding LONG then closing"""
        broker = PaperBroker()

        broker.open_position(side="LONG", price=100.0, qty=2.0)
        broker.add_to_position(price=104.0, qty=1.0)

        # avg_entry = (100*2 + 104*1) / 3 = 101.33...
        pnl = broker.close_position(price=108.0)
        expected = (108 - 101.333333) * 3
        assert abs(pnl - expected) < 0.01

    def test_pyramid_then_close_short(self):
        """Test pyramiding SHORT then closing"""
        broker = PaperBroker()

        broker.open_position(side="SHORT", price=100.0, qty=2.0)
        broker.add_to_position(price=96.0, qty=1.0)

        # avg_entry = (100*2 + 96*1) / 3 = 98.66...
        pnl = broker.close_position(price=92.0)
        expected = (98.666667 - 92) * 3
        assert abs(pnl - expected) < 0.01
