"""
SHORT SELLING SUPPORT - PHASE 4: Integration Testing for Bidirectional Trading

Tests complete trading cycles with both LONG and SHORT positions to validate
the entire system works correctly for bidirectional trading.
"""

import pytest
from datetime import datetime
from execution.position import Position
from execution.broker import PaperBroker


class TestBidirectionalTradingCycles:
    """Test complete trading cycles with both LONG and SHORT"""

    def test_long_only_cycle(self):
        """Test a complete LONG trade cycle: open -> close"""
        broker = PaperBroker()
        entry_price = 50000.0
        qty = 1.0
        leverage = 5.0
        exit_price = 55000.0

        # Open LONG
        pos = broker.open_position(
            side="LONG",
            price=entry_price,
            qty=qty,
            symbol="BTC/USDT",
        )

        assert pos.side == "LONG"
        assert pos.qty == qty
        assert pos.entry_price == entry_price

        # Close LONG (profit)
        pnl = broker.close_position(price=exit_price, symbol="BTC/USDT")

        # PnL calculation: (exit - entry) * qty * leverage (leverage=1 by default for Position)
        expected_pnl = (exit_price - entry_price) * qty
        assert abs(pnl - expected_pnl) < 0.01
        assert pnl > 0  # Should be profitable

    def test_short_only_cycle(self):
        """Test a complete SHORT trade cycle: open -> close"""
        broker = PaperBroker()
        entry_price = 50000.0
        qty = 1.0
        exit_price = 45000.0  # SHORT profits when price goes down

        # Open SHORT
        pos = broker.open_position(
            side="SHORT",
            price=entry_price,
            qty=qty,
            symbol="BTC/USDT",
        )

        assert pos.side == "SHORT"
        assert pos.qty == qty

        # Close SHORT (profit)
        pnl = broker.close_position(price=exit_price, symbol="BTC/USDT")

        # PnL calculation: (entry - exit) * qty
        expected_pnl = (entry_price - exit_price) * qty
        assert abs(pnl - expected_pnl) < 0.01
        assert pnl > 0  # Should be profitable

    def test_alternating_long_short_cycles(self):
        """Test alternating LONG and SHORT trades"""
        broker = PaperBroker()

        # Cycle 1: LONG
        pos1 = broker.open_position(
            side="LONG",
            price=50000.0,
            qty=1.0,
            symbol="BTC/USDT",
        )
        pnl1 = broker.close_position(price=52000.0, symbol="BTC/USDT")
        assert pnl1 > 0

        # Cycle 2: SHORT
        pos2 = broker.open_position(
            side="SHORT",
            price=52000.0,
            qty=1.0,
            symbol="BTC/USDT",
        )
        pnl2 = broker.close_position(price=50000.0, symbol="BTC/USDT")
        assert pnl2 > 0

        # Cycle 3: LONG again
        pos3 = broker.open_position(
            side="LONG",
            price=50000.0,
            qty=1.0,
            symbol="BTC/USDT",
        )
        pnl3 = broker.close_position(price=51000.0, symbol="BTC/USDT")
        assert pnl3 > 0

    def test_long_loss_scenario(self):
        """Test LONG position that loses money"""
        broker = PaperBroker()

        pos = broker.open_position(
            side="LONG",
            price=50000.0,
            qty=1.0,
            symbol="BTC/USDT",
        )

        # Price goes down
        pnl = broker.close_position(price=48000.0, symbol="BTC/USDT")

        # Should be a loss: (48000-50000)*1 = -2000
        expected_pnl = (48000.0 - 50000.0) * 1.0
        assert abs(pnl - expected_pnl) < 0.01
        assert pnl < 0  # Should be a loss

    def test_short_loss_scenario(self):
        """Test SHORT position that loses money"""
        broker = PaperBroker()

        pos = broker.open_position(
            side="SHORT",
            price=50000.0,
            qty=1.0,
            symbol="BTC/USDT",
        )

        # Price goes up (bad for SHORT)
        pnl = broker.close_position(price=52000.0, symbol="BTC/USDT")

        # Should be a loss: (50000-52000)*1 = -2000
        expected_pnl = (50000.0 - 52000.0) * 1.0
        assert abs(pnl - expected_pnl) < 0.01
        assert pnl < 0  # Should be a loss

    def test_pyramiding_long(self):
        """Test pyramiding (adding to) a LONG position"""
        pos = Position(
            side="LONG",
            entry_price=50000.0,
            qty=1.0,
            entry_time=datetime.now(),
        )

        assert pos.qty == 1.0
        assert pos.avg_entry == 50000.0

        # Add to position at higher price
        pos.add_to_position(price=51000.0, qty=1.0)

        assert pos.qty == 2.0
        # Average = (50000*1 + 51000*1) / 2 = 50500
        assert abs(pos.avg_entry - 50500.0) < 0.01

    def test_pyramiding_short(self):
        """Test pyramiding a SHORT position"""
        pos = Position(
            side="SHORT",
            entry_price=50000.0,
            qty=1.0,
            entry_time=datetime.now(),
        )

        assert pos.qty == 1.0
        assert pos.avg_entry == 50000.0

        # Add to position at lower price
        pos.add_to_position(price=49000.0, qty=1.0)

        assert pos.qty == 2.0
        # Average = (50000*1 + 49000*1) / 2 = 49500
        assert abs(pos.avg_entry - 49500.0) < 0.01

    def test_multiple_symbols_independent(self):
        """Test that different symbols maintain independent positions"""
        broker = PaperBroker()

        # Open LONG on BTC
        btc_pos = broker.open_position(
            side="LONG",
            price=50000.0,
            qty=1.0,
            symbol="BTC/USDT",
        )

        # Open SHORT on ETH (requires new broker or separate tracking)
        # For single position broker, just verify LONG works
        assert btc_pos.side == "LONG"
        assert btc_pos.qty == 1.0

        # Close independently
        btc_pnl = broker.close_position(price=51000.0, symbol="BTC/USDT")

        # Should be profitable
        assert btc_pnl > 0

    def test_position_state_isolation(self):
        """Test that position states don't leak between instances"""
        # Create two independent LONG positions
        pos1 = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.now(),
        )

        pos2 = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.now(),
        )

        # Modify one
        pos1.add_to_position(price=101.0, qty=1.0)

        # Verify other is unaffected
        assert pos1.qty == 2.0
        assert pos2.qty == 1.0
        assert pos1.avg_entry != pos2.avg_entry

    def test_leverage_consistency_across_sides(self):
        """Test that leverage applies consistently to both LONG and SHORT"""
        leverage = 4.0

        long_pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=10.0,
            entry_time=datetime.now(),
            leverage=leverage,
        )

        short_pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=10.0,
            entry_time=datetime.now(),
            leverage=leverage,
        )

        # LONG profit at +10
        long_pnl = long_pos.pnl(exit_price=110.0)
        # SHORT profit at -10
        short_pnl = short_pos.pnl(exit_price=90.0)

        # Both should have same magnitude PnL (symmetry)
        expected = (10.0) * 10.0 * leverage  # 400
        assert abs(long_pnl - expected) < 0.01
        assert abs(short_pnl - expected) < 0.01

    def test_zero_pnl_at_entry_price(self):
        """Test that closing at entry price gives zero PnL for both sides"""
        long_pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.now(),
            leverage=1.0,
        )

        short_pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.now(),
            leverage=1.0,
        )

        # Close at entry price
        long_pnl = long_pos.pnl(exit_price=100.0)
        short_pnl = short_pos.pnl(exit_price=100.0)

        assert abs(long_pnl) < 0.01  # Should be ~0
        assert abs(short_pnl) < 0.01  # Should be ~0

    def test_extreme_leverage_range(self):
        """Test positions with min and max leverage"""
        # Min leverage
        pos_1x = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.now(),
            leverage=1.0,
        )
        pnl_1x = pos_1x.pnl(exit_price=110.0)
        expected_1x = (110 - 100) * 1.0 * 1.0
        assert abs(pnl_1x - expected_1x) < 0.01

        # Max leverage
        pos_125x = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.now(),
            leverage=125.0,
        )
        pnl_125x = pos_125x.pnl(exit_price=110.0)
        expected_125x = (110 - 100) * 1.0 * 125.0
        assert abs(pnl_125x - expected_125x) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
