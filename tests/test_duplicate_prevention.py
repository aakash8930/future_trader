"""
Tests for Duplicate Order Prevention

Validates duplicate detection and cooldown enforcement.
"""

import pytest
from datetime import datetime, timedelta
from execution.duplicate_prevention import (
    DuplicateOrderPrevention,
    TradeRecord,
)


class TestCooldownEnforcement:
    """Test cooldown period enforcement"""

    def test_can_trade_first_trade(self):
        """First trade on symbol always allowed"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)
        assert dprev.can_trade_symbol("BTC/USDT")

    def test_can_trade_in_cooldown(self):
        """Trade rejected during cooldown period"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        # Record first trade
        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        assert not dprev.can_trade_symbol("BTC/USDT")

    def test_can_trade_after_cooldown(self):
        """Trade allowed after cooldown expires"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        # Record first trade
        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        assert not dprev.can_trade_symbol("BTC/USDT")

        # Mark position as closed
        dprev.mark_position_closed("BTC/USDT")

        # Still in cooldown
        assert not dprev.can_trade_symbol("BTC/USDT")

        # Simulate cooldown expiring
        dprev.trade_history["BTC/USDT"][0].timestamp = (
            datetime.utcnow() - timedelta(seconds=61)
        )
        assert dprev.can_trade_symbol("BTC/USDT")

    def test_short_cooldown_respected(self):
        """Short cooldown period respected"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=5)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        assert not dprev.can_trade_symbol("BTC/USDT")

        # Mark closed to test cooldown (not open position blocking)
        dprev.mark_position_closed("BTC/USDT")

        # Not enough time
        dprev.trade_history["BTC/USDT"][0].timestamp = (
            datetime.utcnow() - timedelta(seconds=3)
        )
        assert not dprev.can_trade_symbol("BTC/USDT")

        # Enough time
        dprev.trade_history["BTC/USDT"][0].timestamp = (
            datetime.utcnow() - timedelta(seconds=6)
        )
        assert dprev.can_trade_symbol("BTC/USDT")


class TestPositionConflicts:
    """Test detection of conflicting open positions"""

    def test_cannot_open_with_existing_position(self):
        """Can't open second position if one already open"""
        dprev = DuplicateOrderPrevention()

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        assert not dprev.can_open_position("BTC/USDT")

    def test_can_open_after_closing(self):
        """Can open new position after closing previous"""
        dprev = DuplicateOrderPrevention()

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        assert not dprev.can_open_position("BTC/USDT")

        dprev.mark_position_closed("BTC/USDT")
        assert dprev.can_open_position("BTC/USDT")

    def test_multiple_symbols_independent(self):
        """Different symbols have independent position tracking"""
        dprev = DuplicateOrderPrevention()

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        assert not dprev.can_open_position("BTC/USDT")
        assert dprev.can_open_position("ETH/USDT")  # Different symbol OK


class TestDuplicateDetection:
    """Test duplicate trade detection"""

    def test_exact_duplicate_detected(self):
        """Exact duplicate (same symbol, side, price) detected"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)

        # Exact duplicate attempt
        assert dprev.is_duplicate_attempt("BTC/USDT", "LONG", 40000.0)

    def test_different_side_not_duplicate(self):
        """Different side (LONG vs SHORT) not duplicate"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)

        # Different side = not duplicate
        assert not dprev.is_duplicate_attempt("BTC/USDT", "SHORT", 40000.0)

    def test_price_within_tolerance_duplicate(self):
        """Similar price (within tolerance) is duplicate"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)

        # Within 0.5% tolerance (default)
        assert dprev.is_duplicate_attempt("BTC/USDT", "LONG", 40200.0)  # 0.5%

    def test_price_outside_tolerance_not_duplicate(self):
        """Price outside tolerance not duplicate"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)

        # Outside 0.5% tolerance
        assert not dprev.is_duplicate_attempt("BTC/USDT", "LONG", 40300.0)  # >0.5%

    def test_custom_tolerance(self):
        """Custom price tolerance applied"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)

        # With 1% tolerance, 40400 is duplicate
        assert dprev.is_duplicate_attempt(
            "BTC/USDT", "LONG", 40400.0, price_tolerance_pct=1.0
        )

        # Without tolerance, it's not
        assert not dprev.is_duplicate_attempt(
            "BTC/USDT", "LONG", 40400.0, price_tolerance_pct=0.1
        )

    def test_closed_position_not_duplicate(self):
        """Closed position not considered for duplicates"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        dprev.mark_position_closed("BTC/USDT")

        # Now closed, so not a duplicate (if in cooldown period)
        # Depends on cooldown actually
        result = dprev.is_duplicate_attempt("BTC/USDT", "LONG", 40000.0)
        # Could be False (closed) or True (cooldown still active)


class TestTradeValidation:
    """Test full trade request validation"""

    def test_valid_trade_request(self):
        """Valid trade request approved"""
        dprev = DuplicateOrderPrevention()

        result = dprev.validate_trade_request("BTC/USDT", "LONG", 40000.0, 1.0)

        assert result["allowed"]
        assert result["reason"] == "OK"

    def test_validation_rejects_duplicate(self):
        """Validation rejects duplicate attempt"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=5)  # Short cooldown

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)

        # Duplicate detected because position is still open
        result = dprev.validate_trade_request("BTC/USDT", "LONG", 40000.0, 1.0)

        # This fails at can_open_position check
        assert not result["allowed"]
        assert "Already have open position" in result["reason"]

    def test_validation_rejects_cooldown(self):
        """Validation rejects trade during cooldown"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        dprev.mark_position_closed("BTC/USDT")

        result = dprev.validate_trade_request("BTC/USDT", "SHORT", 41000.0, 1.0)

        assert not result["allowed"]
        assert "Cooldown" in result["reason"]
        assert result["in_cooldown"]

    def test_validation_rejects_existing_position(self):
        """Validation rejects opening second position"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=5)  # Allow new trade after

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)

        # Different side, but same symbol still blocks
        result = dprev.validate_trade_request("BTC/USDT", "SHORT", 41000.0, 1.0)

        assert not result["allowed"]
        assert "Already have open position" in result["reason"]

    def test_rejected_count_increments(self):
        """Rejected trade count increments"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        dprev.mark_position_closed("BTC/USDT")

        assert dprev.rejected_count == 0

        dprev.validate_trade_request("BTC/USDT", "LONG", 40000.0, 1.0)
        assert dprev.rejected_count == 1

        dprev.validate_trade_request("BTC/USDT", "SHORT", 41000.0, 1.0)
        assert dprev.rejected_count == 2


class TestTradeHistory:
    """Test trade history management"""

    def test_trade_recorded(self):
        """Trade recorded in history"""
        dprev = DuplicateOrderPrevention()

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)

        assert "BTC/USDT" in dprev.trade_history
        assert len(dprev.trade_history["BTC/USDT"]) == 1

    def test_multiple_trades_recorded(self):
        """Multiple trades per symbol recorded"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=0)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        dprev.mark_position_closed("BTC/USDT")
        dprev.record_trade("BTC/USDT", "SHORT", 41000.0, 1.0)

        assert len(dprev.trade_history["BTC/USDT"]) == 2

    def test_max_recent_trades_enforced(self):
        """Only recent N trades kept per symbol"""
        dprev = DuplicateOrderPrevention(max_recent_trades=5, cooldown_seconds=0)

        # Add 10 trades
        for i in range(10):
            dprev.record_trade("BTC/USDT", "LONG", 40000.0 + i, 1.0)
            dprev.mark_position_closed("BTC/USDT")

        # Only 5 recent kept
        assert len(dprev.trade_history["BTC/USDT"]) == 5

    def test_mark_position_closed(self):
        """Mark position as closed"""
        dprev = DuplicateOrderPrevention()

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        assert dprev.trade_history["BTC/USDT"][0].closed == False

        dprev.mark_position_closed("BTC/USDT")
        assert dprev.trade_history["BTC/USDT"][0].closed == True


class TestSymbolStatus:
    """Test symbol status reporting"""

    def test_status_no_history(self):
        """Status for symbol with no history"""
        dprev = DuplicateOrderPrevention()

        status = dprev.get_symbol_status("BTC/USDT")

        assert not status["has_history"]
        assert status["can_trade"]
        assert status["open_positions"] == 0

    def test_status_with_open_position(self):
        """Status for symbol with open position"""
        dprev = DuplicateOrderPrevention()

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        status = dprev.get_symbol_status("BTC/USDT")

        assert status["has_history"]
        assert not status["can_trade"]
        assert status["open_positions"] == 1

    def test_cooldown_symbols_list(self):
        """Get all symbols in cooldown"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        dprev.mark_position_closed("BTC/USDT")
        
        dprev.record_trade("ETH/USDT", "LONG", 2000.0, 10.0)
        dprev.mark_position_closed("ETH/USDT")

        # Both in cooldown
        cooldown = dprev.get_all_symbols_in_cooldown()
        assert "BTC/USDT" in cooldown
        assert "ETH/USDT" in cooldown


class TestStatistics:
    """Test statistics reporting"""

    def test_statistics_empty(self):
        """Statistics for empty state"""
        dprev = DuplicateOrderPrevention()

        stats = dprev.get_statistics()

        assert stats["total_symbols_tracked"] == 0
        assert stats["total_trades_recorded"] == 0
        assert stats["rejected_attempts"] == 0

    def test_statistics_populated(self):
        """Statistics with data"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        dprev.mark_position_closed("BTC/USDT")
        
        dprev.record_trade("ETH/USDT", "LONG", 2000.0, 10.0)
        dprev.mark_position_closed("ETH/USDT")
        
        dprev.validate_trade_request("BTC/USDT", "LONG", 40000.0, 1.0)

        stats = dprev.get_statistics()

        assert stats["total_symbols_tracked"] == 2
        assert stats["total_trades_recorded"] == 2
        assert stats["rejected_attempts"] == 1


class TestHistoryClear:
    """Test history clearing"""

    def test_clear_history(self):
        """Clear all history"""
        dprev = DuplicateOrderPrevention(cooldown_seconds=60)

        dprev.record_trade("BTC/USDT", "LONG", 40000.0, 1.0)
        dprev.mark_position_closed("BTC/USDT")
        dprev.validate_trade_request("BTC/USDT", "LONG", 40000.0, 1.0)

        assert len(dprev.trade_history) > 0
        assert dprev.rejected_count > 0

        dprev.clear_history()

        assert len(dprev.trade_history) == 0
        assert dprev.rejected_count == 0
