"""
Tests for Emergency Shutdown Mode

Validates emergency shutdown and recovery behavior.
"""

import pytest
from datetime import datetime
from execution.emergency_shutdown import (
    EmergencyShutdown,
    ShutdownEvent,
)


class TestShutdownInitiation:
    """Test emergency shutdown initiation"""

    def test_initiate_shutdown(self):
        """Initiate emergency shutdown"""
        es = EmergencyShutdown()

        event = es.initiate_shutdown(
            reason="Circuit breaker triggered",
            positions_to_close=3,
            orders_to_cancel=2,
        )

        assert es.is_active
        assert es.shutdown_reason == "Circuit breaker triggered"
        assert event.positions_to_close == 3
        assert event.orders_to_cancel == 2
        assert event.status == "initiated"

    def test_cannot_trade_during_shutdown(self):
        """Cannot trade during shutdown"""
        es = EmergencyShutdown()

        assert es.can_trade()

        es.initiate_shutdown("Test shutdown")

        assert not es.can_trade()

    def test_shutdown_timestamp(self):
        """Shutdown records timestamp"""
        es = EmergencyShutdown()

        before = datetime.utcnow()
        es.initiate_shutdown("Test")
        after = datetime.utcnow()

        assert before <= es.initiated_at <= after


class TestPositionClosing:
    """Test position closure tracking"""

    def test_queue_position_close(self):
        """Queue position for closure"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        result = es.close_position("BTC/USDT")

        assert result  # Success
        assert "BTC/USDT" in es.pending_close_orders

    def test_multiple_positions_queued(self):
        """Queue multiple positions"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        es.close_position("BTC/USDT")
        es.close_position("ETH/USDT")
        es.close_position("SOL/USDT")

        assert len(es.pending_close_orders) == 3

    def test_process_position_closure(self):
        """Process position closure"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        es.close_position("BTC/USDT")
        assert len(es.pending_close_orders) == 1

        es.process_close("BTC/USDT")

        assert len(es.pending_close_orders) == 0
        assert "BTC/USDT" in es.closed_positions

    def test_cannot_reclose_position(self):
        """Cannot close same position twice"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        result1 = es.close_position("BTC/USDT")
        es.process_close("BTC/USDT")

        result2 = es.close_position("BTC/USDT")

        assert result1  # First close OK
        assert not result2  # Second close rejected


class TestOrderCancellation:
    """Test order cancellation tracking"""

    def test_queue_order_cancel(self):
        """Queue order for cancellation"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        result = es.cancel_order("order_123")

        assert result  # Success
        assert "order_123" in es.cancelled_orders

    def test_multiple_orders_cancelled(self):
        """Cancel multiple orders"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        es.cancel_order("order_1")
        es.cancel_order("order_2")
        es.cancel_order("order_3")

        assert len(es.cancelled_orders) == 3

    def test_process_order_cancellation(self):
        """Process order cancellation"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        es.cancel_order("order_123")
        assert len(es.cancelled_orders) == 1

        es.process_cancel("order_123")

        assert "order_123" in es.cancelled_orders

    def test_cannot_recancel_order(self):
        """Cannot cancel same order twice"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        result1 = es.cancel_order("order_123")
        result2 = es.cancel_order("order_123")

        assert result1  # First cancel OK
        assert not result2  # Second cancel rejected


class TestShutdownStatus:
    """Test status reporting"""

    def test_status_before_shutdown(self):
        """Status when no shutdown"""
        es = EmergencyShutdown()

        status = es.get_shutdown_status()

        assert not status["is_active"]
        assert status["can_trade"]
        assert status["positions_pending_close"] == 0

    def test_status_during_shutdown(self):
        """Status during shutdown"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test", positions_to_close=2)
        es.close_position("BTC/USDT")
        es.close_position("ETH/USDT")

        status = es.get_shutdown_status()

        assert status["is_active"]
        assert not status["can_trade"]
        assert status["reason"] == "Test"
        assert status["positions_pending_close"] == 2


class TestRecovery:
    """Test shutdown recovery"""

    def test_recovery_disabled_by_default(self):
        """Recovery disabled by default"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        assert not es.recovery_allowed()
        assert not es.enable_auto_recovery

    def test_recovery_enabled(self):
        """Enable auto recovery"""
        es = EmergencyShutdown(enable_auto_recovery=True)
        es.initiate_shutdown("Test")

        # All clean, should allow recovery
        assert es.recovery_allowed()

    def test_recovery_blocked_by_pending_closes(self):
        """Recovery blocked if positions pending close"""
        es = EmergencyShutdown(enable_auto_recovery=True)
        es.initiate_shutdown("Test")
        es.close_position("BTC/USDT")

        assert not es.recovery_allowed()

    def test_recovery_blocked_by_pending_cancels(self):
        """Recovery blocked if orders pending cancel"""
        es = EmergencyShutdown(enable_auto_recovery=True)
        es.initiate_shutdown("Test")
        es.cancel_order("order_123")

        assert not es.recovery_allowed()

    def test_successful_recovery(self):
        """Successful recovery from shutdown"""
        es = EmergencyShutdown(enable_auto_recovery=True)
        es.initiate_shutdown("Test")

        assert es.is_active

        result = es.attempt_recovery()

        assert result  # Success
        assert not es.is_active
        assert es.can_trade()

    def test_recovery_fails_without_cleanup(self):
        """Recovery fails if cleanup incomplete"""
        es = EmergencyShutdown(enable_auto_recovery=True)
        es.initiate_shutdown("Test")
        es.close_position("BTC/USDT")

        result = es.attempt_recovery()

        assert not result  # Failed
        assert es.is_active  # Still shutdown


class TestForceComplete:
    """Test forced shutdown completion"""

    def test_force_complete_shutdown(self):
        """Force complete shutdown"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")
        es.close_position("BTC/USDT")
        es.cancel_order("order_123")

        assert len(es.pending_close_orders) > 0
        assert len(es.cancelled_orders) > 0

        es.force_complete_shutdown()

        assert len(es.pending_close_orders) == 0
        assert len(es.cancelled_orders) == 0
        assert es.shutdown_events[-1].status == "complete"


class TestPendingActions:
    """Test pending actions reporting"""

    def test_get_pending_actions(self):
        """Get all pending actions"""
        es = EmergencyShutdown()
        es.initiate_shutdown("Test")

        es.close_position("BTC/USDT")
        es.close_position("ETH/USDT")
        es.cancel_order("order_1")
        es.cancel_order("order_2")

        pending = es.get_pending_actions()

        assert len(pending["positions_to_close"]) == 2
        assert len(pending["orders_to_cancel"]) == 2
        assert "BTC/USDT" in pending["positions_to_close"]
        assert "order_1" in pending["orders_to_cancel"]

    def test_pending_actions_empty(self):
        """Pending actions empty when none"""
        es = EmergencyShutdown()

        pending = es.get_pending_actions()

        assert len(pending["positions_to_close"]) == 0
        assert len(pending["orders_to_cancel"]) == 0


class TestShutdownHistory:
    """Test shutdown event history"""

    def test_single_shutdown_recorded(self):
        """Single shutdown recorded"""
        es = EmergencyShutdown()

        es.initiate_shutdown("Test reason")

        history = es.get_shutdown_history()

        assert len(history) == 1
        assert history[0].reason == "Test reason"

    def test_multiple_shutdowns_recorded(self):
        """Multiple shutdowns recorded"""
        es = EmergencyShutdown(enable_auto_recovery=True)

        es.initiate_shutdown("Reason 1")
        es.attempt_recovery()

        es.initiate_shutdown("Reason 2")
        es.attempt_recovery()

        history = es.get_shutdown_history()

        assert len(history) == 2
        assert history[0].reason == "Reason 1"
        assert history[1].reason == "Reason 2"


class TestShutdownEvent:
    """Test shutdown event structure"""

    def test_shutdown_event_fields(self):
        """Shutdown event has all fields"""
        es = EmergencyShutdown()

        event = es.initiate_shutdown(
            reason="Test",
            positions_to_close=5,
            orders_to_cancel=3,
        )

        assert event.timestamp is not None
        assert event.reason == "Test"
        assert event.positions_to_close == 5
        assert event.orders_to_cancel == 3
        assert event.status == "initiated"
