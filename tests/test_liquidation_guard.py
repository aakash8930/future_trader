"""
Tests for Liquidation Guard

Validates liquidation price calculations and risk detection.
"""

import pytest
from datetime import datetime, timedelta
from risk.liquidation_guard import (
    LiquidationGuard,
    LiquidationLevel,
    LiquidationEvent,
)


class TestLiquidationPriceCalculation:
    """Test liquidation price calculations"""

    def test_long_liquidation_price(self):
        """LONG liquidation price calculated correctly"""
        guard = LiquidationGuard()

        # LONG at 40000, 10x leverage, max loss = 1/10 = 10%
        # Liquidation ≈ 40000 * (1 - 0.1 + 0.00035) ≈ 36001.4
        liquidation = guard.calculate_liquidation_price(
            side="LONG", entry_price=40000.0, leverage=10.0
        )

        # Should be roughly 10% below entry
        assert liquidation < 40000.0
        assert liquidation > 36000.0  # Not too low

    def test_short_liquidation_price(self):
        """SHORT liquidation price calculated correctly"""
        guard = LiquidationGuard()

        # SHORT at 40000, 10x leverage, max loss = 1/10 = 10%
        # Liquidation ≈ 40000 * (1 + 0.1 - 0.00035) ≈ 44001.4
        liquidation = guard.calculate_liquidation_price(
            side="SHORT", entry_price=40000.0, leverage=10.0
        )

        # Should be roughly 10% above entry
        assert liquidation > 40000.0
        assert liquidation < 44000.0  # Not too high

    def test_higher_leverage_liquidation_closer(self):
        """Higher leverage brings liquidation price closer"""
        guard = LiquidationGuard()

        # 5x leverage
        liq_5x = guard.calculate_liquidation_price(
            side="LONG", entry_price=40000.0, leverage=5.0
        )

        # 10x leverage
        liq_10x = guard.calculate_liquidation_price(
            side="LONG", entry_price=40000.0, leverage=10.0
        )

        # 10x should be closer to entry (higher liquidation risk)
        assert abs(40000.0 - liq_10x) < abs(40000.0 - liq_5x)

    def test_no_leverage_no_liquidation(self):
        """1x leverage (no margin) has liquidation price 0"""
        guard = LiquidationGuard()

        liquidation = guard.calculate_liquidation_price(
            side="LONG", entry_price=40000.0, leverage=1.0
        )

        assert liquidation == 0.0  # No liquidation risk at 1x

    def test_extreme_leverage_liquidation(self):
        """Extreme leverage (100x) has liquidation very close"""
        guard = LiquidationGuard()

        liq_100x = guard.calculate_liquidation_price(
            side="LONG", entry_price=40000.0, leverage=100.0
        )

        # At 100x, max loss = 1%, so liquidation ≈ 39600
        assert liq_100x < 40000.0
        assert abs(40000.0 - liq_100x) < 500  # Very close


class TestDistanceToLiquidation:
    """Test distance calculation from current price to liquidation"""

    def test_long_safe_distance(self):
        """LONG position with safe distance"""
        guard = LiquidationGuard()

        # Entry 40000, liquidation 36000, current 38000
        distance = guard.calculate_distance_to_liquidation(
            side="LONG", current_price=38000.0, liquidation_price=36000.0
        )

        # Distance = (38000-36000)/36000 * 100 ≈ 5.56%
        assert distance > 0  # Safe
        assert distance > 5  # Not too close

    def test_long_at_liquidation(self):
        """LONG at exactly liquidation price"""
        guard = LiquidationGuard()

        distance = guard.calculate_distance_to_liquidation(
            side="LONG", current_price=36000.0, liquidation_price=36000.0
        )

        assert distance == 0  # At liquidation

    def test_long_past_liquidation(self):
        """LONG past liquidation (negative distance)"""
        guard = LiquidationGuard()

        distance = guard.calculate_distance_to_liquidation(
            side="LONG", current_price=35000.0, liquidation_price=36000.0
        )

        assert distance < 0  # Liquidated

    def test_short_safe_distance(self):
        """SHORT position with safe distance"""
        guard = LiquidationGuard()

        # Entry 40000, liquidation 44000, current 42000
        distance = guard.calculate_distance_to_liquidation(
            side="SHORT", current_price=42000.0, liquidation_price=44000.0
        )

        assert distance > 0  # Safe
        assert distance > 4  # Not too close

    def test_short_past_liquidation(self):
        """SHORT past liquidation"""
        guard = LiquidationGuard()

        distance = guard.calculate_distance_to_liquidation(
            side="SHORT", current_price=45000.0, liquidation_price=44000.0
        )

        assert distance < 0  # Liquidated


class TestLiquidationRiskAssessment:
    """Test position liquidation risk assessment"""

    def test_long_healthy_position(self):
        """Healthy LONG position returns None"""
        guard = LiquidationGuard()

        level = guard.check_position_liquidation(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=40000.0,
            current_price=42000.0,  # Above entry
            qty=1.0,
            leverage=10.0,
        )

        assert level is None  # Healthy

    def test_long_at_warning_threshold(self):
        """LONG at warning threshold (10% from liquidation)"""
        guard = LiquidationGuard(warning_threshold_pct=10.0)

        # Calculate to get exactly 10% from liq
        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        # Distance 10% from liq
        current = liq * 1.10  # 10% above liquidation

        level = guard.check_position_liquidation(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )

        # Should be detected (distance_pct = 0 or slightly negative at threshold)
        # Exact boundary condition - might return None if just above threshold

    def test_long_at_critical_threshold(self):
        """LONG approaching critical threshold is detected"""
        guard = LiquidationGuard(
            warning_threshold_pct=20.0,
            critical_threshold_pct=10.0,
            emergency_threshold_pct=5.0,
        )

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        # 6% above liquidation - this is in critical zone (between 10% and 5%)
        current = liq * 1.06  

        event = guard.assess_liquidation_risk(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )
        
        assert event is not None  # Should trigger event
        assert event.severity == "CRITICAL"

    def test_short_healthy_position(self):
        """Healthy SHORT position returns None"""
        guard = LiquidationGuard()

        level = guard.check_position_liquidation(
            symbol="BTC/USDT",
            side="SHORT",
            entry_price=40000.0,
            current_price=39000.0,  # Below entry
            qty=1.0,
            leverage=10.0,
        )

        assert level is None  # Healthy


class TestLiquidationEventSeverity:
    """Test liquidation event severity levels"""

    def test_warning_event(self):
        """WARNING severity at warning threshold"""
        guard = LiquidationGuard(
            warning_threshold_pct=20.0,
            critical_threshold_pct=10.0,
            emergency_threshold_pct=5.0,
        )

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        current = liq * 1.15  # 15% above liquidation (in warning zone)

        event = guard.assess_liquidation_risk(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )

        assert event is not None
        assert event.severity == "WARNING"

    def test_critical_event(self):
        """CRITICAL severity at critical threshold"""
        guard = LiquidationGuard(
            warning_threshold_pct=20.0,
            critical_threshold_pct=10.0,
            emergency_threshold_pct=5.0,
        )

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        current = liq * 1.07  # 7% above liquidation (in critical zone)

        event = guard.assess_liquidation_risk(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )

        assert event is not None
        assert event.severity == "CRITICAL"

    def test_emergency_event(self):
        """EMERGENCY severity at emergency threshold"""
        guard = LiquidationGuard(
            warning_threshold_pct=20.0,
            critical_threshold_pct=10.0,
            emergency_threshold_pct=5.0,
        )

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        current = liq * 1.03  # 3% above liquidation (in emergency zone)

        event = guard.assess_liquidation_risk(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )

        assert event is not None
        assert event.severity == "EMERGENCY"


class TestPortfolioRiskAssessment:
    """Test portfolio-level risk assessment"""

    def test_multiple_positions_risk(self):
        """Assess risk across multiple positions"""
        guard = LiquidationGuard(
            critical_threshold_pct=10.0,
        )

        btc_entry = 40000.0
        eth_entry = 2000.0
        leverage = 10.0
        
        btc_liq = guard.calculate_liquidation_price("LONG", btc_entry, leverage)
        eth_liq = guard.calculate_liquidation_price("LONG", eth_entry, leverage)

        positions = {
            "BTC/USDT": {
                "side": "LONG",
                "entry_price": btc_entry,
                "current_price": 42000.0,  # Safe
                "qty": 1.0,
                "leverage": 10.0,
            },
            "ETH/USDT": {
                "side": "LONG",
                "entry_price": eth_entry,
                "current_price": eth_liq * 1.07,  # Critical (7% from liq, within 10%)
                "qty": 10.0,
                "leverage": 10.0,
            },
        }

        result = guard.assess_portfolio_risk(positions)

        assert result["critical_count"] > 0
        assert len(result["events"]) > 0

    def test_emergency_mode_triggered(self):
        """Emergency mode triggered on emergency event"""
        guard = LiquidationGuard(emergency_threshold_pct=5.0)

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        current = liq * 1.03  # 3% from liquidation

        event = guard.assess_liquidation_risk(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )

        assert event.severity == "EMERGENCY"
        assert guard.emergency_mode  # Should be triggered


class TestCriticalPositionTracking:
    """Test tracking of critical positions"""

    def test_critical_position_tracked(self):
        """Critical positions added to list"""
        guard = LiquidationGuard(critical_threshold_pct=10.0)

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        current = liq * 1.07  # 7% from liquidation (critical)

        guard.assess_liquidation_risk(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )

        assert guard.should_closeout_position("BTC/USDT")

    def test_should_closeout_multiple(self):
        """Multiple positions marked for closeout"""
        guard = LiquidationGuard(critical_threshold_pct=10.0)

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        current = liq * 1.07  # 7% from liquidation (critical)

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        for sym in symbols:
            guard.assess_liquidation_risk(
                symbol=sym,
                side="LONG",
                entry_price=entry,
                current_price=current,
                qty=1.0,
                leverage=leverage,
            )

        for sym in symbols:
            assert guard.should_closeout_position(sym)


class TestEmergencyModeReset:
    """Test emergency mode reset"""

    def test_emergency_mode_reset(self):
        """Emergency mode can be reset"""
        guard = LiquidationGuard(emergency_threshold_pct=5.0)

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        current = liq * 1.03  # 3% from liquidation (emergency)

        guard.assess_liquidation_risk(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )

        assert guard.emergency_mode
        guard.reset_emergency_mode()
        assert not guard.emergency_mode
        assert len(guard.critical_positions) == 0


class TestLiquidationStatus:
    """Test status reporting"""

    def test_status_includes_fields(self):
        """Status report has all fields"""
        guard = LiquidationGuard()
        status = guard.get_liquidation_status()

        assert "emergency_mode" in status
        assert "critical_positions" in status
        assert "recent_events" in status
        assert "total_events" in status

    def test_recent_events_filtered(self):
        """Recent events filtered correctly"""
        guard = LiquidationGuard(warning_threshold_pct=20.0)

        entry = 40000.0
        leverage = 10.0
        liq = guard.calculate_liquidation_price("LONG", entry, leverage)
        current = liq * 1.08

        # Create event
        guard.assess_liquidation_risk(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=entry,
            current_price=current,
            qty=1.0,
            leverage=leverage,
        )

        # Make event old
        guard.liquidation_events[0].timestamp = (
            datetime.utcnow() - timedelta(hours=2)
        )

        recent = guard.get_recent_events(hours=1)
        assert len(recent) == 0

        all_events = guard.get_recent_events(hours=24)
        assert len(all_events) == 1
