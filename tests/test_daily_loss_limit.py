"""
Tests for Daily Loss Limit Enforcement

Validates daily loss limit and shutdown behavior.
"""

import pytest
from risk.daily_loss_limit import DailyLossLimit, DailyLossRecord


class TestDailyLossInitialization:
    """Test daily loss limit initialization"""

    def test_initialize_day(self):
        """Initialize trading day"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)

        dll.initialize_day(1000.0)

        assert dll.starting_balance == 1000.0
        assert dll.daily_pnl == 0.0
        assert dll.trade_count == 0
        assert not dll.is_shutdown

    def test_reset_on_new_day(self):
        """Counters reset on new day"""
        dll = DailyLossLimit()

        dll.initialize_day(1000.0)
        dll.update_pnl(50.0)
        assert dll.daily_pnl == 50.0

        # Simulate new day by changing internal date
        # (in practice this would naturally happen at midnight)
        old_date = dll.today_date
        dll.today_date = "2099-01-02"  # Different date

        dll.initialize_day(1050.0)

        assert dll.daily_pnl == 0.0  # Reset
        assert dll.trade_count == 0  # Reset


class TestPnLTracking:
    """Test PnL tracking"""

    def test_update_pnl_positive(self):
        """Track positive PnL"""
        dll = DailyLossLimit()
        dll.initialize_day(1000.0)

        dll.update_pnl(50.0)

        assert dll.daily_pnl == 50.0
        assert dll.trade_count == 1

    def test_update_pnl_negative(self):
        """Track negative PnL"""
        dll = DailyLossLimit()
        dll.initialize_day(1000.0)

        dll.update_pnl(-30.0)

        assert dll.daily_pnl == -30.0
        assert dll.trade_count == 1

    def test_accumulate_pnl(self):
        """Accumulate multiple trades"""
        dll = DailyLossLimit()
        dll.initialize_day(1000.0)

        dll.update_pnl(50.0)
        dll.update_pnl(-20.0)
        dll.update_pnl(30.0)

        assert dll.daily_pnl == 60.0
        assert dll.trade_count == 3


class TestLossLimitEnforcement:
    """Test loss limit enforcement"""

    def test_within_limit(self):
        """Check within limit"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        dll.update_pnl(-50.0)

        check = dll.check_loss_limit()

        assert check["within_limit"]
        assert not check["emergency"]
        assert check["current_loss"] == 50.0

    def test_at_warning_threshold(self):
        """Warning at 50% of limit"""
        dll = DailyLossLimit(
            daily_loss_limit_usd=100.0, warning_threshold_pct=50.0
        )
        dll.initialize_day(1000.0)

        dll.update_pnl(-50.0)

        check = dll.check_loss_limit()

        assert check["within_limit"]
        assert check["warning"]
        assert not check["emergency"]

    def test_exceeds_limit(self):
        """Emergency when limit exceeded"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        dll.update_pnl(-150.0)

        check = dll.check_loss_limit()

        assert not check["within_limit"]
        assert check["emergency"]
        assert dll.is_shutdown
        assert dll.shutdown_reason is not None

    def test_exactly_at_limit(self):
        """Trigger at exact limit"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        dll.update_pnl(-100.0)

        check = dll.check_loss_limit()

        assert not check["within_limit"]
        assert check["emergency"]


class TestCanTrade:
    """Test can_trade status"""

    def test_can_trade_healthy(self):
        """Can trade when healthy"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        assert dll.can_trade()

    def test_cannot_trade_after_shutdown(self):
        """Cannot trade after shutdown"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        dll.update_pnl(-150.0)
        dll.enforce_loss_limit()

        assert not dll.can_trade()
        assert dll.is_shutdown


class TestShutdownEnforcement:
    """Test shutdown mechanism"""

    def test_enforce_triggers_shutdown(self):
        """Enforce method triggers shutdown"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        dll.update_pnl(-150.0)
        reason = dll.enforce_loss_limit()

        assert reason is not None
        assert dll.is_shutdown
        assert "exceeded" in reason.lower()

    def test_enforce_warning_no_shutdown(self):
        """Enforce at warning doesn't shutdown"""
        dll = DailyLossLimit(
            daily_loss_limit_usd=100.0, warning_threshold_pct=50.0
        )
        dll.initialize_day(1000.0)

        dll.update_pnl(-60.0)
        reason = dll.enforce_loss_limit()

        assert reason is None  # Warning, not shutdown
        assert not dll.is_shutdown

    def test_reset_shutdown(self):
        """Reset shutdown (admin function)"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        dll.update_pnl(-150.0)
        dll.enforce_loss_limit()

        assert dll.is_shutdown

        dll.reset_daily_shutdown()

        assert not dll.is_shutdown
        assert dll.shutdown_reason is None


class TestDailyStatus:
    """Test status reporting"""

    def test_status_includes_fields(self):
        """Status has all required fields"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        status = dll.get_daily_status()

        assert "date" in status
        assert "current_pnl" in status
        assert "current_loss" in status
        assert "daily_limit" in status
        assert "num_trades" in status
        assert "is_shutdown" in status

    def test_status_values(self):
        """Status values are correct"""
        dll = DailyLossLimit(daily_loss_limit_usd=100.0)
        dll.initialize_day(1000.0)

        dll.update_pnl(-50.0)
        dll.update_pnl(-20.0)

        status = dll.get_daily_status()

        assert status["current_pnl"] == -70.0
        assert status["current_loss"] == 70.0
        assert status["num_trades"] == 2
        assert status["loss_pct_of_limit"] == 70.0  # 70% of 100


class TestDailyRecording:
    """Test daily record tracking"""

    def test_record_daily_close(self):
        """Record day closing"""
        dll = DailyLossLimit()
        dll.initialize_day(1000.0)

        dll.update_pnl(50.0)

        record = dll.record_daily_close(1050.0)

        assert record.daily_pnl == 50.0
        assert record.starting_balance == 1000.0
        assert record.current_balance == 1050.0
        assert record.num_trades == 1

    def test_records_accumulated(self):
        """Multiple days recorded"""
        dll = DailyLossLimit()

        # Day 1
        dll.initialize_day(1000.0)
        dll.update_pnl(50.0)
        record1 = dll.record_daily_close(1050.0)

        assert record1.daily_pnl == 50.0

        # Simulate new day by modifying the date
        dll.today_date = "2099-01-02"

        # Day 2
        dll.initialize_day(1050.0)
        dll.update_pnl(-30.0)
        record2 = dll.record_daily_close(1020.0)

        assert len(dll.daily_records) == 2
        assert dll.daily_records[0].daily_pnl == 50.0
        assert dll.daily_records[1].daily_pnl == -30.0


class TestHistoryRetrieval:
    """Test history and statistics"""

    def test_get_daily_history_empty(self):
        """Empty history when no records"""
        dll = DailyLossLimit()

        history = dll.get_daily_history(days=7)

        assert len(history) == 0

    def test_get_daily_history_limited(self):
        """History limited to requested days"""
        dll = DailyLossLimit()

        # Create 5 days of history
        for i in range(5):
            dll.initialize_day(1000.0)
            dll.update_pnl(10.0 * i)
            dll.record_daily_close(1000.0 + 10.0 * i)

        history = dll.get_daily_history(days=3)

        assert len(history) == 3  # Only last 3 days

    def test_weekly_stats(self):
        """Weekly statistics calculation"""
        dll = DailyLossLimit()

        # 5 days: 3 winning, 2 losing
        days_data = [50.0, -30.0, 100.0, -20.0, 80.0]

        for idx, pnl in enumerate(days_data):
            # Set different date for each day
            dll.today_date = f"2099-01-{idx+1:02d}"
            dll.initialize_day(1000.0)
            dll.update_pnl(pnl)
            dll.trade_count = 1
            dll.record_daily_close(1000.0 + pnl)

        stats = dll.get_weekly_stats(days=7)

        assert stats["days_traded"] == 5
        assert abs(stats["total_pnl"] - 180.0) < 0.1  # Sum of PnLs
        assert stats["total_trades"] == 5
        assert stats["winning_days"] == 3
        assert stats["losing_days"] == 2
        assert abs(stats["win_rate_pct"] - 60.0) < 0.1  # 3/5 = 60%

    def test_weekly_stats_empty(self):
        """Weekly stats when no history"""
        dll = DailyLossLimit()

        stats = dll.get_weekly_stats(days=7)

        assert stats["days_traded"] == 0
        assert stats["total_pnl"] == 0.0
