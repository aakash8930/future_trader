"""
Tests for Exchange Reconciliation Engine

Validates position and balance reconciliation against exchange state.
"""

import pytest
from datetime import datetime, timedelta
from execution.exchange_reconciliator import (
    ExchangeReconciliator,
    ExchangePosition,
    DesyncEvent,
    ReconciliationReport,
)


class TestPositionReconciliation:
    """Test position reconciliation against exchange"""

    def test_exact_position_match(self):
        """Positions match exactly - healthy"""
        reconciliator = ExchangeReconciliator()

        local_positions = {
            "BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}
        }

        exchange_positions = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT",
                side="LONG",
                qty=1.0,
                entry_price=40000.0,
                unrealized_pnl=1000.0,
            )
        }

        report = reconciliator.reconcile_positions(
            local_positions, exchange_positions
        )

        assert report.is_healthy
        assert report.mismatches_found == 0
        assert len(report.desync_events) == 0

    def test_qty_mismatch_warning(self):
        """Qty mismatch detected and logged"""
        reconciliator = ExchangeReconciliator(max_qty_diff=0.01)

        local_positions = {
            "BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}
        }

        exchange_positions = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT",
                side="LONG",
                qty=1.05,  # 0.05 diff, ~5% - warning
                entry_price=40000.0,
                unrealized_pnl=1000.0,
            )
        }

        report = reconciliator.reconcile_positions(
            local_positions, exchange_positions
        )

        assert not report.is_healthy
        assert report.mismatches_found == 1
        assert len(report.desync_events) == 1

        event = report.desync_events[0]
        assert event.type == "QTY_MISMATCH"
        assert event.local_value == 1.0
        assert event.exchange_value == 1.05
        assert event.severity == "WARNING"

    def test_position_missing_on_exchange(self):
        """Local position exists but not on exchange - critical"""
        reconciliator = ExchangeReconciliator()

        local_positions = {
            "BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0},
            "ETH/USDT": {"qty": 10.0, "side": "LONG", "entry_price": 2000.0},
        }

        exchange_positions = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT",
                side="LONG",
                qty=1.0,
                entry_price=40000.0,
                unrealized_pnl=1000.0,
            )
        }

        report = reconciliator.reconcile_positions(
            local_positions, exchange_positions
        )

        assert not report.is_healthy
        assert report.mismatches_found == 1

        event = report.desync_events[0]
        assert event.symbol == "ETH/USDT"
        assert event.type == "POSITION_MISMATCH"
        assert event.severity == "CRITICAL"

    def test_position_on_exchange_not_local(self):
        """Exchange has position not in local - critical"""
        reconciliator = ExchangeReconciliator()

        local_positions = {
            "BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}
        }

        exchange_positions = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT",
                side="LONG",
                qty=1.0,
                entry_price=40000.0,
                unrealized_pnl=1000.0,
            ),
            "ETH/USDT": ExchangePosition(
                symbol="ETH/USDT",
                side="SHORT",
                qty=5.0,
                entry_price=2500.0,
                unrealized_pnl=-200.0,
            ),
        }

        report = reconciliator.reconcile_positions(
            local_positions, exchange_positions
        )

        assert not report.is_healthy
        assert report.mismatches_found == 1

        event = report.desync_events[0]
        assert event.symbol == "ETH/USDT"
        assert event.type == "POSITION_MISMATCH"
        assert event.exchange_value == 5.0
        assert event.local_value == 0.0

    def test_multiple_mismatches_tracked(self):
        """Multiple mismatches tracked and reported"""
        reconciliator = ExchangeReconciliator(max_qty_diff=0.01)

        local_positions = {
            "BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0},
            "ETH/USDT": {"qty": 10.0, "side": "LONG", "entry_price": 2000.0},
            "SOL/USDT": {"qty": 100.0, "side": "LONG", "entry_price": 100.0},
        }

        exchange_positions = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            ),
            "ETH/USDT": ExchangePosition(
                symbol="ETH/USDT", side="LONG", qty=0.0, entry_price=2000.0, unrealized_pnl=0
            ),
            # SOL/USDT missing on exchange
        }

        report = reconciliator.reconcile_positions(
            local_positions, exchange_positions
        )

        assert not report.is_healthy
        assert report.mismatches_found == 3  # BTC qty, ETH missing, SOL missing


class TestConsecutiveDesync:
    """Test consecutive desync counter and critical threshold"""

    def test_consecutive_desync_increments(self):
        """Consecutive desync counter increments on each mismatch"""
        reconciliator = ExchangeReconciliator(desync_threshold=3)

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            )
        }

        # First desync
        report1 = reconciliator.reconcile_positions(local, exchange)
        assert reconciliator.consecutive_desync_count == 1
        assert not report1.is_healthy

        # Second desync
        report2 = reconciliator.reconcile_positions(local, exchange)
        assert reconciliator.consecutive_desync_count == 2
        assert not report2.is_healthy

        # Third desync - hits threshold
        report3 = reconciliator.reconcile_positions(local, exchange)
        assert reconciliator.consecutive_desync_count == 3
        assert reconciliator.is_critical_desync()

    def test_desync_counter_resets_on_healthy(self):
        """Counter resets to 0 on healthy reconciliation"""
        reconciliator = ExchangeReconciliator(desync_threshold=3)

        local_bad = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange_bad = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            )
        }

        # Trigger desync
        reconciliator.reconcile_positions(local_bad, exchange_bad)
        assert reconciliator.consecutive_desync_count == 1

        # Now reconcile healthy
        local_good = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange_good = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.0, entry_price=40000.0, unrealized_pnl=0
            )
        }

        reconciliator.reconcile_positions(local_good, exchange_good)
        assert reconciliator.consecutive_desync_count == 0
        assert not reconciliator.is_critical_desync()

    def test_recommendations_on_critical_desync(self):
        """Critical alert recommendation added when threshold hit"""
        reconciliator = ExchangeReconciliator(desync_threshold=1)

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            )
        }

        report = reconciliator.reconcile_positions(local, exchange)

        assert reconciliator.is_critical_desync()
        assert any(
            "emergency closeout" in rec.lower() for rec in report.recommendations
        )


class TestBalanceReconciliation:
    """Test account balance verification"""

    def test_exact_balance_match(self):
        """Balances match exactly"""
        reconciliator = ExchangeReconciliator()

        result = reconciliator.reconcile_balance(
            local_balance=1000.0, exchange_balance=1000.0
        )

        assert result["match"]
        assert result["difference"] == 0.0
        assert result["difference_pct"] == 0.0

    def test_small_balance_difference_accepted(self):
        """Small difference within threshold"""
        reconciliator = ExchangeReconciliator(max_balance_diff_pct=0.5)

        result = reconciliator.reconcile_balance(
            local_balance=1000.0, exchange_balance=1004.0  # 0.4% diff
        )

        assert result["match"]
        assert abs(result["difference_pct"] - 0.4) < 0.01

    def test_large_balance_difference_flagged(self):
        """Large difference flagged as mismatch"""
        reconciliator = ExchangeReconciliator(max_balance_diff_pct=0.5)

        result = reconciliator.reconcile_balance(
            local_balance=1000.0, exchange_balance=1020.0  # ~2% diff
        )

        assert not result["match"]
        assert result["difference_pct"] > 1.5  # More than 1.5%
        assert "recommendation" in result

    def test_balance_difference_calculation(self):
        """Balance difference calculated correctly"""
        reconciliator = ExchangeReconciliator()

        result = reconciliator.reconcile_balance(
            local_balance=5000.0, exchange_balance=4950.0
        )

        assert result["difference"] == 50.0
        assert abs(result["difference_pct"] - 1.01) < 0.01


class TestReconciliationTiming:
    """Test reconciliation interval enforcement"""

    def test_should_reconcile_on_first_call(self):
        """First reconciliation always allowed"""
        reconciliator = ExchangeReconciliator()
        assert reconciliator.should_reconcile()

    def test_should_not_reconcile_immediately(self):
        """Second reconciliation within interval rejected"""
        reconciliator = ExchangeReconciliator(reconciliation_interval_seconds=60)

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.0, entry_price=40000.0, unrealized_pnl=0
            )
        }

        reconciliator.reconcile_positions(local, exchange)
        assert not reconciliator.should_reconcile()

    def test_should_reconcile_after_interval(self):
        """Reconciliation allowed after interval passes"""
        reconciliator = ExchangeReconciliator(reconciliation_interval_seconds=60)

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.0, entry_price=40000.0, unrealized_pnl=0
            )
        }

        reconciliator.reconcile_positions(local, exchange)
        reconciliator.last_reconciliation = datetime.utcnow() - timedelta(seconds=61)

        assert reconciliator.should_reconcile()


class TestDesyncHistory:
    """Test desync event history tracking"""

    def test_desync_events_recorded(self):
        """Desync events added to history"""
        reconciliator = ExchangeReconciliator()

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            )
        }

        reconciliator.reconcile_positions(local, exchange)
        assert len(reconciliator.desync_history) == 1
        assert reconciliator.desync_history[0].symbol == "BTC/USDT"

    def test_desync_history_filtered_by_time(self):
        """History filtered by time range"""
        reconciliator = ExchangeReconciliator()

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            )
        }

        # Create old event
        reconciliator.reconcile_positions(local, exchange)
        reconciliator.desync_history[0].timestamp = datetime.utcnow() - timedelta(
            hours=2
        )

        # Create new event
        reconciliator.reconcile_positions(local, exchange)

        recent = reconciliator.get_desync_history(hours=1)
        assert len(recent) == 1
        assert recent[0].timestamp > datetime.utcnow() - timedelta(hours=1)

    def test_desync_history_unlimited_retrieval(self):
        """Can retrieve all desync events"""
        reconciliator = ExchangeReconciliator()

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            )
        }

        for _ in range(3):
            reconciliator.reconcile_positions(local, exchange)

        all_events = reconciliator.get_desync_history(hours=24)
        assert len(all_events) == 3


class TestReconciliationStatus:
    """Test status reporting"""

    def test_status_includes_all_fields(self):
        """Status report contains all required fields"""
        reconciliator = ExchangeReconciliator()
        status = reconciliator.get_status()

        assert "last_reconciliation" in status
        assert "consecutive_desync_events" in status
        assert "is_critical" in status
        assert "desync_history_count" in status
        assert "should_reconcile_now" in status

    def test_status_reflects_desync_state(self):
        """Status accurately reflects desync"""
        reconciliator = ExchangeReconciliator(desync_threshold=2)

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            )
        }

        reconciliator.reconcile_positions(local, exchange)
        status = reconciliator.get_status()

        assert status["consecutive_desync_events"] == 1
        assert not status["is_critical"]

    def test_reset_desync_counter(self):
        """Manual reset of counter"""
        reconciliator = ExchangeReconciliator()

        local = {"BTC/USDT": {"qty": 1.0, "side": "LONG", "entry_price": 40000.0}}
        exchange = {
            "BTC/USDT": ExchangePosition(
                symbol="BTC/USDT", side="LONG", qty=1.1, entry_price=40000.0, unrealized_pnl=0
            )
        }

        reconciliator.reconcile_positions(local, exchange)
        assert reconciliator.consecutive_desync_count == 1

        reconciliator.reset_desync_counter()
        assert reconciliator.consecutive_desync_count == 0
