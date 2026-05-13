"""
Tests for Phase 5: Observability
"""

import pytest
import json
import tempfile
from pathlib import Path

from execution.observability import StructuredLogger, PerformanceAnalytics


class TestStructuredLogger:
    """Test structured logging"""

    def test_logger_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            assert logger.log_dir == Path(tmpdir)

    def test_log_event_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            logger.log_event("test_event", key="value", number=42)

            log_file = Path(tmpdir) / "trading_structured.jsonl"
            assert log_file.exists()

            # Read log file
            with open(log_file) as f:
                line = f.readline()
                data = json.loads(line)

            assert data["event_type"] == "test_event"
            assert data["key"] == "value"
            assert data["number"] == 42
            assert "timestamp" in data

    def test_log_trade_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            logger.log_trade_entry(
                symbol="BTC/USDT",
                side="LONG",
                qty=1.0,
                entry_price=50000.0,
                leverage=2.0,
                stop_loss=49000.0,
                take_profit=52000.0,
                confidence=0.75,
                reason="ML signal",
            )

            log_file = Path(tmpdir) / "trading_structured.jsonl"
            with open(log_file) as f:
                data = json.loads(f.readline())

            assert data["event_type"] == "trade_entry"
            assert data["symbol"] == "BTC/USDT"
            assert data["side"] == "LONG"
            assert data["model_confidence"] == 0.75

    def test_log_trade_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            logger.log_trade_exit(
                symbol="BTC/USDT",
                side="LONG",
                entry_price=50000.0,
                exit_price=52000.0,
                qty=1.0,
                realized_pnl=2000.0,
                realized_pnl_pct=0.04,
                reason="TP hit",
                bars_held=10,
            )

            log_file = Path(tmpdir) / "trading_structured.jsonl"
            with open(log_file) as f:
                data = json.loads(f.readline())

            assert data["event_type"] == "trade_exit"
            assert data["realized_pnl_usd"] == 2000.0
            assert data["realized_pnl_pct"] == 0.04
            assert data["bars_held"] == 10

    def test_log_signal_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            logger.log_signal_rejected(
                symbol="BTC/USDT",
                side="LONG",
                confidence=0.48,
                reason="Confidence too low",
                filters_failed={"confidence": False, "volatility": True},
            )

            log_file = Path(tmpdir) / "trading_structured.jsonl"
            with open(log_file) as f:
                data = json.loads(f.readline())

            assert data["event_type"] == "signal_rejected"
            assert data["proposed_side"] == "LONG"
            assert data["model_confidence"] == 0.48

    def test_log_risk_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            logger.log_risk_guard_triggered(
                guard_type="leverage_guard",
                severity="WARNING",
                message="Position leverage at 90% of limit",
                leverage_ratio=0.90,
            )

            log_file = Path(tmpdir) / "trading_structured.jsonl"
            with open(log_file) as f:
                data = json.loads(f.readline())

            assert data["event_type"] == "risk_guard_triggered"
            assert data["guard_type"] == "leverage_guard"

    def test_log_balance_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            logger.log_balance_update(
                previous_balance=10000.0,
                current_balance=10500.0,
                pnl=500.0,
                pnl_pct=0.05,
                reason="Trade exit",
            )

            log_file = Path(tmpdir) / "trading_structured.jsonl"
            with open(log_file) as f:
                data = json.loads(f.readline())

            assert data["event_type"] == "balance_update"
            assert data["pnl_usd"] == 500.0
            assert data["pnl_pct"] == 0.05


class TestPerformanceAnalytics:
    """Test performance analytics"""

    def test_expectancy_no_trades(self):
        analytics = PerformanceAnalytics([])
        expectancy = analytics.calculate_expectancy()

        assert expectancy["expectancy"] == 0.0
        assert expectancy["trades"] == 0

    def test_expectancy_single_win(self):
        trades = [{"realized_pnl": 100.0}]
        analytics = PerformanceAnalytics(trades)
        expectancy = analytics.calculate_expectancy()

        assert expectancy["win_rate"] == 1.0
        assert expectancy["total_trades"] == 1
        assert expectancy["expectancy_per_trade"] == 100.0

    def test_expectancy_single_loss(self):
        trades = [{"realized_pnl": -100.0}]
        analytics = PerformanceAnalytics(trades)
        expectancy = analytics.calculate_expectancy()

        assert expectancy["loss_rate"] == 1.0
        assert expectancy["expectancy_per_trade"] == -100.0

    def test_expectancy_mixed(self):
        trades = [
            {"realized_pnl": 100.0},
            {"realized_pnl": 100.0},
            {"realized_pnl": -50.0},
        ]
        analytics = PerformanceAnalytics(trades)
        expectancy = analytics.calculate_expectancy()

        assert expectancy["total_trades"] == 3
        assert expectancy["win_rate"] == 2 / 3
        assert expectancy["loss_rate"] == 1 / 3
        assert expectancy["avg_win"] == 100.0
        assert expectancy["avg_loss"] == -50.0

    def test_sharpe_ratio_single_trade(self):
        trades = [{"realized_pnl_pct": 0.05}]
        analytics = PerformanceAnalytics(trades)
        sharpe = analytics.calculate_sharpe_ratio()

        # Single trade, Sharpe should be 0
        assert sharpe == 0.0

    def test_sharpe_ratio_consistent_wins(self):
        trades = [
            {"realized_pnl_pct": 0.05},
            {"realized_pnl_pct": 0.05},
            {"realized_pnl_pct": 0.05},
        ]
        analytics = PerformanceAnalytics(trades)
        sharpe = analytics.calculate_sharpe_ratio()

        # Consistent returns = low/zero volatility = high or zero Sharpe
        assert sharpe >= 0

    def test_sharpe_ratio_volatile_returns(self):
        trades = [
            {"realized_pnl_pct": 0.10},
            {"realized_pnl_pct": -0.05},
            {"realized_pnl_pct": 0.15},
            {"realized_pnl_pct": -0.08},
        ]
        analytics = PerformanceAnalytics(trades)
        sharpe = analytics.calculate_sharpe_ratio()

        # Volatile returns = higher volatility
        assert sharpe is not None

    def test_max_drawdown_no_trades(self):
        analytics = PerformanceAnalytics([])
        dd = analytics.calculate_max_drawdown()

        assert dd["max_drawdown_pct"] == 0.0

    def test_max_drawdown_monotonic_gains(self):
        trades = [
            {"realized_pnl": 100.0},
            {"realized_pnl": 100.0},
            {"realized_pnl": 100.0},
        ]
        analytics = PerformanceAnalytics(trades)
        dd = analytics.calculate_max_drawdown()

        # No drawdown on consistent gains
        assert dd["max_drawdown_pct"] == 0.0

    def test_max_drawdown_with_loss(self):
        trades = [
            {"realized_pnl": 100.0},
            {"realized_pnl": 100.0},
            {"realized_pnl": -50.0},  # Drawdown
        ]
        analytics = PerformanceAnalytics(trades)
        dd = analytics.calculate_max_drawdown()

        # Should have some drawdown
        assert dd["max_drawdown_pct"] > 0

    def test_summary_stats(self):
        trades = [
            {"realized_pnl": 100.0},
            {"realized_pnl": 100.0},
            {"realized_pnl": -50.0},
        ]
        analytics = PerformanceAnalytics(trades)
        stats = analytics.get_summary_stats()

        assert stats["total_trades"] == 3
        assert stats["total_pnl_usd"] == 150.0
        assert stats["avg_pnl_per_trade"] == 50.0
        assert "win_rate" in stats
        assert "expectancy" in stats
        assert "sharpe_ratio" in stats
        assert "max_drawdown_pct" in stats
