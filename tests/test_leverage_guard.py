"""
Tests for Leverage Guard

Verifies leverage enforcement and position sizing validation.
"""

import pytest
from risk.leverage_guard import LeverageGuard, validate_leverage_for_entry
from execution.position import Position
from datetime import datetime


class TestLeverageValidation:
    """Test per-trade leverage validation."""
    
    def test_valid_leverage_within_limits(self):
        """Accept leverage within safe limits."""
        guard = LeverageGuard(max_per_trade=10.0)
        
        is_valid, reason = guard.validate_leverage_for_trade(5.0, "BTC/USDT")
        assert is_valid
        assert reason is None
    
    def test_reject_leverage_exceeds_max(self):
        """Reject leverage above maximum."""
        guard = LeverageGuard(max_per_trade=10.0)
        
        is_valid, reason = guard.validate_leverage_for_trade(15.0, "ETH/USDT")
        assert not is_valid
        assert "exceeds max" in reason.lower()
    
    def test_reject_leverage_too_low(self):
        """Reject leverage below minimum (< 1.0)."""
        guard = LeverageGuard(min_leverage=1.0)
        
        is_valid, reason = guard.validate_leverage_for_trade(0.5, "SOL/USDT")
        assert not is_valid
        assert "below minimum" in reason.lower()
    
    def test_accept_exact_max_leverage(self):
        """Accept leverage exactly at maximum."""
        guard = LeverageGuard(max_per_trade=10.0)
        
        is_valid, reason = guard.validate_leverage_for_trade(10.0, "AVAX/USDT")
        assert is_valid
        assert reason is None
    
    def test_accept_exact_min_leverage(self):
        """Accept leverage exactly at minimum."""
        guard = LeverageGuard(min_leverage=1.0)
        
        is_valid, reason = guard.validate_leverage_for_trade(1.0, "LINK/USDT")
        assert is_valid
        assert reason is None
    
    def test_warning_on_high_leverage(self):
        """Warn when leverage approaches maximum."""
        guard = LeverageGuard(max_per_trade=10.0, warn_threshold=0.80)
        
        is_valid, reason = guard.validate_leverage_for_trade(8.5, "BTC/USDT")
        assert is_valid
        assert len(guard.warnings) > 0
        assert "using" in guard.warnings[0].lower() and "max leverage" in guard.warnings[0].lower()


class TestAccountLeverage:
    """Test account-level leverage validation."""
    
    def test_no_positions_is_valid(self):
        """Account with no positions is always valid."""
        guard = LeverageGuard(max_account=5.0)
        
        is_valid, status = guard.validate_account_leverage({}, 100000)
        assert is_valid
    
    def test_single_position_account_leverage(self):
        """Calculate account leverage from single position."""
        guard = LeverageGuard(max_account=5.0)
        
        # Create position: 1 BTC at $50k with 5x leverage
        position = Position(
            side="LONG",
            entry_price=50000,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=5.0,
        )
        position.symbol = "BTC/USDT"  # Manually add symbol
        
        # Account equity: $100k
        # Notional: 1 * 50000 * 5 = $250k
        # Account leverage: 250k / 100k = 2.5x (should be OK)
        
        is_valid, status = guard.validate_account_leverage(
            {"BTC/USDT": position},
            account_equity=100000
        )
        assert is_valid
        assert "2.5x" in status or "2.50x" in status
    
    def test_exceeds_account_leverage_limit(self):
        """Reject when account leverage exceeds maximum."""
        guard = LeverageGuard(max_account=5.0)
        
        # Position: 5 BTC at $50k with 10x leverage
        position = Position(
            side="LONG",
            entry_price=50000,
            qty=5.0,
            entry_time=datetime.utcnow(),
            leverage=10.0,
        )
        position.symbol = "BTC/USDT"
        
        # Notional: 5 * 50000 * 10 = $2.5M
        # Account leverage: 2.5M / 100k = 25x (WAY over max of 5x)
        
        is_valid, reason = guard.validate_account_leverage(
            {"BTC/USDT": position},
            account_equity=100000
        )
        assert not is_valid
        assert "exceeds max" in reason.lower()
    
    def test_multiple_positions_account_leverage(self):
        """Calculate account leverage from multiple positions."""
        guard = LeverageGuard(max_account=5.0)
        
        pos_btc = Position(
            side="LONG",
            entry_price=50000,
            qty=1.0,
            entry_time=datetime.utcnow(),
            leverage=3.0,
        )
        pos_btc.symbol = "BTC/USDT"
        
        pos_eth = Position(
            side="LONG",
            entry_price=3000,
            qty=10.0,
            entry_time=datetime.utcnow(),
            leverage=2.0,
        )
        pos_eth.symbol = "ETH/USDT"
        
        # BTC notional: 1 * 50000 * 3 = $150k
        # ETH notional: 10 * 3000 * 2 = $60k
        # Total notional: $210k
        # Account leverage: 210k / 100k = 2.1x (OK)
        
        is_valid, status = guard.validate_account_leverage(
            {"BTC/USDT": pos_btc, "ETH/USDT": pos_eth},
            account_equity=100000
        )
        assert is_valid
        assert "2.10x" in status or "2.1x" in status


class TestPositionSizing:
    """Test position size validation."""
    
    def test_valid_position_size(self):
        """Accept appropriately sized position."""
        guard = LeverageGuard()
        
        # Position: 0.5 BTC at $50k with 5x leverage
        # Notional: 0.5 * 50000 = $25k
        # Required margin: 25k / 5 = $5k
        # Account equity: $100k
        # Margin used: 5k / 100k = 5% (OK)
        
        is_valid, reason = guard.validate_position_size(
            qty=0.5,
            entry_price=50000,
            leverage=5.0,
            account_equity=100000,
            max_risk_pct=0.02,
        )
        assert is_valid
    
    def test_reject_position_exceeds_equity(self):
        """Reject position requiring more equity than available."""
        guard = LeverageGuard()
        
        # Position: 100 BTC at $50k with 1x leverage
        # Notional: 100 * 50000 = $5M
        # Required margin: 5M / 1 = $5M
        # Account equity: $100k (NOT ENOUGH)
        
        is_valid, reason = guard.validate_position_size(
            qty=100.0,
            entry_price=50000,
            leverage=1.0,
            account_equity=100000,
        )
        assert not is_valid
        assert "too large" in reason.lower()
    
    def test_warn_position_high_risk(self):
        """Warn when position uses high % of equity."""
        guard = LeverageGuard()
        
        # Position: 10 BTC at $50k with 1x leverage
        # Notional: 10 * 50000 = $500k
        # Required margin: 500k / 1 = $500k
        # Account equity: $100k (uses 500% - way over!)
        # This should actually fail the equity check too
        
        is_valid, reason = guard.validate_position_size(
            qty=10.0,
            entry_price=50000,
            leverage=1.0,
            account_equity=100000,
        )
        assert not is_valid


class TestQuickValidation:
    """Test quick leverage validation function."""
    
    def test_quick_validate_within_limits(self):
        """Quick validation accepts valid leverage."""
        is_valid, reason = validate_leverage_for_entry(5.0, "BTC/USDT", max_leverage=10.0)
        assert is_valid
        assert "approved" in reason.lower()
    
    def test_quick_validate_exceeds_max(self):
        """Quick validation rejects excessive leverage."""
        is_valid, reason = validate_leverage_for_entry(15.0, "ETH/USDT", max_leverage=10.0)
        assert not is_valid
        assert "exceeds max" in reason.lower()
    
    def test_quick_validate_invalid_low(self):
        """Quick validation rejects leverage < 1.0x."""
        is_valid, reason = validate_leverage_for_entry(0.5, "SOL/USDT", max_leverage=10.0)
        assert not is_valid
        assert "invalid" in reason.lower()


class TestViolationTracking:
    """Test violation and warning tracking."""
    
    def test_track_violations(self):
        """Track all violations."""
        guard = LeverageGuard(max_per_trade=10.0)
        
        guard.validate_leverage_for_trade(15.0, "BTC/USDT")
        guard.validate_leverage_for_trade(20.0, "ETH/USDT")
        guard.validate_leverage_for_trade(12.0, "SOL/USDT")
        
        report = guard.get_violations_report()
        assert report["total_violations"] == 3
        assert len(report["violations"]) == 3
    
    def test_track_warnings(self):
        """Track all warnings."""
        guard = LeverageGuard(max_per_trade=10.0, warn_threshold=0.80)
        
        guard.validate_leverage_for_trade(8.5, "BTC/USDT")
        guard.validate_leverage_for_trade(8.0, "ETH/USDT")
        
        report = guard.get_violations_report()
        assert report["total_warnings"] >= 2
    
    def test_clear_history(self):
        """Clear violation and warning history."""
        guard = LeverageGuard(max_per_trade=10.0)
        
        guard.validate_leverage_for_trade(15.0, "BTC/USDT")
        assert len(guard.violations) > 0
        
        guard.clear_history()
        assert len(guard.violations) == 0
        assert len(guard.warnings) == 0


class TestConfiguration:
    """Test leverage guard configuration."""
    
    def test_custom_max_per_trade(self):
        """Accept custom max per-trade leverage."""
        guard = LeverageGuard(max_per_trade=20.0)
        
        is_valid, _ = guard.validate_leverage_for_trade(15.0, "BTC/USDT")
        assert is_valid
        
        is_valid, _ = guard.validate_leverage_for_trade(25.0, "BTC/USDT")
        assert not is_valid
    
    def test_custom_max_account(self):
        """Accept custom max account leverage."""
        guard = LeverageGuard(max_account=10.0)
        
        pos = Position(
            side="LONG",
            entry_price=50000,
            qty=2.0,
            entry_time=datetime.utcnow(),
            leverage=5.0,
        )
        pos.symbol = "BTC/USDT"
        
        # Notional: 2 * 50000 * 5 = $500k
        # Account leverage: 500k / 100k = 5x (OK for max 10x)
        is_valid, _ = guard.validate_account_leverage(
            {"BTC/USDT": pos},
            account_equity=100000
        )
        assert is_valid
    
    def test_custom_min_leverage(self):
        """Accept custom minimum leverage."""
        guard = LeverageGuard(min_leverage=2.0)
        
        is_valid, _ = guard.validate_leverage_for_trade(1.5, "BTC/USDT")
        assert not is_valid
        
        is_valid, _ = guard.validate_leverage_for_trade(2.0, "BTC/USDT")
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
