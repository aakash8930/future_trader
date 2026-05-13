"""
Tests for the trade reconciliation engine.
"""

import pytest
from datetime import datetime
from execution.reconciliation import TradeRecord, ReconciliationEngine


class TestTradeRecord:
    """Test TradeRecord creation and validation."""
    
    def test_create_trade_record(self):
        """Create a valid trade record."""
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            entry_price_local=50000.0,
            entry_qty=0.01,
            entry_leverage=1.0,
            entry_fee_usd=5.0,
            exit_price_local=51000.0,
            exit_qty=0.01,
            exit_fee_usd=5.1,
            pnl_before_fees=10.0,
            pnl_net=10.0 - 5.0 - 5.1,
            balance_before=1000.0,
            balance_after=1000.0 + (10.0 - 5.0 - 5.1),
        )
        
        is_valid, error = trade.validate()
        assert is_valid, f"Trade should be valid: {error}"
    
    def test_validate_missing_symbol(self):
        """Reject trade without symbol."""
        trade = TradeRecord(entry_side="LONG")
        is_valid, error = trade.validate()
        assert not is_valid
        assert "symbol" in error.lower()
    
    def test_validate_invalid_side(self):
        """Reject trade with invalid side."""
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="INVALID"
        )
        is_valid, error = trade.validate()
        assert not is_valid
        assert "side" in error.lower()
    
    def test_validate_zero_qty(self):
        """Reject trade with zero quantity."""
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            entry_qty=0.0
        )
        is_valid, error = trade.validate()
        assert not is_valid
        assert "qty" in error.lower()
    
    def test_validate_invalid_leverage(self):
        """Reject trade with invalid leverage."""
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            entry_qty=0.01,
            entry_price_local=50000.0,
            exit_price_local=51000.0,
            entry_leverage=150.0,  # Too high
        )
        is_valid, error = trade.validate()
        assert not is_valid
        assert "leverage" in error.lower()
    
    def test_to_dict_and_from_dict(self):
        """Test serialization/deserialization."""
        original = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            entry_price_local=50000.0,
            entry_qty=0.01,
            exit_price_local=51000.0,
            pnl_net=5.0,
            balance_before=1000.0,
            balance_after=1005.0,
        )
        
        data = original.to_dict()
        restored = TradeRecord.from_dict(data)
        
        assert restored.symbol == original.symbol
        assert restored.entry_side == original.entry_side
        assert restored.pnl_net == original.pnl_net
    
    def test_to_json(self):
        """Test JSON serialization."""
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            pnl_net=5.0,
        )
        json_str = trade.to_json()
        assert isinstance(json_str, str)
        assert "BTC/USDT" in json_str
        assert "LONG" in json_str


class TestPnLCalculation:
    """Test PnL calculations within TradeRecord."""
    
    def test_calculate_long_profit_1x(self):
        """Calculate PnL for long profit at 1x leverage."""
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            entry_price_local=100.0,
            entry_qty=1.0,
            entry_leverage=1.0,
            exit_price_local=110.0,
            exit_qty=1.0,
            entry_fee_usd=0.1,
            exit_fee_usd=0.11,
            slippage_usd=0.0,
        )
        
        pnl = trade.calculate_pnl()
        # Raw PnL: (110-100)*1 = 10
        # Fees: 0.1 + 0.11 = 0.21
        # Net: 10 - 0.21 = 9.79
        expected = 10.0 - 0.1 - 0.11
        assert pnl == pytest.approx(expected)
    
    def test_calculate_long_profit_5x_leverage(self):
        """Calculate PnL with 5x leverage."""
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            entry_price_local=100.0,
            entry_qty=1.0,
            entry_leverage=5.0,
            exit_price_local=105.0,
            exit_qty=1.0,
            entry_fee_usd=0.5,
            exit_fee_usd=0.525,
            slippage_usd=0.0,
        )
        
        pnl = trade.calculate_pnl()
        # Raw PnL: (105-100)*1*5 = 25
        # Fees: 0.5 + 0.525 = 1.025
        # Net: 25 - 1.025 = 23.975
        expected = (105.0 - 100.0) * 1.0 * 5.0 - 0.5 - 0.525
        assert pnl == pytest.approx(expected)
    
    def test_calculate_short_profit(self):
        """Calculate PnL for short profit."""
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="SHORT",
            entry_price_local=100.0,
            entry_qty=1.0,
            entry_leverage=1.0,
            exit_price_local=90.0,
            exit_qty=1.0,
            entry_fee_usd=0.1,
            exit_fee_usd=0.09,
            slippage_usd=0.0,
        )
        
        pnl = trade.calculate_pnl()
        # Raw PnL: (100-90)*1 = 10
        # Fees: 0.1 + 0.09 = 0.19
        # Net: 10 - 0.19 = 9.81
        expected = (100.0 - 90.0) * 1.0 - 0.1 - 0.09
        assert pnl == pytest.approx(expected)


class TestReconciliationEngine:
    """Test the reconciliation engine (basic - uses in-memory only)."""
    
    def test_engine_initialization(self):
        """Engine initializes without error."""
        engine = ReconciliationEngine()
        assert engine is not None
        assert isinstance(engine.trades, dict)
    
    def test_record_valid_trade(self):
        """Record a valid trade."""
        engine = ReconciliationEngine()
        
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            entry_price_local=50000.0,
            entry_qty=0.01,
            exit_price_local=51000.0,
            exit_qty=0.01,
            pnl_net=5.0,
            balance_before=1000.0,
            balance_after=1005.0,
        )
        
        success = engine.record_trade(trade)
        assert success
        assert trade.trade_id in engine.trades
    
    def test_record_invalid_trade(self):
        """Reject invalid trades."""
        engine = ReconciliationEngine()
        
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="INVALID",  # Invalid side
        )
        
        success = engine.record_trade(trade)
        assert not success
    
    def test_get_trade(self):
        """Retrieve recorded trade."""
        engine = ReconciliationEngine()
        
        trade = TradeRecord(
            symbol="BTC/USDT",
            entry_side="LONG",
            entry_price_local=50000.0,
            entry_qty=0.01,
            exit_price_local=51000.0,
            exit_qty=0.01,
            pnl_net=5.0,
            balance_before=1000.0,
            balance_after=1005.0,
        )
        
        engine.record_trade(trade)
        retrieved = engine.get_trade(trade.trade_id)
        
        assert retrieved is not None
        assert retrieved.symbol == "BTC/USDT"
        assert retrieved.pnl_net == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
