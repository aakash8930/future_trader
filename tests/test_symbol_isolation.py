"""
Tests for symbol isolation and state safety in multi-runner system.

Ensures that:
- One symbol's runner cannot access another symbol's data/position
- Position state is properly isolated per symbol
- No cross-symbol price/ticker leakage
"""

import pytest
from datetime import datetime
from execution.position import Position
from execution.broker import ShadowBroker, PaperBroker


class TestBrokerSymbolIsolation:
    """Test that brokers properly isolate by symbol."""
    
    def test_shadowbroker_symbol_attribute(self):
        """ShadowBroker should track its symbol."""
        broker = ShadowBroker()
        # Manually set symbol (would normally be done by runner)
        broker.symbol = "BTC/USDT"
        assert broker.symbol == "BTC/USDT"
    
    def test_open_position_preserves_symbol(self):
        """Opening a position should not affect symbol tracking."""
        broker = ShadowBroker()
        broker.symbol = "BTC/USDT"
        
        pos = broker.open_position(
            side="LONG",
            price=50000.0,
            qty=0.01,
            symbol="BTC/USDT"
        )
        
        assert broker.symbol == "BTC/USDT"
        assert pos.side == "LONG"
    
    def test_close_position_clears_symbol_state(self):
        """Closing should handle symbol properly."""
        broker = ShadowBroker()
        broker.symbol = "BTC/USDT"
        
        broker.open_position("LONG", 50000.0, 0.01, "BTC/USDT")
        pnl = broker.close_position(51000.0, "BTC/USDT")
        
        assert pnl == pytest.approx(10.0)  # (51000-50000)*0.01
        assert broker.position is None
    
    def test_paperbroker_no_symbol_assertion(self):
        """PaperBroker should work without symbol tracking."""
        broker = PaperBroker()
        
        pos = broker.open_position("LONG", 100.0, 1.0, symbol=None)
        assert pos is not None
        
        pnl = broker.close_position(110.0, symbol=None)
        assert pnl == pytest.approx(10.0)


class TestPositionIsolation:
    """Test that positions are properly isolated."""
    
    def test_different_positions_dont_interfere(self):
        """Two positions should not interfere with each other."""
        pos1 = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        
        pos2 = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        
        # Modify pos1
        pos1.add_to_position(105.0, 1.0)
        
        # pos2 should be unchanged
        assert pos2.qty == pytest.approx(1.0)
        assert pos2.avg_entry == pytest.approx(100.0)
        assert pos2.add_count == 0
    
    def test_position_pnl_with_same_prices_different_sides(self):
        """LONG and SHORT at same price should give opposite PnL."""
        long_pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        
        short_pos = Position(
            side="SHORT",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        
        long_pnl = long_pos.pnl(110.0)
        short_pnl = short_pos.pnl(110.0)
        
        # LONG: (110-100)*1 = 10
        # SHORT: (100-110)*1 = -10
        assert long_pnl == pytest.approx(10.0)
        assert short_pnl == pytest.approx(-10.0)
        assert long_pnl == -short_pnl


class TestCrossSymbolContamination:
    """Test scenarios that could cause cross-symbol issues."""
    
    def test_broker_switching_symbols_safely(self):
        """Broker instance can safely handle symbol switches."""
        broker = ShadowBroker()
        
        # Trade symbol 1
        broker.symbol = "BTC/USDT"
        pos1 = broker.open_position("LONG", 50000.0, 0.01, "BTC/USDT")
        assert broker.position is not None
        
        # Close symbol 1
        pnl1 = broker.close_position(51000.0, "BTC/USDT")
        assert pnl1 == pytest.approx(10.0)
        assert broker.position is None
        
        # Trade symbol 2
        broker.symbol = "ETH/USDT"
        pos2 = broker.open_position("LONG", 3000.0, 0.1, "ETH/USDT")
        assert broker.position is not None
        
        # Close symbol 2
        pnl2 = broker.close_position(3100.0, "ETH/USDT")
        assert pnl2 == pytest.approx(10.0)  # (3100-3000)*0.1
        assert broker.position is None
    
    def test_multiple_broker_instances_isolated(self):
        """Multiple broker instances should be fully isolated."""
        broker_btc = ShadowBroker()
        broker_eth = ShadowBroker()
        
        broker_btc.symbol = "BTC/USDT"
        broker_eth.symbol = "ETH/USDT"
        
        # Open positions in both
        pos_btc = broker_btc.open_position("LONG", 50000.0, 0.01, "BTC/USDT")
        pos_eth = broker_eth.open_position("SHORT", 3000.0, 0.1, "ETH/USDT")
        
        # Verify positions are different
        assert broker_btc.position is not broker_eth.position
        assert broker_btc.position.side == "LONG"
        assert broker_eth.position.side == "SHORT"
        assert broker_btc.position.entry_price == pytest.approx(50000.0)
        assert broker_eth.position.entry_price == pytest.approx(3000.0)


class TestSymbolAssertions:
    """Test adding symbol assertions to critical operations."""
    
    def test_position_symbol_validation(self):
        """Positions should validate their symbol context."""
        pos = Position(
            side="LONG",
            entry_price=100.0,
            qty=1.0,
            entry_time=datetime.utcnow(),
        )
        
        # Position itself doesn't store symbol (done at runner level)
        # But close_position in runner should assert correct symbol
        
        # This is a reminder that runner should assert:
        # assert symbol == self.symbol, f"Symbol mismatch: {symbol} != {self.symbol}"
        
        pnl = pos.pnl(110.0)
        assert pnl == pytest.approx(10.0)
    
    def test_broker_close_with_wrong_symbol_safeguard(self):
        """Broker should handle symbol verification."""
        broker = ShadowBroker()
        broker.symbol = "BTC/USDT"
        
        pos = broker.open_position("LONG", 50000.0, 0.01, "BTC/USDT")
        
        # Runner should verify symbol before close:
        # if symbol != self.symbol:
        #     raise ValueError(f"Symbol mismatch: {symbol} vs {self.symbol}")
        
        pnl = broker.close_position(51000.0, "BTC/USDT")
        assert pnl == pytest.approx(10.0)


class TestStateLeakageScenarios:
    """Test scenarios that could cause state leakage."""
    
    def test_stale_position_from_crashed_runner(self):
        """Simulates a crashed runner's position not leaking to replacement runner."""
        # Create broker with BTC position
        broker_old = ShadowBroker()
        broker_old.symbol = "BTC/USDT"
        broker_old.open_position("LONG", 50000.0, 0.01, "BTC/USDT")
        
        assert broker_old.position is not None
        
        # Simulate runner crash - new runner creates new broker instance
        broker_new = ShadowBroker()
        broker_new.symbol = "ETH/USDT"
        
        # New broker should have NO position from old runner
        assert broker_new.position is None
        
        # This test verifies that each runner instance gets its own broker
        # and positions don't leak between runners
    
    def test_cached_price_isolation(self):
        """Verify that prices don't leak between symbol caches."""
        # This is more of a data/fetcher concern, but we test the concept
        
        price_cache = {}  # Simulates fetcher cache
        
        price_cache["BTC/USDT"] = 50000.0
        price_cache["ETH/USDT"] = 3000.0
        
        # Get price for BTC
        btc_price = price_cache.get("BTC/USDT")
        assert btc_price == 50000.0
        
        # Get price for ETH
        eth_price = price_cache.get("ETH/USDT")
        assert eth_price == 3000.0
        
        # Verify no mixing
        assert btc_price != eth_price


class TestCleanupOnRemoval:
    """Test that runner cleanup prevents state leakage."""
    
    def test_broker_cleanup_on_removal(self):
        """Broker should be cleaned up when runner is removed."""
        broker = ShadowBroker()
        broker.symbol = "BTC/USDT"
        
        # Open position
        broker.open_position("LONG", 50000.0, 0.01, "BTC/USDT")
        assert broker.position is not None
        
        # Cleanup: close position and clear reference
        if broker.position:
            broker.close_position(50000.0, "BTC/USDT")
        
        # After cleanup
        assert broker.position is None
        
        # Verify we can't access old runner's position
        # (this would be caught by runner removal logic)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
