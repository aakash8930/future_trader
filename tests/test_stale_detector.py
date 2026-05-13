"""
Tests for Stale Data Detector

Verifies data freshness monitoring and stream health detection.
"""

import pytest
import time
from datetime import datetime, timedelta
from data.stale_detector import StaleDataDetector, check_data_age


class TestDataFreshness:
    """Test data freshness checking."""
    
    def test_fresh_data_within_limit(self):
        """Accept data within freshness limit."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=10)
        
        is_fresh, reason = detector.is_data_fresh("BTC/USDT")
        assert is_fresh
        assert reason is None
    
    def test_reject_stale_data(self):
        """Reject data older than max age."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["ETH/USDT"] = now - timedelta(seconds=120)
        
        is_fresh, reason = detector.is_data_fresh("ETH/USDT")
        assert not is_fresh
        assert "stale" in reason.lower()
    
    def test_exact_max_age_accepted(self):
        """Accept data at exactly max age or slightly more is stale."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        # Create timestamp exactly 59 seconds old (safe for test timing variance)
        detector.last_update["SOL/USDT"] = now - timedelta(seconds=59)
        
        is_fresh, reason = detector.is_data_fresh("SOL/USDT")
        assert is_fresh
        assert reason is None
    
    def test_no_data_received_yet(self):
        """Reject symbol with no data received."""
        detector = StaleDataDetector()
        
        is_fresh, reason = detector.is_data_fresh("UNKNOWN/USDT")
        assert not is_fresh
        assert "No data" in reason
    
    def test_warn_approaching_stale(self):
        """Warn when data approaching stale threshold."""
        detector = StaleDataDetector(
            max_age_seconds=60,
            warn_threshold_seconds=30,
        )
        
        now = datetime.utcnow()
        detector.last_update["AVAX/USDT"] = now - timedelta(seconds=31)
        detector.warning_events = []  # Clear any prior warnings
        
        is_fresh, reason = detector.is_data_fresh("AVAX/USDT")
        assert is_fresh  # Still fresh but warned
        assert len(detector.warning_events) > 0
        assert "monitor data source" in detector.warning_events[0].lower()


class TestFreshnessReport:
    """Test freshness reporting."""
    
    def test_generate_report_multiple_symbols(self):
        """Generate freshness report for multiple symbols."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=5)
        detector.last_update["ETH/USDT"] = now - timedelta(seconds=120)
        # SOL/USDT has no update
        
        report = detector.get_freshness_report([
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
        ])
        
        assert len(report) == 3
        assert report[0].is_fresh is True  # BTC fresh
        assert report[1].is_fresh is False  # ETH stale
        assert report[2].is_fresh is False  # SOL no data
    
    def test_freshness_report_contains_ages(self):
        """Freshness report includes age information."""
        detector = StaleDataDetector()
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=15)
        
        report = detector.get_freshness_report(["BTC/USDT"])
        
        assert report[0].age_seconds > 10
        assert report[0].age_seconds < 20


class TestStaleSymbols:
    """Test stale symbol detection."""
    
    def test_get_stale_symbols(self):
        """Identify which symbols have stale data."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=10)
        detector.last_update["ETH/USDT"] = now - timedelta(seconds=120)
        detector.last_update["SOL/USDT"] = now - timedelta(seconds=5)
        # AVAX has no data
        
        stale = detector.get_stale_symbols([
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "AVAX/USDT",
        ])
        
        assert "ETH/USDT" in stale
        assert "AVAX/USDT" in stale
        assert "BTC/USDT" not in stale
        assert "SOL/USDT" not in stale
    
    def test_get_stale_symbols_empty_list(self):
        """Handle empty symbol list."""
        detector = StaleDataDetector()
        
        stale = detector.get_stale_symbols([])
        assert stale == []


class TestStreamHealth:
    """Test WebSocket stream health detection."""
    
    def test_healthy_stream(self):
        """Healthy stream with fresh data."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=5)
        detector.last_update["ETH/USDT"] = now - timedelta(seconds=10)
        
        is_alive, reason = detector.detect_stream_death([
            "BTC/USDT",
            "ETH/USDT",
        ])
        
        assert is_alive is True
        assert reason is None
    
    def test_dead_stream_all_symbols_stale(self):
        """Stream declared dead if all symbols stale."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=120)
        detector.last_update["ETH/USDT"] = now - timedelta(seconds=120)
        detector.last_update["SOL/USDT"] = now - timedelta(seconds=120)
        
        is_alive, reason = detector.detect_stream_death([
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
        ])
        
        assert is_alive is False
        assert "DEAD" in reason
    
    def test_stream_degraded_warning(self):
        """Warn if majority of symbols are stale."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        # 3 symbols stale out of 4
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=120)
        detector.last_update["ETH/USDT"] = now - timedelta(seconds=120)
        detector.last_update["SOL/USDT"] = now - timedelta(seconds=120)
        detector.last_update["AVAX/USDT"] = now - timedelta(seconds=5)
        
        is_alive, reason = detector.detect_stream_death([
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "AVAX/USDT",
        ])
        
        assert is_alive is True  # Not completely dead
        assert len(detector.warning_events) > 0
        assert "degraded" in detector.warning_events[0].lower()
    
    def test_stream_empty_symbol_list(self):
        """Handle empty symbol list in stream health check."""
        detector = StaleDataDetector()
        
        is_alive, reason = detector.detect_stream_death([])
        
        assert is_alive is True
        assert reason is None


class TestAssertions:
    """Test assertion methods."""
    
    def test_assert_data_fresh_passes(self):
        """Assert passes for fresh data."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=10)
        
        # Should not raise
        detector.assert_data_fresh("BTC/USDT")
    
    def test_assert_data_fresh_fails(self):
        """Assert raises for stale data."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["ETH/USDT"] = now - timedelta(seconds=120)
        
        with pytest.raises(ValueError, match="stale"):
            detector.assert_data_fresh("ETH/USDT")


class TestQuickValidation:
    """Test quick data age check."""
    
    def test_quick_check_fresh(self):
        """Quick check for fresh data."""
        now = datetime.utcnow()
        old_time = now - timedelta(seconds=10)
        
        is_fresh, age = check_data_age(
            "BTC/USDT",
            old_time,
            max_age_seconds=60,
        )
        
        assert is_fresh
        assert age > 9 and age < 11
    
    def test_quick_check_stale(self):
        """Quick check for stale data."""
        now = datetime.utcnow()
        old_time = now - timedelta(seconds=120)
        
        is_fresh, age = check_data_age(
            "ETH/USDT",
            old_time,
            max_age_seconds=60,
        )
        
        assert not is_fresh
        assert age > 119


class TestEventTracking:
    """Test stale event tracking."""
    
    def test_track_stale_events(self):
        """Track all stale events."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        
        # Trigger 3 stale events
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=120)
        detector.is_data_fresh("BTC/USDT")
        
        detector.last_update["ETH/USDT"] = now - timedelta(seconds=120)
        detector.is_data_fresh("ETH/USDT")
        
        detector.is_data_fresh("SOL/USDT")  # Never updated
        
        assert len(detector.stale_events) >= 3
    
    def test_get_report(self):
        """Get detector report."""
        detector = StaleDataDetector(max_age_seconds=60)
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=120)
        detector.is_data_fresh("BTC/USDT")
        
        report = detector.get_report()
        
        assert "total_stale_events" in report
        assert "stale_events" in report
        assert "warning_events" in report
        assert "config" in report
    
    def test_clear_history(self):
        """Clear event history."""
        detector = StaleDataDetector()
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=120)
        detector.is_data_fresh("BTC/USDT")
        
        assert len(detector.stale_events) > 0
        
        detector.clear_history()
        assert len(detector.stale_events) == 0
        assert len(detector.warning_events) == 0


class TestConfiguration:
    """Test stale detector configuration."""
    
    def test_custom_max_age(self):
        """Accept custom max age."""
        detector = StaleDataDetector(max_age_seconds=30)
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=25)
        
        is_fresh, _ = detector.is_data_fresh("BTC/USDT")
        assert is_fresh
        
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=35)
        is_fresh, _ = detector.is_data_fresh("BTC/USDT")
        assert not is_fresh
    
    def test_custom_warn_threshold(self):
        """Accept custom warning threshold."""
        detector = StaleDataDetector(
            max_age_seconds=60,
            warn_threshold_seconds=10,
        )
        
        now = datetime.utcnow()
        detector.last_update["BTC/USDT"] = now - timedelta(seconds=15)
        detector.warning_events = []
        
        detector.is_data_fresh("BTC/USDT")
        
        assert len(detector.warning_events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
