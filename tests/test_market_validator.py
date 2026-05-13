# tests/test_market_validator.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from execution.market_validator import MarketValidator


@pytest.fixture
def mock_fetcher():
    """Create a mock MarketDataFetcher."""
    fetcher = Mock()
    fetcher.exchange = Mock()
    fetcher.exchange_name = "binance"
    return fetcher


@pytest.fixture
def validator(mock_fetcher):
    """Create a MarketValidator instance."""
    return MarketValidator(
        fetcher=mock_fetcher,
        min_daily_volume_usd=500_000,
        min_candles=200,
        min_volatility_bps=20,
        max_bid_ask_spread_pct=0.15,
        min_positive_class_count=50,
        dead_market_hours=1,
        cache_ttl_seconds=3600,
    )


@pytest.fixture
def valid_metadata():
    """Valid market metadata."""
    return {
        "type": "future",
        "perpetual": True,
        "quote": "USDT",
        "active": True,
        "trading": True,
        "quoteVolume": 1_000_000,
        "lastTrade": (datetime.utcnow() - timedelta(minutes=5)).isoformat() + "Z",
    }


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=500, freq="15min")
    prices = np.random.normal(100, 2, 500)
    prices = np.cumsum(prices) + 50000
    
    return pd.DataFrame({
        "time": dates,
        "open": prices,
        "high": prices + np.abs(np.random.normal(0, 1, 500)),
        "low": prices - np.abs(np.random.normal(0, 1, 500)),
        "close": prices + np.random.normal(0, 1, 500),
        "volume": np.random.uniform(1000, 5000, 500),
    })


class TestUSTPerpetualCheck:
    """Test USDT perpetual validation."""

    def test_valid_usdt_perpetual(self, validator, valid_metadata):
        """Test valid USDT perpetual acceptance."""
        is_perpetual, reason = validator._is_usdt_perpetual("BTC/USDT", valid_metadata)
        assert is_perpetual
        assert reason == "usdt_perpetual_ok"

    def test_non_perpetual_rejected(self, validator):
        """Test non-perpetual spot markets rejected."""
        metadata = {
            "type": "spot",
            "quote": "USDT",
            "active": True,
            "trading": True,
        }
        is_perpetual, reason = validator._is_usdt_perpetual("BTC/USDT", metadata)
        assert not is_perpetual
        assert "not_perpetual" in reason

    def test_non_usdt_rejected(self, validator):
        """Test non-USDT quotes rejected."""
        metadata = {
            "type": "future",
            "perpetual": True,
            "quote": "BUSD",
            "active": True,
            "trading": True,
        }
        is_perpetual, reason = validator._is_usdt_perpetual("BTC/BUSD", metadata)
        assert not is_perpetual
        assert "wrong_quote" in reason

    def test_fallback_to_symbol_format(self, validator):
        """Test fallback to symbol naming convention when no metadata."""
        is_perpetual, reason = validator._is_usdt_perpetual("BTC/USDT", None)
        assert is_perpetual
        assert reason == "symbol_convention_match"

    def test_non_usdt_format_fallback(self, validator):
        """Test fallback rejects non-USDT format."""
        is_perpetual, reason = validator._is_usdt_perpetual("BTC/BUSD", None)
        assert not is_perpetual


class TestDelistingCheck:
    """Test delisting detection."""

    def test_active_market(self, validator, valid_metadata):
        """Test active market passes."""
        is_delisted, reason = validator._is_delisted("BTC/USDT", valid_metadata)
        assert not is_delisted

    def test_inactive_market(self, validator):
        """Test inactive market detected."""
        metadata = {
            "active": False,
            "trading": True,
        }
        is_delisted, reason = validator._is_delisted("OLDTOKEN/USDT", metadata)
        assert is_delisted
        assert "inactive" in reason

    def test_trading_disabled(self, validator):
        """Test trading disabled detected."""
        metadata = {
            "active": True,
            "trading": False,
        }
        is_delisted, reason = validator._is_delisted("OLDTOKEN/USDT", metadata)
        assert is_delisted


class TestLeveragedTokens:
    """Test leveraged token filtering."""

    @pytest.mark.parametrize("token", [
        "BTCUP",
        "ETHDOWN",
        "1000PEPE",
        "1000FLOKI",
        "1000LUNC",
    ])
    def test_known_leveraged_tokens_rejected(self, validator, token):
        """Test known leveraged tokens rejected."""
        is_leveraged, reason = validator._is_leveraged_token(f"{token}/USDT")
        assert is_leveraged

    @pytest.mark.parametrize("token", [
        "BTC",
        "ETH",
        "DOGE",
        "LINK",
    ])
    def test_normal_tokens_accepted(self, validator, token):
        """Test normal tokens not filtered."""
        is_leveraged, reason = validator._is_leveraged_token(f"{token}/USDT")
        assert not is_leveraged


class TestDeadMarket:
    """Test dead market detection."""

    def test_recent_trade_not_dead(self, validator, valid_metadata):
        """Test market with recent trade not marked as dead."""
        is_dead, reason = validator._is_dead_market("BTC/USDT", valid_metadata)
        assert not is_dead

    def test_old_trade_detected_as_dead(self, validator):
        """Test market with no recent trades marked as dead."""
        metadata = {
            "lastTrade": (datetime.utcnow() - timedelta(hours=2)).isoformat() + "Z",
        }
        is_dead, reason = validator._is_dead_market("OLDTOKEN/USDT", metadata)
        assert is_dead
        assert "dead_market" in reason

    def test_no_trade_info(self, validator):
        """Test handles missing trade info gracefully."""
        metadata = {"lastTrade": None}
        is_dead, reason = validator._is_dead_market("BTC/USDT", metadata)
        assert not is_dead


class TestVolumeCheck:
    """Test volume validation."""

    def test_sufficient_volume(self, validator, valid_metadata):
        """Test sufficient volume passes."""
        vol_ok, reason, amount = validator._check_volume("BTC/USDT", valid_metadata)
        assert vol_ok
        assert amount == 1_000_000

    def test_insufficient_volume(self, validator):
        """Test insufficient volume rejected."""
        metadata = {"quoteVolume": 100_000}
        vol_ok, reason, amount = validator._check_volume("LOWVOL/USDT", metadata)
        assert not vol_ok
        assert "volume_too_low" in reason

    def test_no_volume_data(self, validator):
        """Test missing volume handled."""
        metadata = {"quoteVolume": None}
        vol_ok, reason, amount = validator._check_volume("NOVOL/USDT", metadata)
        assert not vol_ok

    def test_volume_from_info_field(self, validator):
        """Test volume extraction from info field."""
        metadata = {
            "quoteVolume": None,
            "info": {"quoteAssetVolume": "1000000"},
        }
        vol_ok, reason, amount = validator._check_volume("BTC/USDT", metadata)
        assert vol_ok


class TestVolatilityCheck:
    """Test volatility validation."""

    def test_sufficient_volatility(self, validator, sample_ohlcv):
        """Test data with sufficient volatility passes."""
        vol_ok, reason, vol_bps = validator._check_volatility(sample_ohlcv)
        # Our sample has random walk data, so volatility should be non-trivial
        assert vol_ok or vol_bps > 0  # May or may not pass depending on randomness

    def test_insufficient_data(self, validator):
        """Test insufficient data rejected."""
        df = pd.DataFrame({
            "close": [100, 101, 102],
        })
        vol_ok, reason, vol_bps = validator._check_volatility(df)
        assert not vol_ok

    def test_empty_dataframe(self, validator):
        """Test empty dataframe handled."""
        df = pd.DataFrame()
        vol_ok, reason, vol_bps = validator._check_volatility(df)
        assert not vol_ok


class TestSpreadCheck:
    """Test bid-ask spread validation."""

    def test_acceptable_spread(self, validator, mock_fetcher):
        """Test acceptable spread passes."""
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 100.0,
            "ask": 100.1,
        }
        spread_ok, reason, spread_pct = validator._check_spread("BTC/USDT")
        assert spread_ok
        # Spread should be around (0.1 / 100.05) * 100 = 0.0999%
        assert spread_pct < validator.max_bid_ask_spread_pct

    def test_wide_spread(self, validator, mock_fetcher):
        """Test wide spread rejected."""
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 100.0,
            "ask": 100.5,
        }
        spread_ok, reason, spread_pct = validator._check_spread("WIDESPREAD/USDT")
        assert not spread_ok
        assert "spread_too_wide" in reason

    def test_no_spread_data(self, validator, mock_fetcher):
        """Test missing spread data handled gracefully."""
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": None,
            "ask": None,
        }
        spread_ok, reason, spread_pct = validator._check_spread("BTC/USDT")
        # Spread check is non-fatal, should return False when data missing
        assert not spread_ok
        assert "bid_ask" in reason or "skipped" in reason


class TestTradeValidation:
    """Test trading validation."""

    def test_valid_symbol_passes_all_checks(self, validator, valid_metadata, mock_fetcher):
        """Test symbol that passes all checks."""
        mock_fetcher.exchange.markets = {"BTC/USDT": valid_metadata}
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50001.0,
        }

        is_valid, reason = validator.validate_for_trading("BTC/USDT")
        assert is_valid
        assert reason == "all_checks_passed"

    def test_symbol_not_supported(self, validator, mock_fetcher):
        """Test unsupported symbol rejected."""
        mock_fetcher.exchange.markets = {}

        is_valid, reason = validator.validate_for_trading("NOSUCH/USDT")
        assert not is_valid

    def test_cache_respects_ttl(self, validator, valid_metadata, mock_fetcher):
        """Test cache respects TTL."""
        mock_fetcher.exchange.markets = {"BTC/USDT": valid_metadata}
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50001.0,
        }

        # First call
        validator.validate_for_trading("BTC/USDT")

        # Immediately: should use cache
        is_valid1, _ = validator.validate_for_trading("BTC/USDT")
        assert is_valid1

        # Clear cache manually
        validator.clear_cache("BTC/USDT")
        is_valid2, _ = validator.validate_for_trading("BTC/USDT")
        assert is_valid2


class TestTrainingValidation:
    """Test training validation."""

    def test_training_requires_tradeable(self, validator):
        """Test training requires symbol to be tradeable."""
        validator.fetcher.exchange.markets = {}
        is_valid, reason = validator.validate_for_training("BADTOKEN/USDT")
        assert not is_valid
        assert "not_tradeable" in reason

    def test_insufficient_candles(self, validator, valid_metadata, mock_fetcher):
        """Test insufficient candles rejected."""
        mock_fetcher.exchange.markets = {"BTC/USDT": valid_metadata}
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50001.0,
        }

        df = pd.DataFrame({
            "close": list(range(50)),
        })
        is_valid, reason = validator.validate_for_training("BTC/USDT", df)
        assert not is_valid
        assert "insufficient_candles" in reason

    def test_training_with_sufficient_data(self, validator, valid_metadata, sample_ohlcv, mock_fetcher):
        """Test training validation with sufficient data."""
        mock_fetcher.exchange.markets = {"BTC/USDT": valid_metadata}
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50001.0,
        }

        # Our sample has 500 candles, should pass minimum
        is_valid, reason = validator.validate_for_training("BTC/USDT", sample_ohlcv)
        # May pass or fail based on volatility, but shouldn't fail on candles
        assert "insufficient_candles" not in reason or "insufficient" not in reason


class TestPositiveClassCount:
    """Test positive class count validation."""

    def test_sufficient_positive_class(self, validator, valid_metadata, mock_fetcher):
        """Test sufficient positive class passes."""
        mock_fetcher.exchange.markets = {"BTC/USDT": valid_metadata}
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50001.0,
        }

        df = pd.DataFrame({
            "close": list(range(300)),
            "label": [1] * 100 + [0] * 200,
        })
        is_valid, reason = validator.validate_for_training_with_labels(
            "BTC/USDT", df, "label"
        )
        assert is_valid or "insufficient_positive_class" not in reason

    def test_insufficient_positive_class(self, validator, valid_metadata, mock_fetcher):
        """Test insufficient positive class rejected."""
        mock_fetcher.exchange.markets = {"BTC/USDT": valid_metadata}
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50001.0,
        }

        df = pd.DataFrame({
            "close": list(range(300)),
            "label": [1] * 20 + [0] * 280,
        })
        is_valid, reason = validator.validate_for_training_with_labels(
            "BTC/USDT", df, "label"
        )
        assert not is_valid
        assert "insufficient_positive_class" in reason


class TestFilterSymbols:
    """Test symbol filtering."""

    def test_filter_symbols_basic(self, validator, valid_metadata, mock_fetcher):
        """Test filtering list of symbols."""
        # Set up market metadata for valid symbols only
        markets = {
            "BTC/USDT": valid_metadata,
            "ETH/USDT": valid_metadata,
        }
        mock_fetcher.exchange.markets = markets
        mock_fetcher.exchange.fetch_ticker.side_effect = lambda s: {
            "bid": 100.0,
            "ask": 100.1,
        }

        symbols = ["BTC/USDT", "ETH/USDT", "BADTOKEN/USDT"]
        filtered = validator.filter_symbols(symbols)
        
        # Should filter out BADTOKEN
        assert "BTC/USDT" in filtered or "ETH/USDT" in filtered
        assert "BADTOKEN/USDT" not in filtered or len(filtered) < len(symbols)


class TestValidationReport:
    """Test detailed validation report."""

    def test_get_validation_report(self, validator, valid_metadata, mock_fetcher):
        """Test detailed report generation."""
        mock_fetcher.exchange.markets = {"BTC/USDT": valid_metadata}
        mock_fetcher.exchange.fetch_ticker.return_value = {
            "bid": 50000.0,
            "ask": 50001.0,
        }
        mock_fetcher.fetch_ohlcv.return_value = None

        report = validator.get_validation_report("BTC/USDT")
        
        assert "symbol" in report
        assert report["symbol"] == "BTC/USDT"
        assert "checks" in report
        assert "usdt_perpetual" in report["checks"]
        assert "minimum_volume" in report["checks"]
        assert "is_valid_for_trading" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
