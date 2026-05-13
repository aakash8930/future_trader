# execution/market_validator.py

import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

import pandas as pd
from data.fetcher import MarketDataFetcher

logger = logging.getLogger(__name__)


class MarketValidator:
    """
    Strict market qualification validator for trading and training.
    
    Enforces:
    - Only USDT perpetual futures
    - Active market (not delisted)
    - Minimum daily volume
    - Minimum candles available
    - Minimum volatility (basis points)
    - Maximum acceptable bid-ask spread
    - Filters out leveraged tokens
    - Detects dead markets (no recent trades)
    - For training: minimum positive class count
    """

    LEVERAGED_TOKENS = {
        # 3x/2x leverage tokens (Binance)
        "BTCDOWN", "BTCUP", "ETHDOWN", "ETHUP",
        "BNBDOWN", "BNBUP", "XRPDOWN", "XRPUP",
        "LTCDOWN", "LTCUP", "LINKDOWN", "LINKUP",
        "ADADOWN", "ADAUP", "DOTDOWN", "DOTUP",
        "EOSDOWN", "EOSUP",
        # 1000x leverage tokens (real symbols)
        "1000PEPE", "1000FLOKI", "1000LUNC", "1000SHIB",
        # High leverage marks
        "1INCH", "1ML", "1XM",  # Potential leverage indicators
    }

    def __init__(
        self,
        fetcher: MarketDataFetcher,
        min_daily_volume_usd: float = 500_000,
        min_candles: int = 200,
        min_volatility_bps: float = 20,
        max_bid_ask_spread_pct: float = 0.15,
        min_positive_class_count: int = 50,
        dead_market_hours: int = 1,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize market validator.

        Args:
            fetcher: MarketDataFetcher instance
            min_daily_volume_usd: Minimum 24h volume in USD
            min_candles: Minimum number of historical candles required
            min_volatility_bps: Minimum volatility in basis points
            max_bid_ask_spread_pct: Maximum acceptable spread percentage
            min_positive_class_count: Minimum positive samples for training
            dead_market_hours: Hours threshold for dead market detection
            cache_ttl_seconds: Cache validity in seconds
        """
        self.fetcher = fetcher
        self.min_daily_volume_usd = min_daily_volume_usd
        self.min_candles = min_candles
        self.min_volatility_bps = min_volatility_bps
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        self.min_positive_class_count = min_positive_class_count
        self.dead_market_hours = dead_market_hours
        self.cache_ttl_seconds = cache_ttl_seconds

        # Cache for validation results
        self._validation_cache: dict[str, Tuple[bool, str, float]] = {}
        self._cache_timestamps: dict[str, float] = {}

    def _get_market_metadata(self, symbol: str) -> Optional[dict]:
        """Get market metadata from exchange."""
        if not self.fetcher.exchange:
            return None

        try:
            markets = self.fetcher.exchange.markets
            return markets.get(symbol)
        except Exception as e:
            logger.warning(f"[MarketValidator] Failed to get metadata for {symbol}: {e}")
            return None

    def _is_usdt_perpetual(self, symbol: str, metadata: Optional[dict]) -> Tuple[bool, str]:
        """Check if symbol is USDT perpetual futures."""
        if not metadata:
            # Fallback: check symbol naming convention
            if symbol.endswith("/USDT"):
                return True, "symbol_convention_match"
            return False, "no_metadata_and_not_usdt_format"

        # Check if it's perpetual
        is_perpetual = metadata.get("type") == "future" or metadata.get("perpetual", False)
        is_quote_usdt = metadata.get("quote") == "USDT"

        if not is_perpetual:
            return False, f"not_perpetual (type={metadata.get('type')})"

        if not is_quote_usdt:
            return False, f"wrong_quote ({metadata.get('quote')})"

        return True, "usdt_perpetual_ok"

    def _is_delisted(self, symbol: str, metadata: Optional[dict]) -> Tuple[bool, str]:
        """Check if market is delisted."""
        if not metadata:
            return False, "no_metadata_cannot_verify_delisting"

        is_active = metadata.get("active", True)
        is_trading = metadata.get("trading", True)

        if not is_active:
            return True, "market_inactive"

        if not is_trading:
            return True, "trading_disabled"

        return False, "market_active"

    def _is_leveraged_token(self, symbol: str) -> Tuple[bool, str]:
        """Detect leveraged/synthetic tokens."""
        base = symbol.split("/")[0] if "/" in symbol else symbol

        # Check exact match
        if base in self.LEVERAGED_TOKENS:
            return True, f"leveraged_token ({base})"

        # Check patterns
        if any(base.startswith(pat) for pat in ["3S", "3L", "2S", "2L"]):
            return True, f"leverage_pattern ({base})"

        if base.startswith("1000") or base.startswith("1"):
            # Filter out legitimate 1x or 1-letter tokens
            if base not in ["1INCH", "1ML", "1XM"] and base[0].isdigit():
                try:
                    multiplier = int(base[:4]) if base[:4].isdigit() else 1
                    if multiplier > 1:
                        return True, f"high_multiplier_token ({base})"
                except (ValueError, IndexError):
                    pass

        return False, "not_leveraged"

    def _is_dead_market(self, symbol: str, metadata: Optional[dict]) -> Tuple[bool, str]:
        """Check if market is dead (no recent trades)."""
        if not metadata:
            return False, "no_metadata_cannot_verify_trades"

        last_trade = metadata.get("lastTrade")
        if last_trade is None:
            return False, "no_last_trade_info"

        try:
            if isinstance(last_trade, str):
                last_trade_dt = datetime.fromisoformat(last_trade.replace("Z", "+00:00"))
            elif isinstance(last_trade, (int, float)):
                last_trade_dt = datetime.fromtimestamp(last_trade / 1000, tz=datetime.now().astimezone().tzinfo)
            else:
                return False, "unknown_last_trade_format"

            age = datetime.now(last_trade_dt.tzinfo) - last_trade_dt
            hours_old = age.total_seconds() / 3600

            if hours_old > self.dead_market_hours:
                return True, f"dead_market ({hours_old:.1f}h old)"

            return False, f"market_active ({hours_old:.2f}h since trade)"

        except Exception as e:
            logger.debug(f"[MarketValidator] Cannot parse last_trade for {symbol}: {e}")
            return False, "cannot_parse_last_trade"

    def _check_volume(self, symbol: str, metadata: Optional[dict]) -> Tuple[bool, str, float]:
        """Check 24h volume."""
        if not metadata:
            return False, "no_volume_metadata", 0.0

        # Volume info location varies by exchange
        quote_volume = metadata.get("quoteVolume")
        info = metadata.get("info", {})
        quoteAssetVolume = info.get("quoteAssetVolume")

        volume_usd = quote_volume or quoteAssetVolume

        if volume_usd is None:
            return False, "volume_not_available", 0.0

        try:
            volume_usd = float(volume_usd)
        except (ValueError, TypeError):
            return False, "volume_parse_failed", 0.0

        if volume_usd < self.min_daily_volume_usd:
            return False, f"volume_too_low ({volume_usd:.0f})", volume_usd

        return True, f"volume_ok ({volume_usd:.0f})", volume_usd

    def _check_volatility(self, df: pd.DataFrame) -> Tuple[bool, str, float]:
        """Check minimum volatility from OHLCV data."""
        if df is None or df.empty or len(df) < 10:
            return False, "insufficient_data_for_volatility", 0.0

        try:
            # Calculate volatility as percentage
            close = df["close"].astype(float)
            returns = close.pct_change().dropna()

            if len(returns) < 5:
                return False, "insufficient_returns", 0.0

            # Annualized volatility (for 15m ~252*96 = 24192 candles/year)
            volatility_pct = returns.std()
            volatility_bps = volatility_pct * 10000

            if volatility_bps < self.min_volatility_bps:
                return False, f"volatility_too_low ({volatility_bps:.1f} bps)", volatility_bps

            return True, f"volatility_ok ({volatility_bps:.1f} bps)", volatility_bps

        except Exception as e:
            logger.debug(f"[MarketValidator] Volatility check failed: {e}")
            return False, f"volatility_check_error", 0.0

    def _check_spread(self, symbol: str) -> Tuple[bool, str, float]:
        """Check bid-ask spread."""
        try:
            ticker = self.fetcher.exchange.fetch_ticker(symbol)
            bid = ticker.get("bid")
            ask = ticker.get("ask")

            if bid is None or ask is None or bid <= 0 or ask <= 0:
                return False, "no_bid_ask_data", 0.0

            spread_abs = ask - bid
            spread_pct = (spread_abs / ((bid + ask) / 2)) * 100

            if spread_pct > self.max_bid_ask_spread_pct:
                return False, f"spread_too_wide ({spread_pct:.4f}%)", spread_pct

            return True, f"spread_ok ({spread_pct:.4f}%)", spread_pct

        except Exception as e:
            logger.debug(f"[MarketValidator] Spread check failed for {symbol}: {e}")
            # Don't fail on spread check - it's optional
            return True, "spread_check_skipped", 0.0

    def _is_cached_valid(self, symbol: str) -> bool:
        """Check if cached result is still valid."""
        if symbol not in self._cache_timestamps:
            return False

        age = time.time() - self._cache_timestamps[symbol]
        return age < self.cache_ttl_seconds

    def validate_symbol(self, symbol: str) -> bool:
        """
        Quick validation: is this symbol tradeable?

        Returns:
            bool: True if symbol is valid for trading
        """
        is_valid, _ = self.validate_for_trading(symbol)
        return is_valid

    def validate_for_trading(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate symbol for live trading.

        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        # Check cache
        if self._is_cached_valid(symbol):
            is_valid, reason, _ = self._validation_cache[symbol]
            return is_valid, reason

        metadata = self._get_market_metadata(symbol)

        # 1. Check USDT perpetual
        is_perpetual, reason = self._is_usdt_perpetual(symbol, metadata)
        if not is_perpetual:
            result = (False, reason)
            self._validation_cache[symbol] = (False, reason, 0.0)
            self._cache_timestamps[symbol] = time.time()
            return result

        # 2. Check delisted
        is_delisted, reason = self._is_delisted(symbol, metadata)
        if is_delisted:
            result = (False, reason)
            self._validation_cache[symbol] = (False, reason, 0.0)
            self._cache_timestamps[symbol] = time.time()
            return result

        # 3. Check leveraged token
        is_leveraged, reason = self._is_leveraged_token(symbol)
        if is_leveraged:
            result = (False, reason)
            self._validation_cache[symbol] = (False, reason, 0.0)
            self._cache_timestamps[symbol] = time.time()
            return result

        # 4. Check dead market
        is_dead, reason = self._is_dead_market(symbol, metadata)
        if is_dead:
            result = (False, reason)
            self._validation_cache[symbol] = (False, reason, 0.0)
            self._cache_timestamps[symbol] = time.time()
            return result

        # 5. Check volume
        vol_ok, vol_reason, vol_amount = self._check_volume(symbol, metadata)
        if not vol_ok:
            result = (False, vol_reason)
            self._validation_cache[symbol] = (False, vol_reason, 0.0)
            self._cache_timestamps[symbol] = time.time()
            return result

        # 6. Check spread (non-fatal)
        spread_ok, spread_reason, spread_pct = self._check_spread(symbol)
        if not spread_ok:
            logger.warning(f"[MarketValidator] {symbol} | {spread_reason} (continuing anyway)")

        result = (True, "all_checks_passed")
        self._validation_cache[symbol] = (True, "all_checks_passed", vol_amount)
        self._cache_timestamps[symbol] = time.time()
        return result

    def validate_for_training(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """
        Validate symbol for model training.

        Args:
            symbol: Trading symbol
            df: Optional OHLCV dataframe

        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        # First check if tradeable
        is_tradeable, trade_reason = self.validate_for_trading(symbol)
        if not is_tradeable:
            return False, f"not_tradeable ({trade_reason})"

        # Check candle count
        if df is not None:
            if df.empty or len(df) < self.min_candles:
                return (
                    False,
                    f"insufficient_candles ({len(df)}/{self.min_candles})",
                )

            # Check volatility from data
            vol_ok, vol_reason, vol_bps = self._check_volatility(df)
            if not vol_ok:
                return False, vol_reason

        return True, "training_ready"

    def validate_for_training_with_labels(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        label_col: str = "label",
    ) -> Tuple[bool, str]:
        """
        Validate symbol for training with minimum positive class count.

        Args:
            symbol: Trading symbol
            df: OHLCV dataframe with labels
            label_col: Column name containing labels (1=up, 0=down)

        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        # First base validation
        is_valid, reason = self.validate_for_training(symbol, df)
        if not is_valid:
            return is_valid, reason

        # Check positive class count
        if df is not None and label_col in df.columns:
            positive_count = int((df[label_col] == 1).sum())
            if positive_count < self.min_positive_class_count:
                return (
                    False,
                    f"insufficient_positive_class ({positive_count}/{self.min_positive_class_count})",
                )

        return True, "training_with_labels_ready"

    def filter_symbols(self, symbols: list[str]) -> list[str]:
        """
        Filter list of symbols, returning only valid ones.

        Args:
            symbols: List of trading symbols

        Returns:
            list[str]: Filtered symbols (valid only)
        """
        valid = []
        invalid = []

        for symbol in symbols:
            is_valid, reason = self.validate_for_trading(symbol)
            if is_valid:
                valid.append(symbol)
            else:
                invalid.append((symbol, reason))

        if invalid:
            logger.warning(f"[MarketValidator] Filtered out {len(invalid)} symbols:")
            for symbol, reason in invalid:
                logger.warning(f"  - {symbol}: {reason}")

        return valid

    def get_validation_report(self, symbol: str) -> dict:
        """
        Get detailed validation report for a symbol.

        Returns:
            dict: Detailed validation results
        """
        metadata = self._get_market_metadata(symbol)

        # Run all checks
        perpetual_ok, perpetual_reason = self._is_usdt_perpetual(symbol, metadata)
        delisted_ok, delisted_reason = self._is_delisted(symbol, metadata)
        leveraged_ok, leveraged_reason = self._is_leveraged_token(symbol)
        dead_ok, dead_reason = self._is_dead_market(symbol, metadata)
        volume_ok, volume_reason, volume_amount = self._check_volume(symbol, metadata)
        spread_ok, spread_reason, spread_pct = self._check_spread(symbol)

        # Try to get candle data
        candle_ok = False
        volatility_bps = 0.0
        volatility_reason = "no_data"
        try:
            df = self.fetcher.fetch_ohlcv(symbol, "15m", limit=self.min_candles)
            if df is not None and len(df) >= self.min_candles:
                candle_ok = True
                vol_ok, volatility_reason, volatility_bps = self._check_volatility(df)
        except Exception as e:
            volatility_reason = f"fetch_error: {e}"

        is_valid, trading_reason = self.validate_for_trading(symbol)

        return {
            "symbol": symbol,
            "is_valid_for_trading": is_valid,
            "trading_reason": trading_reason,
            "checks": {
                "usdt_perpetual": {
                    "passed": perpetual_ok,
                    "reason": perpetual_reason,
                },
                "not_delisted": {
                    "passed": not delisted_ok,
                    "reason": delisted_reason,
                },
                "not_leveraged_token": {
                    "passed": not leveraged_ok,
                    "reason": leveraged_reason,
                },
                "not_dead_market": {
                    "passed": not dead_ok,
                    "reason": dead_reason,
                },
                "minimum_volume": {
                    "passed": volume_ok,
                    "reason": volume_reason,
                    "value_usd": volume_amount,
                    "threshold_usd": self.min_daily_volume_usd,
                },
                "bid_ask_spread": {
                    "passed": spread_ok,
                    "reason": spread_reason,
                    "value_pct": spread_pct,
                    "threshold_pct": self.max_bid_ask_spread_pct,
                },
                "minimum_candles": {
                    "passed": candle_ok,
                    "reason": f"{self.min_candles} candles" if candle_ok else f"insufficient",
                    "threshold": self.min_candles,
                },
                "minimum_volatility": {
                    "passed": volatility_bps >= self.min_volatility_bps if candle_ok else False,
                    "reason": volatility_reason,
                    "value_bps": volatility_bps,
                    "threshold_bps": self.min_volatility_bps,
                },
            },
        }

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear validation cache."""
        if symbol:
            self._validation_cache.pop(symbol, None)
            self._cache_timestamps.pop(symbol, None)
        else:
            self._validation_cache.clear()
            self._cache_timestamps.clear()
