
#data/fetcher.py

import time
import ccxt
import pandas as pd
from ccxt.base.errors import (
    RequestTimeout,
    NetworkError,
    ExchangeNotAvailable,
    ExchangeError,
)


class MarketDataFetcher:
    """
    Shared market data fetcher with retry & timeout safety.
    Supports multiple exchanges with automatic fallback on geo-restrictions.
    """

    _exchange = None  # 🔑 singleton
    _exchange_name = None  # Track which exchange is active
    _supported_symbols = set()  # Cache of symbols supported by active exchange

    def __init__(
        self,
        exchange_name: str = "binance",
        fallback_exchanges: list[str] = None,
        timeout_ms: int = 20000,
    ):
        if MarketDataFetcher._exchange is None:
            if fallback_exchanges is None:
                fallback_exchanges = ["bybit", "kraken", "okx"]

            MarketDataFetcher._exchange, MarketDataFetcher._exchange_name = (
                self._init_exchange_with_fallback(
                    exchange_name,
                    fallback_exchanges,
                    timeout_ms,
                )
            )

            # Load supported symbols
            if MarketDataFetcher._exchange:
                try:
                    markets = MarketDataFetcher._exchange.markets
                    MarketDataFetcher._supported_symbols = set(markets.keys())
                except Exception as e:
                    print(f"[FETCHER] Warning: could not load market symbols: {e}")
                    MarketDataFetcher._supported_symbols = set()

        self.exchange = MarketDataFetcher._exchange
        self.exchange_name = MarketDataFetcher._exchange_name
        self.supported_symbols = MarketDataFetcher._supported_symbols

    def _init_exchange_with_fallback(
        self,
        primary_exchange: str,
        fallbacks: list[str],
        timeout_ms: int,
    ) -> tuple:
        """
        Initialize exchange with fallback support.
        Returns (exchange_object, exchange_name) or raises RuntimeError.
        """
        attempts = [primary_exchange] + fallbacks
        errors = {}

        for exchange_name in attempts:
            try:
                print(f"[FETCHER] attempting to connect to {exchange_name}...")
                exchange = self._create_exchange(exchange_name, timeout_ms)

                # Critical: load_markets() can fail with 451 or NetworkError
                exchange.load_markets()

                print(f"[FETCHER] ✓ using exchange: {exchange_name}")
                return exchange, exchange_name

            except ExchangeNotAvailable as e:
                msg = str(e)
                if "451" in msg or "geo" in msg.lower():
                    errors[exchange_name] = f"geo-restricted (HTTP 451)"
                else:
                    errors[exchange_name] = f"unavailable: {e}"
                print(f"[FETCHER] {exchange_name} unavailable: {errors[exchange_name]}")

            except (NetworkError, RequestTimeout) as e:
                errors[exchange_name] = f"network error: {e}"
                print(f"[FETCHER] {exchange_name} network error: {e}")

            except ExchangeError as e:
                errors[exchange_name] = f"exchange error: {e}"
                print(f"[FETCHER] {exchange_name} exchange error: {e}")

            except Exception as e:
                errors[exchange_name] = f"unexpected error: {e}"
                print(f"[FETCHER] {exchange_name} unexpected error: {e}")

        # All exchanges failed - generate clean error message
        error_summary = "\n".join(f"  - {name}: {err}" for name, err in errors.items())
        raise RuntimeError(
            f"[FETCHER] All exchanges failed:\n{error_summary}\n\n"
            f"This deployment region may block crypto exchanges.\n"
            f"Try setting EXCHANGE_FALLBACKS to different exchanges or deploy in a different region."
        )

    def _create_exchange(self, exchange_name: str, timeout_ms: int):
        """Create exchange instance dynamically from ccxt."""
        exchange_name = exchange_name.lower().strip()

        # Map exchange names to ccxt classes
        exchange_classes = {
            "binance": ccxt.binance,
            "bybit": ccxt.bybit,
            "kraken": ccxt.kraken,
            "okx": ccxt.okx,
            "coinbase": ccxt.coinbase,
            "kucoin": ccxt.kucoin,
            "huobi": ccxt.huobi,
            "gateio": ccxt.gateio,
        }

        if exchange_name not in exchange_classes:
            supported = ", ".join(exchange_classes.keys())
            raise ValueError(
                f"Unsupported exchange: {exchange_name}. Supported: {supported}"
            )

        exchange_class = exchange_classes[exchange_name]
        return exchange_class({
            "enableRateLimit": True,
            "timeout": timeout_ms,
        })

    def is_symbol_supported(self, symbol: str) -> bool:
        """Check if symbol is supported on the active exchange."""
        return symbol in self.supported_symbols

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        retries: int = 3,
    ) -> pd.DataFrame | None:
        """
        Fetch OHLCV data with retry logic.
        Returns None if symbol is not supported on this exchange.
        """
        # Check symbol compatibility
        if not self.is_symbol_supported(symbol):
            print(f"[FETCHER] symbol {symbol} not supported on {self.exchange_name}, skipping")
            return None

        for attempt in range(1, retries + 1):
            try:
                bars = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    limit=limit,
                )

                if not bars:
                    raise RuntimeError("empty OHLCV")

                return pd.DataFrame(
                    bars,
                    columns=["time", "open", "high", "low", "close", "volume"],
                )

            except (RequestTimeout, NetworkError) as e:
                if attempt == retries:
                    raise
                time.sleep(2 * attempt)

        raise RuntimeError("fetch_ohlcv failed after retries")
