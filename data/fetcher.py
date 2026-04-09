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


def _sanitize_error_msg(error: Exception) -> str:
    msg = str(error)

    if "<html" in msg.lower() or "<!doctype" in msg.lower() or "<body" in msg.lower():
        if "403" in msg or "forbidden" in msg.lower():
            return "access blocked (HTTP 403)"
        if "451" in msg or "unavailable for legal reasons" in msg.lower():
            return "geo-restricted (HTTP 451)"
        if "503" in msg or "service unavailable" in msg.lower():
            return "service unavailable (HTTP 503)"
        if "cloudfront" in msg.lower():
            return "blocked by CloudFront"
        return "blocked (HTML error page)"

    if "451" in msg:
        return "geo-restricted (HTTP 451)"
    if "403" in msg:
        return "access forbidden (HTTP 403)"
    if "503" in msg:
        return "service unavailable (HTTP 503)"

    if len(msg) > 150:
        return msg[:150] + "..."

    return msg


class MarketDataFetcher:
    """
    Shared market data fetcher with retry & timeout safety.
    Supports multiple exchanges with automatic fallback on geo-restrictions.
    """

    _exchange = None
    _exchange_name = None
    _supported_symbols = set()

    def __init__(
        self,
        exchange_name: str = "binance",
        fallback_exchanges: list[str] | None = None,
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

            if MarketDataFetcher._exchange:
                try:
                    markets = MarketDataFetcher._exchange.markets
                    MarketDataFetcher._supported_symbols = set(markets.keys())
                except Exception as e:
                    sanitized = _sanitize_error_msg(e)
                    print(f"[FETCHER] Warning: could not load market symbols: {sanitized}")
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
        attempts = [primary_exchange] + fallbacks
        errors = {}

        for idx, exchange_name in enumerate(attempts):
            try:
                if idx > 0:
                    print(f"[FETCHER] trying fallback exchange: {exchange_name}")
                else:
                    print(f"[FETCHER] attempting to connect to {exchange_name}...")

                exchange = self._create_exchange(exchange_name, timeout_ms)
                exchange.load_markets()

                print(f"[FETCHER] ✓ using exchange: {exchange_name}")
                return exchange, exchange_name

            except ExchangeNotAvailable as e:
                sanitized = _sanitize_error_msg(e)
                errors[exchange_name] = f"unavailable: {sanitized}"
                print(f"[FETCHER] {exchange_name} unavailable: {sanitized}")

            except (NetworkError, RequestTimeout) as e:
                sanitized = _sanitize_error_msg(e)
                errors[exchange_name] = f"network error: {sanitized}"
                print(f"[FETCHER] {exchange_name} network error: {sanitized}")

            except ExchangeError as e:
                sanitized = _sanitize_error_msg(e)
                errors[exchange_name] = f"exchange error: {sanitized}"
                print(f"[FETCHER] {exchange_name} exchange error: {sanitized}")

            except Exception as e:
                sanitized = _sanitize_error_msg(e)
                errors[exchange_name] = f"unexpected error: {sanitized}"
                print(f"[FETCHER] {exchange_name} unexpected error: {sanitized}")

        error_summary = "\n".join(f"  - {name}: {err}" for name, err in errors.items())
        raise RuntimeError(
            f"[FETCHER] All exchanges failed:\n{error_summary}\n\n"
            f"This deployment region may block crypto exchanges.\n"
            f"Try setting EXCHANGE_FALLBACKS to different exchanges or deploy in a different region."
        )

    def _create_exchange(self, exchange_name: str, timeout_ms: int):
        exchange_name = exchange_name.lower().strip()

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
        return symbol in self.supported_symbols

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        retries: int = 3,
    ) -> pd.DataFrame | None:
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

            except (RequestTimeout, NetworkError):
                if attempt == retries:
                    raise
                time.sleep(2 * attempt)

        raise RuntimeError("fetch_ohlcv failed after retries")

    def fetch_last_price(
        self,
        symbol: str,
        retries: int = 3,
    ) -> float | None:
        """
        Real-time price for managing open positions.
        Uses ticker last/close/bid/ask fallback.
        """
        if not self.is_symbol_supported(symbol):
            return None

        for attempt in range(1, retries + 1):
            try:
                ticker = self.exchange.fetch_ticker(symbol)

                candidates = [
                    ticker.get("last"),
                    ticker.get("close"),
                    ticker.get("bid"),
                    ticker.get("ask"),
                ]

                for value in candidates:
                    if value is not None:
                        price = float(value)
                        if price > 0:
                            return price

                raise RuntimeError("ticker returned no usable price")

            except (RequestTimeout, NetworkError):
                if attempt == retries:
                    return None
                time.sleep(1 * attempt)
            except Exception:
                return None

        return None