# execution/universe_manager.py

import time
from collections import deque
from typing import List

from execution.coin_selector import CoinSelector, _has_trained_model


class UniverseManager:
    """
    Manages the tradable symbol universe autonomously.
    """

    def __init__(
        self,
        all_symbols: List[str],
        timeframe: str,
        max_active: int,
        refresh_minutes: int = 60,
        exchange_name: str = "binance",
        exchange_fallbacks: List[str] = None,
        exchange_timeout_ms: int = 20000,
    ):
        self.all_symbols = all_symbols
        self.timeframe = timeframe
        self.max_active = max_active
        self.refresh_seconds = refresh_minutes * 60
        self.exchange_name = exchange_name.lower()

        self.selector = CoinSelector(
            timeframe=timeframe,
            top_k=max_active * 2,
            exchange_name=exchange_name,
            exchange_fallbacks=exchange_fallbacks,
            exchange_timeout_ms=exchange_timeout_ms,
        )

        self.active_symbols: List[str] = []
        self.last_refresh = 0

    # ----------------------------------
    def _active_exchange_name(self) -> str:
        """
        Read the currently active exchange from the shared MarketDataFetcher
        singleton via the selector's fetcher instance.
        """
        active = getattr(self.selector.fetcher, "exchange_name", None)
        if not active:
            active = self.exchange_name
        return str(active).lower()

    # ----------------------------------
    def _select_configured_symbols_direct(self, active_exchange: str) -> List[str]:
        """
        On non-binance exchanges skip CoinSelector ranking entirely and
        use configured symbols directly after compatibility/model filtering.
        """
        print(
            f"[Universe] fallback exchange {active_exchange} detected "
            f"→ using configured symbols directly"
        )

        fetcher = self.selector.fetcher

        supported = [s for s in self.all_symbols if fetcher.is_symbol_supported(s)]
        eligible = [s for s in supported if _has_trained_model(s)]
        return eligible[: self.max_active]

    # ----------------------------------
    def refresh_if_needed(self) -> List[str]:
        now = time.time()
        if now - self.last_refresh < self.refresh_seconds:
            return self.active_symbols

        self.last_refresh = now
        active_exchange = self._active_exchange_name()

        # Always try selector first regardless of exchange
        ranked = self.selector.select(self.all_symbols)
        if ranked:
            self.active_symbols = ranked[: self.max_active]
        else:
            # Fallback to configured symbols if selector returns empty
            print(f"[Universe] selector returned empty on {active_exchange} → using configured symbols")
            self.active_symbols = self._select_configured_symbols_direct(active_exchange)

        print(f"🔄 Universe updated → {self.active_symbols}")

        return self.active_symbols