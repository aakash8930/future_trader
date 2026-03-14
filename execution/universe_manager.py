# execution/universe_manager.py

import time
from collections import deque
from typing import List

from execution.coin_selector import CoinSelector, _has_trained_model

# Exchanges where CoinSelector volume/ATR ranking is unreliable.
# On these exchanges we use configured symbols directly.
_FALLBACK_EXCHANGES = {"kraken", "bybit", "okx"}


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
    def _select_for_fallback_exchange(self) -> List[str]:
        """
        On fallback exchanges (kraken, bybit, okx, …) skip CoinSelector
        ranking entirely.  Apply only symbol-support and trained-model checks
        against the configured symbol list, then return it directly.
        """
        print(
            f"[Universe] fallback exchange {self.exchange_name} detected "
            f"→ using configured symbols directly"
        )

        fetcher = self.selector.fetcher

        supported = [
            s for s in self.all_symbols if fetcher.is_symbol_supported(s)
        ]
        unsupported = [s for s in self.all_symbols if s not in supported]
        if unsupported:
            print(
                f"[Universe] Unsupported on {self.exchange_name}: {unsupported}"
            )

        eligible = [s for s in supported if _has_trained_model(s)]
        skipped = [s for s in supported if s not in eligible]
        if skipped:
            print(f"[Universe] Skipped (no model): {skipped}")

        return eligible[: self.max_active]

    # ----------------------------------
    def refresh_if_needed(self) -> List[str]:
        now = time.time()
        if now - self.last_refresh < self.refresh_seconds:
            return self.active_symbols

        self.last_refresh = now

        if self.exchange_name in _FALLBACK_EXCHANGES:
            self.active_symbols = self._select_for_fallback_exchange()
        else:
            ranked = self.selector.select(self.all_symbols)
            self.active_symbols = ranked[: self.max_active]

        print(f"🔄 Universe updated → {self.active_symbols}")

        return self.active_symbols