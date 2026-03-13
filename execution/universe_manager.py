# execution/universe_manager.py

import time
from collections import deque
from typing import List

from execution.coin_selector import CoinSelector


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
    def refresh_if_needed(self) -> List[str]:
        now = time.time()
        if now - self.last_refresh < self.refresh_seconds:
            return self.active_symbols

        self.last_refresh = now
        ranked = self.selector.select(self.all_symbols)

        self.active_symbols = ranked[: self.max_active]
        print(f"🔄 Universe updated → {self.active_symbols}")

        return self.active_symbols