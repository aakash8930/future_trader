import time
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
        selector_top_k_multiplier: int = 2,
        selector_min_atr_pct: float = 0.001,
        selector_soft_min_volume_ratio: float = 0.15,
        exchange_name: str = "binance",
        exchange_fallbacks: List[str] | None = None,
        exchange_timeout_ms: int = 20000,
    ):
        self.all_symbols = all_symbols
        self.timeframe = timeframe
        self.max_active = max_active
        self.refresh_seconds = refresh_minutes * 60
        self.last_refresh = 0.0
        self.active_symbols: List[str] = []

        self.selector = CoinSelector(
            timeframe=timeframe,
            top_k=max_active * selector_top_k_multiplier,
            min_atr_pct=selector_min_atr_pct,
            soft_min_volume_ratio=selector_soft_min_volume_ratio,
            exchange_name=exchange_name,
            exchange_fallbacks=exchange_fallbacks,
            exchange_timeout_ms=exchange_timeout_ms,
        )

    def _fallback_symbols(self) -> List[str]:
        fetcher = self.selector.fetcher
        supported = [s for s in self.all_symbols if fetcher.is_symbol_supported(s)]
        eligible = [s for s in supported if _has_trained_model(s)]
        return eligible[: self.max_active]

    def refresh_if_needed(self) -> List[str]:
        now = time.time()
        if now - self.last_refresh < self.refresh_seconds and self.active_symbols:
            return self.active_symbols

        self.last_refresh = now

        ranked = self.selector.select(self.all_symbols)
        selected = ranked[: self.max_active]

        if not selected:
            print("[Universe] selector returned empty → using configured fallback symbols")
            selected = self._fallback_symbols()

        if not selected:
            print("[Universe] no eligible symbols available")
            self.active_symbols = []
            return self.active_symbols

        if selected != self.active_symbols:
            print(f"🔄 Universe updated → {selected}")

        self.active_symbols = selected
        return self.active_symbols