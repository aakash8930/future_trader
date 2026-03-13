
# execution/coin_selector.py

import os
import ta
import numpy as np
from data.fetcher import MarketDataFetcher


def _has_trained_model(symbol: str) -> bool:
    """Return True only if both model.pt and scaler.save exist for this symbol."""
    folder = os.path.join("models", symbol.replace("/", "_"))
    return (
        os.path.exists(os.path.join(folder, "model.pt"))
        and os.path.exists(os.path.join(folder, "scaler.save"))
    )


class CoinSelector:
    """
    Selects the best trading symbols based on volatility, volume,
    and trend strength.

    This mimics how hedge-fund crypto bots rank markets.
    """

    DEFAULT_SYMBOLS = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "DOGE/USDT",
        "BNB/USDT",
    ]

    def __init__(
        self,
        timeframe: str = "15m",
        lookback: int = 200,
        top_k: int = 4,
        min_atr_pct: float = 0.001,
        min_volume_ratio: float = 0.7,
    ):
        self.timeframe = timeframe
        self.lookback = lookback
        self.top_k = top_k
        self.min_atr_pct = min_atr_pct
        self.min_volume_ratio = min_volume_ratio
        self.fetcher = MarketDataFetcher()

    def _score_symbol(self, symbol: str) -> float | None:

        try:
            df = self.fetcher.fetch_ohlcv(
                symbol,
                self.timeframe,
                limit=self.lookback,
            )

            if df is None or len(df) < 120:
                print(f"[CoinSelector] {symbol}: insufficient data ({len(df) if df is not None else 0} rows)")
                return None

            atr = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=14
            ).average_true_range()

            adx = ta.trend.ADXIndicator(
                df["high"], df["low"], df["close"], window=14
            ).adx()

            vol_ma = df["volume"].rolling(20).mean()

            atr_pct = atr.iloc[-1] / df["close"].iloc[-1]
            volume_ratio = df["volume"].iloc[-1] / vol_ma.iloc[-1]
            trend_strength = min(adx.iloc[-1], 40)

            if atr_pct < self.min_atr_pct:
                print(f"[CoinSelector] {symbol}: atr_pct too low ({atr_pct:.4f})")
                return None

            if volume_ratio < self.min_volume_ratio:
                print(f"[CoinSelector] {symbol}: volume_ratio too low ({volume_ratio:.2f})")
                return None

            score = atr_pct * volume_ratio * trend_strength

            return float(score)

        except Exception as exc:
            print(f"[CoinSelector] {symbol}: error during scoring — {exc}")
            return None

    def select(self, symbols: list[str]) -> list[str]:

        configured_symbols = symbols  # Store original input before fallback
        if not symbols:
            symbols = self.DEFAULT_SYMBOLS

        # Only consider symbols that have a trained model on disk.
        eligible = [s for s in symbols if _has_trained_model(s)]
        skipped  = [s for s in symbols if s not in eligible]
        if skipped:
            print(f"[CoinSelector] Skipped (no model): {skipped}")

        scores = {}

        for symbol in eligible:

            score = self._score_symbol(symbol)

            if score is not None:
                scores[symbol] = score

        ranked = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        if not ranked:
            # Prefer configured symbols, only fall back to DEFAULT_SYMBOLS if none provided
            if configured_symbols:
                print("⚠️ CoinSelector empty → fallback to configured symbols with trained models")
                fallback = [s for s in configured_symbols if _has_trained_model(s)]
            else:
                print("⚠️ CoinSelector empty → fallback to default symbols with trained models")
                fallback = [s for s in self.DEFAULT_SYMBOLS if _has_trained_model(s)]
            return fallback[: self.top_k] or configured_symbols[: self.top_k] or self.DEFAULT_SYMBOLS[: self.top_k]

        selected = ranked[: self.top_k]

        return selected
