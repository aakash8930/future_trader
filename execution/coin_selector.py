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
        exchange_name: str = "binance",
        exchange_fallbacks: list[str] = None,
        exchange_timeout_ms: int = 20000,
    ):
        self.timeframe = timeframe
        self.lookback = lookback
        self.top_k = top_k
        self.min_atr_pct = min_atr_pct
        self.min_volume_ratio = min_volume_ratio
        self.fetcher = MarketDataFetcher(
            exchange_name=exchange_name,
            fallback_exchanges=exchange_fallbacks,
            timeout_ms=exchange_timeout_ms,
        )

    def _score_symbol(self, symbol: str) -> float | None:

        try:
            df = self.fetcher.fetch_ohlcv(
                symbol,
                self.timeframe,
                limit=self.lookback,
            )

            # Symbol not supported on this exchange
            if df is None:
                return None

            if len(df) < 120:
                print(f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: insufficient data ({len(df)} rows)")
                return None

            atr = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=14
            ).average_true_range()

            adx = ta.trend.ADXIndicator(
                df["high"], df["low"], df["close"], window=14
            ).adx()

            vol_ma = df["volume"].rolling(20).mean()

            # Use most recent CLOSED candle (iloc[-1])
            last_volume = df["volume"].iloc[-1]
            last_vol_ma = vol_ma.iloc[-1]

            # Validate volume data before computing ratio
            if (
                last_volume <= 0
                or last_vol_ma <= 0
                or not np.isfinite(last_volume)
                or not np.isfinite(last_vol_ma)
            ):
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"invalid volume data (last={last_volume:.2f}, ma={last_vol_ma:.2f})"
                )
                return None

            atr_pct = atr.iloc[-1] / df["close"].iloc[-1]
            volume_ratio = last_volume / last_vol_ma
            trend_strength = min(adx.iloc[-1], 40)

            # SAFETY CHECKS: Keep existing filters
            if atr_pct < self.min_atr_pct:
                print(f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: atr_pct too low ({atr_pct:.4f})")
                return None

            if volume_ratio < self.min_volume_ratio:
                print(f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: volume_ratio too low ({volume_ratio:.2f})")
                return None

            # ========== NEW SCORING LOGIC ==========  

            # 1. Get model probability
            prob_up = 0.5
            try:
                from models.direction import DirectionModel
                model_path = os.path.join("models", symbol.replace("/", "_"), "model.pt")
                scaler_path = os.path.join("models", symbol.replace("/", "_"), "scaler.save")
                model = DirectionModel(model_path, scaler_path)
                prob_up = model.predict_proba(df)
            except Exception as e:
                prob_up = 0.5

            # 2. Calculate trend bonuses
            price = df["close"].iloc[-1]
            ema_fast = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator().iloc[-1]
            ema_slow = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator().iloc[-1]
            ema200 = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator().iloc[-1]

            above_ema200 = price > ema200
            bullish_cross = ema_fast > ema_slow

            trend_bonus = 0.0
            if above_ema200 and bullish_cross:
                trend_bonus = 1.0
            elif above_ema200:
                trend_bonus = 0.5

            # 3. Normalize indicators to [0, 1]
            adx_score = min(trend_strength / 40.0, 1.0)
            atr_score = min(atr_pct / 0.01, 1.0)
            vol_score = min(volume_ratio / 2.0, 1.0)

            # 4. Weighted composite score
            score = (
                prob_up * 0.4
                + adx_score * 0.2
                + atr_score * 0.15
                + vol_score * 0.15
                + trend_bonus * 0.1
            )

            return float(score)

        except Exception as exc:
            print(f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: error during scoring — {exc}")
            return None

    def select(self, symbols: list[str]) -> list[str]:

        configured_symbols = symbols  # Store original input before fallback
        if not symbols:
            symbols = self.DEFAULT_SYMBOLS

        # Filter out symbols not supported on this exchange
        supported = [s for s in symbols if self.fetcher.is_symbol_supported(s)]
        unsupported = [s for s in symbols if s not in supported]
        if unsupported:
            print(f"[CoinSelector] Unsupported on {self.fetcher.exchange_name}: {unsupported}")

        # Only consider symbols that have a trained model on disk.
        eligible = [s for s in supported if _has_trained_model(s)]
        skipped  = [s for s in supported if s not in eligible]
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