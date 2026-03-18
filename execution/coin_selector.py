# execution/coin_selector.py

import os
import ta
import numpy as np

from data.fetcher import MarketDataFetcher


def _has_trained_model(symbol: str) -> bool:
    """Return True only if model, scaler, and metadata exist for this symbol."""
    folder = os.path.join("models", symbol.replace("/", "_"))
    return (
        os.path.exists(os.path.join(folder, "model.pt"))
        and os.path.exists(os.path.join(folder, "scaler.save"))
        and os.path.exists(os.path.join(folder, "metadata.json"))
    )


class CoinSelector:
    """
    Selects the best trading symbols based on model confidence,
    volatility, volume, and trend strength.
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
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"insufficient data ({len(df)} rows)"
                )
                return None

            atr = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=14
            ).average_true_range()

            adx = ta.trend.ADXIndicator(
                df["high"], df["low"], df["close"], window=14
            ).adx()

            vol_ma = df["volume"].rolling(20).mean()

            # Use most recent closed candle
            last_volume = df["volume"].iloc[-1]
            last_vol_ma = vol_ma.iloc[-1]

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

            last_close = float(df["close"].iloc[-1])
            atr_pct = float(atr.iloc[-1] / last_close)
            volume_ratio = float(last_volume / last_vol_ma)
            adx_value = float(adx.iloc[-1])
            trend_strength = min(adx_value, 40.0)

            # Keep existing safety filters
            if atr_pct < self.min_atr_pct:
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"atr_pct too low ({atr_pct:.4f})"
                )
                return None

            if volume_ratio < self.min_volume_ratio:
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"volume_ratio too low ({volume_ratio:.2f})"
                )
                return None

            # -------- Model probability --------
            prob_up = 0.5
            try:
                from models.direction import DirectionModel

                model_dir = os.path.join("models", symbol.replace("/", "_"))
                model_path = os.path.join(model_dir, "model.pt")
                scaler_path = os.path.join(model_dir, "scaler.save")
                metadata_path = os.path.join(model_dir, "metadata.json")

                model = DirectionModel(model_path, scaler_path, metadata_path)
                prob_up = float(model.predict_proba(df))
            except Exception:
                prob_up = 0.5

            # -------- Trend bonus --------
            price = float(df["close"].iloc[-1])
            ema_fast = float(
                ta.trend.EMAIndicator(df["close"], window=9).ema_indicator().iloc[-1]
            )
            ema_slow = float(
                ta.trend.EMAIndicator(df["close"], window=21).ema_indicator().iloc[-1]
            )
            ema200 = float(
                ta.trend.EMAIndicator(df["close"], window=200).ema_indicator().iloc[-1]
            )

            above_ema200 = price > ema200
            bullish_cross = ema_fast > ema_slow

            trend_bonus = 0.0
            if above_ema200 and bullish_cross:
                trend_bonus = 1.0
            elif above_ema200:
                trend_bonus = 0.5

            # -------- Normalized scores --------
            adx_score = min(trend_strength / 40.0, 1.0)
            atr_score = min(atr_pct / 0.01, 1.0)
            vol_score = min(volume_ratio / 2.0, 1.0)

            # -------- Composite score --------
            score = (
                prob_up * 0.40
                + adx_score * 0.20
                + atr_score * 0.15
                + vol_score * 0.15
                + trend_bonus * 0.10
            )

            return float(score)

        except Exception as exc:
            print(
                f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                f"error during scoring — {exc}"
            )
            return None

    def select(self, symbols: list[str]) -> list[str]:
        configured_symbols = symbols
        if not symbols:
            symbols = self.DEFAULT_SYMBOLS

        supported = [s for s in symbols if self.fetcher.is_symbol_supported(s)]
        unsupported = [s for s in symbols if s not in supported]
        if unsupported:
            print(
                f"[CoinSelector] Unsupported on {self.fetcher.exchange_name}: {unsupported}"
            )

        eligible = [s for s in supported if _has_trained_model(s)]
        skipped = [s for s in supported if s not in eligible]
        if skipped:
            print(f"[CoinSelector] Skipped (no model): {skipped}")

        scores = {}

        for symbol in eligible:
            score = self._score_symbol(symbol)
            if score is not None:
                scores[symbol] = score

        ranked = sorted(scores, key=scores.get, reverse=True)

        if not ranked:
            if configured_symbols:
                print("⚠️ CoinSelector empty → fallback to configured symbols with trained models")
                fallback = [s for s in configured_symbols if _has_trained_model(s)]
            else:
                print("⚠️ CoinSelector empty → fallback to default symbols with trained models")
                fallback = [s for s in self.DEFAULT_SYMBOLS if _has_trained_model(s)]

            return (
                fallback[: self.top_k]
                or configured_symbols[: self.top_k]
                or self.DEFAULT_SYMBOLS[: self.top_k]
            )

        selected = ranked[: self.top_k]
        return selected