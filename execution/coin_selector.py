# execution/coin_selector.py

import os
import numpy as np

from data.fetcher import MarketDataFetcher
from features.technicals import compute_core_features


def _has_trained_model(symbol: str) -> bool:
    folder = os.path.join("models", symbol.replace("/", "_"))
    return (
        os.path.exists(os.path.join(folder, "model.pt"))
        and os.path.exists(os.path.join(folder, "scaler.save"))
        and os.path.exists(os.path.join(folder, "metadata.json"))
    )


class CoinSelector:

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
        timeframe="15m",
        lookback=240,
        top_k=4,
        min_atr_pct=0.001,
        rsi_long_min=38,
        rsi_long_max=75,
    ):

        self.timeframe = timeframe
        self.lookback = lookback
        self.top_k = top_k
        self.min_atr_pct = min_atr_pct
        self.rsi_long_min = rsi_long_min
        self.rsi_long_max = rsi_long_max

        self.fetcher = MarketDataFetcher()
        self._model_cache = {}

    # -----------------------------

    def _get_model(self, symbol):

        if symbol in self._model_cache:
            return self._model_cache[symbol]

        from models.direction import DirectionModel

        model = DirectionModel.for_symbol(symbol)
        self._model_cache[symbol] = model
        return model

    # -----------------------------

    def _score_symbol(self, symbol):

        raw = self.fetcher.fetch_ohlcv(
            symbol,
            self.timeframe,
            limit=320,
        )

        if raw is None or len(raw) < 260:
            return None

        df = raw.iloc[:-1].copy()
        df = compute_core_features(df)

        last = df.iloc[-1]

        price = float(last["close"])
        adx = float(last["adx"])
        atr_pct = float(last["atr_pct"])
        ema_fast = float(last["ema_fast"])
        ema_slow = float(last["ema_slow"])
        ema200 = float(last["ema200"])
        rsi = float(last["rsi"])

        if atr_pct < self.min_atr_pct:
            return None

        model = self._get_model(symbol)
        prob = float(model.predict_proba(df))
        th = float(getattr(model, "long_threshold", 0.50))

        above_ema200 = price > ema200
        bullish_cross = ema_fast > ema_slow

        ema_gap = (price - ema200) / ema200

        rsi_ok = self.rsi_long_min <= rsi <= self.rsi_long_max

        # -------- HARD REJECTION --------

        if ema_gap < -0.03:
            return -999

        # -------- STRUCTURE --------

        structure = 0

        if above_ema200:
            structure += 0.6
        elif ema_gap > -0.015:
            structure += 0.25

        if bullish_cross:
            structure += 0.2

        if rsi_ok:
            structure += 0.1

        if prob >= th:
            structure += 0.1

        # -------- TREND POWER --------

        adx_score = min(adx / 40, 1)

        atr_score = min(atr_pct / 0.01, 1)

        # -------- FINAL SCORE --------

        score = (
            prob * 0.25
            + structure * 0.45
            + adx_score * 0.15
            + atr_score * 0.15
        )

        return float(score)

    # -----------------------------

    def select(self, symbols):

        if not symbols:
            symbols = self.DEFAULT_SYMBOLS

        scores = {}

        for s in symbols:

            if not _has_trained_model(s):
                continue

            score = self._score_symbol(s)

            if score is not None:
                scores[s] = score

        ranked = sorted(scores, key=scores.get, reverse=True)

        ranked = [s for s in ranked if scores[s] > -100]

        return ranked[: self.top_k]