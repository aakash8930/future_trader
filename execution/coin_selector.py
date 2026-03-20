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
        exchange_name="binance",
        exchange_fallbacks=None,
        exchange_timeout_ms=20000,
    ):

        self.timeframe = timeframe
        self.lookback = lookback
        self.top_k = top_k

        self.fetcher = MarketDataFetcher(
            exchange_name=exchange_name,
            fallback_exchanges=exchange_fallbacks,
            timeout_ms=exchange_timeout_ms,
        )

        self._model_cache = {}

    # --------------------------------------------------------

    def _get_model(self, symbol):

        if symbol in self._model_cache:
            return self._model_cache[symbol]

        from models.direction import DirectionModel

        model = DirectionModel.for_symbol(symbol)

        self._model_cache[symbol] = model

        return model

    # --------------------------------------------------------

    def _calc_threshold(self, model, adx):

        base = getattr(model, "long_threshold", 0.47)

        long_th = min(base, 0.47)

        if adx >= 30:
            long_th -= 0.02
        elif adx >= 20:
            long_th -= 0.01

        long_th = max(0.46, min(long_th, 0.60))

        return long_th

    # --------------------------------------------------------

    def _expected_edge(self, prob, price, atr):

        stop = price - atr * 1.7
        tp = price + atr * 3.0

        tp_ret = max((tp - price) / price, 0)
        sl_ret = max((price - stop) / price, 0)

        cost = 2 * (0.0010 + 0.0008)

        return prob * tp_ret - (1 - prob) * sl_ret - cost

    # --------------------------------------------------------

    def _score_symbol(self, symbol):

        try:

            raw = self.fetcher.fetch_ohlcv(
                symbol,
                self.timeframe,
                limit=max(self.lookback + 80, 320),
            )

            if raw is None:
                return None

            df = raw.iloc[:-1].copy()

            df = compute_core_features(df)

            if len(df) < 50:
                return None

            last = df.iloc[-1]

            price = float(last["close"])
            adx = float(last["adx"])
            atr = float(last["atr"])
            atr_pct = float(last["atr_pct"])
            ema_fast = float(last["ema_fast"])
            ema_slow = float(last["ema_slow"])
            ema200 = float(last["ema200"])
            rsi = float(last["rsi"])

            model = self._get_model(symbol)

            prob = float(model.predict_proba(df))

            th = self._calc_threshold(model, adx)

            above_ema200 = price > ema200
            bullish = ema_fast > ema_slow

            ema_gap = (price - ema200) / ema200 if ema200 else 0

            momentum_override = (
                bullish
                and adx >= 18
                and rsi >= 42
                and prob >= th - 0.01
                and ema_gap >= -0.035
            )

            edge = self._expected_edge(prob, price, atr)

            score = 0.0

            # prob
            score += prob * 0.35

            # trend
            if above_ema200:
                score += 0.25
            elif momentum_override:
                score += 0.15
            else:
                score -= 0.25

            # momentum
            if bullish:
                score += 0.15
            else:
                score -= 0.10

            # rsi
            if 38 <= rsi <= 75:
                score += 0.10
            else:
                score -= 0.10

            # volatility
            score += min(atr_pct / 0.01, 1) * 0.10

            # adx
            score += min(adx / 40, 1) * 0.10

            # edge
            score += edge * 10

            print(
                f"[CoinSelector] {symbol} "
                f"score={score:.3f} "
                f"prob={prob:.3f}/{th:.3f} "
                f"adx={adx:.1f} "
                f"rsi={rsi:.1f} "
                f"edge={edge:.5f} "
                f"above200={above_ema200}"
            )

            return score

        except Exception as e:

            print(f"[CoinSelector] error {symbol}: {e}")

            return None

    # --------------------------------------------------------

    def select(self, symbols):

        symbols = symbols or self.DEFAULT_SYMBOLS

        supported = [
            s for s in symbols
            if self.fetcher.is_symbol_supported(s)
        ]

        trained = [
            s for s in supported
            if _has_trained_model(s)
        ]

        scores = {}

        for s in trained:

            sc = self._score_symbol(s)

            if sc is not None:
                scores[s] = sc

        if not scores:

            return trained[: self.top_k]

        ranked = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        return ranked[: self.top_k]