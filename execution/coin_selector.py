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
    """
    Scores symbols for live trading.

    Design goals:
    - uses only closed candles
    - avoids selecting symbols the runner will almost certainly reject
    - strongly prefers symbols already aligned with strategy rules
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
        lookback: int = 240,
        top_k: int = 4,
        min_atr_pct: float = 0.0008,
        soft_min_volume_ratio: float = 0.15,
        rsi_long_min: float = 45.0,
        rsi_long_max: float = 64.0,
        exchange_name: str = "binance",
        exchange_fallbacks: list[str] | None = None,
        exchange_timeout_ms: int = 20000,
    ):
        self.timeframe = timeframe
        self.lookback = lookback
        self.top_k = top_k
        self.min_atr_pct = min_atr_pct
        self.soft_min_volume_ratio = soft_min_volume_ratio
        self.rsi_long_min = rsi_long_min
        self.rsi_long_max = rsi_long_max

        self.fetcher = MarketDataFetcher(
            exchange_name=exchange_name,
            fallback_exchanges=exchange_fallbacks,
            timeout_ms=exchange_timeout_ms,
        )

        self._model_cache: dict[str, object] = {}

    def _get_model(self, symbol: str):
        if symbol in self._model_cache:
            return self._model_cache[symbol]

        from models.direction import DirectionModel

        model = DirectionModel.for_symbol(symbol)
        self._model_cache[symbol] = model
        return model

    def _model_probability_and_threshold(self, symbol: str, df) -> tuple[float, float]:
        try:
            model = self._get_model(symbol)
            prob = float(model.predict_proba(df))
            long_th = float(getattr(model, "long_threshold", 0.50))

            if not (0.0 <= prob <= 1.0):
                prob = 0.5

            return prob, long_th
        except Exception:
            return 0.5, 0.50

    def _score_symbol(self, symbol: str) -> float | None:
        try:
            raw_fetch_limit = max(self.lookback + 80, 320)

            raw_df = self.fetcher.fetch_ohlcv(
                symbol,
                self.timeframe,
                limit=raw_fetch_limit,
            )

            if raw_df is None:
                return None

            if len(raw_df) < 260:
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"insufficient raw data ({len(raw_df)} rows)"
                )
                return None

            df = raw_df.iloc[:-1].copy()
            df = compute_core_features(df)

            if df.empty or len(df) < 50:
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"feature window too small ({len(df)} rows)"
                )
                return None

            last = df.iloc[-1]
            price = float(last["close"])
            adx = float(last["adx"])
            atr_pct = float(last["atr_pct"])
            ema_fast = float(last["ema_fast"])
            ema_slow = float(last["ema_slow"])
            ema200 = float(last["ema200"])
            rsi = float(last["rsi"])

            vol_series = df["volume"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vol_series) < 30:
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"not enough clean volume data"
                )
                return None

            recent_window = vol_series.tail(3)
            baseline_window = vol_series.iloc[-23:-3]

            if len(baseline_window) < 10:
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"baseline volume window too small"
                )
                return None

            recent_vol = float(recent_window.median())
            baseline_vol = float(baseline_window.median())

            if recent_vol <= 0 or baseline_vol <= 0:
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"invalid volume data (recent={recent_vol:.2f}, base={baseline_vol:.2f})"
                )
                return None

            volume_ratio = recent_vol / baseline_vol

            if atr_pct < self.min_atr_pct:
                print(
                    f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                    f"atr_pct too low ({atr_pct:.4f})"
                )
                return None

            prob_up, model_long_th = self._model_probability_and_threshold(symbol, df)
            selector_long_th = max(model_long_th, 0.50)

            if prob_up < 0.45:
                print(
                    f"[CoinSelector] {symbol} | "
                    f"score=-999.000 prob={prob_up:.3f}/{selector_long_th:.3f} "
                    f"adx={adx:.1f} atr_pct={atr_pct:.4f} rsi={rsi:.1f} "
                    f"vol_ratio={volume_ratio:.2f} reasons=['prob_too_low']"
                )
                return -999.0

            above_ema200 = price > ema200
            bullish_cross = ema_fast > ema_slow
            rsi_ok = self.rsi_long_min <= rsi <= self.rsi_long_max
            prob_ok = prob_up >= selector_long_th
            adx_ok = adx >= 22.0

            ema_gap_pct = (price - ema200) / ema200 if ema200 > 0 else -1.0
            ema_fast_vs_slow_pct = (
                (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0.0
            )

            adx_score = min(max(adx, 0.0) / 40.0, 1.0)
            atr_score = min(max(atr_pct, 0.0) / 0.01, 1.0)

            volume_score = min(max(volume_ratio, 0.0) / 1.5, 1.0)
            if volume_ratio < self.soft_min_volume_ratio:
                volume_score *= 0.35

            reasons: list[str] = []

            # Below-EMA names must look like true recovery setups.
            recovery_candidate = (
                bullish_cross
                and adx >= 26
                and prob_up >= selector_long_th + 0.012
                and rsi_ok
                and ema_gap_pct >= -0.006
                and ema_fast_vs_slow_pct >= 0.0010
            )

            if not above_ema200 and not recovery_candidate:
                reasons.append("far_below_ema200")
                print(
                    f"[CoinSelector] {symbol} | "
                    f"score=-999.000 prob={prob_up:.3f}/{selector_long_th:.3f} "
                    f"adx={adx:.1f} atr_pct={atr_pct:.4f} rsi={rsi:.1f} "
                    f"vol_ratio={volume_ratio:.2f} "
                    f"above_ema200={above_ema200} bullish_cross={bullish_cross} "
                    f"reasons={reasons}"
                )
                return -999.0

            near_cross = ema_fast >= ema_slow * 0.9993
            continuation_candidate = (
                above_ema200
                and not bullish_cross
                and near_cross
                and adx_ok
                and prob_up >= selector_long_th + 0.008
                and ema_gap_pct >= 0.0015
                and rsi_ok
            )

            if above_ema200 and not bullish_cross and not continuation_candidate:
                reasons.append("bearish_cross_hard_reject")
                print(
                    f"[CoinSelector] {symbol} | "
                    f"score=-999.000 prob={prob_up:.3f}/{selector_long_th:.3f} "
                    f"adx={adx:.1f} atr_pct={atr_pct:.4f} rsi={rsi:.1f} "
                    f"vol_ratio={volume_ratio:.2f} "
                    f"above_ema200={above_ema200} bullish_cross={bullish_cross} "
                    f"reasons={reasons}"
                )
                return -999.0

            # Hard-align with strategy for above-EMA selections.
            if above_ema200 and bullish_cross:
                if not adx_ok:
                    reasons.append("adx_low")
                    print(
                        f"[CoinSelector] {symbol} | "
                        f"score=-999.000 prob={prob_up:.3f}/{selector_long_th:.3f} "
                        f"adx={adx:.1f} atr_pct={atr_pct:.4f} rsi={rsi:.1f} "
                        f"vol_ratio={volume_ratio:.2f} "
                        f"above_ema200={above_ema200} bullish_cross={bullish_cross} "
                        f"reasons={reasons}"
                    )
                    return -999.0

                if not rsi_ok:
                    reasons.append("rsi_bad")
                    print(
                        f"[CoinSelector] {symbol} | "
                        f"score=-999.000 prob={prob_up:.3f}/{selector_long_th:.3f} "
                        f"adx={adx:.1f} atr_pct={atr_pct:.4f} rsi={rsi:.1f} "
                        f"vol_ratio={volume_ratio:.2f} "
                        f"above_ema200={above_ema200} bullish_cross={bullish_cross} "
                        f"reasons={reasons}"
                    )
                    return -999.0

                if not prob_ok:
                    reasons.append("prob_low")
                    print(
                        f"[CoinSelector] {symbol} | "
                        f"score=-999.000 prob={prob_up:.3f}/{selector_long_th:.3f} "
                        f"adx={adx:.1f} atr_pct={atr_pct:.4f} rsi={rsi:.1f} "
                        f"vol_ratio={volume_ratio:.2f} "
                        f"above_ema200={above_ema200} bullish_cross={bullish_cross} "
                        f"reasons={reasons}"
                    )
                    return -999.0

            structure_score = 0.0
            if above_ema200 and bullish_cross:
                structure_score += 0.82
            elif continuation_candidate:
                structure_score += 0.46
            elif recovery_candidate:
                structure_score += 0.22

            if bullish_cross:
                structure_score += 0.08
            if rsi_ok:
                structure_score += 0.05
            if prob_ok:
                structure_score += 0.07

            penalty = 0.0

            if not above_ema200:
                penalty += 0.12
                reasons.append("below_ema200")

            if not bullish_cross:
                penalty += 0.18
                reasons.append("bearish_cross")

            if not rsi_ok:
                penalty += 0.18
                reasons.append("rsi_bad")

            if not prob_ok:
                penalty += 0.08
                reasons.append("prob_low")

            score = (
                prob_up * 0.16
                + adx_score * 0.12
                + atr_score * 0.10
                + volume_score * 0.08
                + structure_score * 0.54
                - penalty
            )

            print(
                f"[CoinSelector] {symbol} | "
                f"score={score:.3f} prob={prob_up:.3f}/{selector_long_th:.3f} "
                f"adx={adx:.1f} atr_pct={atr_pct:.4f} rsi={rsi:.1f} "
                f"vol_ratio={volume_ratio:.2f} "
                f"above_ema200={above_ema200} bullish_cross={bullish_cross} "
                f"reasons={reasons if reasons else ['ok']}"
            )

            return float(score)

        except Exception as exc:
            print(
                f"[CoinSelector] {symbol} on {self.fetcher.exchange_name}: "
                f"error during scoring — {exc}"
            )
            return None

    def select(self, symbols: list[str]) -> list[str]:
        configured_symbols = symbols[:] if symbols else self.DEFAULT_SYMBOLS[:]

        supported = [s for s in configured_symbols if self.fetcher.is_symbol_supported(s)]
        unsupported = [s for s in configured_symbols if s not in supported]
        if unsupported:
            print(
                f"[CoinSelector] Unsupported on {self.fetcher.exchange_name}: {unsupported}"
            )

        eligible = [s for s in supported if _has_trained_model(s)]
        skipped = [s for s in supported if s not in eligible]
        if skipped:
            print(f"[CoinSelector] Skipped (no model): {skipped}")

        scores: dict[str, float] = {}

        for symbol in eligible:
            score = self._score_symbol(symbol)
            if score is not None:
                scores[symbol] = score

        ranked = sorted(scores, key=scores.get, reverse=True)
        filtered_ranked = [s for s in ranked if scores[s] > -100]

        if not filtered_ranked:
            print("⚠️ CoinSelector found no long-ready symbols")
            return []

        return filtered_ranked[: self.top_k]
