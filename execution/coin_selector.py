import os
import numpy as np

from data.fetcher import MarketDataFetcher
from features.technicals import compute_core_features


def _has_trained_model(symbol: str) -> bool:
    folder = symbol.replace("/", "_")
    # Check legacy MLP path and any model subdirectories
    for subfolder in [f"models/{folder}", f"models/{folder}_lstm", f"models/{folder}_xgb"]:
        if os.path.exists(os.path.join(subfolder, "metadata.json")):
            if (os.path.exists(os.path.join(subfolder, "model.pt")) or
                os.path.exists(os.path.join(subfolder, "model.save"))):
                return True
    return False


class CoinSelector:
    """
    Scores symbols for live trading.

    Design goals:
    - use only closed candles
    - avoid selecting symbols the runner will almost certainly reject
    - prefer symbols already aligned with strategy rules
    - allow strong-trend overrides so the system does not become too idle
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
        rsi_long_min: float = 40.0,
        rsi_long_max: float = 76.0,
        exchange_name: str = "binance",
        exchange_fallbacks: list[str] | None = None,
        exchange_timeout_ms: int = 20000,
        demo_mode: bool = False,
    ):
        self.timeframe = timeframe
        self.lookback = lookback
        self.top_k = top_k
        self.min_atr_pct = min_atr_pct
        self.soft_min_volume_ratio = soft_min_volume_ratio
        self.demo_mode = demo_mode
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

        try:
            model = DirectionModel.for_symbol(symbol)
            self._model_cache[symbol] = model
            return model
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # Windows cp1252 encoding issue with metadata files
            print(f"[CoinSelector] {symbol} | encoding_error: {e}")
            return None
        except Exception as e:
            print(f"[CoinSelector] {symbol} | model_load_error: {e}")
            return None

    def _model_probability_and_threshold(self, symbol: str, df) -> tuple[float, float]:
        try:
            model = self._get_model(symbol)
            if model is None:
                return 0.5, 0.50
            prob = float(model.predict_proba(df))
            long_th = float(getattr(model, "long_threshold", 0.50))

            if not (0.0 <= prob <= 1.0):
                prob = 0.5

            return prob, long_th
        except Exception:
            return 0.5, 0.50

    def _score_symbol(self, symbol: str) -> float | None:
        try:
            raw_fetch_limit = max(self.lookback + 220, 420)
            df = self.fetcher.fetch_ohlcv(symbol, self.timeframe, limit=raw_fetch_limit)
            if df is None or df.empty or len(df) < self.lookback:
                print(f"[CoinSelector] {symbol} | insufficient_data")
                return None

            # Use only historical closed candles for selection.
            df = df.iloc[:-1].copy()
            if len(df) < self.lookback:
                print(f"[CoinSelector] {symbol} | insufficient_closed_candles")
                return None

            df = compute_core_features(df)
            df = df.dropna().copy()
            if df.empty:
                print(f"[CoinSelector] {symbol} | no_features_after_dropna")
                return None

            row = df.iloc[-1]

            price = float(row["close"])
            ema200 = float(row["ema200"])
            ema_fast = float(row["ema_fast"])
            ema_slow = float(row["ema_slow"])
            adx = float(row["adx"])
            atr_pct = float(row["atr_pct"])
            rsi = float(row["rsi"])
            dist_ema200 = float(row["dist_ema200"])

            if not np.isfinite(price) or price <= 0:
                print(f"[CoinSelector] {symbol} | invalid_price")
                return None

            if not np.isfinite(ema200) or ema200 <= 0:
                print(f"[CoinSelector] {symbol} | invalid_ema200")
                return None

            if not np.isfinite(adx):
                adx = 0.0
            if not np.isfinite(atr_pct):
                atr_pct = 0.0
            if not np.isfinite(rsi):
                rsi = 50.0
            if not np.isfinite(dist_ema200):
                dist_ema200 = 0.0

            vol_ratio = 1.0
            if "volume" in df.columns:
                vol_ma = float(df["volume"].rolling(20).mean().iloc[-1])
                cur_vol = float(row["volume"])
                if np.isfinite(vol_ma) and vol_ma > 0 and np.isfinite(cur_vol):
                    vol_ratio = cur_vol / vol_ma

            bullish_cross = ema_fast > ema_slow
            above_ema200 = price > ema200

            prob_up, model_long_th = self._model_probability_and_threshold(symbol, df)

            # Slightly more permissive than before.
            # Previous logs show many good trend symbols dying on RSI and strict thresholds.
            adaptive_long_th = max(0.48, model_long_th - 0.01)

            strong_trend = (
                above_ema200
                and bullish_cross
                and adx >= 26.0
                and atr_pct >= self.min_atr_pct
            )

            momentum_override = (
                bullish_cross
                and adx >= 30.0
                and atr_pct >= self.min_atr_pct
                and prob_up >= adaptive_long_th + 0.01
                and dist_ema200 > -0.020
            )

            reasons: list[str] = []

            # Hard rejects only for clearly bad cases.
            if atr_pct < self.min_atr_pct:
                print(f"[ATR DEBUG] atr_pct={atr_pct:.6f} < min_atr_pct={self.min_atr_pct:.6f} -> blocked")
                reasons.append(f"atr_pct_low({atr_pct:.4f}<{self.min_atr_pct:.4f})")

            # Demo mode: skip volume and far_below_ema200 blocks
            if not self.demo_mode:
                if vol_ratio < self.soft_min_volume_ratio * 0.60:
                    reasons.append("volume_too_low")

                if not above_ema200 and dist_ema200 <= -0.035 and not momentum_override:
                    reasons.append("far_below_ema200")

            # Soft scoring instead of hard RSI/prob blocks.
            # RSI: lower is better in the 30-60 recovery zone; penalize extremes.
            if rsi < 30.0:
                rsi_score = -0.08
                reasons.append("rsi_oversold")
            elif rsi < self.rsi_long_min:
                rsi_score = -0.04
            elif rsi > 78.0:
                rsi_score = -0.06
            else:
                rsi_score = 0.0

            # Probability: score contribution, no hard block
            if prob_up < adaptive_long_th - 0.035:
                reasons.append("prob_below_thresh")

            if reasons:
                # Only hard-reject on atr/volume/far_ema; everything else gets a score
                if "atr_too_low" in reasons or "volume_too_low" in reasons or "far_below_ema200" in reasons:
                    print(
                        f"[CoinSelector] {symbol} | "
                        f"score=-999.000 "
                        f"prob={prob_up:.3f}/{adaptive_long_th:.3f} "
                        f"adx={adx:.1f} atr_pct={atr_pct:.4f} "
                        f"rsi={rsi:.1f} vol_ratio={vol_ratio:.2f} "
                        f"above_ema200={above_ema200} bullish_cross={bullish_cross} "
                        f"reasons={reasons}"
                    )
                    return -999.0

            # Soft score:
            # probability matters most, then regime alignment, then momentum/liquidity.
            score = 0.0
            score += prob_up * 0.95
            score += min(adx, 50.0) / 100.0
            score += min(atr_pct, 0.0100) * 18.0
            score += min(max(vol_ratio, 0.0), 3.0) * 0.04
            # RSI soft penalty
            score += rsi_score
            # Extra credit for bullish RSI zone (not overbought, not oversold)
            if 40.0 <= rsi <= 65.0:
                score += 0.05

            if above_ema200:
                score += 0.12
            else:
                score -= 0.06

            if bullish_cross:
                score += 0.08

            if strong_trend:
                score += 0.06

            if momentum_override and not above_ema200:
                score += 0.03

            # Mild penalty for extreme RSI, but no automatic rejection.
            if rsi > 78.0:
                score -= min((rsi - 78.0) * 0.005, 0.06)

            score = float(max(score, 0.0))

            reason = "ok"
            if momentum_override and not above_ema200:
                reason = "ok_momentum_override"
            elif strong_trend:
                reason = "ok_strong_trend"

            print(
                f"[CoinSelector] {symbol} | "
                f"score={score:.3f} "
                f"prob={prob_up:.3f}/{adaptive_long_th:.3f} "
                f"adx={adx:.1f} atr_pct={atr_pct:.4f} "
                f"rsi={rsi:.1f} vol_ratio={vol_ratio:.2f} "
                f"above_ema200={above_ema200} bullish_cross={bullish_cross} "
                f"reason=['{reason}']"
            )
            return score

        except Exception as exc:
            print(
                f"[CoinSelector] {symbol} | "
                f"error on {self.fetcher.exchange_name}: {exc}"
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
            print("[WARN] CoinSelector found no long-ready symbols")
            return []

        return filtered_ranked[: self.top_k]