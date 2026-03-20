# execution/ regime_controller.py

from enum import Enum
import pandas as pd


class MarketRegime(str, Enum):
    TREND_STRONG = "trend_strong"
    TREND_WEAK = "trend_weak"
    SIDEWAYS = "sideways"


class RegimeController:
    """
    Detect market regime and adjust risk/trading rules.
    Tuned to avoid over-blocking valid recovery / weak-trend conditions.
    """

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        last = df.iloc[-1]

        adx = float(last["adx"])
        atr_pct = float(last["atr_pct"])
        ema_fast = float(last["ema_fast"])
        ema_slow = float(last["ema_slow"])
        price = float(last["close"])
        ema200 = float(last["ema200"])

        bullish_cross = ema_fast > ema_slow
        near_ema200 = ema200 > 0 and abs(price - ema200) / ema200 <= 0.03

        if adx >= 26 and atr_pct >= 0.003:
            return MarketRegime.TREND_STRONG

        if adx >= 16 and atr_pct >= 0.0025:
            return MarketRegime.TREND_WEAK

        if bullish_cross and adx >= 14 and atr_pct >= 0.0025 and near_ema200:
            return MarketRegime.TREND_WEAK

        return MarketRegime.SIDEWAYS

    def risk_multiplier(self, regime: MarketRegime) -> float:
        if regime == MarketRegime.TREND_STRONG:
            return 1.15

        if regime == MarketRegime.TREND_WEAK:
            return 1.0

        return 0.75

    def trading_allowed(self, regime: MarketRegime) -> bool:
        return regime != MarketRegime.SIDEWAYS

    def skip_reason(self, df: pd.DataFrame) -> str:
        last = df.iloc[-1]
        adx = float(last["adx"])
        atr_pct = float(last["atr_pct"])
        ema_fast = float(last["ema_fast"])
        ema_slow = float(last["ema_slow"])
        price = float(last["close"])
        ema200 = float(last["ema200"])

        bullish_cross = ema_fast > ema_slow
        near_ema200 = ema200 > 0 and abs(price - ema200) / ema200 <= 0.03

        return (
            f"regime_sideways(adx={adx:.1f}, atr_pct={atr_pct:.4f}, "
            f"bullish_cross={bullish_cross}, near_ema200={near_ema200}, "
            f"price={price:.4f}, ema200={ema200:.4f})"
        )