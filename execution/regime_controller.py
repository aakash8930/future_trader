from enum import Enum
import pandas as pd


class MarketRegime(str, Enum):
    TREND_STRONG = "trend_strong"
    TREND_WEAK = "trend_weak"
    SIDEWAYS = "sideways"


class RegimeController:
    """
    Detect market regime and adjust risk/trading rules.
    """

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        adx = float(df.iloc[-1]["adx"])
        atr_pct = float(df.iloc[-1]["atr_pct"])

        if adx >= 28 and atr_pct >= 0.0030:
            return MarketRegime.TREND_STRONG

        if adx >= 16 and atr_pct >= 0.0020:
            return MarketRegime.TREND_WEAK

        return MarketRegime.SIDEWAYS

    def risk_multiplier(self, regime: MarketRegime) -> float:
        if regime == MarketRegime.TREND_STRONG:
            return 1.15
        if regime == MarketRegime.TREND_WEAK:
            return 1.0
        return 0.0

    def trading_allowed(self, regime: MarketRegime) -> bool:
        return regime != MarketRegime.SIDEWAYS

    def skip_reason(self, df: pd.DataFrame) -> str:
        adx = float(df.iloc[-1]["adx"])
        atr_pct = float(df.iloc[-1]["atr_pct"])
        return f"regime_sideways(adx={adx:.1f}, atr_pct={atr_pct:.4f})"