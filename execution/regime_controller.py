# execution/regime_controller.py

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

        if adx >= 30 and atr_pct >= 0.003:
            return MarketRegime.TREND_STRONG

        if adx >= 15:
            return MarketRegime.TREND_WEAK

        return MarketRegime.SIDEWAYS

    def risk_multiplier(self, regime: MarketRegime) -> float:

        if regime == MarketRegime.TREND_STRONG:
            return 1.2

        if regime == MarketRegime.TREND_WEAK:
            return 1.0

        return 0.6

    def trading_allowed(self, regime: MarketRegime) -> bool:

        if regime == MarketRegime.SIDEWAYS:
            print("DEBUG | SKIP sideways")
            return False

        return True