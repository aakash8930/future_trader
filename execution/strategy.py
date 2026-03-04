# execution/strategy.py

import pandas as pd

from models.direction import DirectionModel
from risk.sizing import fixed_fractional_size


class StrategyEngine:
    """
    StrategyEngine responsibilities:
    - Indicator interpretation
    - Entry signal generation
    - Position sizing

    ❌ Does NOT manage exits or trailing stops
    """

    def __init__(
        self,
        model: DirectionModel,
        risk_per_trade: float = 0.01,
        min_adx: float = 25.0,  # Strong trend filter
    ):
        self.model = model
        self.risk_per_trade = risk_per_trade
        self.min_adx = min_adx

        # Logging / analytics
        self.last_entry_price = None
        self.last_entry_prob = None

    # ==================================================
    # SIGNAL GENERATION
    # ==================================================

    def generate_signal(self, df: pd.DataFrame):
        if len(df) < 5:
            return None, 0.0

        price = df.iloc[-1]["close"]
        ema200 = df.iloc[-1]["ema200"]
        atr = df.iloc[-1]["atr"]
        adx = df.iloc[-1]["adx"]

        prob_up = self.model.predict_proba(df)

        # -----------------------------
        # OOD protection
        # -----------------------------
        if prob_up < 0.05 or prob_up > 0.95:
            return None, prob_up

        atr_pct = atr / price

        # -----------------------------
        # HARD TREND FILTER
        # -----------------------------
        if adx < self.min_adx:
            print(
                f"DEBUG | SKIP (sideways) "
                f"adx={adx:.1f}"
            )
            return None, prob_up

        # -----------------------------
        # Dynamic threshold
        # -----------------------------
        long_th = 0.55

        if adx >= 40:
            long_th -= 0.02

        if price > ema200:
            long_th -= 0.01

        long_th = max(0.52, min(long_th, 0.60))

        # -----------------------------
        # LONG ENTRY
        # -----------------------------
        if (
            prob_up >= long_th
            and atr_pct > 0.0012
        ):
            self.last_entry_price = price
            self.last_entry_prob = prob_up

            return "LONG", prob_up

        # -----------------------------
        # Debug logging
        # -----------------------------
        print(
            f"DEBUG | prob={prob_up:.3f} | "
            f"f1={self.model.f1:.2f} | "
            f"adx={adx:.1f} | "
            f"atr_pct={atr_pct:.4f} | "
            f"long_th={long_th:.3f}"
        )

        return None, prob_up

    # ==================================================
    # POSITION SIZING
    # ==================================================

    def position_size(
        self,
        balance: float,
        entry_price: float,
        side: str,
        max_position_notional_pct: float = 1.0,
    ) -> float:

        stop_price = entry_price * (0.99 if side == "LONG" else 1.01)

        return fixed_fractional_size(
            balance=balance,
            risk_pct=self.risk_per_trade,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional_pct=max_position_notional_pct,
        )

    # ==================================================
    # SYMBOL SCORING
    # ==================================================

    def score_symbol(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0

        prob_up = self.model.predict_proba(df)

        atr_pct = df.iloc[-1]["atr"] / df.iloc[-1]["close"]
        adx = df.iloc[-1]["adx"]

        return float(prob_up * atr_pct * adx)