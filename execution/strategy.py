# execution/strategy.py

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from models.direction import DirectionModel
from risk.sizing import fixed_fractional_size


@dataclass
class StrategyConfig:

    min_prob = 0.48
    min_adx = 14
    min_atr_pct = 0.0011

    rsi_min = 40
    rsi_max = 72

    stop_atr = 1.6
    take_atr = 3.0

    min_edge = 0.00015


@dataclass
class SignalDecision:

    side: Optional[str]
    reason: str
    price: float
    stop: float
    take: float


class StrategyEngine:

    def __init__(self, model):

        self.model = model
        self.cfg = StrategyConfig()

    # -----------------------------

    def generate_signal(self, df):

        row = df.iloc[-1]

        price = row["close"]
        ema200 = row["ema200"]
        ema_fast = row["ema_fast"]
        ema_slow = row["ema_slow"]

        adx = row["adx"]
        atr = row["atr"]
        atr_pct = row["atr_pct"]
        rsi = row["rsi"]

        prob = float(self.model.predict_proba(df))

        # ---------- basic filters ----------

        if adx < self.cfg.min_adx:
            return SignalDecision(None, "adx", 0, 0, 0)

        if atr_pct < self.cfg.min_atr_pct:
            return SignalDecision(None, "atr", 0, 0, 0)

        if not (self.cfg.rsi_min <= rsi <= self.cfg.rsi_max):
            return SignalDecision(None, "rsi", 0, 0, 0)

        above = price > ema200
        cross = ema_fast > ema_slow

        gap = (price - ema200) / ema200

        # ---------- structure ----------

        if above:

            if not cross and gap < 0.01:
                return SignalDecision(None, "bear_cross", 0, 0, 0)

        else:

            # allow recovery trades

            if not (cross and gap > -0.01 and adx > 20):
                return SignalDecision(None, "below200", 0, 0, 0)

        # ---------- probability ----------

        if prob < self.cfg.min_prob:
            return SignalDecision(None, "prob", 0, 0, 0)

        stop = price - atr * self.cfg.stop_atr
        take = price + atr * self.cfg.take_atr

        edge = prob * (take - price) - (1 - prob) * (price - stop)

        if edge <= self.cfg.min_edge:
            return SignalDecision(None, "edge", 0, 0, 0)

        return SignalDecision(
            "LONG",
            "ok",
            price,
            stop,
            take,
        )

    # -----------------------------

    def position_size(
        self,
        balance,
        entry,
        stop,
    ):

        return fixed_fractional_size(
            balance,
            0.01,
            entry,
            stop,
        )