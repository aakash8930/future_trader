#execution/strategy.py

from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd

from models.direction import DirectionModel
from risk.sizing import fixed_fractional_size


@dataclass
class StrategyConfig:
    """Single source of truth for all strategy parameters."""

    # Core signal quality
    min_adx: float = 22.0
    min_atr_pct: float = 0.0011
    rsi_long_min: float = 45.0
    rsi_long_max: float = 64.0

    # Threshold handling
    base_long_threshold: float = 0.50

    # Risk / reward
    stop_atr_mult: float = 1.30
    take_atr_mult: float = 3.80

    # Trading costs
    fee_pct_per_side: float = 0.0010
    slippage_pct_per_side: float = 0.0008

    # Positive edge only
    min_expected_edge: float = 0.00020

    # Profit management
    trail_activate_atr_mult: float = 1.0
    trail_atr_mult: float = 1.0

    # Cooldown
    cooldown_minutes: int = 30

    # Pyramiding disabled
    max_pyramid_adds: int = 0
    pyramid_trigger_pct: float = 0.005
    pyramid_qty_scales: List[float] = field(
        default_factory=lambda: [0.6, 0.4, 0.25]
    )


@dataclass
class SignalDecision:
    side: Optional[str]
    prob: float
    threshold: float
    reason: str
    adx: float
    atr: float
    atr_pct: float
    regime: str
    ema_fast: float
    ema_slow: float
    price: float
    stop_loss: float
    take_profit: float
    expected_edge: float


class StrategyEngine:
    def __init__(
        self,
        model: DirectionModel,
        risk_per_trade: float = 0.01,
        config: Optional[StrategyConfig] = None,
    ):
        self.model = model
        self.risk_per_trade = risk_per_trade
        self.cfg = config or StrategyConfig()

    def generate_signal(self, df: pd.DataFrame, regime: str = "") -> SignalDecision:
        _null = SignalDecision(
            side=None,
            prob=0.0,
            threshold=0.0,
            reason="",
            adx=0.0,
            atr=0.0,
            atr_pct=0.0,
            regime=regime,
            ema_fast=0.0,
            ema_slow=0.0,
            price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            expected_edge=0.0,
        )

        if len(df) < 5:
            _null.reason = "insufficient_data"
            return _null

        row = df.iloc[-1]

        price = float(row["close"])
        ema200 = float(row["ema200"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        atr = float(row["atr"])
        adx = float(row["adx"])
        atr_pct = float(row["atr_pct"])
        rsi = float(row["rsi"])

        prob_up = float(self.model.predict_proba(df))

        model_th = float(
            getattr(self.model, "long_threshold", self.cfg.base_long_threshold)
        )
        long_th = max(model_th, self.cfg.base_long_threshold)

        # Slight relaxation only in stronger trends
        if adx >= 38:
            long_th -= 0.010
        elif adx >= 30:
            long_th -= 0.005

        long_th = max(0.49, min(long_th, 0.58))

        stop_loss = price - atr * self.cfg.stop_atr_mult
        take_profit = price + atr * self.cfg.take_atr_mult

        base = SignalDecision(
            side=None,
            prob=prob_up,
            threshold=long_th,
            reason="",
            adx=adx,
            atr=atr,
            atr_pct=atr_pct,
            regime=regime,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_edge=0.0,
        )

        if adx < self.cfg.min_adx:
            base.reason = f"adx_low({adx:.1f}<{self.cfg.min_adx})"
            return base

        if atr_pct < self.cfg.min_atr_pct:
            base.reason = f"atr_pct_low({atr_pct:.4f}<{self.cfg.min_atr_pct:.4f})"
            return base

        if not (self.cfg.rsi_long_min <= rsi <= self.cfg.rsi_long_max):
            base.reason = (
                f"rsi_out_of_range({rsi:.1f} not in "
                f"[{self.cfg.rsi_long_min:.1f},{self.cfg.rsi_long_max:.1f}])"
            )
            return base

        # Strong-trend-only block
        if adx < 22.0:
            base.reason = f"weak_trend_block({adx:.1f}<22.0)"
            return base

        above_ema200 = price > ema200
        bullish_cross = ema_fast > ema_slow
        ema_gap_pct = (price - ema200) / ema200 if ema200 > 0 else 0.0
        ema_fast_vs_slow_pct = (
            (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0.0
        )

        if above_ema200:
            if not bullish_cross:
                near_cross = ema_fast >= ema_slow * 0.999
                continuation_override = (
                    near_cross
                    and adx >= 26
                    and prob_up >= long_th + 0.010
                    and ema_gap_pct >= 0.0020
                )
                if not continuation_override:
                    base.reason = (
                        f"ema_cross_bearish(fast={ema_fast:.4f}<=slow={ema_slow:.4f}, "
                        f"price={price:.4f}, ema200={ema200:.4f})"
                    )
                    return base
        else:
            # Rare, high-quality recovery entries only
            momentum_override = (
                bullish_cross
                and adx >= 32
                and prob_up >= long_th + 0.025
                and rsi >= 50
                and ema_gap_pct >= -0.004
                and ema_fast_vs_slow_pct >= 0.0018
            )
            if not momentum_override:
                base.reason = (
                    f"price_below_ema200(price={price:.4f}<=ema200={ema200:.4f}, "
                    f"ema_fast={ema_fast:.4f}, ema_slow={ema_slow:.4f}, adx={adx:.1f})"
                )
                return base

        if prob_up < long_th:
            base.reason = (
                f"prob_low(prob={prob_up:.3f}<th={long_th:.3f}, "
                f"adx={adx:.1f}, atr_pct={atr_pct:.4f}, rsi={rsi:.1f})"
            )
            return base

        expected_edge = self._expected_edge(
            prob_up=prob_up,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        base.expected_edge = expected_edge

        if expected_edge < self.cfg.min_expected_edge:
            base.reason = (
                f"edge_low({expected_edge:.5f}<{self.cfg.min_expected_edge:.5f})"
            )
            return base

        trend_label = "above_ema200" if above_ema200 else "momentum_override_below_ema200"
        base.side = "LONG"
        base.reason = f"ok:{trend_label}"
        return base

    def _expected_edge(
        self,
        prob_up: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> float:
        if entry_price <= 0:
            return -1.0

        tp_return = max((take_profit - entry_price) / entry_price, 0.0)
        sl_return = max((entry_price - stop_loss) / entry_price, 0.0)

        round_trip_cost = 2.0 * (
            self.cfg.fee_pct_per_side + self.cfg.slippage_pct_per_side
        )

        return float(
            prob_up * tp_return
            - (1.0 - prob_up) * sl_return
            - round_trip_cost
        )

    def position_size(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
        max_position_notional_pct: float = 1.0,
    ) -> float:
        return fixed_fractional_size(
            balance=balance,
            risk_pct=self.risk_per_trade,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional_pct=max_position_notional_pct,
        )

    def score_symbol(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0

        prob_up = float(self.model.predict_proba(df))
        atr_pct = float(df.iloc[-1]["atr_pct"])
        adx = float(df.iloc[-1]["adx"])
        return float(prob_up * atr_pct * adx)
