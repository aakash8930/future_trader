from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd

from models.direction import DirectionModel
from risk.sizing import fixed_fractional_size


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Single source of truth for all strategy parameters."""
    # Signal filters
    min_prob:              float = 0.47
    min_adx:               float = 10.0
    min_atr_pct:           float = 0.0008
    rsi_long_min:          float = 35.0
    rsi_long_max:          float = 72.0

    # Threshold adjustment (model base threshold is used as starting point)
    base_long_threshold:   float = 0.48

    # ATR-based stop / take-profit
    stop_atr_mult:         float = 1.7
    take_atr_mult:         float = 3.0

    # Execution cost / edge filter
    fee_pct_per_side:      float = 0.0010
    slippage_pct_per_side: float = 0.0008
    min_expected_edge:     float = -0.00020

    # Trailing stop
    trail_activate_atr_mult: float = 1.0
    trail_atr_mult:          float = 1.0

    # Cooldown
    cooldown_minutes:      int   = 30

    # Pyramiding
    max_pyramid_adds:      int   = 0
    pyramid_trigger_pct:   float = 0.005
    pyramid_qty_scales:    List[float] = field(default_factory=lambda: [0.6, 0.4, 0.25])


# ---------------------------------------------------------------------------
# Signal Decision
# ---------------------------------------------------------------------------

@dataclass
class SignalDecision:
    """Full signal output returned by StrategyEngine.generate_signal()."""
    side:          Optional[str]
    prob:          float
    threshold:     float
    reason:        str
    adx:           float
    atr:           float
    atr_pct:       float
    regime:        str
    ema_fast:      float
    ema_slow:      float
    price:         float
    stop_loss:     float
    take_profit:   float
    expected_edge: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

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

        model_th = float(getattr(self.model, "long_threshold", self.cfg.base_long_threshold))
        long_th = min(model_th, self.cfg.base_long_threshold)

        # Adaptive easing in stronger trend
        if adx >= 30:
            long_th -= 0.02
        elif adx >= 20:
            long_th -= 0.01

        long_th = max(self.cfg.min_prob, min(long_th, 0.62))

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
            base.reason = f"atr_pct_low({atr_pct:.4f}<{self.cfg.min_atr_pct})"
            return base

        if not (self.cfg.rsi_long_min <= rsi <= self.cfg.rsi_long_max):
            base.reason = (
                f"rsi_out_of_range({rsi:.1f} not in "
                f"[{self.cfg.rsi_long_min:.1f},{self.cfg.rsi_long_max:.1f}])"
            )
            return base

        above_ema200 = price > ema200
        bullish_cross = ema_fast > ema_slow

        # Allow a controlled override when momentum is improving even below EMA200
        momentum_override = (
            bullish_cross
            and adx >= 18
            and prob_up >= max(self.cfg.min_prob, long_th - 0.01)
            and rsi >= max(self.cfg.rsi_long_min, 38.0)
        )

        if not above_ema200 and not momentum_override:
            base.reason = (
                f"price_below_ema200(price={price:.4f}<=ema200={ema200:.4f}, "
                f"ema_fast={ema_fast:.4f}, ema_slow={ema_slow:.4f}, adx={adx:.1f})"
            )
            return base

        if above_ema200 and not bullish_cross:
            # allow strong-trend continuation if fast/slow are very close and model is confident
            weak_bearish_cross = ema_fast > (ema_slow * 0.9985)
            continuation_override = (
                adx >= 22
                and prob_up >= long_th + 0.01
                and weak_bearish_cross
            )
            if not continuation_override:
                base.reason = (
                    f"ema_cross_bearish(fast={ema_fast:.4f}<=slow={ema_slow:.4f}, "
                    f"price={price:.4f}, ema200={ema200:.4f})"
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

        prob_up = self.model.predict_proba(df)
        atr_pct = df.iloc[-1]["atr_pct"]
        adx = df.iloc[-1]["adx"]
        return float(prob_up * atr_pct * adx)