# execution/strategy.py

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
    min_prob:              float = 0.58
    min_adx:               float = 15.0
    min_atr_pct:           float = 0.001
    rsi_long_min:          float = 48.0
    rsi_long_max:          float = 72.0

    # Threshold adjustment (model base threshold is used as starting point)
    base_long_threshold:   float = 0.58   # overridden by model.long_threshold at runtime

    # ATR-based stop / take-profit
    stop_atr_mult:         float = 2.0
    take_atr_mult:         float = 3.0

    # Execution cost / edge filter (round-trip estimate)
    fee_pct_per_side:      float = 0.0010
    slippage_pct_per_side: float = 0.0008
    min_expected_edge:     float = 0.0002

    # Trailing stop
    trail_activate_atr_mult: float = 1.0   # activate trail when move >= 1 ATR
    trail_atr_mult:          float = 1.0   # trail by 1 ATR below peak

    # Cooldown
    cooldown_minutes:      int   = 30

    # Pyramiding (max_pyramid_adds=0 means disabled)
    max_pyramid_adds:      int   = 1
    pyramid_trigger_pct:   float = 0.005
    pyramid_qty_scales:    List[float] = field(default_factory=lambda: [0.6, 0.4, 0.25])


# ---------------------------------------------------------------------------
# Signal Decision
# ---------------------------------------------------------------------------

@dataclass
class SignalDecision:
    """Full signal output returned by StrategyEngine.generate_signal()."""
    side:       Optional[str]  # "LONG" or None
    prob:       float
    threshold:  float
    reason:     str            # human-readable skip reason when side=None
    adx:        float
    atr:        float
    atr_pct:    float
    regime:     str
    ema_fast:   float
    ema_slow:   float
    price:      float
    stop_loss:  float
    take_profit: float
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

    # ------------------------------------------------------------------
    def generate_signal(self, df: pd.DataFrame, regime: str = "") -> SignalDecision:
        """
        Evaluate entry conditions and return a SignalDecision.
        All threshold logic lives here — runner must NOT apply a second filter.
        """
        _null = SignalDecision(
            side=None, prob=0.0, threshold=0.0, reason="",
            adx=0.0, atr=0.0, atr_pct=0.0, regime=regime,
            ema_fast=0.0, ema_slow=0.0, price=0.0,
            stop_loss=0.0, take_profit=0.0, expected_edge=0.0,
        )

        if len(df) < 5:
            _null.reason = "insufficient_data"
            return _null

        row      = df.iloc[-1]
        price    = float(row["close"])
        ema200   = float(row["ema200"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        atr      = float(row["atr"])
        adx      = float(row["adx"])
        atr_pct  = float(row["atr_pct"])
        rsi      = float(row["rsi"])

        prob_up = self.model.predict_proba(df)

        # Use model's optimised threshold as the base; fall back to config.
        long_th = getattr(self.model, "long_threshold", self.cfg.base_long_threshold)

        # Slight regime adjustments (keep modest — avoid over-fitting).
        if adx >= 35:
            long_th -= 0.02
        long_th = max(self.cfg.min_prob, min(long_th, 0.65))

        stop_loss  = price - atr * self.cfg.stop_atr_mult
        take_profit = price + atr * self.cfg.take_atr_mult

        base = SignalDecision(
            side=None, prob=prob_up, threshold=long_th, reason="",
            adx=adx, atr=atr, atr_pct=atr_pct, regime=regime,
            ema_fast=ema_fast, ema_slow=ema_slow, price=price,
            stop_loss=stop_loss, take_profit=take_profit, expected_edge=0.0,
        )

        # ---- Filters (order: fastest to cheapest to eliminate) ----
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

        # --- Trend condition: two valid paths for a long entry ---
        # Path 1 (classic):   price is above the 200-period EMA
        # Path 2 (override):  strong short-term momentum even if price is still
        #                     slightly below ema200 — requires faster EMA cross,
        #                     meaningful trend strength, and higher model confidence.
        above_ema200 = price > ema200
        momentum_override = (
            ema_fast > ema_slow
            and adx >= 25
            and prob_up >= long_th + 0.02
        )

        if not above_ema200 and not momentum_override:
            base.reason = f"price_below_ema200({price:.4f}<={ema200:.4f})"
            return base

        # EMA cross is still required on the classic path;
        # momentum_override already enforces ema_fast > ema_slow.
        if above_ema200 and ema_fast <= ema_slow:
            base.reason = f"ema_cross_bearish(fast={ema_fast:.4f}<=slow={ema_slow:.4f})"
            return base

        if prob_up < long_th:
            base.reason = f"prob_low({prob_up:.3f}<{long_th:.3f})"
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
        round_trip_cost = 2.0 * (self.cfg.fee_pct_per_side + self.cfg.slippage_pct_per_side)

        return float(prob_up * tp_return - (1.0 - prob_up) * sl_return - round_trip_cost)

    # ------------------------------------------------------------------
    def position_size(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
        max_position_notional_pct: float = 1.0,
    ) -> float:
        """
        Fixed-fractional sizing using the actual ATR-based stop distance.
        stop_price must come from the SignalDecision, not a hardcoded 1%.
        """
        return fixed_fractional_size(
            balance=balance,
            risk_pct=self.risk_per_trade,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional_pct=max_position_notional_pct,
        )

    # ------------------------------------------------------------------
    def score_symbol(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        prob_up  = self.model.predict_proba(df)
        atr_pct  = df.iloc[-1]["atr_pct"]
        adx      = df.iloc[-1]["adx"]
        return float(prob_up * atr_pct * adx)
