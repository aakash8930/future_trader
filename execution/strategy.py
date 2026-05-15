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
    min_adx: float = 12.0
    min_atr_pct: float = 0.0003
    rsi_long_min: float = 34.0
    rsi_long_max: float = 78.0

    # Threshold handling
    base_long_threshold: float = 0.48

    # Risk / reward
    stop_atr_mult: float = 1.30
    take_atr_mult: float = 3.80

    # Trading costs
    fee_pct_per_side: float = 0.0010
    slippage_pct_per_side: float = 0.0008

    # Positive edge only
    min_expected_edge: float = -0.0025

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

    # Demo mode relaxations
    demo_mode: bool = False


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
        dist_ema200 = float(row["dist_ema200"])

        prob_up = float(self.model.predict_proba(df))
        prob_down = 1.0 - prob_up

        # User's strategy_min_prob from shared state sets the floor for threshold.
        # Model's long_threshold (from ensemble) is only used as a ceiling
        # (never raises above what the user configured).
        model_th = float(
            getattr(self.model, "long_threshold", self.cfg.base_long_threshold)
        )
        user_th = max(self.cfg.base_long_threshold, 0.40)
        long_th = min(user_th, model_th)

        # SHORT threshold is inverted (use 1 - long_th)
        short_th = 1.0 - long_th

        # Slight relaxation only in stronger trends
        if adx >= 38:
            long_th -= 0.010
            short_th += 0.010
        elif adx >= 32:
            long_th -= 0.008
            short_th += 0.008
        elif adx >= 28:
            long_th -= 0.005
            short_th += 0.005
        elif adx >= 24:
            long_th -= 0.003
            short_th += 0.003

        long_th = max(0.44, min(long_th, 0.58))
        short_th = max(0.42, min(short_th, 0.56))

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

        # ─── ADX and ATR filters (apply to both LONG and SHORT) ───────────────────
        
        if adx < self.cfg.min_adx:
            base.reason = f"adx_low({adx:.1f}<{self.cfg.min_adx})"
            return base

        if atr_pct < self.cfg.min_atr_pct:
            print(f"[ATR DEBUG] atr_pct={atr_pct:.6f} < min_atr_pct={self.cfg.min_atr_pct:.6f} -> blocked")
            base.reason = f"atr_pct_low({atr_pct:.4f}<{self.cfg.min_atr_pct:.4f})"
            return base

        # RSI range check (allow wider range)
        rsi_lower = 20.0 if self.cfg.demo_mode else max(30.0, self.cfg.rsi_long_min)
        if not (rsi_lower <= rsi <= 82.0):
            base.reason = (
                f"rsi_out_of_range({rsi:.1f} not in [{rsi_lower:.1f}, 82.0])"
            )
            return base

        # Lower trend threshold — allow entries in developing trends
        if adx < 15.0 and not self.cfg.demo_mode:
            base.reason = f"weak_trend_block({adx:.1f}<15.0)"
            return base

        above_ema200 = price > ema200
        bullish_cross = ema_fast > ema_slow
        bearish_cross = ema_fast < ema_slow
        ema_gap_pct = (price - ema200) / ema200 if ema200 > 0 else 0.0
        ema_fast_vs_slow_pct = (
            (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0.0
        )

        # ─── LONG Signal Logic ──────────────────────────────────────────────────────
        
        # Try LONG first
        long_signal = self._evaluate_long_signal(
            base, price, ema200, ema_fast, ema_slow, prob_up, long_th,
            above_ema200, bullish_cross, ema_gap_pct, ema_fast_vs_slow_pct, rsi, adx
        )
        
        if long_signal.side == "LONG":
            return long_signal

        # ─── SHORT Signal Logic ─────────────────────────────────────────────────────
        
        # Try SHORT if LONG failed (demo mode: try SHORT more aggressively)
        if not self.cfg.demo_mode or prob_down >= 0.35:
            short_signal = self._evaluate_short_signal(
                base, price, ema200, ema_fast, ema_slow, prob_down, short_th,
                above_ema200, bearish_cross, ema_gap_pct, ema_fast_vs_slow_pct, rsi, adx
            )
            
            if short_signal.side == "SHORT":
                return short_signal

        # No valid signal
        return base

    def _evaluate_long_signal(
        self,
        base: SignalDecision,
        price: float,
        ema200: float,
        ema_fast: float,
        ema_slow: float,
        prob_up: float,
        long_th: float,
        above_ema200: bool,
        bullish_cross: bool,
        ema_gap_pct: float,
        ema_fast_vs_slow_pct: float,
        rsi: float,
        adx: float,
    ) -> SignalDecision:
        """Evaluate if conditions are good for a LONG entry."""
        stop_loss = price - base.atr * self.cfg.stop_atr_mult
        take_profit = price + base.atr * self.cfg.take_atr_mult

        if above_ema200:
            if not bullish_cross:
                # Soft check: allow above-EMA200 continuation with decent trend
                near_cross = ema_fast >= ema_slow * 0.998
                # Demo mode: lower thresholds for cross and prob
                demo_relax = self.cfg.demo_mode
                continuation_override = (
                    near_cross
                    and adx >= (20 if demo_relax else 22)
                    and prob_up >= (long_th - 0.01 if demo_relax else long_th + 0.005)
                    and ema_gap_pct >= 0.0005
                )
                if not continuation_override:
                    base.reason = (
                        f"ema_cross_bearish(fast={ema_fast:.4f}<=slow={ema_slow:.4f}, "
                        f"price={price:.4f}, ema200={ema200:.4f})"
                    )
                    return base
        else:
            # High-quality recovery entries only
            # Demo mode: lower ADX and prob requirements
            demo_relax = self.cfg.demo_mode
            momentum_override = (
                bullish_cross
                and adx >= (22 if demo_relax else 28)
                and prob_up >= (long_th - 0.02 if demo_relax else long_th + 0.025)
                and rsi >= 30
                and ema_gap_pct >= -0.006
                and ema_fast_vs_slow_pct >= 0.0010
            )
            if not momentum_override:
                # Only hard block if FAR below EMA200
                if ema_gap_pct < -0.030:
                    base.reason = (
                        f"price_below_ema200(price={price:.4f}<=ema200={ema200:.4f}, "
                        f"dist={ema_gap_pct:.4f})"
                    )
                    return base

        # Demo mode: allow trades with prob >= 0.35 (force execution)
        # Relax prob threshold by 0.05 for demo (allow prob >= long_th - 0.05)
        if self.cfg.demo_mode:
            if prob_up < 0.35:
                base.reason = (
                    f"prob_low(prob={prob_up:.3f}<0.35, "
                    f"adx={adx:.1f}, atr_pct={base.atr_pct:.4f}, rsi={rsi:.1f})"
                )
                return base
        else:
            if prob_up < long_th:
                base.reason = (
                    f"prob_low(prob={prob_up:.3f}<long_th={long_th:.3f}, "
                    f"adx={adx:.1f}, atr_pct={base.atr_pct:.4f}, rsi={rsi:.1f})"
                )
                return base

        expected_edge = self._expected_edge(
            prob_up=prob_up,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # Demo mode: allow small negative edge (>= -0.005), skip edge check otherwise
        if self.cfg.demo_mode:
            if expected_edge < -0.005:
                base.reason = (
                    f"edge_low({expected_edge:.5f}<-0.003)"
                )
                return base
        else:
            if expected_edge < self.cfg.min_expected_edge:
                base.reason = (
                    f"edge_low({expected_edge:.5f}<{self.cfg.min_expected_edge:.5f})"
                )
                return base

        trend_label = "above_ema200" if above_ema200 else "momentum_override_below_ema200"
        base.side = "LONG"
        base.reason = f"ok:{trend_label}"
        base.stop_loss = stop_loss
        base.take_profit = take_profit
        base.expected_edge = expected_edge
        return base

    def _evaluate_short_signal(
        self,
        base: SignalDecision,
        price: float,
        ema200: float,
        ema_fast: float,
        ema_slow: float,
        prob_down: float,
        short_th: float,
        above_ema200: bool,
        bearish_cross: bool,
        ema_gap_pct: float,
        ema_fast_vs_slow_pct: float,
        rsi: float,
        adx: float,
    ) -> SignalDecision:
        """Evaluate if conditions are good for a SHORT entry."""
        stop_loss = price + base.atr * self.cfg.stop_atr_mult  # SHORT: SL is above entry
        take_profit = price - base.atr * self.cfg.take_atr_mult  # SHORT: TP is below entry

        # SHORT enters below EMA200 with bearish signal
        if not above_ema200:
            if not bearish_cross:
                # Soft check: allow below-EMA200 continuation with decent trend
                near_cross = ema_fast <= ema_slow * 1.002
                demo_relax = self.cfg.demo_mode
                continuation_override = (
                    near_cross
                    and adx >= (20 if demo_relax else 22)
                    and prob_down >= (short_th - 0.01 if demo_relax else short_th + 0.005)
                    and ema_gap_pct <= -0.0005
                )
                if not continuation_override:
                    base.reason = (
                        f"ema_cross_bullish(fast={ema_fast:.4f}>=slow={ema_slow:.4f}, "
                        f"price={price:.4f}, ema200={ema200:.4f})"
                    )
                    return base
        else:
            # High-quality recovery entries only (above EMA200 but going down)
            demo_relax = self.cfg.demo_mode
            momentum_override = (
                bearish_cross
                and adx >= (22 if demo_relax else 28)
                and prob_down >= (short_th - 0.02 if demo_relax else short_th + 0.025)
                and rsi <= 70
                and ema_gap_pct <= 0.006
                and ema_fast_vs_slow_pct <= -0.0010
            )
            if not momentum_override:
                # Only hard block if FAR above EMA200
                if ema_gap_pct > 0.030:
                    base.reason = (
                        f"price_above_ema200(price={price:.4f}>=ema200={ema200:.4f}, "
                        f"dist={ema_gap_pct:.4f})"
                    )
                    return base

        # Probability check for SHORT
        if self.cfg.demo_mode:
            if prob_down < 0.35:
                base.reason = (
                    f"prob_down_low(prob={prob_down:.3f}<0.35, "
                    f"adx={adx:.1f}, atr_pct={base.atr_pct:.4f}, rsi={rsi:.1f})"
                )
                return base
        else:
            if prob_down < short_th:
                base.reason = (
                    f"prob_down_low(prob={prob_down:.3f}<short_th={short_th:.3f}, "
                    f"adx={adx:.1f}, atr_pct={base.atr_pct:.4f}, rsi={rsi:.1f})"
                )
                return base

        expected_edge = self._expected_edge_short(
            prob_down=prob_down,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # Demo mode: allow small negative edge, skip edge check otherwise
        if self.cfg.demo_mode:
            if expected_edge < -0.005:
                base.reason = (
                    f"edge_low({expected_edge:.5f}<-0.003)"
                )
                return base
        else:
            if expected_edge < self.cfg.min_expected_edge:
                base.reason = (
                    f"edge_low({expected_edge:.5f}<{self.cfg.min_expected_edge:.5f})"
                )
                return base

        trend_label = "below_ema200" if not above_ema200 else "momentum_override_above_ema200"
        base.side = "SHORT"
        base.prob = prob_down  # Update prob to prob_down for SHORT
        base.threshold = short_th
        base.reason = f"ok:{trend_label}"
        base.stop_loss = stop_loss
        base.take_profit = take_profit
        base.expected_edge = expected_edge
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

    def _expected_edge_short(
        self,
        prob_down: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> float:
        """Calculate expected edge for SHORT trades."""
        if entry_price <= 0:
            return -1.0

        # For shorts: TP is below entry, SL is above entry
        tp_return = max((entry_price - take_profit) / entry_price, 0.0)
        sl_return = max((stop_loss - entry_price) / entry_price, 0.0)

        round_trip_cost = 2.0 * (
            self.cfg.fee_pct_per_side + self.cfg.slippage_pct_per_side
        )

        return float(
            prob_down * tp_return
            - (1.0 - prob_down) * sl_return
            - round_trip_cost
        )

    def position_size(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
        max_position_notional_pct: float = 1.0,
        leverage: float = 1.0,
    ) -> float:
        return fixed_fractional_size(
            balance=balance,
            risk_pct=self.risk_per_trade,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional_pct=max_position_notional_pct,
            leverage=leverage,
        )

    def score_symbol(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0

        prob_up = float(self.model.predict_proba(df))
        atr_pct = float(df.iloc[-1]["atr_pct"])
        adx = float(df.iloc[-1]["adx"])
        return float(prob_up * atr_pct * adx)
