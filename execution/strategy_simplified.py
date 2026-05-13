"""
Phase 3: Strategy Simplification

Reduce complexity by consolidating overlapping filters and indicators.
Focus on core signal quality without redundancy.

Core Principle: Simpler strategies are more robust and generalizable.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedStrategyConfig:
    """
    Minimal, non-redundant strategy configuration.
    
    Removed:
    - Redundant ADX/ATR checks (use simple volatility)
    - Redundant EMA checks (use simple trend)
    - Excessive pyramid logic
    - Regime-specific thresholds (add complexity)
    - Cascading filters (hidden overfitting)
    
    Kept:
    - ML model confidence (primary signal)
    - Volatility gate (prevents thin markets)
    - Trend bias (removes counter-trend whipsaws)
    - Basic risk/reward (SL/TP levels)
    """

    # ML Model Parameters
    min_confidence: float = 0.55  # Only trade high-confidence signals
    confidence_percentile: float = 50.0  # Filter by model quality distribution

    # Volatility Gating (prevents thin market trades)
    min_volatility_pct: float = 0.002  # Min 0.2% daily volatility
    max_volatility_pct: float = 0.15  # Max 15% daily volatility (avoid gaps)

    # Trend Bias (simple EMA, no regime complexity)
    use_trend_filter: bool = True
    ema_period: int = 50  # Simple 50-period trend

    # Risk/Reward (exit strategy)
    stop_loss_atr_mult: float = 1.5  # Stop loss = ATR * 1.5
    take_profit_atr_mult: float = 3.0  # TP = ATR * 3.0

    # Trading Costs (realistic)
    fee_pct_per_side: float = 0.0010  # Maker fee
    slippage_pct_per_side: float = 0.0005  # Conservative slippage

    # Cooldown (prevent oversignaling)
    min_bars_between_trades: int = 10  # At least 10 bars between signals

    # Risk Management
    risk_per_trade: float = 0.01  # 1% per trade
    max_open_trades: int = 1  # Only 1 position at a time (no pyramiding)

    # Leverage (Futures trading)
    default_leverage: float = 1.0  # Default leverage for positions (1.0-125.0)
    max_leverage: float = 10.0  # Maximum allowed leverage per trade


@dataclass
class SimplifiedSignal:
    """Clean signal output (removed complexity)"""

    symbol: str
    timestamp: pd.Timestamp
    side: Optional[str]  # 'LONG', 'SHORT', None
    confidence: float  # Model probability
    volatility: float  # Current ATR % of price
    trend_bias: str  # 'UP', 'DOWN', 'NEUTRAL'
    entry_price: float
    stop_loss: float
    take_profit: float
    expected_pnl_pct: float
    reason: str  # Why was this signal generated?
    filters_passed: Dict[str, bool]


class SimplifiedStrategyEngine:
    """
    Non-redundant strategy engine.
    
    Single responsibility: Convert ML confidence + risk filters into trade decisions.
    
    Removes:
    - Redundant indicator stacking
    - Regime-dependent logic (increases overfitting)
    - Complex filter cascades
    - Adaptive thresholds
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[SimplifiedStrategyConfig] = None,
    ):
        self.model = model
        self.config = config or SimplifiedStrategyConfig()
        self.last_signal_bar = -999  # Track last signal to enforce cooldown

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_bar_idx: Optional[int] = None,
    ) -> SimplifiedSignal:
        """
        Generate trading signal with minimal complexity.
        
        Args:
            symbol: Trading symbol
            df: OHLCV dataframe with indicators (should have ATR, EMA)
            current_bar_idx: Index of current bar (for cooldown)
            
        Returns:
            SimplifiedSignal with decision + reasoning
        """
        if len(df) < 5:
            return self._null_signal(symbol, df, "Insufficient data")

        # Get current bar
        bar = df.iloc[-1]
        timestamp = bar.name if hasattr(bar.name, "to_pydatetime") else pd.Timestamp.now()

        filters = {}

        # === FILTER 1: Volatility Gate ===
        # Prevents trading thin, gappy markets
        atr = bar.get("atr", 0.0)
        volatility_pct = (atr / bar["close"]) if bar["close"] > 0 else 0.0

        filters["volatility_ok"] = (
            self.config.min_volatility_pct <= volatility_pct <= self.config.max_volatility_pct
        )

        if not filters["volatility_ok"]:
            return self._null_signal(
                symbol, df, f"Volatility {volatility_pct:.3%} outside range"
            )

        # === FILTER 2: Trend Bias ===
        # Simple EMA, prevents counter-trend trades
        trend_bias = "NEUTRAL"
        if self.config.use_trend_filter:
            ema = bar.get("ema_50", bar["close"])
            if bar["close"] > ema * 1.01:
                trend_bias = "UP"
            elif bar["close"] < ema * 0.99:
                trend_bias = "DOWN"

        filters["trend_ok"] = trend_bias in ["UP", "DOWN"]

        if not filters["trend_ok"]:
            return self._null_signal(symbol, df, f"No trend bias: {trend_bias}")

        # === FILTER 3: ML Model Confidence ===
        # Primary signal source
        if self.model is None:
            confidence = 0.5
            pred_side = None
        else:
            try:
                proba = self.model.predict_proba(df.iloc[[-1]])
                if isinstance(proba, (tuple, list)):
                    proba_short, proba_long = proba
                    confidence = max(proba_short, proba_long)
                    pred_side = "LONG" if proba_long > proba_short else "SHORT"
                else:
                    confidence = float(proba[0, 1]) if len(proba.shape) > 1 else float(proba)
                    pred_side = "LONG" if confidence > 0.5 else "SHORT"
            except Exception as e:
                logger.warning(f"Model prediction error: {e}")
                return self._null_signal(symbol, df, f"Model error: {e}")

        filters["confidence_ok"] = confidence >= self.config.min_confidence

        if not filters["confidence_ok"]:
            return self._null_signal(
                symbol, df, f"Confidence {confidence:.3f} < {self.config.min_confidence}"
            )

        # === FILTER 4: Cooldown ===
        # Prevent oversignaling
        if current_bar_idx is not None:
            bars_since_last = current_bar_idx - self.last_signal_bar
            filters["cooldown_ok"] = bars_since_last >= self.config.min_bars_between_trades

            if not filters["cooldown_ok"]:
                return self._null_signal(
                    symbol, df, f"Cooldown: {bars_since_last} bars since last signal"
                )

        # === All filters passed - Generate exit levels ===

        entry = bar["close"]
        stop_dist = atr * self.config.stop_loss_atr_mult
        tp_dist = atr * self.config.take_profit_atr_mult

        if pred_side == "LONG":
            stop_loss = entry - stop_dist
            take_profit = entry + tp_dist
        else:  # SHORT
            stop_loss = entry + stop_dist
            take_profit = entry - tp_dist

        # Expected PnL (assuming model accuracy)
        if pred_side == "LONG":
            expected_pnl_pct = (take_profit - entry) / entry - (
                self.config.fee_pct_per_side * 2 + self.config.slippage_pct_per_side * 2
            )
        else:
            expected_pnl_pct = (entry - take_profit) / entry - (
                self.config.fee_pct_per_side * 2 + self.config.slippage_pct_per_side * 2
            )

        # Record signal for cooldown
        if current_bar_idx is not None:
            self.last_signal_bar = current_bar_idx

        # All filters passed
        for key, val in filters.items():
            if not val:
                return self._null_signal(symbol, df, f"Filter failed: {key}")

        return SimplifiedSignal(
            symbol=symbol,
            timestamp=timestamp,
            side=pred_side,
            confidence=confidence,
            volatility=volatility_pct,
            trend_bias=trend_bias,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_pnl_pct=expected_pnl_pct,
            reason=f"{pred_side} with {confidence:.1%} confidence, {volatility_pct:.1%} vol, {trend_bias} trend",
            filters_passed=filters,
        )

    def _null_signal(
        self, symbol: str, df: pd.DataFrame, reason: str
    ) -> SimplifiedSignal:
        """Return null signal with reasoning"""
        bar = df.iloc[-1]
        return SimplifiedSignal(
            symbol=symbol,
            timestamp=bar.name if hasattr(bar.name, "to_pydatetime") else pd.Timestamp.now(),
            side=None,
            confidence=0.0,
            volatility=0.0,
            trend_bias="NEUTRAL",
            entry_price=bar["close"],
            stop_loss=0.0,
            take_profit=0.0,
            expected_pnl_pct=0.0,
            reason=reason,
            filters_passed={},
        )

    def validate_signal(self, signal: SimplifiedSignal) -> bool:
        """Validate signal quality before execution"""
        if signal.side is None:
            return False

        # Check RR ratio (at least 2:1)
        if signal.side == "LONG":
            rr_ratio = (signal.take_profit - signal.entry_price) / max(
                signal.entry_price - signal.stop_loss, 0.0001
            )
        else:
            rr_ratio = (signal.entry_price - signal.take_profit) / max(
                signal.stop_loss - signal.entry_price, 0.0001
            )

        return rr_ratio >= 2.0 and signal.expected_pnl_pct > 0.0
