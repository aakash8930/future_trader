# backtest/simulator.py
"""
Historical simulator — uses IDENTICAL logic to execution/runner.py:
  - Same StrategyEngine filters and entry thresholds
  - Same ATR-based stop loss / take profit
  - Same TP1-triggered profit-lock model and trailing distance
  - Same cooldown
"""

from datetime import datetime, timedelta

import pandas as pd
import ta

from execution.broker import PaperBroker
from execution.strategy import StrategyEngine, StrategyConfig
from execution.regime_controller import RegimeController
from models.direction import DirectionModel
from risk.limits import RiskLimits, RiskState
from features.technicals import compute_core_features


class HistoricalSimulator:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        metadata_path: str,
        starting_balance: float = 500.0,
        lookback: int = 300,
        config: StrategyConfig | None = None,
        risk_per_trade: float = 0.01,
    ):
        self.model = DirectionModel(model_path, scaler_path, metadata_path)
        self.cfg   = config or StrategyConfig()

        self.strategy     = StrategyEngine(self.model, risk_per_trade, self.cfg)
        self.regime_ctrl  = RegimeController()

        self.risk_limits  = RiskLimits(max_daily_loss_pct=1.0, max_consecutive_losses=1_000_000)
        self.risk_state   = RiskState(starting_balance)
        self.broker       = PaperBroker()

        self.lookback      = lookback
        self.risk_per_trade = risk_per_trade

        self.trades: list[dict] = []

        # Position state (mirrors runner.py)
        self.last_entry_time:  datetime | None = None
        self.last_entry_price: float   | None = None
        self.last_entry_prob:  float   | None = None
        self.last_entry_side:  str     | None = None
        self.last_decision = None

        self.stop_loss:    float | None = None
        self.take_profit:  float | None = None
        self.take_profit_1: float | None = None
        self._profit_lock_activated: bool = False

        # Candle-close deduplication: skip if this candle timestamp was already processed.
        self.last_processed_candle_time: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    def step(self, df: pd.DataFrame) -> None:
        # The caller passes a rolling window; iloc[-1] is the candle under evaluation.
        # In a properly stepped backtest loop each window ends on a closed candle, but
        # guard against accidental duplicate calls with the same window.
        if len(df) < 2:
            return

        raw_ts = df.iloc[-1]["time"] if "time" in df.columns else None
        if raw_ts is not None:
            candle_ts = pd.Timestamp(raw_ts, unit="ms")
            if candle_ts == self.last_processed_candle_time:
                return
            self.last_processed_candle_time = candle_ts

        df = compute_core_features(df)

        if df.empty:
            return

        ts    = pd.to_datetime(df.iloc[-1]["time"], unit="ms") if "time" in df.columns else datetime.utcnow()
        today = ts.date()
        self.risk_state.reset_if_new_day(today)

        price = float(df.iloc[-1]["close"])
        atr   = float(df.iloc[-1]["atr"])

        regime = self.regime_ctrl.detect(df)

        # ================= MANAGE OPEN POSITION =================
        if self.broker.position:
            pos = self.broker.position

            # TP1 profit lock (mirrors runner.py)
            if not self._profit_lock_activated and price >= self.take_profit_1:
                self._profit_lock_activated = True
                profit_lock_sl = self.last_entry_price + atr * 0.5
                self.stop_loss = max(self.stop_loss, profit_lock_sl)

            # Trailing (post-TP1 only, SL can only move upward)
            if self._profit_lock_activated:
                new_sl = price - self.cfg.trail_atr_mult * atr
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl

            # Stop loss
            if price <= self.stop_loss:
                self._close(price, ts, "stop_loss")
                return

            # Take profit
            if price >= self.take_profit:
                self._close(price, ts, "take_profit")
                return

            return

        # ================= ENTRY =================
        if self.last_entry_time:
            cooldown = timedelta(minutes=self.cfg.cooldown_minutes)
            if (ts - self.last_entry_time) < cooldown:
                return

        if not self.risk_state.trading_allowed(self.risk_limits):
            return

        dec = self.strategy.generate_signal(df, regime=str(regime))

        if not dec.side:
            return

        qty = self.strategy.position_size(
            balance=self.risk_state.current_balance,
            entry_price=dec.price,
            stop_price=dec.stop_loss,
        )

        if qty <= 0:
            return

        self.broker.open_position(dec.side, dec.price, qty)

        self.last_entry_time   = ts
        self.last_entry_price  = dec.price
        self.last_entry_prob   = dec.prob
        self.last_entry_side   = dec.side
        self.last_decision     = dec
        self.stop_loss         = dec.stop_loss
        self.take_profit       = dec.take_profit
        self.take_profit_1     = dec.price + dec.atr * (self.cfg.take_atr_mult / 2)
        self._profit_lock_activated = False

    # ------------------------------------------------------------------
    def _close(self, price: float, ts: datetime, exit_reason: str):
        pos       = self.broker.position
        add_count = pos.add_count
        avg_entry = pos.avg_entry
        total_qty = pos.qty
        dec       = self.last_decision

        pnl = self.broker.close_position(price)
        self.risk_state.register_trade(pnl)

        self.trades.append({
            "entry_time":  self.last_entry_time,
            "exit_time":   ts,
            "symbol":      "",
            "side":        self.last_entry_side,
            "entry_price": self.last_entry_price,
            "avg_entry":   avg_entry,
            "exit_price":  price,
            "qty":         total_qty,
            "pnl":         pnl,
            "balance":     self.risk_state.current_balance,
            "prob":        self.last_entry_prob,
            "threshold":   dec.threshold if dec else 0.0,
            "atr":         dec.atr if dec else 0.0,
            "atr_pct":     dec.atr_pct if dec else 0.0,
            "adx":         dec.adx if dec else 0.0,
            "regime":      dec.regime if dec else "",
            "stop_loss":   self.stop_loss,
            "take_profit": self.take_profit,
            "exit_reason": exit_reason,
            "add_count":   add_count,
        })

    # ------------------------------------------------------------------
    def export(self, path: str) -> None:
        pd.DataFrame(self.trades).to_csv(path, index=False)


