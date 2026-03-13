# execution/runner.py

import time
import pandas as pd
from datetime import datetime, timedelta

from data.fetcher import MarketDataFetcher
from models.direction import DirectionModel
from models.ensemble import EnsembleDirectionModel
from execution.strategy import StrategyEngine, StrategyConfig
from execution.shadow_broker import ShadowBroker
from execution.regime_controller import RegimeController
from execution.ai_supervisor import AISupervisor
from execution.market_guard import MarketGuard
from risk.limits import RiskLimits, RiskState
from features.technicals import compute_core_features
from metrics.self_report import DailyAIReport
from logs.logger import TradeLogger


def _fmt_price(price: float, symbol: str) -> str:
    """Symbol-aware price formatting (low-price assets need more decimals)."""
    if price < 0.01:
        return f"{price:.6f}"
    if price < 1.0:
        return f"{price:.4f}"
    return f"{price:.2f}"


class TradingRunner:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 300,
        mode: str = "shadow",
        starting_balance_usdt: float = 500.0,
        cooldown_minutes: int = 30,
        risk_per_trade: float = 0.01,
        config: StrategyConfig | None = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback

        self.data = MarketDataFetcher()

        base_model = DirectionModel.for_symbol(symbol)
        models = [base_model]

        if symbol != "BTC/USDT":
            try:
                models.append(DirectionModel.for_symbol("BTC/USDT"))
            except Exception:
                pass

        self.model = EnsembleDirectionModel(models)

        self.cfg = config or StrategyConfig(cooldown_minutes=cooldown_minutes)
        self.strategy = StrategyEngine(self.model, risk_per_trade, self.cfg)

        self.supervisor = AISupervisor()
        self.regime_ctrl = RegimeController()
        self.market_guard = MarketGuard()

        self.risk_limits = RiskLimits()
        self.risk_state = RiskState(starting_balance_usdt)

        self.broker = ShadowBroker()
        self.logger = TradeLogger()

        self.report = DailyAIReport()

        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_trade_time: datetime | None = None

        # Position state
        self.last_entry_price: float | None = None
        self.last_entry_prob: float | None = None
        self.last_entry_side: str | None = None
        self.last_decision = None                      # SignalDecision snapshot at entry

        # Stops
        self.stop_loss: float | None = None
        self.take_profit: float | None = None
        self.take_profit_1: float | None = None
        self._profit_lock_activated: bool = False

        # Candle-close deduplication: only act once per fully closed candle.
        self.last_processed_candle_time: datetime | None = None

        print(f"[AUTONOMOUS AI] {symbol} ready")

    # --------------------------------------------------
    def run_once(self):

        df = self.data.fetch_ohlcv(self.symbol, self.timeframe, self.lookback)

        # ---- Candle-close guard ----
        # iloc[-1] is the still-forming (live) candle — use iloc[-2] as the last
        # fully closed candle so signals are never based on incomplete data.
        if len(df) < 3:
            return
        closed_candle_time = pd.Timestamp(df.iloc[-2]["time"], unit="ms", tz="UTC")
        if closed_candle_time == self.last_processed_candle_time:
            return   # already processed this candle, nothing new to evaluate
        self.last_processed_candle_time = closed_candle_time

        # Use only confirmed closed candles (drop the live/forming last row).
        df = df.iloc[:-1].copy()
        df = compute_core_features(df)

        today = datetime.utcnow().date()
        self.risk_state.reset_if_new_day(today)
        self.supervisor.update_equity(self.risk_state.current_balance)

        # ---------------- GLOBAL SAFETY ----------------
        if not self.market_guard.allow_trading(
            balance=self.risk_state.current_balance,
            today=today,
        ):
            return

        regime = self.regime_ctrl.detect(df)

        if not self.regime_ctrl.trading_allowed(regime):
            return

        supervisor_dec = self.supervisor.decide()
        if not supervisor_dec.trade_allowed:
            return

        price = float(df.iloc[-1]["close"])
        atr   = float(df.iloc[-1]["atr"])

        # ================= MANAGE OPEN POSITION =================
        if self.broker.position:
            pos = self.broker.position
            fp  = _fmt_price(price, self.symbol)

            # ---- TP1 Profit Lock ----
            if not self._profit_lock_activated and price >= self.take_profit_1:
                self._profit_lock_activated = True
                profit_lock_sl = self.last_entry_price + atr * 0.5
                self.stop_loss = max(self.stop_loss, profit_lock_sl)
                print(f"🎯 TP1 HIT → profit lock activated")
                print(f"🔒 PROFIT LOCK SL → {_fmt_price(self.stop_loss, self.symbol)}")

            # ---- Trailing (post-TP1 only, SL can only move upward) ----
            if self._profit_lock_activated:
                new_sl = price - self.cfg.trail_atr_mult * atr
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
                    print(f"🔁 TRAILING SL → {_fmt_price(self.stop_loss, self.symbol)}")

            # ---- Stop Loss ----
            if price <= self.stop_loss:
                self._close_position(price, "stop_loss")
                return

            # ---- Take Profit ----
            if price >= self.take_profit:
                self._close_position(price, "take_profit")
                return

            return

        # ================= ENTRY =================
        if self.last_trade_time and datetime.utcnow() - self.last_trade_time < self.cooldown:
            return

        # All filtering logic is inside generate_signal — runner trusts it fully.
        dec = self.strategy.generate_signal(df, regime=str(regime))

        if not dec.side:
            print(f"[{self.symbol}] SKIP | {dec.reason}")
            return

        risk_mult = supervisor_dec.risk_multiplier * self.regime_ctrl.risk_multiplier(regime)

        qty = self.strategy.position_size(
            balance=self.risk_state.current_balance * risk_mult,
            entry_price=dec.price,
            stop_price=dec.stop_loss,          # ATR-based stop, not hardcoded 1%
        )

        if qty <= 0:
            return

        self.broker.open_position(dec.side, dec.price, qty, self.symbol)

        self.last_trade_time   = datetime.utcnow()
        self.last_entry_price  = dec.price
        self.last_entry_prob   = dec.prob
        self.last_entry_side   = dec.side
        self.last_decision     = dec
        self.stop_loss         = dec.stop_loss
        self.take_profit       = dec.take_profit
        self.take_profit_1     = dec.price + dec.atr * (self.cfg.take_atr_mult / 2)
        self._profit_lock_activated = False

        fp = _fmt_price(dec.price, self.symbol)
        print(
            f"📈 OPEN {dec.side} {self.symbol} | price={fp} | qty={qty:.6f} | "
            f"SL={_fmt_price(dec.stop_loss, self.symbol)} | "
            f"TP={_fmt_price(dec.take_profit, self.symbol)} | "
            f"prob={dec.prob:.3f} | adx={dec.adx:.1f} | regime={dec.regime}"
        )

    # --------------------------------------------------
    def _close_position(self, price: float, exit_reason: str):
        """Shared close logic for SL and TP exits."""
        pos       = self.broker.position
        add_count = pos.add_count
        avg_entry = pos.avg_entry
        total_qty = pos.qty
        dec       = self.last_decision

        pnl = self.broker.close_position(price, self.symbol)

        self.market_guard.register_trade(pnl)
        self.risk_state.register_trade(pnl)
        self.supervisor.register_trade(pnl)

        try:
            self.logger.log(
                symbol=self.symbol,
                side=self.last_entry_side,
                entry_price=self.last_entry_price,
                avg_entry=avg_entry,
                exit_price=price,
                qty=total_qty,
                pnl=pnl,
                balance=self.risk_state.current_balance,
                prob_up=self.last_entry_prob,
                threshold=dec.threshold if dec else 0.0,
                atr=dec.atr if dec else 0.0,
                atr_pct=dec.atr_pct if dec else 0.0,
                adx=dec.adx if dec else 0.0,
                regime=dec.regime if dec else "",
                stop_loss=self.stop_loss,
                take_profit=self.take_profit,
                exit_reason=exit_reason,
                add_count=add_count,
            )
        except Exception as exc:
            print(f"[LOGGER ERROR] {exc}")

        icon = "🛑" if exit_reason == "stop_loss" else "🎯"
        print(
            f"{icon} {exit_reason.upper().replace('_', ' ')} | "
            f"pnl={pnl:.4f} | balance={self.risk_state.current_balance:.2f} | "
            f"avg_entry={_fmt_price(avg_entry, self.symbol)} | adds={add_count}"
        )

    # --------------------------------------------------
    def run_loop(self, sleep_seconds: int = 60):
        """Single-symbol continuous loop — mirrors multi_runner behaviour."""
        print(f"🚀 {self.symbol} loop started (sleep={sleep_seconds}s)")
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                print("Stopped by user")
                break
            except Exception as exc:
                print(f"[{self.symbol}] run error: {exc}")
            time.sleep(sleep_seconds)
