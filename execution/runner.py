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
        self.last_entry_qty: float | None = None      # initial qty (base for pyramid scale)
        self.last_entry_prob: float | None = None
        self.last_entry_side: str | None = None
        self.last_pyramid_price: float | None = None  # price at last pyramid add
        self.last_decision = None                      # SignalDecision snapshot at entry

        # Stops
        self.stop_loss: float | None = None
        self.take_profit: float | None = None
        self._trail_activated: bool = False

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

            # ---- Trailing Stop (ATR-based, activates after 1 ATR of profit) ----
            unrealised_move = price - self.last_entry_price
            if unrealised_move >= self.cfg.trail_activate_atr_mult * atr:
                if not self._trail_activated:
                    # First activation: move stop to breakeven + small fee buffer
                    self._trail_activated = True
                    self.stop_loss = max(self.stop_loss, self.last_entry_price * 1.0005)
                    print(f"🔒 TRAIL ACTIVATED → SL moved to breakeven ({_fmt_price(self.stop_loss, self.symbol)})")

                new_sl = price - self.cfg.trail_atr_mult * atr
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
                    print(f"🔁 TRAILING SL → {_fmt_price(self.stop_loss, self.symbol)}")

            # ---- Pyramiding ----
            ref_price = self.last_pyramid_price or self.last_entry_price
            move_pct  = (price - ref_price) / ref_price

            if (
                self._trail_activated                           # only add when protected
                and move_pct >= self.cfg.pyramid_trigger_pct
                and pos.add_count < self.cfg.max_pyramid_adds
            ):
                scale   = self.cfg.pyramid_qty_scales[pos.add_count]
                add_qty = self.last_entry_qty * scale

                # Notional safety cap
                max_notional = self.last_entry_price * self.last_entry_qty * 2.0
                if pos.avg_entry * (pos.qty + add_qty) <= max_notional:
                    self.broker.add_to_position(price, add_qty)
                    self.last_pyramid_price = price
                    print(
                        f"➕ PYRAMID ADD #{pos.add_count} | price={fp} "
                        f"qty={add_qty:.6f} avg_entry={_fmt_price(pos.avg_entry, self.symbol)} "
                        f"total_qty={pos.qty:.6f}"
                    )

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
        self.last_entry_qty    = qty
        self.last_entry_prob   = dec.prob
        self.last_entry_side   = dec.side
        self.last_pyramid_price = None
        self.last_decision     = dec
        self.stop_loss         = dec.stop_loss
        self.take_profit       = dec.take_profit
        self._trail_activated  = False

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

        self.last_pyramid_price = None
        self._trail_activated   = False

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
