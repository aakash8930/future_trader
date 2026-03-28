#execution/runner.py

import time
from datetime import datetime, timedelta

import pandas as pd

from data.fetcher import MarketDataFetcher
from execution.ai_supervisor import AISupervisor
from execution.market_guard import MarketGuard
from execution.regime_controller import RegimeController
from execution.shadow_broker import ShadowBroker
from execution.strategy import StrategyConfig, StrategyEngine
from features.technicals import compute_core_features
from logs.logger import TradeLogger
from metrics.self_report import DailyAIReport
from models.direction import DirectionModel
from risk.limits import RiskLimits, RiskState


def _fmt_price(price: float, symbol: str) -> str:
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
        exchange_name: str = "binance",
        exchange_fallbacks: list[str] | None = None,
        exchange_timeout_ms: int = 20000,
        logger: TradeLogger | None = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback

        self.data = MarketDataFetcher(
            exchange_name=exchange_name,
            fallback_exchanges=exchange_fallbacks,
            timeout_ms=exchange_timeout_ms,
        )

        self.model = DirectionModel.for_symbol(symbol)

        self.cfg = config or StrategyConfig(cooldown_minutes=cooldown_minutes)
        self.strategy = StrategyEngine(self.model, risk_per_trade, self.cfg)

        self.supervisor = AISupervisor()
        self.regime_ctrl = RegimeController()
        self.market_guard = MarketGuard()

        self.risk_limits = RiskLimits()
        self.risk_state = RiskState(starting_balance_usdt)

        self.broker = ShadowBroker()
        self.logger = logger if logger is not None else TradeLogger()
        self.report = DailyAIReport()

        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_trade_time: datetime | None = None
        self.last_processed_candle_time: pd.Timestamp | None = None
        self.last_fetch_wallclock: datetime | None = None

        self.last_entry_price: float | None = None
        self.last_entry_prob: float | None = None
        self.last_entry_side: str | None = None
        self.last_decision = None

        self.stop_loss: float | None = None
        self.take_profit: float | None = None
        self.take_profit_1: float | None = None
        self._profit_lock_activated = False

        print(f"[AUTONOMOUS AI] {symbol} ready")

    def _should_skip_fetch(self) -> bool:
        if self.last_fetch_wallclock is None:
            return False

        elapsed = (datetime.utcnow() - self.last_fetch_wallclock).total_seconds()

        tf = (self.timeframe or "").lower().strip()
        min_gap_seconds = {
            "1m": 10,
            "3m": 20,
            "5m": 30,
            "15m": 60,
            "30m": 90,
            "1h": 120,
            "4h": 300,
            "1d": 900,
        }.get(tf, 60)

        return elapsed < min_gap_seconds

    def run_once(self):
        if self._should_skip_fetch():
            return

        self.last_fetch_wallclock = datetime.utcnow()

        raw_df = self.data.fetch_ohlcv(self.symbol, self.timeframe, self.lookback + 5)

        if raw_df is None:
            print(f"[{self.symbol}] not supported on {self.data.exchange_name}, skipping")
            return

        if len(raw_df) < 3:
            return

        closed_candle_time = pd.Timestamp(raw_df.iloc[-2]["time"], unit="ms", tz="UTC")
        if closed_candle_time == self.last_processed_candle_time:
            return

        self.last_processed_candle_time = closed_candle_time

        df = raw_df.iloc[:-1].copy()
        df = compute_core_features(df)

        if df.empty:
            return

        today = datetime.utcnow().date()
        self.risk_state.reset_if_new_day(today)
        self.supervisor.update_equity(self.risk_state.current_balance)

        if not self.market_guard.allow_trading(
            balance=self.risk_state.current_balance,
            today=today,
        ):
            return

        regime = self.regime_ctrl.detect(df)
        if not self.regime_ctrl.trading_allowed(regime):
            if hasattr(self.regime_ctrl, "skip_reason"):
                print(f"[{self.symbol}] SKIP | {self.regime_ctrl.skip_reason(df)}")
            else:
                print(f"[{self.symbol}] SKIP | regime_block({regime})")
            return

        supervisor_dec = self.supervisor.decide()
        if not supervisor_dec.trade_allowed:
            print(f"[{self.symbol}] SKIP | supervisor_block({supervisor_dec.reason})")
            return

        closed_price = float(df.iloc[-1]["close"])
        atr = float(df.iloc[-1]["atr"])

        live_price = None
        if hasattr(self.data, "fetch_last_price"):
            try:
                live_price = self.data.fetch_last_price(self.symbol)
            except Exception:
                live_price = None

        manage_price = live_price if live_price is not None else closed_price

        if self.broker.position:
            if not self._profit_lock_activated and manage_price >= self.take_profit_1:
                self._profit_lock_activated = True
                profit_lock_sl = self.last_entry_price + atr * 0.5
                self.stop_loss = max(self.stop_loss, profit_lock_sl)
                print(
                    f"[{self.symbol}] TP1 HIT | "
                    f"manage_price={_fmt_price(manage_price, self.symbol)} | "
                    f"lock_sl={_fmt_price(self.stop_loss, self.symbol)}"
                )

            if self._profit_lock_activated:
                new_sl = manage_price - self.cfg.trail_atr_mult * atr
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
                    print(f"[{self.symbol}] TRAILING SL | {_fmt_price(self.stop_loss, self.symbol)}")

            if manage_price <= self.stop_loss:
                self._close_position(manage_price, "stop_loss")
                return

            if manage_price >= self.take_profit:
                self._close_position(manage_price, "take_profit")
                return

            return

        if self.last_trade_time and datetime.utcnow() - self.last_trade_time < self.cooldown:
            return

        dec = self.strategy.generate_signal(df, regime=str(regime))

        if not dec.side:
            print(f"[{self.symbol}] SKIP | {dec.reason}")
            return

        risk_mult = supervisor_dec.risk_multiplier * self.regime_ctrl.risk_multiplier(regime)

        qty = self.strategy.position_size(
            balance=self.risk_state.current_balance * risk_mult,
            entry_price=dec.price,
            stop_price=dec.stop_loss,
        )

        if qty <= 0:
            print(f"[{self.symbol}] SKIP | qty_zero")
            return

        self.broker.open_position(dec.side, dec.price, qty, self.symbol)

        self.last_trade_time = datetime.utcnow()
        self.last_entry_price = dec.price
        self.last_entry_prob = dec.prob
        self.last_entry_side = dec.side
        self.last_decision = dec
        self.stop_loss = dec.stop_loss
        self.take_profit = dec.take_profit
        self.take_profit_1 = dec.price + dec.atr * (self.cfg.take_atr_mult / 2.0)
        self._profit_lock_activated = False

        print(
            f"📈 OPEN {dec.side} {self.symbol} | "
            f"price={_fmt_price(dec.price, self.symbol)} | qty={qty:.6f} | "
            f"SL={_fmt_price(dec.stop_loss, self.symbol)} | "
            f"TP={_fmt_price(dec.take_profit, self.symbol)} | "
            f"prob={dec.prob:.3f} | adx={dec.adx:.1f} | regime={dec.regime}"
        )

    def _close_position(self, price: float, exit_reason: str):
        pos = self.broker.position
        add_count = pos.add_count
        avg_entry = pos.avg_entry
        total_qty = pos.qty
        dec = self.last_decision

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
            f"{icon} {self.symbol} {exit_reason.upper().replace('_', ' ')} | "
            f"pnl={pnl:.4f} | balance={self.risk_state.current_balance:.2f} | "
            f"avg_entry={_fmt_price(avg_entry, self.symbol)} | adds={add_count}"
        )

    def run_loop(self, sleep_seconds: int = 60):
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