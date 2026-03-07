#execution/runner.py

import time
from datetime import datetime, timedelta

from data.fetcher import MarketDataFetcher
from models.direction import DirectionModel
from models.ensemble import EnsembleDirectionModel
from execution.strategy import StrategyEngine
from execution.shadow_broker import ShadowBroker
from execution.regime_controller import RegimeController
from execution.ai_supervisor import AISupervisor
from execution.market_guard import MarketGuard
from risk.limits import RiskLimits, RiskState
from features.technicals import compute_core_features
from metrics.self_report import DailyAIReport
from logs.logger import TradeLogger


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
        self.strategy = StrategyEngine(self.model, risk_per_trade)

        self.supervisor = AISupervisor()
        self.regime_ctrl = RegimeController()
        self.market_guard = MarketGuard()

        self.risk_limits = RiskLimits()
        self.risk_state = RiskState(starting_balance_usdt)

        self.broker = ShadowBroker()
        self.logger = TradeLogger()

        self.report = DailyAIReport()

        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_trade_time = None

        self.last_entry_price = None
        self.last_entry_qty = None
        self.last_entry_prob = None
        self.last_entry_side = None

        self.stop_loss = None
        self.take_profit = None

        print(f"[AUTONOMOUS AI] {symbol} ready")

    # --------------------------------------------------
    def run_once(self):

        df = self.data.fetch_ohlcv(self.symbol, self.timeframe, self.lookback)
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

        decision = self.supervisor.decide()

        if not decision.trade_allowed:
            return

        price = float(df.iloc[-1]["close"])
        atr = float(df.iloc[-1]["atr"])

        # ================= EXIT =================
        if self.broker.position:

            # -------- Trailing Stop --------
            if price - self.last_entry_price > atr:

                new_sl = price - atr

                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
                    print(f"🔁 TRAILING SL UPDATED → {self.stop_loss:.2f}")

            # -------- Stop Loss --------
            if price <= self.stop_loss:

                pnl = self.broker.close_position(price, self.symbol)

                self.market_guard.register_trade(pnl)
                self.risk_state.register_trade(pnl)
                self.supervisor.register_trade(pnl)

                self.logger.log(
                    symbol=self.symbol,
                    side=self.last_entry_side,
                    entry_price=self.last_entry_price,
                    exit_price=price,
                    qty=self.last_entry_qty,
                    pnl=pnl,
                    balance=self.risk_state.current_balance,
                    prob_up=self.last_entry_prob,
                )

                print(f"🛑 STOP LOSS HIT | PnL={pnl:.4f} | Balance={self.risk_state.current_balance:.2f}")
                return

            # -------- Take Profit --------
            if price >= self.take_profit:

                pnl = self.broker.close_position(price, self.symbol)

                self.market_guard.register_trade(pnl)
                self.risk_state.register_trade(pnl)
                self.supervisor.register_trade(pnl)

                self.logger.log(
                    symbol=self.symbol,
                    side=self.last_entry_side,
                    entry_price=self.last_entry_price,
                    exit_price=price,
                    qty=self.last_entry_qty,
                    pnl=pnl,
                    balance=self.risk_state.current_balance,
                    prob_up=self.last_entry_prob,
                )

                print(f"🎯 TAKE PROFIT HIT | PnL={pnl:.4f} | Balance={self.risk_state.current_balance:.2f}")
                return

            return

        # ================= ENTRY =================

        if self.last_trade_time and datetime.utcnow() - self.last_trade_time < self.cooldown:
            return

        signal, prob = self.strategy.generate_signal(df)

        if not signal:
            return

        # -------- Probability Filter --------
        MIN_PROB = 0.50
        LONG_TH = 0.58

        if prob < MIN_PROB:
            print(f"DEBUG | SKIP low probability | prob={prob:.3f}")
            return

        if prob < LONG_TH:
            return

        risk_mult = decision.risk_multiplier * self.regime_ctrl.risk_multiplier(regime)

        qty = self.strategy.position_size(
            balance=self.risk_state.current_balance * risk_mult,
            entry_price=price,
            side=signal,
        )

        if qty <= 0:
            return

        # -------- ATR Risk Model --------
        stop_distance = atr * 2
        take_distance = atr * 3

        self.stop_loss = price - stop_distance
        self.take_profit = price + take_distance

        self.broker.open_position(signal, price, qty, self.symbol)

        self.last_trade_time = datetime.utcnow()
        self.last_entry_price = price
        self.last_entry_qty = qty
        self.last_entry_prob = prob
        self.last_entry_side = signal

        print(
            f"📈 OPEN {signal} | price={price:.2f} | qty={qty:.6f} | "
            f"SL={self.stop_loss:.2f} | TP={self.take_profit:.2f} | prob={prob:.3f}"
        )