# execution/runner.py

import signal
import time
from datetime import datetime, timedelta

import pandas as pd

from data.fetcher import MarketDataFetcher
from execution.ai_supervisor import AISupervisor
from execution.market_guard import MarketGuard
from execution.regime_controller import RegimeController
import os

from execution.broker import LiveBroker, ShadowBroker
from execution.strategy import StrategyConfig, StrategyEngine
from features.technicals import compute_core_features
from logs.logger import TradeLogger
from logs.trade_store import append_equity, append_trade
from logs.event_logger import get_event_logger
from metrics.self_report import DailyAIReport
from models.direction import DirectionModel
from models.ensemble import EnsembleDirectionModel
from risk.limits import RiskLimits, RiskState
from config.shared_state import get_shared_state, update_shared_state


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
        leverage: float = 1.0,
        config: StrategyConfig | None = None,
        exchange_name: str = "binance",
        exchange_fallbacks: list[str] | None = None,
        exchange_timeout_ms: int = 20000,
        exchange_type: str = "cex",
        logger: TradeLogger | None = None,
        event_logger=None,
    ):
        self.symbol = symbol
        self.leverage = leverage  # Position leverage (1.0-125.0)
        self.timeframe = timeframe  # original timeframe (used for CLI)
        self.lookback = lookback
        self._cached_timeframe = timeframe
        self._cached_mode = mode
        self._cached_exchange_type = exchange_type

        # Poll-based config override support
        self._config_override_file = os.getenv("SHARED_STATE_FILE", "logs/shared_state.json")
        self._pending_mode: str | None = None
        self._pending_timeframe: str | None = None

        # Read shared state for initial values (dashboard can set these)
        # NOTE: we only read timeframe from shared state, not mode.
        # Mode changes are a security-sensitive decision - the runner should only
        # enter live/demo mode via explicit configuration, not auto-overwritten.
        try:
            state = get_shared_state()
            if state.timeframe and state.timeframe != timeframe:
                self._cached_timeframe = state.timeframe
        except Exception:
            pass

        self.data = MarketDataFetcher(
            exchange_name=exchange_name,
            fallback_exchanges=exchange_fallbacks,
            timeout_ms=exchange_timeout_ms,
            exchange_type=exchange_type,
        )

        self.model = DirectionModel.for_symbol(symbol)

        # Use ensemble factory for multi-model regime-aware weighting
        try:
            self.ensemble = EnsembleDirectionModel.for_symbol(symbol)
            self.model = self.ensemble
        except Exception as exc:
            # Fall back to single model if ensemble fails
            print(f"[WARNING] Ensemble failed for {symbol}: {exc}")
            self.ensemble = None

        self.cfg = config or StrategyConfig(cooldown_minutes=cooldown_minutes)
        if mode == "demo":
            self.cfg.demo_mode = True
            print(f"[DEMO MODE] Relaxed strategy filters enabled (ADX>=10, EMA/prob relaxed, no sideways block)")
        self.strategy = StrategyEngine(self.model, risk_per_trade, self.cfg)

        self.supervisor = AISupervisor()
        self.regime_ctrl = RegimeController()
        self.market_guard = MarketGuard()

        self.risk_limits = RiskLimits()
        self.risk_state = RiskState(starting_balance_usdt)

        self.broker = self._create_broker(self._cached_mode, exchange_name)

        # In demo/live mode, fetch real balance from broker instead of using hardcoded value
        if self._cached_mode in ("demo", "live") and hasattr(self.broker, "get_balance_usdt"):
            try:
                real_balance = self.broker.get_balance_usdt()
                starting_balance_usdt = real_balance
                print(f"[BROKER] Using live balance: ${real_balance:.2f}")
            except Exception as e:
                print(f"[BROKER] Could not fetch live balance: {e}, using configured: ${starting_balance_usdt:.2f}")

        self.risk_state = RiskState(starting_balance_usdt)

        # Log initial balance snapshot
        try:
            append_equity(datetime.utcnow().isoformat() + "Z", self.risk_state.current_balance)
        except Exception:
            pass
        self.logger = logger if logger is not None else TradeLogger()
        self.event_logger = event_logger if event_logger is not None else get_event_logger()
        self.report = DailyAIReport()

        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_trade_time: datetime | None = None
        self.last_processed_candle_time: pd.Timestamp | None = None
        self.last_fetch_wallclock: datetime | None = None

        self.last_entry_price: float | None = None
        self.last_entry_prob: float | None = None
        self.last_entry_side: str | None = None
        self.last_decision = None

        # Restore SL/TP from broker if position was persisted from previous run
        self.stop_loss: float | None = getattr(self.broker, '_cached_stop_loss', None)
        self.take_profit: float | None = getattr(self.broker, '_cached_take_profit', None)
        self.take_profit_1: float | None = None  # Will be recalculated on next candle
        self._profit_lock_activated = False

        # If we have a persisted position but missing SL/TP, we need to restore other state too
        if self.broker.position and self.stop_loss is not None and self.take_profit is not None:
            self.last_entry_price = self.broker.position.avg_entry
            self.last_entry_side = self.broker.position.side
            # Calculate take_profit_1 (50% of TP distance) for restored position
            if self.last_entry_price and self.take_profit:
                tp_distance = self.take_profit - self.last_entry_price
                self.take_profit_1 = self.last_entry_price + (tp_distance / 2.0)
            print(f"[RESTORED] {symbol} position: entry={self.last_entry_price:.4f}, SL={self.stop_loss:.4f}, TP={self.take_profit:.4f}, TP1={self.take_profit_1:.4f}")

        print(f"[AUTONOMOUS AI] {symbol} ready (mode={self._cached_mode}, timeframe={self._cached_timeframe})")

    def _create_broker(self, mode: str, exchange_name: str):
        """Create broker for the given mode."""
        if mode in ("live", "demo"):
            api_key = os.getenv("EXCHANGE_API_KEY", "")
            api_secret = os.getenv("EXCHANGE_API_SECRET", "")
            if not api_key or not api_secret:
                raise ValueError(
                    f"{mode} trading requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET env vars"
                )
            testnet = (mode == "demo")
            return LiveBroker(
                exchange_name=exchange_name,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                exchange_type=self._cached_exchange_type,
            )
        else:
            return ShadowBroker()

    def _check_config_updates(self):
        """Check shared state file for timeframe, pause, and strategy config changes.

        NOTE: Mode changes are intentionally NOT processed here.
        Switching between paper/shadow/live/demo affects broker creation
        and API credentials - we only want to process those changes when
        the user explicitly restarts the runner with the new mode.
        """
        if not os.path.exists(self._config_override_file):
            return

        try:
            state = get_shared_state()

            # Timeframe changes
            new_timeframe = state.timeframe
            if new_timeframe != self._cached_timeframe and new_timeframe:
                print(f"[CONFIG] Timeframe changed: {self._cached_timeframe} -> {new_timeframe}")
                self._cached_timeframe = new_timeframe
                self.last_fetch_wallclock = None
                self.last_processed_candle_time = None

            # Pause state
            if state.paused != getattr(self, "_cached_paused", False):
                self._cached_paused = state.paused
                if state.paused:
                    print("[CONFIG] Trading PAUSED")
                else:
                    print("[CONFIG] Trading RESUMED")

            # Strategy config changes
            new_risk = state.risk_per_trade
            if new_risk != getattr(self, "_cached_risk_per_trade", None) and new_risk:
                self._cached_risk_per_trade = new_risk
                self.strategy.risk_per_trade = new_risk
                print(f"[CONFIG] Risk per trade: {new_risk * 100:.2f}%")

            new_min_prob = state.strategy_min_prob
            if new_min_prob != getattr(self, "_cached_strategy_min_prob", None) and new_min_prob:
                self._cached_strategy_min_prob = new_min_prob
                self.cfg.base_long_threshold = new_min_prob
                print(f"[CONFIG] Min probability threshold: {new_min_prob:.2f}")

            new_cooldown = state.cooldown_minutes
            if new_cooldown != getattr(self, "_cached_cooldown_minutes", None) and new_cooldown:
                self._cached_cooldown_minutes = new_cooldown
                self.cooldown = timedelta(minutes=new_cooldown)
                print(f"[CONFIG] Cooldown: {new_cooldown}m")

        except Exception as e:
            print(f"[CONFIG] Error reading shared state: {e}")

    def _apply_pending_mode(self):
        """Apply any pending mode change after position closes."""
        if self._pending_mode and not self.broker.position:
            print(f"[CONFIG] Applying pending mode change: {self._cached_mode} -> {self._pending_mode}")
            self._cached_mode = self._pending_mode
            self._pending_mode = None
            self.broker = self._create_broker(self._cached_mode, self.data.exchange_name)

    def _should_skip_fetch(self) -> bool:
        if self.last_fetch_wallclock is None:
            return False

        elapsed = (datetime.utcnow() - self.last_fetch_wallclock).total_seconds()

        tf = (self._cached_timeframe or "").lower().strip()
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
        self._check_config_updates()

        # Check if paused via shared state
        try:
            state = get_shared_state()
            if state.paused:
                return
        except Exception:
            pass

        if self._should_skip_fetch():
            return

        self.last_fetch_wallclock = datetime.utcnow()

        raw_df = self.data.fetch_ohlcv(self.symbol, self._cached_timeframe, self.lookback + 5)

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
            reason = self.regime_ctrl.skip_reason(df) if hasattr(self.regime_ctrl, "skip_reason") else f"regime_block({regime})"
            print(f"[{self.symbol}] SKIP | {reason}")
            self.event_logger.skip(symbol=self.symbol, message=f"SKIP {self.symbol}: {reason}", reason=reason)
            return

        supervisor_dec = self.supervisor.decide()
        if not supervisor_dec.trade_allowed:
            reason = f"supervisor_block({supervisor_dec.reason})"
            print(f"[{self.symbol}] SKIP | {reason}")
            self.event_logger.skip(symbol=self.symbol, message=f"SKIP {self.symbol}: {reason}", reason=reason)
            return

        closed_price = float(df.iloc[-1]["close"])
        atr = float(df.iloc[-1]["atr"])

        if closed_price != closed_price or closed_price <= 0:  # NaN check or invalid price
            print(f"[{self.symbol}] SKIP | invalid closed_price={closed_price}")
            return

        live_price = None
        if hasattr(self.data, "fetch_last_price"):
            try:
                live_price = self.data.fetch_last_price(self.symbol)
            except Exception:
                live_price = None

        manage_price = live_price if live_price is not None else closed_price

        if self.broker.position:
            # Guard: skip management if TP/SL not yet initialized
            if self.take_profit_1 is None or self.stop_loss is None or self.take_profit is None:
                print(
                    f"[{self.symbol}] MG | price={_fmt_price(manage_price, self.symbol)} | "
                    f"TP1={self.take_profit_1} TP={self.take_profit} SL={self.stop_loss} | not ready, skipping"
                )
                return

            if not self._profit_lock_activated and manage_price >= self.take_profit_1:
                self._profit_lock_activated = True
                profit_lock_sl = self.last_entry_price + atr * 0.5
                self.stop_loss = max(self.stop_loss, profit_lock_sl)
                if hasattr(self.broker, 'update_levels'):
                    self.broker.update_levels(stop_loss=self.stop_loss, take_profit=self.take_profit)
                print(
                    f"[{self.symbol}] TP1 HIT | "
                    f"manage_price={_fmt_price(manage_price, self.symbol)} | "
                    f"lock_sl={_fmt_price(self.stop_loss, self.symbol)}"
                )

            if self._profit_lock_activated:
                new_sl = manage_price - self.cfg.trail_atr_mult * atr
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
                    if hasattr(self.broker, 'update_levels'):
                        self.broker.update_levels(stop_loss=self.stop_loss, take_profit=self.take_profit)
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
            reason = dec.reason
            print(f"[{self.symbol}] SKIP | {reason}")
            self.event_logger.skip(symbol=self.symbol, message=f"SKIP {self.symbol}: {reason}", reason=reason)
            return

        risk_mult = supervisor_dec.risk_multiplier * self.regime_ctrl.risk_multiplier(regime)

        qty = self.strategy.position_size(
            balance=self.risk_state.current_balance * risk_mult,
            entry_price=dec.price,
            stop_price=dec.stop_loss,
        )

        if qty <= 0:
            reason = "qty_zero"
            print(f"[{self.symbol}] SKIP | qty_zero")
            self.event_logger.skip(symbol=self.symbol, message=f"SKIP {self.symbol}: qty_zero", reason="qty_zero")
            return

        # Validate leverage
        if self.leverage < 1.0 or self.leverage > 125.0:
            reason = f"invalid_leverage_{self.leverage}x"
            print(f"[{self.symbol}] SKIP | {reason}")
            self.event_logger.skip(symbol=self.symbol, message=f"SKIP {self.symbol}: {reason}", reason=reason)
            return

        self.broker.open_position(dec.side, dec.price, qty, self.symbol, leverage=self.leverage)

        self.last_trade_time = datetime.utcnow()
        self.last_entry_price = dec.price
        self.last_entry_prob = dec.prob
        self.last_entry_side = dec.side
        self.last_decision = dec
        self.stop_loss = dec.stop_loss
        self.take_profit = dec.take_profit
        self.take_profit_1 = dec.price + dec.atr * (self.cfg.take_atr_mult / 2.0)
        self._profit_lock_activated = False

        # Persist SL/TP to broker state immediately
        if hasattr(self.broker, 'update_levels'):
            self.broker.update_levels(stop_loss=self.stop_loss, take_profit=self.take_profit)

        # Log open trade to trade store so frontend shows it immediately
        try:
            ts = datetime.utcnow().isoformat() + "Z"
            open_record = {
                "timestamp": ts,
                "symbol": self.symbol,
                "side": dec.side,
                "entry_price": dec.price,
                "avg_entry": dec.price,
                "exit_price": 0.0,
                "qty": qty,
                "pnl": 0.0,
                "balance": self.risk_state.current_balance,
                "prob": dec.prob,
                "threshold": dec.threshold,
                "atr": dec.atr,
                "atr_pct": dec.atr_pct,
                "adx": dec.adx,
                "regime": dec.regime,
                "stop_loss": dec.stop_loss,
                "take_profit": dec.take_profit,
                "exit_reason": "",
                "add_count": 0,
                "status": "OPEN",
            }
            append_trade(open_record)
        except Exception as exc:
            print(f"[TRADE STORE ERROR] {exc}")

        # Emit live event for trade open
        self.event_logger.trade_open(
            symbol=self.symbol,
            message=f"OPEN {dec.side} {self.symbol} qty={qty:.6f} @ {dec.price:.4f}",
            side=dec.side,
            entry_price=dec.price,
            qty=qty,
            prob=dec.prob,
            adx=dec.adx,
            regime=dec.regime,
        )

        print(
            f"[^^] OPEN {dec.side} {self.symbol} | "
            f"price={_fmt_price(dec.price, self.symbol)} | qty={qty:.6f} | "
            f"SL={_fmt_price(dec.stop_loss, self.symbol)} | "
            f"TP={_fmt_price(dec.take_profit, self.symbol)} | "
            f"prob={dec.prob:.3f} | adx={dec.adx:.1f} | regime={dec.regime}"
        )

    def _close_position(self, price: float, exit_reason: str):
        """Close position with proper fee and slippage accounting."""
        pos = self.broker.position
        add_count = pos.add_count
        avg_entry = pos.avg_entry
        total_qty = pos.qty
        dec = self.last_decision

        # Calculate raw PnL (before fees/slippage)
        pnl_raw = self.broker.close_position(price, self.symbol)
        
        # Deduct fees from PnL
        # Entry fee (already paid when opening)
        entry_fee_usd = avg_entry * total_qty * self.cfg.fee_pct_per_side
        # Exit fee
        exit_fee_usd = price * total_qty * self.cfg.fee_pct_per_side
        total_fees_usd = entry_fee_usd + exit_fee_usd
        
        # Deduct slippage from PnL
        slippage_usd = price * total_qty * self.cfg.slippage_pct_per_side
        
        # Net PnL after fees and slippage
        pnl_net = pnl_raw - total_fees_usd - slippage_usd
        
        # Use net PnL for balance tracking
        pnl = pnl_net

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

        # Log equity snapshot and trade to JSONL store
        try:
            ts = datetime.utcnow().isoformat() + "Z"
            append_equity(ts, self.risk_state.current_balance, pnl)
            trade_record = {
                "timestamp": ts,
                "symbol": self.symbol,
                "side": self.last_entry_side,
                "entry_price": self.last_entry_price,
                "avg_entry": avg_entry,
                "exit_price": price,
                "qty": total_qty,
                "pnl": pnl,
                "balance": self.risk_state.current_balance,
                "prob": self.last_entry_prob,
                "threshold": dec.threshold if dec else 0.0,
                "atr": dec.atr if dec else 0.0,
                "atr_pct": dec.atr_pct if dec else 0.0,
                "adx": dec.adx if dec else 0.0,
                "regime": dec.regime if dec else "",
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "exit_reason": exit_reason,
                "add_count": add_count,
                "status": "CLOSED",
            }
            append_trade(trade_record)
        except Exception as exc:
            print(f"[TRADE STORE ERROR] {exc}")

        # Emit live event for trade close
        self.event_logger.trade_close(
            symbol=self.symbol,
            message=f"CLOSE {self.symbol} {exit_reason.upper()} pnl={pnl:.4f}",
            side=self.last_entry_side,
            entry_price=self.last_entry_price,
            qty=total_qty,
            pnl=pnl,
            exit_reason=exit_reason,
            balance=self.risk_state.current_balance,
        )

        icon = "[##]" if exit_reason == "stop_loss" else "[*]"
        print(
            f"{icon} {self.symbol} {exit_reason.upper().replace('_', ' ')} | "
            f"pnl={pnl:.4f} | balance={self.risk_state.current_balance:.2f} | "
            f"avg_entry={_fmt_price(avg_entry, self.symbol)} | adds={add_count}"
        )

    def run_loop(self, sleep_seconds: int = 60):
        print(f"[>>] {self.symbol} loop started (timeframe={self._cached_timeframe}, mode={self._cached_mode}, sleep={sleep_seconds}s)")

        def _sig_handler(signum, frame):
            print(f"\n[{self.symbol}] Received signal {signum}, shutting down gracefully...")
            self._persist_state()
            print(f"[{self.symbol}] Shutdown complete.")
            raise KeyboardInterrupt("Graceful shutdown")

        old_sigint = signal.signal(signal.SIGINT, _sig_handler)
        old_sigterm = signal.signal(signal.SIGTERM, _sig_handler)

        try:
            while True:
                try:
                    self.run_once()
                except KeyboardInterrupt:
                    print(f"[{self.symbol}] Stopped by user")
                    break
                except Exception as exc:
                    import traceback
                    print(f"[{self.symbol}] run error: {exc}")
                    print(f"[{self.symbol}] traceback: {traceback.format_exc()[-500:]}")
                time.sleep(sleep_seconds)
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

    def _persist_state(self):
        """Persist open position and state before shutdown."""
        try:
            if hasattr(self.broker, '_persist_position'):
                self.broker._persist_position()
            print(f"[{self.symbol}] Position state persisted.")
        except Exception as exc:
            print(f"[{self.symbol}] Failed to persist state: {exc}")