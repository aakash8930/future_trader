#execution/multi_runner.py

import time
import json
from pathlib import Path

from execution.runner import TradingRunner
from execution.universe_manager import UniverseManager
from config.live import LiveSettings
from logs.logger import TradeLogger
from execution.strategy import StrategyConfig


class MultiSymbolTradingSystem:
    """
    Fully autonomous multi-symbol trading system.
    """

    def __init__(
        self,
        settings: LiveSettings,
        logger: TradeLogger | None = None,
        strategy_config: StrategyConfig | None = None,
    ):
        self.settings = settings
        self.logger = logger if logger is not None else TradeLogger()
        self.strategy_config = strategy_config
        self.runners: dict[str, TradingRunner] = {}

        self.universe = UniverseManager(
            all_symbols=settings.symbols,
            timeframe=settings.timeframe,
            max_active=settings.max_active_positions,
            refresh_minutes=settings.universe_refresh_minutes,
            selector_top_k_multiplier=settings.selector_top_k_multiplier,
            selector_min_atr_pct=settings.selector_min_atr_pct,
            selector_soft_min_volume_ratio=settings.selector_soft_min_volume_ratio,
            exchange_name=settings.exchange_name,
            exchange_fallbacks=settings.exchange_fallbacks,
            exchange_timeout_ms=settings.exchange_timeout_ms,
        )

    def _model_quality_ok(self, symbol: str) -> bool:
        metadata_path = Path("models") / symbol.replace("/", "_") / "metadata.json"
        if not metadata_path.exists():
            return False

        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
        except Exception:
            return False

        return (
            float(metrics.get("val_f1", 0.0)) >= self.settings.min_model_val_f1
            and float(metrics.get("val_precision", 0.0)) >= self.settings.min_model_val_precision
            and float(metrics.get("val_recall", 0.0)) >= self.settings.min_model_val_recall
        )

    def _filtered_active_symbols(self, symbols: list[str]) -> list[str]:
        filtered: list[str] = []

        for symbol in symbols:
            if self.settings.require_model_quality and not self._model_quality_ok(symbol):
                print(f"[{symbol}] rejected by model-quality gate")
                continue
            filtered.append(symbol)

        return filtered

    def _ensure_runner(self, symbol: str):
        if symbol in self.runners:
            return

        runner = TradingRunner(
            symbol=symbol,
            timeframe=self.settings.timeframe,
            lookback=self.settings.lookback,
            mode=self.settings.mode,
            starting_balance_usdt=self.settings.starting_balance_usdt,
            cooldown_minutes=self.settings.cooldown_minutes,
            risk_per_trade=self.settings.risk_per_trade,
            config=self.strategy_config,
            exchange_name=self.settings.exchange_name,
            exchange_fallbacks=self.settings.exchange_fallbacks,
            exchange_timeout_ms=self.settings.exchange_timeout_ms,
            logger=self.logger,
        )

        self.runners[symbol] = runner
        print(f"➕ Runner added for {symbol}")

    def _remove_inactive_runners(self, active_symbols: list[str]):
        inactive = [symbol for symbol in self.runners if symbol not in active_symbols]
        for symbol in inactive:
            del self.runners[symbol]
            print(f"➖ Runner removed for {symbol}")

    def run_loop(self):
        print(f"🚀 Autonomous trading system started [MODE={self.settings.mode}]")

        while True:
            try:
                active_symbols = self.universe.refresh_if_needed()
                active_symbols = self._filtered_active_symbols(active_symbols)

                if not active_symbols:
                    self._remove_inactive_runners([])
                    print("[SYSTEM] no active tradable symbols → flat mode")
                    time.sleep(max(self.settings.sleep_seconds, 120))
                    continue

                for symbol in active_symbols:
                    self._ensure_runner(symbol)

                self._remove_inactive_runners(active_symbols)

                for symbol, runner in list(self.runners.items()):
                    if symbol not in active_symbols:
                        continue
                    runner.run_once()

                time.sleep(self.settings.sleep_seconds)

            except KeyboardInterrupt:
                print("Stopped by user")
                break

            except RuntimeError as e:
                error_msg = str(e)
                if "FETCHER" in error_msg or "exchange" in error_msg.lower():
                    print(f"\n❌ FATAL: {error_msg}")
                    print("\nSystem cannot start due to exchange connectivity issues.")
                    break
                raise

            except Exception as e:
                print(f"System error: {e}")
                time.sleep(30)