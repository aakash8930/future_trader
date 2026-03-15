# execution/multi_runner.py

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
            exchange_name=settings.exchange_name,
            exchange_fallbacks=settings.exchange_fallbacks,
            exchange_timeout_ms=settings.exchange_timeout_ms,
        )

    # ----------------------------------
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

    # ----------------------------------
    def _ensure_runner(self, symbol: str):
        if symbol in self.runners:
            return

        if self.settings.require_model_quality and not self._model_quality_ok(symbol):
            print(f"[{symbol}] rejected by model-quality gate")
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

    # ----------------------------------
    def run_loop(self):
        print(f"🚀 Autonomous trading system started [MODE={self.settings.mode}]")

        while True:
            try:
                active_symbols = self.universe.refresh_if_needed()

                for symbol in active_symbols:
                    self._ensure_runner(symbol)

                for symbol, runner in list(self.runners.items()):
                    if symbol not in active_symbols:
                        continue
                    runner.run_once()

                time.sleep(self.settings.sleep_seconds)

            except KeyboardInterrupt:
                print("Stopped by user")
                break
            except RuntimeError as e:
                # Clean error for exchange/config issues
                error_msg = str(e)
                if "FETCHER" in error_msg or "exchange" in error_msg.lower():
                    print(f"\n❌ FATAL: {error_msg}")
                    print("\nSystem cannot start due to exchange connectivity issues.")
                    break
                # Re-raise other runtime errors
                raise
            except Exception as e:
                print(f"System error: {e}")
                time.sleep(30)