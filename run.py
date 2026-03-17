# run.py

import os
import sys

from config.env_loader import load_env_file
from config.live import LiveSettings
from execution.runner import TradingRunner
from execution.multi_runner import MultiSymbolTradingSystem
from execution.strategy import StrategyConfig
from logs.logger import TradeLogger


def _ensure_project_root():
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


def main():
    _ensure_project_root()
    load_env_file()

    settings = LiveSettings.from_env()
    settings.validate()

    logger = TradeLogger()

    strategy_cfg = StrategyConfig(
        cooldown_minutes=settings.cooldown_minutes,
        rsi_long_min=settings.strategy_rsi_long_min,
        rsi_long_max=settings.strategy_rsi_long_max,
        fee_pct_per_side=settings.strategy_fee_pct_per_side,
        slippage_pct_per_side=settings.strategy_slippage_pct_per_side,
        min_expected_edge=settings.strategy_min_expected_edge,
    )

    try:
        if len(settings.symbols) > 1:
            system = MultiSymbolTradingSystem(settings, logger=logger, strategy_config=strategy_cfg)
            system.run_loop()
            return

        symbol = settings.symbols[0]

        runner = TradingRunner(
            symbol=symbol,
            timeframe=settings.timeframe,
            lookback=settings.lookback,
            mode=settings.mode,
            starting_balance_usdt=settings.starting_balance_usdt,
            cooldown_minutes=settings.cooldown_minutes,
            risk_per_trade=settings.risk_per_trade,
            config=strategy_cfg,
            exchange_name=settings.exchange_name,
            exchange_fallbacks=settings.exchange_fallbacks,
            exchange_timeout_ms=settings.exchange_timeout_ms,
            logger=logger,
        )

        runner.run_loop(sleep_seconds=settings.sleep_seconds)

    except RuntimeError as e:
        error_msg = str(e)
        if "FETCHER" in error_msg or "exchange" in error_msg.lower():
            print(f"\n{'='*60}")
            print("❌ FATAL ERROR - Cannot Start Trading System")
            print('='*60)
            print(error_msg)
            print('='*60)
            print("\nPossible solutions:")
            print("  1. Set EXCHANGE_NAME=bybit (or another exchange)")
            print("  2. Set EXCHANGE_FALLBACKS=kraken,okx,coinbase")
            print("  3. Deploy to a different region if geo-blocked")
            print('='*60)
            return
        raise


if __name__ == "__main__":
    main()