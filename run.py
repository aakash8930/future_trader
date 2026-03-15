# run.py

import os
import sys

from config.env_loader import load_env_file
from config.live import LiveSettings
from execution.runner import TradingRunner
from execution.multi_runner import MultiSymbolTradingSystem
from execution.strategy import StrategyConfig


def _ensure_project_root():
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


def main():
    _ensure_project_root()
    load_env_file()

    settings = LiveSettings.from_env()
    settings.validate()

    strategy_cfg = StrategyConfig(
        cooldown_minutes=settings.cooldown_minutes,
        rsi_long_min=settings.strategy_rsi_long_min,
        rsi_long_max=settings.strategy_rsi_long_max,
        fee_pct_per_side=settings.strategy_fee_pct_per_side,
        slippage_pct_per_side=settings.strategy_slippage_pct_per_side,
        min_expected_edge=settings.strategy_min_expected_edge,
    )

    if len(settings.symbols) > 1:
        system = MultiSymbolTradingSystem(settings, strategy_config=strategy_cfg)
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
    )

    runner.run_loop(sleep_seconds=settings.sleep_seconds)


if __name__ == "__main__":
    main()