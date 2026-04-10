#main.py

from config.env_loader import load_env_file
from config.live import LiveSettings
from execution.multi_runner import MultiSymbolTradingSystem
from execution.runner import TradingRunner
from execution.strategy import StrategyConfig
from logs.logger import TradeLogger


def build_strategy_config(settings: LiveSettings) -> StrategyConfig:
    return StrategyConfig(
        min_adx=20.0,
        min_atr_pct=0.0010,
        rsi_long_min=40.0,
        rsi_long_max=75.0,
        fee_pct_per_side=settings.strategy_fee_pct_per_side,
        slippage_pct_per_side=settings.strategy_slippage_pct_per_side,
        stop_atr_mult=1.30,
        take_atr_mult=3.80,
        trail_activate_atr_mult=1.0,
        trail_atr_mult=1.0,
        cooldown_minutes=settings.cooldown_minutes,
        min_expected_edge=0.00005,
        base_long_threshold=0.49,
    )


def main():
    load_env_file()

    settings = LiveSettings.from_env()
    settings.validate()

    logger = TradeLogger()
    strategy_cfg = build_strategy_config(settings)

    try:
        if len(settings.symbols) > 1:
            system = MultiSymbolTradingSystem(
                settings=settings,
                logger=logger,
                strategy_config=strategy_cfg,
            )

            system.universe.refresh_seconds = settings.universe_refresh_minutes * 60
            system.universe.selector.top_k = (
                settings.max_active_positions * settings.selector_top_k_multiplier
            )
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
            print(f"\n{'=' * 60}")
            print("❌ FATAL ERROR - Cannot Start Trading System")
            print("=" * 60)
            print(error_msg)
            print("=" * 60)
            print("\nPossible solutions:")
            print("  1. Set EXCHANGE_NAME to a different exchange")
            print("  2. Set EXCHANGE_FALLBACKS=kraken,okx,coinbase")
            print("  3. Deploy to a different region if geo-blocked")
            print("=" * 60)
            return
        raise


if __name__ == "__main__":
    main()
