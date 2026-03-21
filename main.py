from config.env_loader import load_env_file
from config.live import LiveSettings
from execution.multi_runner import MultiSymbolTradingSystem
from execution.runner import TradingRunner
from execution.strategy import StrategyConfig
from logs.logger import TradeLogger


def build_strategy_config(settings: LiveSettings) -> StrategyConfig:
    return StrategyConfig(
        min_prob=settings.strategy_min_prob,
        min_adx=settings.strategy_min_adx,
        min_atr_pct=settings.strategy_min_atr_pct,
        rsi_long_min=settings.strategy_rsi_long_min,
        rsi_long_max=settings.strategy_rsi_long_max,
        fee_pct_per_side=settings.strategy_fee_pct_per_side,
        slippage_pct_per_side=settings.strategy_slippage_pct_per_side,
        stop_atr_mult=settings.strategy_stop_atr_mult,
        take_atr_mult=settings.strategy_take_atr_mult,
        trail_atr_mult=settings.strategy_trail_atr_mult,
        cooldown_minutes=settings.cooldown_minutes,
        min_expected_edge=-0.0020,  # settings.strategy_min_expected_edge,
        base_long_threshold=0.46,  # settings.strategy_base_long_threshold,
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

            # Inject selector/universe settings cleanly
            system.universe.refresh_seconds = settings.universe_refresh_minutes * 60
            system.universe.selector.top_k = settings.max_active_positions * settings.selector_top_k_multiplier
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