# main.py

from config.env_loader import load_env_file
from config.live import LiveSettings
from execution.runner import TradingRunner
from execution.multi_runner import MultiSymbolTradingSystem
from logs.logger import TradeLogger


def main():
    # Load .env
    load_env_file()

    # Load settings
    settings = LiveSettings.from_env()
    settings.validate()

    # Create single shared logger
    logger = TradeLogger()

    try:
        # -------------------------------
        # MULTI-SYMBOL AUTONOMOUS MODE
        # -------------------------------
        if len(settings.symbols) > 1:
            system = MultiSymbolTradingSystem(settings, logger)
            system.run_loop()
            return

        # -------------------------------
        # SINGLE-SYMBOL MODE
        # -------------------------------
        symbol = settings.symbols[0]

        runner = TradingRunner(
            symbol=symbol,
            timeframe=settings.timeframe,
            lookback=settings.lookback,
            mode=settings.mode,
            starting_balance_usdt=settings.starting_balance_usdt,
            cooldown_minutes=settings.cooldown_minutes,
            risk_per_trade=settings.risk_per_trade,
            exchange_name=settings.exchange_name,
            exchange_fallbacks=settings.exchange_fallbacks,
            exchange_timeout_ms=settings.exchange_timeout_ms,
            logger=logger,
        )

        runner.run_loop(sleep_seconds=settings.sleep_seconds)

    except RuntimeError as e:
        # Clean fatal error messages (exchange issues, etc.)
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
        # Re-raise other runtime errors
        raise


if __name__ == "__main__":
    main()