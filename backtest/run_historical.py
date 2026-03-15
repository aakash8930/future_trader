# backtest/run_historical.py

import sys
from pathlib import Path

import ccxt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backtest.simulator import HistoricalSimulator
except ModuleNotFoundError:
    from simulator import HistoricalSimulator


SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
CANDLES = 50_000


def fetch_history(symbol: str, timeframe: str, candles: int) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    ms_per_candle = 15 * 60 * 1000

    since = exchange.milliseconds() - candles * ms_per_candle
    all_bars: list[list] = []

    while len(all_bars) < candles:
        bars = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=since,
            limit=1000,
        )
        if not bars:
            break

        all_bars.extend(bars)
        since = bars[-1][0] + 1

    return pd.DataFrame(
        all_bars,
        columns=["time", "open", "high", "low", "close", "volume"],
    )


def main():
    print("Fetching historical data...")
    df = fetch_history(SYMBOL, TIMEFRAME, CANDLES)
    print(f"Fetched {len(df)} candles")

    model_dir = f"models/{SYMBOL.replace('/', '_')}"

    sim = HistoricalSimulator(
        model_path=f"{model_dir}/model.pt",
        scaler_path=f"{model_dir}/scaler.save",
        metadata_path=f"{model_dir}/metadata.json",
    )

    print("Running simulation...")
    for i in range(sim.lookback, len(df)):
        window = df.iloc[i - sim.lookback : i + 1]
        sim.step(window)

    sim.export("v2/data_outputs/v2_backtest_trades.csv")
    print("Simulation finished.")


if __name__ == "__main__":
    main()
