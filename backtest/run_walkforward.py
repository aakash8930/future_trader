# backtest/run_walkforward.py

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
TOTAL_CANDLES = 50_000

TRAIN_SIZE = 8_000
TEST_SIZE = 3_000


def fetch_history() -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.milliseconds() - TOTAL_CANDLES * 15 * 60 * 1000

    all_bars: list[list] = []

    while len(all_bars) < TOTAL_CANDLES:
        bars = exchange.fetch_ohlcv(
            SYMBOL,
            TIMEFRAME,
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


def compute_metrics(trades: pd.DataFrame, initial_balance: float) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "max_dd": 0.0,
            "final_equity": float(initial_balance),
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = float(wins["pnl"].mean()) if not wins.empty else 0.0
    avg_loss = abs(float(losses["pnl"].mean())) if not losses.empty else 0.0

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    equity = trades["balance"].astype(float)
    peak = equity.cummax()
    drawdown = (peak - equity) / peak.replace(0, pd.NA)
    max_dd = float(drawdown.fillna(0.0).max())
    final_equity = float(equity.iloc[-1]) if not equity.empty else float(initial_balance)

    return {
        "trades": len(trades),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "max_dd": max_dd,
        "final_equity": final_equity,
    }


def main():
    print("Fetching historical data...")
    df = fetch_history()
    print(f"Fetched {len(df)} candles")

    results = []
    start = 0
    segment = 1

    while start + TRAIN_SIZE + TEST_SIZE <= len(df):
        test_df = df.iloc[start + TRAIN_SIZE : start + TRAIN_SIZE + TEST_SIZE]

        print(f"\n--- Walk Forward Segment {segment} ---")

        model_dir = f"models/{SYMBOL.replace('/', '_')}"
        initial_balance = 500.0

        sim = HistoricalSimulator(
            model_path=f"{model_dir}/model.pt",
            scaler_path=f"{model_dir}/scaler.save",
            metadata_path=f"{model_dir}/metadata.json",
            starting_balance=initial_balance,
        )

        for i in range(sim.lookback, len(test_df)):
            window = test_df.iloc[i - sim.lookback : i + 1]
            sim.step(window)

        trades = pd.DataFrame(sim.trades)
        metrics = compute_metrics(trades, initial_balance=initial_balance)
        results.append({"segment": segment, **metrics})

        print(
            f"Trades={metrics['trades']} | "
            f"WinRate={metrics['win_rate']:.3f} | "
            f"Expectancy={metrics['expectancy']:.4f} | "
            f"MaxDD={metrics['max_dd']:.4f} | "
            f"FinalEquity={metrics['final_equity']:.2f}"
        )

        start += TEST_SIZE
        segment += 1

    summary = pd.DataFrame(results)
    print("\n========== SUMMARY ==========")
    print(summary)

    if not summary.empty:
        print("\nAverage Expectancy:", summary["expectancy"].mean())
        print("Worst Drawdown:", summary["max_dd"].max())


if __name__ == "__main__":
    main()

