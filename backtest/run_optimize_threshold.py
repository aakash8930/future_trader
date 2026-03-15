# backtest/run_optimize_threshold.py

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backtest.optimize_threshold import optimize_long_threshold
except ModuleNotFoundError:
    from optimize_threshold import optimize_long_threshold


def main():
    symbol = "BTC/USDT"
    model_dir = "models/BTC_USDT"

    best, full = optimize_long_threshold(
        symbol=symbol,
        model_path=f"{model_dir}/model.pt",
        scaler_path=f"{model_dir}/scaler.save",
        metadata_path=f"{model_dir}/metadata.json",
    )

    print("\n===== BEST THRESHOLD =====")
    print(best)

    print("\n===== TOP 5 =====")
    print(full.head())


if __name__ == "__main__":
    main()

# backtest/run_vector.py

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backtest.vector_engine import VectorBacktestEngine
except ModuleNotFoundError:
    from vector_engine import VectorBacktestEngine



def main():
    bt = VectorBacktestEngine(
        symbol="BTC/USDT",
        timeframe="15m",
        model_path="models/BTC_USDT/model.pt",
        scaler_path="models/BTC_USDT/scaler.save",
        metadata_path="models/BTC_USDT/metadata.json",
    )
    


    df = bt.run(limit=20_000)
    output_path = Path("v2/data_outputs/vector_backtest.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    trades = df[df["signal"] != 0]

    win_rate = (trades["net_ret"] > 0).mean()
    expectancy = trades["net_ret"].mean()

    peak = df["equity"].cummax()
    drawdown = (peak - df["equity"]) / peak

    print("========== VECTOR BACKTEST ==========")
    print(f"Signals: {len(trades)}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Expectancy: {expectancy:.5f}")
    print(f"Max DD: {drawdown.max():.3f}")
    print(f"Final equity: {df['equity'].iloc[-1]:.2f}")

    df.to_csv("v2/data_outputs/vector_backtest.csv", index=False)


if __name__ == "__main__":
    main()
