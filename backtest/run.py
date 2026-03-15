# backtest/run.py

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backtest.engine import BacktestEngine
except ModuleNotFoundError:
    from engine import BacktestEngine


def main():
    symbol = "BTC/USDT"
    model_dir = f"models/{symbol.replace('/', '_')}"

    bt = BacktestEngine(
        symbol=symbol,
        timeframe="15m",
        model_path=f"{model_dir}/model.pt",
        scaler_path=f"{model_dir}/scaler.save",
        metadata_path=f"{model_dir}/metadata.json",
    )

    df = bt.run(limit=1000)
    print(df.head())
    print("Signals generated:", len(df))


if __name__ == "__main__":
    main()

