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
