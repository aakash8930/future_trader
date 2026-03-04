import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.vector_engine import VectorBacktestEngine


def run_for_symbol(symbol: str):
    print(f"\n========== BACKTESTING {symbol} ==========\n")

    model_dir = f"models/{symbol.replace('/', '_')}"

    bt = VectorBacktestEngine(
        symbol=symbol,
        timeframe="15m",
        model_path=f"{model_dir}/model.pt",
        scaler_path=f"{model_dir}/scaler.save",
    )

    df = bt.run(limit=20_000)

    trades = df[df["signal"] != 0]

    if trades.empty:
        print("No trades generated.")
        return

    win_rate = (trades["net_ret"] > 0).mean()
    expectancy = trades["net_ret"].mean()

    peak = df["equity"].cummax()
    drawdown = (peak - df["equity"]) / peak

    print(f"Total trades: {len(trades)}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Expectancy: {expectancy:.5f}")
    print(f"Max DD: {drawdown.max():.3f}")
    print(f"Final equity: {df['equity'].iloc[-1]:.2f}")


if __name__ == "__main__":
    run_for_symbol("BTC/USDT")
    run_for_symbol("ETH/USDT")
    run_for_symbol("MATIC/USDT")