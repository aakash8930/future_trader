# backtest/optimize_threshold.py

import numpy as np
import pandas as pd

from backtest.vector_engine import VectorBacktestEngine


def optimize_long_threshold(
    symbol: str,
    model_path: str,
    scaler_path: str,
    metadata_path: str,
    timeframe: str = "15m",
    lookback: int = 300,
    limit: int = 20_000,
):
    bt = VectorBacktestEngine(
        symbol=symbol,
        timeframe=timeframe,
        model_path=model_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        lookback=lookback,
    )

    df = bt.run(limit=limit)

    results = []

    for th in np.arange(0.48, 0.61, 0.01):
        mask = (
            (df["prob_up"] >= th)
            & (df["atr_pct"] > 0.0015)
            & (df["adx"] >= 15)
            & (df["rsi"] >= 48)
            & (df["rsi"] <= 72)
        )

        position = mask.astype(int).shift(1).fillna(0)
        if int(position.sum()) < 50:
            continue

        ret = df["close"].pct_change().shift(-1).fillna(0.0)
        turn = position.diff().abs().fillna(position.abs())
        per_side_cost = bt.fee_pct + bt.slippage_pct
        net_ret = position * ret - turn * per_side_cost

        active = position > 0
        if int(active.sum()) < 50:
            continue

        trade_rets = net_ret[active]

        expectancy = float(trade_rets.mean())
        win_rate = float((trade_rets > 0).mean())

        equity = (1 + net_ret).cumprod().fillna(1.0)
        peak = equity.cummax()
        drawdown = (peak - equity) / peak.replace(0, pd.NA)
        max_dd = float(drawdown.fillna(0.0).max())

        score = expectancy / (max_dd + 1e-6)

        results.append({
            "threshold": round(th, 3),
            "trades": int(active.sum()),
            "expectancy": expectancy,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "score": score,
        })

    res = pd.DataFrame(results).sort_values("score", ascending=False)

    if res.empty:
        raise RuntimeError("No viable thresholds found.")

    return res.iloc[0], res
