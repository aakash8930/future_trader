
# metrics/performance.py

import pandas as pd


def performance_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    win_rate = len(wins) / len(df)
    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl"].abs().mean() if not losses.empty else 0.0

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    equity = df["balance"].astype(float)
    peak = equity.cummax()
    drawdown = (peak - equity) / peak.replace(0, pd.NA)
    max_dd = drawdown.fillna(0.0).max()

    return {
        "trades": len(df),
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "max_drawdown": round(float(max_dd), 3),
    }

