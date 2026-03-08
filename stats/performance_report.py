# stats/performance_report.py
"""
Trading performance report - reads from Railway PostgreSQL trades table.

Usage:
    railway run .venv/Scripts/python.exe stats/performance_report.py
    railway run .venv/Scripts/python.exe stats/performance_report.py --csv report.csv
"""

import os
import sys
import argparse
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import pandas as pd


# ─────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────

def get_connection():
    url = os.getenv("DATABASE_PUBLIC_URL") or os.getenv("DATABASE_URL")
    if not url:
        sys.exit("[ERROR] No DATABASE_PUBLIC_URL or DATABASE_URL set.")
    return psycopg2.connect(url)


def load_trades(conn) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                id, timestamp, symbol, side,
                entry_price, avg_entry, exit_price,
                qty, pnl, balance,
                prob, threshold, atr, atr_pct, adx,
                regime, stop_loss, take_profit,
                exit_reason, add_count
            FROM trades
            WHERE pnl IS NOT NULL
            ORDER BY timestamp ASC;
        """)
        rows = cur.fetchall()

    if not rows:
        sys.exit("No trades found in the database.")

    return pd.DataFrame([dict(r) for r in rows])


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def win_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return (series > 0).sum() / len(series)


def avg_win(series: pd.Series) -> float:
    wins = series[series > 0]
    return wins.mean() if not wins.empty else 0.0


def avg_loss(series: pd.Series) -> float:
    losses = series[series <= 0]
    return losses.mean() if not losses.empty else 0.0


def expectancy(series: pd.Series) -> float:
    """E = (win_rate × avg_win) + (loss_rate × avg_loss)"""
    wr   = win_rate(series)
    lr   = 1.0 - wr
    return wr * avg_win(series) + lr * avg_loss(series)


def profit_factor(series: pd.Series) -> float:
    gross_profit = series[series > 0].sum()
    gross_loss   = abs(series[series <= 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def max_drawdown(balance_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown in absolute terms."""
    peak = balance_series.cummax()
    dd   = peak - balance_series
    return dd.max()


def max_drawdown_pct(balance_series: pd.Series) -> float:
    peak = balance_series.cummax()
    dd   = (peak - balance_series) / peak
    return dd.max() * 100


# ─────────────────────────────────────────────
# Report sections
# ─────────────────────────────────────────────

SEP  = "─" * 60
SEP2 = "═" * 60

def section(title: str):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


def fmt(val, fmt_str=".4f"):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    return format(val, fmt_str)


def print_overall(df: pd.DataFrame):
    section("OVERALL PERFORMANCE")

    pnl = df["pnl"]
    bal = df["balance"]

    print(f"  Period              : {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    print(f"  Total trades        : {len(df)}")
    print(f"  Winning trades      : {(pnl > 0).sum()}")
    print(f"  Losing trades       : {(pnl <= 0).sum()}")
    print(f"  Win rate            : {win_rate(pnl)*100:.2f}%")
    print(f"{SEP}")
    print(f"  Total PnL           : {pnl.sum():.4f} USDT")
    print(f"  Average win         : {avg_win(pnl):.4f} USDT")
    print(f"  Average loss        : {avg_loss(pnl):.4f} USDT")
    print(f"  Expectancy          : {expectancy(pnl):.4f} USDT/trade")
    print(f"  Profit factor       : {fmt(profit_factor(pnl))}")
    print(f"{SEP}")
    print(f"  Starting balance    : {bal.iloc[0]:.2f} USDT")
    print(f"  Final balance       : {bal.iloc[-1]:.2f} USDT")
    print(f"  Net return          : {bal.iloc[-1] - bal.iloc[0]:.2f} USDT  "
          f"({(bal.iloc[-1]/bal.iloc[0]-1)*100:.2f}%)")
    print(f"  Max drawdown        : {max_drawdown(bal):.2f} USDT  "
          f"({max_drawdown_pct(bal):.2f}%)")


def print_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    section("PER-SYMBOL BREAKDOWN")

    rows = []
    for symbol, g in df.groupby("symbol"):
        pnl = g["pnl"]
        rows.append({
            "symbol":      symbol,
            "trades":      len(g),
            "win_rate_%":  round(win_rate(pnl) * 100, 2),
            "total_pnl":   round(pnl.sum(), 4),
            "avg_win":     round(avg_win(pnl), 4),
            "avg_loss":    round(avg_loss(pnl), 4),
            "expectancy":  round(expectancy(pnl), 4),
            "profit_factor": round(profit_factor(pnl), 3) if profit_factor(pnl) != float("inf") else "∞",
        })

    sym_df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
    print(sym_df.to_string(index=False))
    return sym_df


def print_exit_reasons(df: pd.DataFrame) -> pd.DataFrame:
    section("EXIT REASON BREAKDOWN")

    if "exit_reason" not in df.columns or df["exit_reason"].isna().all():
        print("  exit_reason column empty — no data.")
        return pd.DataFrame()

    rows = []
    for reason, g in df.groupby("exit_reason"):
        pnl = g["pnl"]
        rows.append({
            "exit_reason": reason,
            "count":       len(g),
            "win_rate_%":  round(win_rate(pnl) * 100, 2),
            "total_pnl":   round(pnl.sum(), 4),
            "avg_pnl":     round(pnl.mean(), 4),
        })

    er_df = pd.DataFrame(rows).sort_values("count", ascending=False)
    print(er_df.to_string(index=False))
    return er_df


def print_entry_quality(df: pd.DataFrame):
    section("ENTRY QUALITY METRICS")

    def safe_mean(col):
        if col not in df.columns or df[col].isna().all():
            return "N/A"
        return f"{df[col].mean():.4f}"

    print(f"  Avg probability at entry  : {safe_mean('prob')}")
    print(f"  Avg threshold at entry    : {safe_mean('threshold')}")
    print(f"  Avg ADX at entry          : {safe_mean('adx')}")
    print(f"  Avg ATR% at entry         : {safe_mean('atr_pct')}")
    print(f"  Avg add_count (pyramiding): {safe_mean('add_count')}")

    # Prob buckets
    if "prob" in df.columns and not df["prob"].isna().all():
        print(f"\n  Win rate by probability bucket:")
        bins   = [0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
        labels = ["0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75+"]
        df2 = df.copy()
        df2["prob_bucket"] = pd.cut(df2["prob"], bins=bins, labels=labels, right=False)
        for bucket, g in df2.groupby("prob_bucket", observed=True):
            wr = win_rate(g["pnl"])
            print(f"    {bucket:12s}  trades={len(g):4d}  win_rate={wr*100:.1f}%  "
                  f"avg_pnl={g['pnl'].mean():.4f}")


def print_regime_breakdown(df: pd.DataFrame):
    if "regime" not in df.columns or df["regime"].isna().all():
        return

    section("REGIME BREAKDOWN")
    rows = []
    for regime, g in df.groupby("regime"):
        pnl = g["pnl"]
        rows.append({
            "regime":    regime,
            "trades":    len(g),
            "win_rate_%": round(win_rate(pnl) * 100, 2),
            "total_pnl": round(pnl.sum(), 4),
            "expectancy": round(expectancy(pnl), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────

def export_csv(df: pd.DataFrame, sym_df: pd.DataFrame, er_df: pd.DataFrame, path: str):
    summary_rows = []

    pnl = df["pnl"]
    bal = df["balance"]

    summary_rows.append({"section": "OVERALL", "metric": "total_trades",     "value": len(df)})
    summary_rows.append({"section": "OVERALL", "metric": "win_rate_%",        "value": round(win_rate(pnl)*100, 2)})
    summary_rows.append({"section": "OVERALL", "metric": "total_pnl",         "value": round(pnl.sum(), 4)})
    summary_rows.append({"section": "OVERALL", "metric": "avg_win",           "value": round(avg_win(pnl), 4)})
    summary_rows.append({"section": "OVERALL", "metric": "avg_loss",          "value": round(avg_loss(pnl), 4)})
    summary_rows.append({"section": "OVERALL", "metric": "expectancy",        "value": round(expectancy(pnl), 4)})
    pf = profit_factor(pnl)
    summary_rows.append({"section": "OVERALL", "metric": "profit_factor",     "value": round(pf, 3) if pf != float("inf") else "inf"})
    summary_rows.append({"section": "OVERALL", "metric": "max_drawdown_usdt", "value": round(max_drawdown(bal), 4)})
    summary_rows.append({"section": "OVERALL", "metric": "max_drawdown_%",    "value": round(max_drawdown_pct(bal), 2)})
    summary_rows.append({"section": "OVERALL", "metric": "final_balance",     "value": round(bal.iloc[-1], 2)})

    overall_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(path.replace(".csv", ".xlsx"), engine=None) if False else open(path, "w") as _:
        pass  # just to clear the file

    overall_df.to_csv(path, index=False)

    # Append per-symbol sheet as separate CSV
    sym_path = path.replace(".csv", "_by_symbol.csv")
    if not sym_df.empty:
        sym_df.to_csv(sym_path, index=False)
        print(f"\n  Symbol CSV  → {sym_path}")

    er_path = path.replace(".csv", "_exit_reasons.csv")
    if not er_df.empty:
        er_df.to_csv(er_path, index=False)
        print(f"  Exit CSV    → {er_path}")

    print(f"  Summary CSV → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trading performance report")
    parser.add_argument("--csv", metavar="PATH", default=None,
                        help="Export CSV summary to this path (e.g. report.csv)")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  AI CRYPTO TRADER — PERFORMANCE REPORT")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'═'*60}")

    conn = get_connection()
    df   = load_trades(conn)
    conn.close()

    print(f"\n  Loaded {len(df)} trades from database.")

    print_overall(df)
    sym_df = print_by_symbol(df)
    er_df  = print_exit_reasons(df)
    print_entry_quality(df)
    print_regime_breakdown(df)

    print(f"\n{SEP2}\n")

    if args.csv:
        export_csv(df, sym_df, er_df, args.csv)


if __name__ == "__main__":
    main()
