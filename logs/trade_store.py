import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, List

TRADE_LOG_PATH = Path("logs/trades_history.jsonl")
EQUITY_LOG_PATH = Path("logs/equity_curve.jsonl")
SYMBOLS_PATH   = Path("logs/symbols.json")


def _ensure_dir():
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []


def _write_jsonl(path: Path, records: List[dict]):
    _ensure_dir()
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


# ─── Equity ────────────────────────────────────────────────────────────────────

def append_equity(timestamp: str, balance: float, pnl: Optional[float] = None):
    """Append a balance snapshot to the equity curve log."""
    _ensure_dir()
    rec = {
        "timestamp": timestamp,
        "balance": round(balance, 6),
    }
    if pnl is not None:
        rec["pnl"] = round(pnl, 6)
    with open(EQUITY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def get_equity_curve(limit: int = 500) -> List[dict]:
    records = _read_jsonl(EQUITY_LOG_PATH)
    return records[-limit:]


def get_latest_balance() -> float:
    records = _read_jsonl(EQUITY_LOG_PATH)
    if not records:
        return 0.0
    return records[-1].get("balance", 0.0)


# ─── Trades ───────────────────────────────────────────────────────────────────

def append_trade(trade: dict):
    """Append a trade record to the trade history log."""
    _ensure_dir()
    rec = {
        "timestamp": trade.get("timestamp") or datetime.utcnow().isoformat() + "Z",
        "symbol": trade.get("symbol"),
        "side": trade.get("side"),
        "entry_price": trade.get("entry_price"),
        "avg_entry": trade.get("avg_entry"),
        "exit_price": trade.get("exit_price"),
        "qty": trade.get("qty"),
        "pnl": trade.get("pnl"),
        "balance": trade.get("balance"),
        "prob": trade.get("prob"),
        "threshold": trade.get("threshold"),
        "atr": trade.get("atr"),
        "atr_pct": trade.get("atr_pct"),
        "adx": trade.get("adx"),
        "regime": trade.get("regime"),
        "stop_loss": trade.get("stop_loss"),
        "take_profit": trade.get("take_profit"),
        "exit_reason": trade.get("exit_reason"),
        "add_count": trade.get("add_count", 0),
        "status": trade.get("status", "CLOSED"),
    }
    with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    _update_symbols_from_trade(rec)


def _update_symbols_from_trade(trade: dict):
    sym = trade.get("symbol")
    if not sym:
        return
    symbols = set(_read_jsonl(SYMBOLS_PATH))
    symbols.add(sym)
    with open(SYMBOLS_PATH, "w", encoding="utf-8") as f:
        for s in sorted(symbols):
            f.write(json.dumps(s) + "\n")


def get_trades(
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    exit_reason: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[dict]:
    records = _read_jsonl(TRADE_LOG_PATH)

    # Filter
    if symbol:
        records = [r for r in records if r.get("symbol") == symbol]
    if side:
        records = [r for r in records if r.get("side") == side]
    if exit_reason:
        records = [r for r in records if r.get("exit_reason") == exit_reason]

    total = len(records)
    records = records[offset:offset + limit]

    # Attach id-like index for compatibility
    for i, r in enumerate(records):
        r["id"] = offset + i + 1

    return records


def get_distinct_symbols() -> List[str]:
    records = _read_jsonl(TRADE_LOG_PATH)
    symbols = sorted(set(r.get("symbol") for r in records if r.get("symbol")))
    return symbols


# ─── Stats ────────────────────────────────────────────────────────────────────

def get_performance_summary() -> dict:
    records = _read_jsonl(TRADE_LOG_PATH)

    if not records:
        return _empty_stats()

    trades = [r for r in records if r.get("pnl") is not None]
    wins   = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]

    total_trades = len(trades)
    wins_count   = len(wins)
    losses_count = len(losses)

    win_rate = wins_count / total_trades if total_trades > 0 else 0.0

    total_pnl = sum(t.get("pnl", 0) for t in trades)

    win_pnls  = [t.get("pnl", 0) for t in wins]
    loss_pnls = [abs(t.get("pnl", 0)) for t in losses]

    avg_win  = sum(win_pnls)  / len(win_pnls)  if win_pnls  else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0

    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if total_trades > 0 else 0.0

    total_win_pnl  = sum(win_pnls)
    total_loss_pnl = sum(loss_pnls)
    profit_factor  = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 0.0

    # Max drawdown from equity curve
    equity = _read_jsonl(EQUITY_LOG_PATH)
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    peak = 0.0
    for r in equity:
        bal = r.get("balance", 0)
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > max_drawdown:
            max_drawdown = dd
        dd_pct = dd / peak if peak > 0 else 0
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct

    max_win   = max((t.get("pnl", 0) for t in wins),  default=0.0)
    max_loss  = max((t.get("pnl", 0) for t in losses), default=0.0)

    return {
        "total_trades": total_trades,
        "wins": wins_count,
        "losses": losses_count,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "expectancy": round(expectancy, 6),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_drawdown, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "total_pnl": round(total_pnl, 4),
        "max_single_win": round(max_win, 6),
        "max_single_loss": round(max_loss, 6),
    }


def _empty_stats():
    return {
        "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "expectancy": 0.0, "profit_factor": 0.0,
        "max_drawdown": 0.0, "max_drawdown_pct": 0.0, "total_pnl": 0.0,
        "max_single_win": 0.0, "max_single_loss": 0.0,
    }


# ─── Symbols ──────────────────────────────────────────────────────────────────

def get_symbols() -> List[str]:
    return get_distinct_symbols()


def update_symbols(symbols: List[str]):
    _ensure_dir()
    with open(SYMBOLS_PATH, "w", encoding="utf-8") as f:
        for s in sorted(set(symbols)):
            f.write(json.dumps(s) + "\n")