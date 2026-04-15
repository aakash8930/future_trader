import json
import os
from pathlib import Path
from typing import Optional

POSITION_STATE_PATH = Path("logs/position_state.json")


def get_position_state() -> dict:
    """Read the current position state from logs/position_state.json."""
    if not POSITION_STATE_PATH.exists():
        return _empty_position()

    try:
        with open(POSITION_STATE_PATH, "r") as f:
            data = json.load(f)

        # Check if there's an active position
        pos = data.get("position")
        if not pos or pos.get("qty", 0) == 0:
            return _empty_position()

        side = pos.get("side", "")
        entry_price = pos.get("avg_entry") or pos.get("entry_price", 0)
        qty = pos.get("qty", 0)
        current_price = data.get("current_price") or entry_price

        # Calculate unrealized P&L
        unrealized_pnl = None
        pnl_pct = None
        if entry_price and qty and current_price:
            if side == "LONG":
                unrealized_pnl = (current_price - entry_price) * qty
            elif side == "SHORT":
                unrealized_pnl = (entry_price - current_price) * qty
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0

        return {
            "has_position": True,
            "symbol": data.get("symbol", ""),
            "side": side,
            "entry_price": entry_price,
            "avg_entry": pos.get("avg_entry"),
            "qty": qty,
            "entry_time": pos.get("entry_time"),
            "add_count": pos.get("add_count", 0),
            "stop_loss": data.get("stop_loss"),
            "take_profit": data.get("take_profit"),
            "current_price": current_price,
            "unrealized_pnl": round(unrealized_pnl, 6) if unrealized_pnl is not None else None,
            "pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
        }
    except Exception:
        return _empty_position()


def _empty_position() -> dict:
    return {
        "has_position": False,
        "symbol": None,
        "side": None,
        "entry_price": None,
        "avg_entry": None,
        "qty": None,
        "entry_time": None,
        "add_count": 0,
        "stop_loss": None,
        "take_profit": None,
        "current_price": None,
        "unrealized_pnl": None,
        "pnl_pct": None,
    }


def get_active_positions() -> list:
    """Get all active positions from position_state.json."""
    # The position_state.json only holds one position at a time
    pos = get_position_state()
    if pos["has_position"]:
        return [pos]
    return []
