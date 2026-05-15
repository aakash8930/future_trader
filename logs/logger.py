"""
Trade logger for RUDRA-ALPHA trading system.
Logs trades to SQLite database using the execution.database module.
If database is not available, falls back to JSONL logging (disabled by default).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from execution.database import get_db

# JSONL fallback paths (disabled by default)
TRADE_LOG_PATH = Path("logs/trades_history.jsonl")
EQUITY_LOG_PATH = Path("logs/equity_curve.jsonl")
SYMBOLS_PATH   = Path("logs/symbols.json")

# Flag to enable JSONL fallback (enabled to ensure guaranteed persistence)
ENABLE_JSONL_FALLBACK = True


class TradeLogger:
    """
    Logs trade records to SQLite database.
    Falls back to JSONL if database is not available and ENABLE_JSONL_FALLBACK is True.
    """

    def __init__(self):
        self.db = None
        self._lock = Lock()
        self._init_database()
        if self.db is None and ENABLE_JSONL_FALLBACK:
            print("[LOGGER] Database not available — falling back to JSONL trade logging.")
        elif self.db is None:
            print("[LOGGER] Database not available — trade logging disabled (JSONL fallback also disabled).")
        else:
            print("[LOGGER] Trade logger initialized with SQLite database.")

    def _init_database(self):
        """Initialize the database connection."""
        try:
            self.db = get_db()
            # Database migration is handled by execution/database.py
            # The trades table will be created by migration v4
            print("[LOGGER] Connected to centralized database.")
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to initialize database: {e}")
            self.db = None

    def _ensure_connection(self) -> bool:
        """Check if the database connection is healthy."""
        if self.db is None:
            return False
        try:
            # Execute a simple query to check connection
            self.db.fetchone("SELECT 1;")
            return True
        except Exception:
            print("[LOGGER WARN] Database connection unhealthy — attempting to reconnect...")
            self._init_database()
            return self.db is not None

    def _write_jsonl_fallback(self, trade: dict):
        """Write trade to JSONL file as fallback."""
        if not ENABLE_JSONL_FALLBACK:
            return
        try:
            # Ensure directory exists
            TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "timestamp": trade.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "symbol": trade.get("symbol"),
                "side": trade.get("side"),
                "entry_price": trade.get("entry_price"),
                "avg_entry": trade.get("avg_entry"),
                "exit_price": trade.get("exit_price"),
                "qty": trade.get("qty"),
                "pnl": trade.get("pnl"),
                "pnl_pct": trade.get("pnl_pct"),
                "leverage": trade.get("leverage"),
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
                "close_reason": trade.get("close_reason"),
                "holding_time_sec": trade.get("holding_time_sec"),
                "add_count": trade.get("add_count", 0),
            }
            with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to write to JSONL fallback: {e}")

    def log(
        self,
        symbol:         str,
        side:           str,
        entry_price:    float,
        exit_price:     float,
        qty:            float,
        pnl:            float,
        balance:        float,
        prob_up:        float,
        avg_entry:      float | None = None,
        threshold:      float        = 0.0,
        atr:            float        = 0.0,
        atr_pct:        float        = 0.0,
        adx:            float        = 0.0,
        regime:         str          = "",
        stop_loss:      float | None = None,
        take_profit:    float | None = None,
        exit_reason:    str          = "",
        add_count:      int          = 0,
        leverage:       float        = 1.0,
        pnl_pct:        float | None = None,
        close_reason:   str | None   = None,
        holding_time_sec: int | None = None,
    ):
        """Log a trade to the database (or JSONL fallback)."""
        # Use entry_price if avg_entry is not provided
        if avg_entry is None:
            avg_entry = entry_price
        
        # Calculate pnl_pct if not provided
        if pnl_pct is None:
            pnl_pct = (pnl / balance) if balance > 0 else 0.0

        timestamp = datetime.now(timezone.utc).isoformat()
        payload = (
            timestamp,
            symbol, side,
            entry_price, avg_entry,
            exit_price, qty,
            pnl, pnl_pct, leverage, balance, prob_up, threshold,
            atr, atr_pct, adx, regime,
            stop_loss, take_profit, exit_reason, close_reason, holding_time_sec, add_count,
        )

        # Lightweight validation: ensure payload length matches expected columns
        expected_cols = 23
        if len(payload) != expected_cols:
            raise ValueError(f"Payload length {len(payload)} does not match expected columns {expected_cols}")

        # Try to write to database
        if self.db is not None and self._ensure_connection():
            try:
                with self._lock:  # Ensure thread-safe execution
                    # Build placeholders dynamically to avoid mismatch bugs
                    placeholders = ",".join(["?"] * len(payload))
                    self.db.execute(
                        f"""
                        INSERT INTO trades
                            (timestamp, symbol, side,
                             entry_price, avg_entry, exit_price, qty,
                             pnl, pnl_pct, leverage, balance, prob, threshold,
                             atr, atr_pct, adx, regime,
                             stop_loss, take_profit, exit_reason, close_reason, holding_time_sec, add_count)
                        VALUES
                            ({placeholders})
                    """,
                        payload,
                    )
                return
            except Exception as e:
                print(f"[LOGGER ERROR] Database write failed: {e}")
                print(f"[LOGGER ERROR] Attempting JSONL fallback...")
                # Fall through to JSONL fallback if enabled

        # Fallback to JSONL if enabled
        if ENABLE_JSONL_FALLBACK:
            trade_dict = {
                "timestamp": payload[0],
                "symbol": payload[1],
                "side": payload[2],
                "entry_price": payload[3],
                "avg_entry": payload[4],
                "exit_price": payload[5],
                "qty": payload[6],
                "pnl": payload[7],
                "pnl_pct": payload[8],
                "leverage": payload[9],
                "balance": payload[10],
                "prob": payload[11],
                "threshold": payload[12],
                "atr": payload[13],
                "atr_pct": payload[14],
                "adx": payload[15],
                "regime": payload[16],
                "stop_loss": payload[17],
                "take_profit": payload[18],
                "exit_reason": payload[19],
                "close_reason": payload[20],
                "holding_time_sec": payload[21],
                "add_count": payload[22],
            }
            self._write_jsonl_fallback(trade_dict)
        else:
            print("[LOGGER ERROR] Trade logging failed and JSONL fallback is disabled.")

    # Convenience methods for trade open/close (optional)
    def trade_open(self, **kwargs):
        self.log(exit_price=0.0, qty=0.0, pnl=0.0, **kwargs)  # Placeholder

    def trade_close(self, **kwargs):
        self.log(**kwargs)