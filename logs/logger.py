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

# Flag to enable JSONL fallback (set to True if you want both)
ENABLE_JSONL_FALLBACK = False


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
            # Ensure the trades table exists (migrations should have created it)
            # We'll add a simple check and create if not exists (though migrations should handle it)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT      NOT NULL,
                    symbol      TEXT      NOT NULL,
                    side        TEXT      NOT NULL,
                    entry_price REAL      NOT NULL,
                    avg_entry   REAL      NOT NULL,
                    exit_price  REAL,
                    qty         REAL      NOT NULL,
                    pnl         REAL,
                    balance     REAL      NOT NULL,
                    prob        REAL      NOT NULL,
                    threshold   REAL      NOT NULL,
                    atr         REAL      NOT NULL,
                    atr_pct     REAL      NOT NULL,
                    adx         REAL      NOT NULL,
                    regime      TEXT      NOT NULL,
                    stop_loss   REAL,
                    take_profit REAL,
                    exit_reason TEXT,
                    add_count   INTEGER DEFAULT 0
                );
            """)
            # Create indexes for common queries
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, timestamp DESC);")
            print("[LOGGER] Trades table ensured.")
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
            }
            with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to write to JSONL fallback: {e}")

    def log(
        self,
        symbol:      str,
        side:        str,
        entry_price: float,
        exit_price:  float,
        qty:         float,
        pnl:         float,
        balance:     float,
        prob_up:     float,
        avg_entry:   float | None = None,
        threshold:   float        = 0.0,
        atr:         float        = 0.0,
        atr_pct:     float        = 0.0,
        adx:         float        = 0.0,
        regime:      str          = "",
        stop_loss:   float | None = None,
        take_profit: float | None = None,
        exit_reason: str          = "",
        add_count:   int          = 0,
    ):
        """Log a trade to the database (or JSONL fallback)."""
        # Use entry_price if avg_entry is not provided
        if avg_entry is None:
            avg_entry = entry_price

        payload = (
            datetime.now(timezone.utc).isoformat(),
            symbol, side,
            entry_price, avg_entry,
            exit_price, qty,
            pnl, balance, prob_up, threshold,
            atr, atr_pct, adx, regime,
            stop_loss, take_profit, exit_reason, add_count,
        )

        # Try to write to database
        if self.db is not None and self._ensure_connection():
            try:
                with self._lock:  # Ensure thread-safe execution
                    self.db.execute("""
                        INSERT INTO trades
                            (timestamp, symbol, side,
                             entry_price, avg_entry, exit_price, qty,
                             pnl, balance, prob, threshold,
                             atr, atr_pct, adx, regime,
                             stop_loss, take_profit, exit_reason, add_count)
                        VALUES
                            (?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?)
                    """, payload)
                return
            except Exception as e:
                print(f"[LOGGER ERROR] Database write failed: {e}")
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
                "balance": payload[8],
                "prob": payload[9],
                "threshold": payload[10],
                "atr": payload[11],
                "atr_pct": payload[12],
                "adx": payload[13],
                "regime": payload[14],
                "stop_loss": payload[15],
                "take_profit": payload[16],
                "exit_reason": payload[17],
                "add_count": payload[18],
            }
            self._write_jsonl_fallback(trade_dict)
        else:
            print("[LOGGER ERROR] Trade logging failed and JSONL fallback is disabled.")

    # Convenience methods for trade open/close (optional)
    def trade_open(self, **kwargs):
        self.log(exit_price=0.0, qty=0.0, pnl=0.0, **kwargs)  # Placeholder

    def trade_close(self, **kwargs):
        self.log(**kwargs)