# logs/logger.py

import os
import psycopg2
from datetime import datetime, timezone


class TradeLogger:

    def __init__(self):
        database_url = os.getenv("DATABASE_PUBLIC_URL") or os.getenv("DATABASE_URL")
        if not database_url:
            print("[LOGGER] No DATABASE_URL set — trade logging disabled.")
            self.conn = None
            return

        try:
            self.conn = psycopg2.connect(database_url)
            self.conn.autocommit = True
            self._ensure_schema()
            print("[LOGGER] PostgreSQL connected and schema ready.")
        except Exception as exc:
            print(f"[LOGGER] Database connection failed — trade logging disabled. Error: {exc}")
            self.conn = None

    # ------------------------------------------------------------------
    def _ensure_schema(self):
        """
        Idempotently create the trades table and all required columns.
        Safe to call on every startup — uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id          SERIAL PRIMARY KEY,
                    timestamp   TIMESTAMPTZ      DEFAULT NOW(),
                    symbol      TEXT,
                    side        TEXT,
                    entry_price DOUBLE PRECISION,
                    avg_entry   DOUBLE PRECISION,
                    exit_price  DOUBLE PRECISION,
                    qty         DOUBLE PRECISION,
                    pnl         DOUBLE PRECISION,
                    balance     DOUBLE PRECISION,
                    prob        DOUBLE PRECISION,
                    threshold   DOUBLE PRECISION,
                    atr         DOUBLE PRECISION,
                    atr_pct     DOUBLE PRECISION,
                    adx         DOUBLE PRECISION,
                    regime      TEXT,
                    stop_loss   DOUBLE PRECISION,
                    take_profit DOUBLE PRECISION,
                    exit_reason TEXT,
                    add_count   INTEGER DEFAULT 0
                );
            """)

            # Idempotent column additions for live databases that predate a schema update.
            extra_cols = [
                ("avg_entry",   "DOUBLE PRECISION"),
                ("threshold",   "DOUBLE PRECISION"),
                ("atr",         "DOUBLE PRECISION"),
                ("atr_pct",     "DOUBLE PRECISION"),
                ("adx",         "DOUBLE PRECISION"),
                ("regime",      "TEXT"),
                ("stop_loss",   "DOUBLE PRECISION"),
                ("take_profit", "DOUBLE PRECISION"),
                ("exit_reason", "TEXT"),
                ("add_count",   "INTEGER DEFAULT 0"),
            ]
            for col, col_type in extra_cols:
                cur.execute(
                    f"ALTER TABLE trades ADD COLUMN IF NOT EXISTS {col} {col_type};"
                )

            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, timestamp DESC);")

    # ------------------------------------------------------------------
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
        if self.conn is None:
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trades
                        (timestamp, symbol, side,
                         entry_price, avg_entry, exit_price, qty,
                         pnl, balance, prob, threshold,
                         atr, atr_pct, adx, regime,
                         stop_loss, take_profit, exit_reason, add_count)
                    VALUES
                        (%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s)
                    """,
                    (
                        datetime.now(timezone.utc),
                        symbol, side,
                        entry_price, avg_entry if avg_entry is not None else entry_price,
                        exit_price, qty,
                        pnl, balance, prob_up, threshold,
                        atr, atr_pct, adx, regime,
                        stop_loss, take_profit, exit_reason, add_count,
                    ),
                )
        except Exception as exc:
            print(f"[LOGGER ERROR] Failed to write trade: {exc}")
