import os
import psycopg2
from datetime import datetime


class TradeLogger:

    def __init__(self):
        self.conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        self._create_table()

    def _create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    time TIMESTAMP,
                    symbol TEXT,
                    side TEXT,
                    entry_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    qty DOUBLE PRECISION,
                    pnl DOUBLE PRECISION,
                    balance DOUBLE PRECISION,
                    prob_up DOUBLE PRECISION
                );
            """)
            self.conn.commit()

    def log(
        self,
        symbol,
        side,
        entry_price,
        exit_price,
        qty,
        pnl,
        balance,
        prob_up,
    ):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (
                    time, symbol, side,
                    entry_price, exit_price,
                    qty, pnl, balance, prob_up
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """, (
                datetime.utcnow(),
                symbol,
                side,
                entry_price,
                exit_price,
                qty,
                pnl,
                balance,
                prob_up,
            ))
            self.conn.commit()