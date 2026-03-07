import psycopg2
import os
from datetime import datetime


class TradeLogger:

    def __init__(self):

        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            port=os.getenv("DB_PORT", 5432),
        )

        self.conn.autocommit = True


    def log(
        self,
        symbol,
        side,
        entry_price,
        exit_price,
        qty,
        pnl,
        balance,
        prob_up
    ):

        cur = self.conn.cursor()

        cur.execute(
            """
            INSERT INTO trades
            (timestamp, symbol, side, entry_price, exit_price, qty, pnl, balance, prob)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                datetime.utcnow(),
                symbol,
                side,
                entry_price,
                exit_price,
                qty,
                pnl,
                balance,
                prob_up,
            ),
        )

        cur.close()