import psycopg2
import os
from urllib.parse import urlparse
from datetime import datetime


class TradeLogger:

    def __init__(self):

        database_url = os.getenv("DATABASE_URL")

        result = urlparse(database_url)

        self.conn = psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
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
                prob_up
            )
        )

        cur.close()