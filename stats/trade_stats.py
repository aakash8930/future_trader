#stats/trade_stats.py

class TradeStats:

    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.total = 0

    def record(self, pnl):

        self.total += 1

        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

    def win_rate(self):

        if self.total == 0:
            return 0

        return self.wins / self.total