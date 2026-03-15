# risk/limits.py

from datetime import date


class RiskLimits:
    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_consecutive_losses: int = 3,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses


class RiskState:
    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance

        self.today = date.today()
        self.daily_start_balance = starting_balance

        self.consecutive_losses = 0
        self.trading_blocked = False

    # ----------------------------
    # DAILY RESET
    # ----------------------------
    def reset_if_new_day(self, current_date: date):
        if current_date != self.today:
            self.today = current_date
            self.daily_start_balance = self.current_balance
            self.consecutive_losses = 0
            self.trading_blocked = False

    # ----------------------------
    # REGISTER CLOSED TRADE
    # ----------------------------
    def register_trade(self, pnl: float):
        self.current_balance += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    # ----------------------------
    # CHECK IF TRADING ALLOWED
    # ----------------------------
    def trading_allowed(self, limits: RiskLimits) -> bool:
        if self.trading_blocked:
            return False

        # Drawdown kill-switch
        if self.daily_start_balance <= 0:
            drawdown = 0.0
        else:
            drawdown = (
                self.daily_start_balance - self.current_balance
            ) / self.daily_start_balance

        if drawdown >= limits.max_daily_loss_pct:
            self.trading_blocked = True
            print("🛑 DAILY DRAWDOWN KILL-SWITCH TRIGGERED")
            return False

        # Consecutive loss breaker
        if self.consecutive_losses >= limits.max_consecutive_losses:
            self.trading_blocked = True
            print("🛑 CONSECUTIVE LOSS BREAKER TRIGGERED")
            return False

        return True