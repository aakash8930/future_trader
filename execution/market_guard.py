# execution/market_guard.py

from datetime import date


class MarketGuard:
    """
    Global safety guard.
    Enforces:
    - Daily drawdown kill-switch
    - Consecutive loss breaker
    """

    def __init__(
        self,
        max_daily_drawdown_pct: float = 0.02,  # 2%
        max_consecutive_losses: int = 3,
    ):
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses

        self.current_day: date | None = None
        self.starting_balance: float | None = None
        self.consecutive_losses: int = 0
        self.trading_disabled: bool = False

    # --------------------------------------------------
    def _reset_if_new_day(self, today: date, balance: float):
        if self.current_day != today:
            self.current_day = today
            self.starting_balance = balance
            self.consecutive_losses = 0
            self.trading_disabled = False

    # --------------------------------------------------
    def allow_trading(self, balance: float, today: date) -> bool:
        """
        Check whether trading is allowed right now.
        """
        self._reset_if_new_day(today, balance)

        if self.trading_disabled:
            return False

        if self.starting_balance is None:
            self.starting_balance = balance
            return True

        drawdown = (self.starting_balance - balance) / self.starting_balance

        if drawdown >= self.max_daily_drawdown_pct:
            self.trading_disabled = True
            print(
                f"[!!] MARKET GUARD: Daily drawdown {drawdown:.2%} "
                f"exceeded limit → trading stopped"
            )
            return False

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trading_disabled = True
            print(
                f"[!!] MARKET GUARD: {self.consecutive_losses} consecutive losses "
                f"→ trading stopped"
            )
            return False

        return True

    # --------------------------------------------------
    def register_trade(self, pnl: float):
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0