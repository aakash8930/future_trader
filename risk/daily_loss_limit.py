"""
Daily Loss Limit Enforcement

Stops trading when daily losses exceed threshold.

Critical for:
- Protecting equity
- Preventing catastrophic days
- Enforcing discipline
- Risk management
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class DailyLossRecord:
    """Record of daily trading results"""
    date: str  # YYYY-MM-DD format
    starting_balance: float
    current_balance: float
    daily_pnl: float
    num_trades: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DailyLossLimit:
    """
    Enforces maximum daily loss limits.
    
    Features:
    - Track daily PnL
    - Calculate daily loss
    - Enforce hard stop at limit
    - Warning threshold at 50%
    - Auto-shutdown on limit exceeded
    """

    def __init__(
        self,
        daily_loss_limit_usd: float = 100.0,
        warning_threshold_pct: float = 50.0,
    ):
        """
        Args:
            daily_loss_limit_usd: Max loss allowed per day in USD
            warning_threshold_pct: Warn at this % of limit (default 50%)
        """
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self.warning_threshold_pct = warning_threshold_pct

        self.starting_balance: Optional[float] = None
        self.today_date: Optional[str] = None
        self.daily_pnl: float = 0.0
        self.trade_count: int = 0
        self.is_shutdown: bool = False
        self.shutdown_reason: Optional[str] = None

        self.daily_records: List[DailyLossRecord] = []

    def _get_today_date(self) -> str:
        """Get today's date in YYYY-MM-DD format"""
        return datetime.utcnow().strftime("%Y-%m-%d")

    def initialize_day(self, starting_balance: float) -> None:
        """
        Initialize trading day with starting balance.
        
        Args:
            starting_balance: Account balance at start of day
        """
        today = self._get_today_date()

        if self.today_date != today:
            # New day - reset counters
            self.starting_balance = starting_balance
            self.today_date = today
            self.daily_pnl = 0.0
            self.trade_count = 0
            self.is_shutdown = False
            self.shutdown_reason = None

            logger.info(f"Daily loss limit initialized: balance={starting_balance}")

    def update_pnl(self, pnl: float) -> None:
        """
        Update daily PnL after a trade.
        
        Args:
            pnl: Trade PnL (can be positive or negative)
        """
        self.daily_pnl += pnl
        self.trade_count += 1

    def can_trade(self) -> bool:
        """
        Check if trading is allowed based on daily loss limit.
        
        Returns:
            False if shutdown, True otherwise
        """
        return not self.is_shutdown

    def check_loss_limit(self) -> Dict[str, any]:
        """
        Check if daily loss limit exceeded.
        
        Returns:
            Dict with {within_limit, current_loss, limit, warning, emergency, reason}
        """
        current_loss = -self.daily_pnl  # Negative PnL = loss

        result = {
            "within_limit": True,
            "current_loss": current_loss,
            "daily_limit": self.daily_loss_limit_usd,
            "warning": False,
            "emergency": False,
            "reason": "OK",
        }

        # Check if exceeded limit
        if current_loss >= self.daily_loss_limit_usd:
            result["within_limit"] = False
            result["emergency"] = True
            result["reason"] = f"Daily loss limit exceeded: {current_loss:.2f} >= {self.daily_loss_limit_usd:.2f}"
            self.is_shutdown = True
            self.shutdown_reason = result["reason"]
            return result

        # Check if at warning threshold
        loss_pct_of_limit = (current_loss / self.daily_loss_limit_usd) * 100
        if loss_pct_of_limit >= self.warning_threshold_pct:
            result["warning"] = True
            result["reason"] = (
                f"Daily loss approaching limit: "
                f"{current_loss:.2f} ({loss_pct_of_limit:.1f}% of {self.daily_loss_limit_usd:.2f})"
            )

        return result

    def enforce_loss_limit(self) -> Optional[str]:
        """
        Enforce loss limit and return reason if shutdown triggered.
        
        Returns:
            Shutdown reason if limit exceeded, None otherwise
        """
        check = self.check_loss_limit()

        if check["emergency"]:
            logger.critical(f"DAILY LOSS LIMIT EXCEEDED: {check['reason']}")
            return check["reason"]

        if check["warning"]:
            logger.warning(f"Daily loss warning: {check['reason']}")

        return None

    def get_daily_status(self) -> Dict[str, any]:
        """Get current daily trading status"""
        current_loss = -self.daily_pnl

        return {
            "date": self.today_date,
            "starting_balance": self.starting_balance,
            "current_pnl": self.daily_pnl,
            "current_loss": current_loss,
            "daily_limit": self.daily_loss_limit_usd,
            "loss_pct_of_limit": (
                (current_loss / self.daily_loss_limit_usd) * 100
                if self.daily_loss_limit_usd > 0
                else 0
            ),
            "num_trades": self.trade_count,
            "is_shutdown": self.is_shutdown,
            "shutdown_reason": self.shutdown_reason,
        }

    def record_daily_close(self, closing_balance: float) -> DailyLossRecord:
        """
        Record daily close and create day record.
        
        Args:
            closing_balance: Account balance at end of day
            
        Returns:
            DailyLossRecord for the day
        """
        record = DailyLossRecord(
            date=self.today_date,
            starting_balance=self.starting_balance,
            current_balance=closing_balance,
            daily_pnl=self.daily_pnl,
            num_trades=self.trade_count,
        )

        self.daily_records.append(record)
        logger.info(
            f"Daily record: {self.today_date} PnL={self.daily_pnl:.2f} "
            f"Trades={self.trade_count} Balance={closing_balance:.2f}"
        )

        return record

    def reset_daily_shutdown(self) -> None:
        """Reset shutdown flag (admin only)"""
        self.is_shutdown = False
        self.shutdown_reason = None
        logger.info("Daily loss limit shutdown reset")

    def get_daily_history(self, days: int = 7) -> List[DailyLossRecord]:
        """
        Get last N days of records.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of DailyLossRecord
        """
        if not self.daily_records:
            return []

        return self.daily_records[-days:]

    def get_weekly_stats(self, days: int = 7) -> Dict[str, any]:
        """
        Get weekly statistics.
        
        Args:
            days: Number of days for stats
            
        Returns:
            Dict with weekly stats
        """
        history = self.get_daily_history(days)

        if not history:
            return {
                "days_traded": 0,
                "total_pnl": 0.0,
                "total_trades": 0,
                "avg_pnl_per_day": 0.0,
                "avg_trades_per_day": 0.0,
                "winning_days": 0,
                "losing_days": 0,
            }

        total_pnl = sum(r.daily_pnl for r in history)
        total_trades = sum(r.num_trades for r in history)
        winning_days = len([r for r in history if r.daily_pnl > 0])
        losing_days = len([r for r in history if r.daily_pnl < 0])

        return {
            "days_traded": len(history),
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "avg_pnl_per_day": total_pnl / len(history) if history else 0,
            "avg_trades_per_day": total_trades / len(history) if history else 0,
            "winning_days": winning_days,
            "losing_days": losing_days,
            "win_rate_pct": (winning_days / len(history) * 100) if history else 0,
        }
