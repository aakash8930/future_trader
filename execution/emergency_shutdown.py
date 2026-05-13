"""
Emergency Shutdown Mode

Gracefully closes all positions and stops trading on emergency.

Critical for:
- Circuit breaker mechanism
- Catastrophic loss prevention
- Clean position exit
- Order cancellation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ShutdownEvent:
    """Record of shutdown event"""
    timestamp: datetime
    reason: str
    positions_to_close: int
    orders_to_cancel: int
    status: str  # 'initiated', 'in_progress', 'complete'


class EmergencyShutdown:
    """
    Emergency shutdown mode for safety.
    
    Features:
    - Initiate emergency shutdown
    - Track shutdown state
    - Record shutdown events
    - Report shutdown status
    """

    def __init__(
        self,
        enable_auto_recovery: bool = False,
    ):
        """
        Args:
            enable_auto_recovery: Allow system to recover from shutdown
        """
        self.enable_auto_recovery = enable_auto_recovery
        self.is_active: bool = False
        self.shutdown_reason: Optional[str] = None
        self.initiated_at: Optional[datetime] = None
        
        self.shutdown_events: List[ShutdownEvent] = []
        self.pending_close_orders: List[str] = []  # Symbols to close
        self.cancelled_orders: List[str] = []
        self.closed_positions: List[str] = []

    def initiate_shutdown(
        self,
        reason: str,
        positions_to_close: int = 0,
        orders_to_cancel: int = 0,
    ) -> ShutdownEvent:
        """
        Initiate emergency shutdown.
        
        Args:
            reason: Reason for shutdown
            positions_to_close: Number of open positions
            orders_to_cancel: Number of pending orders
            
        Returns:
            ShutdownEvent record
        """
        self.is_active = True
        self.shutdown_reason = reason
        self.initiated_at = datetime.utcnow()

        event = ShutdownEvent(
            timestamp=self.initiated_at,
            reason=reason,
            positions_to_close=positions_to_close,
            orders_to_cancel=orders_to_cancel,
            status="initiated",
        )

        self.shutdown_events.append(event)
        logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")

        return event

    def can_trade(self) -> bool:
        """
        Check if trading is allowed.
        
        Returns:
            False if shutdown active, True otherwise
        """
        return not self.is_active

    def close_position(self, symbol: str) -> bool:
        """
        Mark position for closure (add to queue).
        
        Args:
            symbol: Symbol to close
            
        Returns:
            True if queued, False if already closed
        """
        if symbol in self.closed_positions:
            return False

        self.pending_close_orders.append(symbol)
        logger.info(f"Queued position close: {symbol}")
        return True

    def process_close(self, symbol: str) -> None:
        """
        Record completed position closure.
        
        Args:
            symbol: Symbol that was closed
        """
        if symbol in self.pending_close_orders:
            self.pending_close_orders.remove(symbol)

        if symbol not in self.closed_positions:
            self.closed_positions.append(symbol)
            logger.info(f"Position closed: {symbol}")

    def cancel_order(self, order_id: str) -> bool:
        """
        Mark order for cancellation.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if queued, False if already cancelled
        """
        if order_id in self.cancelled_orders:
            return False

        self.cancelled_orders.append(order_id)
        logger.info(f"Queued order cancel: {order_id}")
        return True

    def process_cancel(self, order_id: str) -> None:
        """
        Record completed order cancellation.
        
        Args:
            order_id: Order ID that was cancelled
        """
        if order_id in self.cancelled_orders:
            # Already in cancelled list
            logger.info(f"Order cancelled: {order_id}")

    def get_shutdown_status(self) -> Dict[str, any]:
        """
        Get detailed shutdown status.
        
        Returns:
            Dict with shutdown status
        """
        pending_count = len(self.pending_close_orders)
        cancelled_count = len(self.cancelled_orders)
        closed_count = len(self.closed_positions)
        total_events = len(self.shutdown_events)

        return {
            "is_active": self.is_active,
            "reason": self.shutdown_reason,
            "initiated_at": self.initiated_at,
            "positions_pending_close": pending_count,
            "positions_closed": closed_count,
            "orders_cancelled": cancelled_count,
            "total_shutdown_events": total_events,
            "can_trade": self.can_trade(),
            "recovery_enabled": self.enable_auto_recovery,
        }

    def recovery_allowed(self) -> bool:
        """
        Check if system can recover from shutdown.
        
        Recovery is allowed only if:
        - All positions are closed
        - All orders are cancelled
        - Recovery is enabled
        
        Returns:
            True if recovery is possible
        """
        if not self.enable_auto_recovery:
            return False

        if len(self.pending_close_orders) > 0:
            return False

        if len(self.cancelled_orders) > 0:
            return False

        return True

    def attempt_recovery(self) -> bool:
        """
        Attempt to recover from shutdown.
        
        Returns:
            True if recovery successful, False otherwise
        """
        if not self.recovery_allowed():
            logger.warning(
                "Recovery not allowed: "
                f"pending_closes={len(self.pending_close_orders)}, "
                f"pending_cancels={len(self.cancelled_orders)}, "
                f"auto_recovery={self.enable_auto_recovery}"
            )
            return False

        self.is_active = False
        self.shutdown_reason = None
        logger.info("Emergency shutdown recovered")
        return True

    def force_complete_shutdown(self) -> None:
        """
        Force shutdown completion (admin only).
        Clear all pending operations and declare shutdown complete.
        """
        self.pending_close_orders.clear()
        self.cancelled_orders.clear()

        if self.shutdown_events:
            self.shutdown_events[-1].status = "complete"

        logger.warning("Emergency shutdown forced complete")

    def get_pending_actions(self) -> Dict[str, list]:
        """
        Get all pending shutdown actions.
        
        Returns:
            Dict with pending positions and orders
        """
        return {
            "positions_to_close": self.pending_close_orders.copy(),
            "orders_to_cancel": self.cancelled_orders.copy(),
            "closed_positions": self.closed_positions.copy(),
        }

    def get_shutdown_history(self) -> List[ShutdownEvent]:
        """
        Get all shutdown events.
        
        Returns:
            List of ShutdownEvent records
        """
        return self.shutdown_events.copy()
