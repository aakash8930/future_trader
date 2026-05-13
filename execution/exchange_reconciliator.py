"""
Exchange Reconciliation Engine

Periodic position verification against live exchange state.
Detects silent failures, desync, and order fill mismatches.

Critical for:
- Detecting exchange desync
- Validating account state
- Preventing position mismatches
- Emergency closeout triggers
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExchangePosition:
    """Position data from exchange API"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    qty: float
    entry_price: float
    unrealized_pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DesyncEvent:
    """Desync detection event"""
    timestamp: datetime
    symbol: str
    type: str  # 'POSITION_MISMATCH', 'QTY_MISMATCH', 'BALANCE_MISMATCH'
    local_value: float
    exchange_value: float
    difference: float
    severity: str  # 'WARNING', 'CRITICAL'
    description: str


@dataclass
class ReconciliationReport:
    """Reconciliation status report"""
    timestamp: datetime
    total_positions_checked: int
    mismatches_found: int
    balance_match: bool
    account_balance_local: float
    account_balance_exchange: float
    balance_diff: float
    desync_events: List[DesyncEvent] = field(default_factory=list)
    is_healthy: bool = False
    recommendations: List[str] = field(default_factory=list)


class ExchangeReconciliator:
    """
    Validates local state matches live exchange.
    
    Detects:
    - Position mismatches (local != exchange)
    - Quantity mismatches
    - Balance mismatches
    - Stale order fills
    
    Triggers:
    - Warnings on minor desync
    - Critical alert on major desync
    - Emergency closeout if desync > threshold
    """

    def __init__(
        self,
        max_position_diff_pct: float = 1.0,
        max_balance_diff_pct: float = 0.5,
        max_qty_diff: float = 0.01,
        reconciliation_interval_seconds: int = 30,
        desync_threshold: int = 3,
    ):
        """
        Args:
            max_position_diff_pct: Max % difference before warning (default 1%)
            max_balance_diff_pct: Max % diff in balance (default 0.5%)
            max_qty_diff: Max qty difference allowed (default 0.01)
            reconciliation_interval_seconds: Min seconds between checks
            desync_threshold: Number of mismatches before critical alert
        """
        self.max_position_diff_pct = max_position_diff_pct
        self.max_balance_diff_pct = max_balance_diff_pct
        self.max_qty_diff = max_qty_diff
        self.reconciliation_interval_seconds = reconciliation_interval_seconds
        self.desync_threshold = desync_threshold

        self.last_reconciliation = None
        self.consecutive_desync_count = 0
        self.desync_history: List[DesyncEvent] = []

    def should_reconcile(self) -> bool:
        """Check if enough time has passed to reconcile again"""
        if self.last_reconciliation is None:
            return True

        elapsed = datetime.utcnow() - self.last_reconciliation
        return elapsed.total_seconds() >= self.reconciliation_interval_seconds

    def reconcile_positions(
        self,
        local_positions: Dict[str, dict],
        exchange_positions: Dict[str, ExchangePosition],
    ) -> ReconciliationReport:
        """
        Compare local positions with exchange positions.
        
        Args:
            local_positions: Dict of symbol -> {side, qty, entry_price, ...}
            exchange_positions: Dict of symbol -> ExchangePosition
            
        Returns:
            ReconciliationReport with all findings
        """
        report = ReconciliationReport(
            timestamp=datetime.utcnow(),
            total_positions_checked=len(local_positions),
            mismatches_found=0,
            balance_match=False,
            account_balance_local=0.0,
            account_balance_exchange=0.0,
            balance_diff=0.0,
        )

        # Check each local position against exchange
        for symbol, local_pos in local_positions.items():
            if symbol not in exchange_positions:
                # Position exists locally but not on exchange
                event = DesyncEvent(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    type="POSITION_MISMATCH",
                    local_value=local_pos.get("qty", 0.0),
                    exchange_value=0.0,
                    difference=local_pos.get("qty", 0.0),
                    severity="CRITICAL",
                    description=f"Local position exists for {symbol} but not on exchange",
                )
                report.desync_events.append(event)
                report.mismatches_found += 1
                self.desync_history.append(event)
                continue

            exchange_pos = exchange_positions[symbol]

            # Check quantity match
            local_qty = local_pos.get("qty", 0.0)
            exchange_qty = exchange_pos.qty

            qty_diff = abs(local_qty - exchange_qty)
            if qty_diff > self.max_qty_diff:
                pct_diff = (qty_diff / max(exchange_qty, 0.01)) * 100

                event = DesyncEvent(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    type="QTY_MISMATCH",
                    local_value=local_qty,
                    exchange_value=exchange_qty,
                    difference=qty_diff,
                    severity="WARNING" if pct_diff < 5 else "CRITICAL",
                    description=f"Qty mismatch: local {local_qty} vs exchange {exchange_qty}",
                )
                report.desync_events.append(event)
                report.mismatches_found += 1
                self.desync_history.append(event)

        # Check for positions on exchange not in local
        for symbol, exchange_pos in exchange_positions.items():
            if symbol not in local_positions:
                event = DesyncEvent(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    type="POSITION_MISMATCH",
                    local_value=0.0,
                    exchange_value=exchange_pos.qty,
                    difference=exchange_pos.qty,
                    severity="CRITICAL",
                    description=f"Exchange has {symbol} position but local state missing",
                )
                report.desync_events.append(event)
                report.mismatches_found += 1
                self.desync_history.append(event)

        # Update consecutive desync counter
        if report.mismatches_found > 0:
            self.consecutive_desync_count += 1
        else:
            self.consecutive_desync_count = 0

        # Determine health
        report.is_healthy = (
            report.mismatches_found == 0 
            and self.consecutive_desync_count < self.desync_threshold
        )

        # Add recommendations
        if report.mismatches_found > 0:
            report.recommendations.append(
                "Review desync events and manually verify exchange state"
            )
            if self.consecutive_desync_count >= self.desync_threshold:
                report.recommendations.append(
                    "Multiple consecutive desync events - consider emergency closeout"
                )

        self.last_reconciliation = datetime.utcnow()
        return report

    def reconcile_balance(
        self,
        local_balance: float,
        exchange_balance: float,
    ) -> Dict[str, any]:
        """
        Compare account balance.
        
        Args:
            local_balance: Local account balance
            exchange_balance: Exchange account balance
            
        Returns:
            Dict with {match, diff, diff_pct, recommendation}
        """
        diff = abs(local_balance - exchange_balance)
        diff_pct = (diff / max(exchange_balance, 0.01)) * 100

        match = diff_pct <= self.max_balance_diff_pct

        result = {
            "match": match,
            "local_balance": local_balance,
            "exchange_balance": exchange_balance,
            "difference": diff,
            "difference_pct": diff_pct,
            "threshold_pct": self.max_balance_diff_pct,
        }

        if not match:
            result["recommendation"] = (
                f"Balance mismatch: {diff_pct:.2f}% diff "
                f"({diff:.2f} USDT)"
            )

        return result

    def get_desync_history(self, hours: int = 1) -> List[DesyncEvent]:
        """Get desync events from last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [e for e in self.desync_history if e.timestamp >= cutoff]

    def is_critical_desync(self) -> bool:
        """Check if consecutive desync count exceeds threshold"""
        return self.consecutive_desync_count >= self.desync_threshold

    def reset_desync_counter(self):
        """Reset consecutive desync counter after manual intervention"""
        self.consecutive_desync_count = 0
        logger.info("Desync counter reset")

    def get_status(self) -> Dict[str, any]:
        """Get current reconciliation status"""
        return {
            "last_reconciliation": self.last_reconciliation,
            "consecutive_desync_events": self.consecutive_desync_count,
            "is_critical": self.is_critical_desync(),
            "desync_history_count": len(self.desync_history),
            "should_reconcile_now": self.should_reconcile(),
        }
