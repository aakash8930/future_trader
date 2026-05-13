"""
Duplicate Order Prevention

Prevents rapid-fire trades on the same symbol.

Critical for:
- Avoiding duplicate order entries  
- Enforcing cooldown between trades
- Detecting retry storms
- Rate limiting per symbol
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Recent trade record for duplicate detection"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    qty: float
    timestamp: datetime
    closed: bool = False


class DuplicateOrderPrevention:
    """
    Prevents duplicate orders on same symbol.
    
    Features:
    - Track recent trades per symbol
    - Enforce cooldown period between trades
    - Detect duplicate attempts
    - Configurable cooldown duration
    - Clear closed positions from tracking
    """

    def __init__(
        self,
        cooldown_seconds: int = 60,
        max_recent_trades: int = 100,
    ):
        """
        Args:
            cooldown_seconds: Minimum seconds before next trade on same symbol
            max_recent_trades: Max trades to track in history
        """
        self.cooldown_seconds = cooldown_seconds
        self.max_recent_trades = max_recent_trades
        
        self.trade_history: Dict[str, list] = {}  # symbol -> [TradeRecord, ...]
        self.rejected_count: int = 0
        self.rejection_reasons: Dict[str, str] = {}

    def record_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
    ) -> None:
        """
        Record a new trade for tracking.
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            qty: Position quantity
        """
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []

        record = TradeRecord(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            qty=qty,
            timestamp=datetime.utcnow(),
        )

        self.trade_history[symbol].append(record)

        # Keep only recent N trades per symbol
        if len(self.trade_history[symbol]) > self.max_recent_trades:
            self.trade_history[symbol] = self.trade_history[symbol][
                -self.max_recent_trades:
            ]

        logger.info(f"Recorded trade: {symbol} {side} @ {entry_price}")

    def can_trade_symbol(self, symbol: str) -> bool:
        """
        Check if symbol can be traded now (not in cooldown).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if can trade, False if in cooldown period
        """
        if symbol not in self.trade_history:
            return True  # No previous trades

        if len(self.trade_history[symbol]) == 0:
            return True

        # Get last open trade
        open_trades = [t for t in self.trade_history[symbol] if not t.closed]
        if open_trades:
            # Can't trade if already have open position
            return False

        # Check cooldown on last trade (open or closed)
        last_trade = self.trade_history[symbol][-1]
        elapsed = (datetime.utcnow() - last_trade.timestamp).total_seconds()

        if elapsed < self.cooldown_seconds:
            return False

        return True

    def can_open_position(self, symbol: str) -> bool:
        """
        Check if can open new position (no existing open position).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if can open, False if already have open position
        """
        if symbol not in self.trade_history:
            return True

        # Check for any open positions
        for trade in self.trade_history[symbol]:
            if not trade.closed:
                return False

        return True

    def is_duplicate_attempt(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        price_tolerance_pct: float = 0.5,
    ) -> bool:
        """
        Detect if this is a duplicate trade attempt.
        
        Duplicate is defined as:
        - Same symbol
        - Same side
        - Similar entry price (within tolerance)
        - Recent timestamp (within cooldown)
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            price_tolerance_pct: Price tolerance for matching (default 0.5%)
            
        Returns:
            True if duplicate, False if unique
        """
        if symbol not in self.trade_history:
            return False

        tolerance = entry_price * (price_tolerance_pct / 100)

        for trade in self.trade_history[symbol]:
            if trade.closed:
                continue  # Ignore closed trades

            # Same symbol and side
            if trade.side != side:
                continue

            # Price within tolerance
            if abs(trade.entry_price - entry_price) > tolerance:
                continue

            # Recent trade
            elapsed = (datetime.utcnow() - trade.timestamp).total_seconds()
            if elapsed < self.cooldown_seconds:
                return True

        return False

    def validate_trade_request(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
    ) -> Dict[str, any]:
        """
        Full validation of trade request.
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            qty: Position quantity
            
        Returns:
            Dict with {allowed, reason, can_open, in_cooldown, duplicate}
        """
        result = {
            "allowed": True,
            "reason": "OK",
            "can_open": True,
            "in_cooldown": False,
            "duplicate": False,
        }

        # Check 1: Can open new position
        if not self.can_open_position(symbol):
            result["allowed"] = False
            result["reason"] = f"Already have open position in {symbol}"
            result["can_open"] = False
            return result

        # Check 2: Not in cooldown
        if not self.can_trade_symbol(symbol):
            result["allowed"] = False
            result["reason"] = f"Cooldown active for {symbol}"
            result["in_cooldown"] = True
            self.rejected_count += 1
            return result

        # Check 3: Not duplicate attempt
        if self.is_duplicate_attempt(symbol, side, entry_price):
            result["allowed"] = False
            result["reason"] = f"Duplicate trade attempt on {symbol}"
            result["duplicate"] = True
            self.rejected_count += 1
            return result

        return result

    def mark_position_closed(self, symbol: str) -> bool:
        """
        Mark all open positions on symbol as closed.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if any positions marked closed
        """
        if symbol not in self.trade_history:
            return False

        marked = False
        for trade in self.trade_history[symbol]:
            if not trade.closed:
                trade.closed = True
                marked = True

        if marked:
            logger.info(f"Marked position closed for {symbol}")

        return marked

    def get_symbol_status(self, symbol: str) -> Dict[str, any]:
        """
        Get detailed status for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with status info
        """
        if symbol not in self.trade_history:
            return {
                "symbol": symbol,
                "has_history": False,
                "open_positions": 0,
                "total_trades": 0,
                "can_trade": True,
                "in_cooldown": False,
            }

        trades = self.trade_history[symbol]
        open_count = len([t for t in trades if not t.closed])
        
        can_trade = self.can_trade_symbol(symbol)
        in_cooldown = not can_trade and open_count == 0

        last_trade = trades[-1] if trades else None
        last_timestamp = last_trade.timestamp if last_trade else None

        return {
            "symbol": symbol,
            "has_history": True,
            "open_positions": open_count,
            "total_trades": len(trades),
            "can_trade": can_trade,
            "in_cooldown": in_cooldown,
            "last_trade_time": last_timestamp,
            "cooldown_seconds": self.cooldown_seconds,
        }

    def get_all_symbols_in_cooldown(self) -> Set[str]:
        """Get all symbols currently in cooldown period"""
        cooldown_symbols = set()

        for symbol, trades in self.trade_history.items():
            if not trades:
                continue

            # Check if can trade - if False and no open positions, it's cooldown
            if not self.can_trade_symbol(symbol):
                open_count = len([t for t in trades if not t.closed])
                if open_count == 0:
                    cooldown_symbols.add(symbol)

        return cooldown_symbols

    def get_statistics(self) -> Dict[str, any]:
        """Get overall statistics"""
        total_trades = sum(len(t) for t in self.trade_history.values())
        total_symbols = len(self.trade_history)
        symbols_in_cooldown = len(self.get_all_symbols_in_cooldown())

        return {
            "total_symbols_tracked": total_symbols,
            "total_trades_recorded": total_trades,
            "symbols_in_cooldown": symbols_in_cooldown,
            "rejected_attempts": self.rejected_count,
            "cooldown_seconds": self.cooldown_seconds,
        }

    def clear_history(self) -> None:
        """Clear all trade history"""
        self.trade_history.clear()
        self.rejected_count = 0
        logger.info("Duplicate prevention history cleared")
