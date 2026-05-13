"""Position model with bidirectional side support."""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


def coerce_side(value: Side | str) -> Side:
    if isinstance(value, Side):
        return value

    normalized = str(value).strip().upper()
    if normalized in {"LONG", "SHORT", "NONE"}:
        return Side[normalized]

    raise ValueError(f"Invalid side: {value}")


@dataclass
class Position:
    """
    Represents a single open trading position.
    Supports pyramiding via add_to_position().
    
    NOTE: This class calculates RAW PnL (before fees/slippage).
    All fees and slippage are handled in the runner/broker layer.
    """
    side: Side | str            # "LONG" or "SHORT"
    entry_price: float   # initial entry price (kept for reference)
    qty: float
    entry_time: datetime
    leverage: float = field(default=1.0)  # Leverage multiplier (1.0 = 1x, 5.0 = 5x)
    avg_entry: float = field(init=False)   # weighted average entry price
    add_count: int = field(default=0)      # number of pyramid adds so far

    def __post_init__(self):
        self.side = coerce_side(self.side)
        if self.side == Side.NONE:
            raise ValueError("Invalid side: NONE")
        self.avg_entry = self.entry_price
        # Validate leverage
        if self.leverage < 1.0 or self.leverage > 125.0:
            raise ValueError(f"Invalid leverage: {self.leverage}. Must be 1.0-125.0")

    def add_to_position(self, price: float, qty: float, leverage: float = None):
        """
        Scale into the position (pyramid add).
        Updates avg_entry, total qty, and add_count.
        
        Args:
            price: Entry price for this add
            qty: Additional quantity
            leverage: If provided, update position leverage (usually constant)
        """
        if leverage is not None and leverage != self.leverage:
            raise ValueError(
                f"Cannot change leverage mid-position: {self.leverage} -> {leverage}"
            )
        
        total_cost = self.avg_entry * self.qty + price * qty
        self.qty += qty
        self.avg_entry = total_cost / self.qty
        self.add_count += 1

    def pnl(self, exit_price: float) -> float:
        """
        Calculate RAW profit / loss at given exit price.
        
        Formula (with leverage):
        - LONG:  (exit_price - avg_entry) * qty * leverage
        - SHORT: (avg_entry - exit_price) * qty * leverage
        
        NOTE: This does NOT include fees or slippage.
        Those are applied by the runner/broker.
        
        Args:
            exit_price: Price at exit
            
        Returns:
            Raw PnL in USD (before fees/slippage)
        """
        if exit_price <= 0 or self.avg_entry <= 0 or self.qty <= 0:
            raise ValueError(
                f"Invalid PnL inputs: exit_price={exit_price}, "
                f"avg_entry={self.avg_entry}, qty={self.qty}"
            )
        
        if self.side == Side.LONG:
            raw_pnl = (exit_price - self.avg_entry) * self.qty
        elif self.side == Side.SHORT:
            raw_pnl = (self.avg_entry - exit_price) * self.qty
        else:
            raise ValueError(f"Invalid side: {self.side}")
        
        # Apply leverage multiplier
        pnl_with_leverage = raw_pnl * self.leverage
        
        return pnl_with_leverage

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized PnL at current market price.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized PnL in USD
        """
        return self.pnl(current_price)

    def __repr__(self) -> str:
        return (
            f"Position({self.side.value}, qty={self.qty:.6f}, "
            f"entry={self.avg_entry:.4f}, leverage={self.leverage}x)"
        )
