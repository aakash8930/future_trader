# execution/position.py

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    """
    Represents a single open trading position.
    Supports pyramiding via add_to_position().
    """
    side: str            # "LONG" or "SHORT"
    entry_price: float   # initial entry price (kept for reference)
    qty: float
    entry_time: datetime
    avg_entry: float = field(init=False)   # weighted average entry price
    add_count: int = field(default=0)      # number of pyramid adds so far

    def __post_init__(self):
        self.avg_entry = self.entry_price

    def add_to_position(self, price: float, qty: float):
        """
        Scale into the position (pyramid add).
        Updates avg_entry, total qty, and add_count.
        """
        total_cost = self.avg_entry * self.qty + price * qty
        self.qty += qty
        self.avg_entry = total_cost / self.qty
        self.add_count += 1

    def pnl(self, exit_price: float) -> float:
        """
        Calculate profit / loss at given exit price using avg_entry.
        """
        if self.side == "LONG":
            return (exit_price - self.avg_entry) * self.qty

        # SHORT (future-proofing)
        return (self.avg_entry - exit_price) * self.qty
