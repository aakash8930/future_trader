# execution/shadow_broker.py

from datetime import datetime
from typing import Optional
from execution.position import Position


class ShadowBroker:
    """
    Shadow trading broker.
    Behaves like live execution but never sends real orders.
    Supports position pyramiding via add_to_position().
    """

    def __init__(self):
        self.position: Optional[Position] = None

    def open_position(self, side: str, price: float, qty: float, symbol: str) -> Position:
        self.position = Position(
            side=side,
            entry_price=price,
            qty=qty,
            entry_time=datetime.utcnow(),
        )
        return self.position

    def add_to_position(self, price: float, qty: float):
        """
        Pyramid into the current open position.
        Delegates to Position.add_to_position() which updates
        avg_entry, total qty, and add_count.
        """
        if not self.position:
            return
        self.position.add_to_position(price, qty)

    def close_position(self, price: float, symbol: str) -> float:
        if not self.position:
            return 0.0

        pnl = self.position.pnl(price)
        self.position = None
        return pnl