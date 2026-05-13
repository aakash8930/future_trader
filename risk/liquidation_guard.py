"""
Liquidation Guard

Prevents liquidation events through early detection and auto-closeout.

Critical for:
- Calculating liquidation price per position
- Early warning system (10% from liquidation)
- Auto-closeout at critical threshold (5%)
- Emergency mode for cascade prevention
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiquidationLevel:
    """Liquidation level for a position"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    qty: float
    leverage: float
    liquidation_price: float
    distance_pct: float  # % away from liquidation
    margin_ratio: float  # Current margin usage
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LiquidationEvent:
    """Liquidation warning event"""
    timestamp: datetime
    symbol: str
    side: str
    liquidation_price: float
    current_price: float
    distance_pct: float
    severity: str  # 'WARNING', 'CRITICAL', 'EMERGENCY'
    description: str
    recommended_action: str


class LiquidationGuard:
    """
    Monitors and prevents liquidation events.
    
    Features:
    - Calculate liquidation price for each position
    - Warn when approaching liquidation (10% default)
    - Auto-close when critical (5% default)
    - Emergency mode if multiple positions at risk
    
    Liquidation Price Formula:
    - LONG: liquidation_price = entry_price * (1 - 1/leverage + fee_margin)
    - SHORT: liquidation_price = entry_price * (1 + 1/leverage - fee_margin)
    
    Where fee_margin accounts for trading fees (maker/taker)
    """

    def __init__(
        self,
        warning_threshold_pct: float = 10.0,
        critical_threshold_pct: float = 5.0,
        emergency_threshold_pct: float = 2.0,
        maker_fee_pct: float = 0.02,
        taker_fee_pct: float = 0.05,
    ):
        """
        Args:
            warning_threshold_pct: Distance from liquidation to warn (default 10%)
            critical_threshold_pct: Distance to trigger closeout (default 5%)
            emergency_threshold_pct: Distance to trigger emergency mode (default 2%)
            maker_fee_pct: Maker fee percentage (0.02 = 0.02%)
            taker_fee_pct: Taker fee percentage (0.05 = 0.05%)
        """
        self.warning_threshold_pct = warning_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self.emergency_threshold_pct = emergency_threshold_pct
        self.maker_fee_pct = maker_fee_pct / 100  # Convert to decimal
        self.taker_fee_pct = taker_fee_pct / 100

        self.liquidation_events: List[LiquidationEvent] = []
        self.critical_positions: List[str] = []  # Symbols at critical threshold
        self.emergency_mode: bool = False

    def calculate_liquidation_price(
        self,
        side: str,
        entry_price: float,
        leverage: float,
    ) -> float:
        """
        Calculate liquidation price for position.
        
        Args:
            side: 'LONG' or 'SHORT'
            entry_price: Entry price for position
            leverage: Leverage multiplier (1.0-125.0)
            
        Returns:
            Liquidation price
        """
        if leverage <= 1.0:
            return 0.0  # No leverage, no liquidation risk

        # Average fee (rough estimate of fees paid)
        avg_fee = (self.maker_fee_pct + self.taker_fee_pct) / 2

        if side == "LONG":
            # LONG: liquidation = entry * (1 - 1/leverage + fee)
            # When price drops to this, margin is exhausted
            liquidation = entry_price * (1.0 - (1.0 / leverage) + avg_fee)
        else:  # SHORT
            # SHORT: liquidation = entry * (1 + 1/leverage - fee)
            # When price rises to this, margin is exhausted
            liquidation = entry_price * (1.0 + (1.0 / leverage) - avg_fee)

        return max(liquidation, 0.0001)  # Prevent zero/negative values

    def calculate_distance_to_liquidation(
        self,
        side: str,
        current_price: float,
        liquidation_price: float,
    ) -> float:
        """
        Calculate % distance from current price to liquidation.
        
        Args:
            side: 'LONG' or 'SHORT'
            current_price: Current market price
            liquidation_price: Liquidation price
            
        Returns:
            Percentage distance (positive = safe, negative = past liquidation)
        """
        if liquidation_price <= 0:
            return 100.0  # Safe if no liquidation price

        if side == "LONG":
            # Distance = (current - liquidation) / liquidation * 100
            distance = ((current_price - liquidation_price) / liquidation_price) * 100
        else:  # SHORT
            # Distance = (liquidation - current) / liquidation * 100
            distance = ((liquidation_price - current_price) / liquidation_price) * 100

        return distance

    def check_position_liquidation(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        qty: float,
        leverage: float,
    ) -> Optional[LiquidationLevel]:
        """
        Check if position is at liquidation risk.
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            current_price: Current price
            qty: Position quantity
            leverage: Leverage multiplier
            
        Returns:
            LiquidationLevel if at risk, None if healthy
        """
        liquidation_price = self.calculate_liquidation_price(side, entry_price, leverage)
        distance_pct = self.calculate_distance_to_liquidation(side, current_price, liquidation_price)

        # Only return if at some level of risk
        if distance_pct > 0:  # Still safe
            return None

        # At risk - return level info
        margin_ratio = 1.0 - (distance_pct / 100.0)  # Margin used

        return LiquidationLevel(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            qty=qty,
            leverage=leverage,
            liquidation_price=liquidation_price,
            distance_pct=distance_pct,
            margin_ratio=margin_ratio,
            timestamp=datetime.utcnow(),
        )

    def assess_liquidation_risk(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        qty: float,
        leverage: float,
    ) -> Optional[LiquidationEvent]:
        """
        Assess position and return event if triggered.
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Entry price
            current_price: Current price
            qty: Position quantity
            leverage: Leverage multiplier
            
        Returns:
            LiquidationEvent if warning/critical/emergency, None if healthy
        """
        liquidation_price = self.calculate_liquidation_price(side, entry_price, leverage)
        distance_pct = self.calculate_distance_to_liquidation(side, current_price, liquidation_price)

        # Use absolute distance for threshold comparison
        abs_distance = abs(distance_pct)

        # Check if within any threshold
        if abs_distance > self.warning_threshold_pct:
            return None  # Safe, outside all thresholds

        # Determine severity (thresholds are smallest to largest)
        if abs_distance <= self.emergency_threshold_pct:
            severity = "EMERGENCY"
            action = "IMMEDIATE CLOSEOUT REQUIRED - Liquidation imminent"
            self.emergency_mode = True
        elif abs_distance <= self.critical_threshold_pct:
            severity = "CRITICAL"
            action = "AUTO-CLOSEOUT TRIGGERED - Position too risky"
            if symbol not in self.critical_positions:
                self.critical_positions.append(symbol)
        else:  # Within warning threshold
            severity = "WARNING"
            action = "REDUCE POSITION - Liquidation approaching"

        event = LiquidationEvent(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            liquidation_price=liquidation_price,
            current_price=current_price,
            distance_pct=distance_pct,
            severity=severity,
            description=f"{symbol} {side} at {abs_distance:.1f}% from liquidation",
            recommended_action=action,
        )

        self.liquidation_events.append(event)
        return event

    def assess_portfolio_risk(
        self,
        positions: Dict[str, dict],
    ) -> Dict[str, any]:
        """
        Assess all positions for liquidation risk.
        
        Args:
            positions: Dict of symbol -> {side, entry_price, current_price, qty, leverage}
            
        Returns:
            Dict with {emergency_mode, critical_count, warning_count, events}
        """
        result = {
            "emergency_mode": self.emergency_mode,
            "critical_count": 0,
            "warning_count": 0,
            "events": [],
        }

        for symbol, pos in positions.items():
            event = self.assess_liquidation_risk(
                symbol=symbol,
                side=pos["side"],
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                qty=pos["qty"],
                leverage=pos["leverage"],
            )

            if event:
                result["events"].append(event)
                if event.severity == "CRITICAL":
                    result["critical_count"] += 1
                elif event.severity == "WARNING":
                    result["warning_count"] += 1

        return result

    def should_closeout_position(self, symbol: str) -> bool:
        """Check if position should be auto-closed"""
        return symbol in self.critical_positions

    def should_enter_emergency_mode(self) -> bool:
        """Check if emergency mode is active"""
        return self.emergency_mode

    def reset_emergency_mode(self):
        """Reset emergency mode after manual intervention"""
        self.emergency_mode = False
        self.critical_positions.clear()
        logger.info("Emergency mode reset")

    def get_liquidation_status(self) -> Dict[str, any]:
        """Get current liquidation status"""
        return {
            "emergency_mode": self.emergency_mode,
            "critical_positions": len(self.critical_positions),
            "recent_events": len([e for e in self.liquidation_events[-10:]]),
            "total_events": len(self.liquidation_events),
        }

    def get_recent_events(self, hours: int = 1) -> List[LiquidationEvent]:
        """Get liquidation events from last N hours"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [e for e in self.liquidation_events if e.timestamp >= cutoff]
