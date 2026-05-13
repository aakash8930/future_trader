# risk/leverage_guard.py
"""
Leverage Guard - Enforce maximum leverage limits per trade and account.

Prevents over-leveraging which leads to quick liquidation and catastrophic losses.
"""

import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LeverageConfig:
    """Leverage limit configuration."""
    max_per_trade: float = 10.0      # Max leverage per single trade
    max_account: float = 5.0          # Max average leverage across all positions
    min_leverage: float = 1.0         # Minimum allowed (must be at least 1x)
    warn_threshold: float = 0.8       # Warn at 80% of max


class LeverageGuard:
    """
    Enforce leverage limits to prevent over-leveraging.
    
    Checks:
    - Per-trade leverage doesn't exceed max
    - Account-level leverage reasonable
    - Position size appropriate for account equity
    - Margin requirements satisfied
    """
    
    def __init__(
        self,
        max_per_trade: float = 10.0,
        max_account: float = 5.0,
        min_leverage: float = 1.0,
        warn_threshold: float = 0.8,
    ):
        """Initialize leverage guard with configurable limits."""
        self.config = LeverageConfig(
            max_per_trade=max_per_trade,
            max_account=max_account,
            min_leverage=min_leverage,
            warn_threshold=warn_threshold,
        )
        self.violations = []
        self.warnings = []
    
    def validate_leverage_for_trade(
        self,
        requested_leverage: float,
        symbol: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if requested leverage is acceptable for a new trade.
        
        Args:
            requested_leverage: Leverage requested (1.0 = 1x, 10.0 = 10x)
            symbol: Trading symbol
            
        Returns:
            (is_valid, reason) tuple
            - is_valid: True if leverage OK
            - reason: Violation reason if invalid
        """
        # Check minimum leverage
        if requested_leverage < self.config.min_leverage:
            reason = f"{symbol}: Leverage {requested_leverage}x below minimum {self.config.min_leverage}x"
            self.violations.append(reason)
            return (False, reason)
        
        # Check maximum per-trade leverage
        if requested_leverage > self.config.max_per_trade:
            reason = (
                f"{symbol}: Leverage {requested_leverage}x exceeds max "
                f"{self.config.max_per_trade}x per trade"
            )
            self.violations.append(reason)
            logger.error(f"[LEVERAGE GUARD] {reason}")
            return (False, reason)
        
        # Check if approaching warning threshold
        leverage_ratio = requested_leverage / self.config.max_per_trade
        if leverage_ratio >= self.config.warn_threshold:
            warn_msg = (
                f"{symbol}: Using {leverage_ratio:.0%} of max leverage "
                f"({requested_leverage}x of {self.config.max_per_trade}x)"
            )
            self.warnings.append(warn_msg)
            logger.warning(f"[LEVERAGE GUARD] {warn_msg}")
        
        return (True, None)
    
    def validate_account_leverage(
        self,
        positions: Dict,
        account_equity: float,
    ) -> Tuple[bool, str]:
        """
        Validate account-level leverage (sum of all open positions).
        
        Args:
            positions: Dict of {symbol: position_object}
            account_equity: Current account equity
            
        Returns:
            (is_valid, status_msg) tuple
        """
        if not positions or account_equity <= 0:
            return (True, "No positions or invalid equity")
        
        # Calculate total position value (sum of qty * entry_price * leverage)
        total_notional = 0.0
        position_details = []
        
        for symbol, position in positions.items():
            if position.qty == 0:
                continue
            
            notional = abs(position.qty * position.avg_entry * position.leverage)
            total_notional += notional
            position_details.append(
                f"  {symbol}: {position.qty:.4f} qty, {position.leverage}x = {notional:,.0f}"
            )
        
        # Calculate effective account leverage
        account_leverage = total_notional / account_equity if account_equity > 0 else 0
        
        # Check against max account leverage
        if account_leverage > self.config.max_account:
            reason = (
                f"Account leverage {account_leverage:.2f}x exceeds max {self.config.max_account}x. "
                f"Total notional: ${total_notional:,.0f}, Equity: ${account_equity:,.0f}"
            )
            self.violations.append(reason)
            logger.error(f"[LEVERAGE GUARD] {reason}")
            for detail in position_details:
                logger.error(detail)
            return (False, reason)
        
        # Warn if approaching threshold
        if account_leverage >= (self.config.max_account * self.config.warn_threshold):
            warn_msg = (
                f"Account leverage {account_leverage:.2f}x approaching max "
                f"{self.config.max_account}x. Total notional: ${total_notional:,.0f}"
            )
            self.warnings.append(warn_msg)
            logger.warning(f"[LEVERAGE GUARD] {warn_msg}")
        
        status = f"Account leverage: {account_leverage:.2f}x (max: {self.config.max_account}x)"
        return (True, status)
    
    def validate_position_size(
        self,
        qty: float,
        entry_price: float,
        leverage: float,
        account_equity: float,
        max_risk_pct: float = 0.02,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate position size doesn't exceed risk limits.
        
        Args:
            qty: Position quantity
            entry_price: Entry price
            leverage: Leverage multiplier
            account_equity: Account equity
            max_risk_pct: Max risk per trade as % of equity
            
        Returns:
            (is_valid, reason) tuple
        """
        if qty <= 0 or entry_price <= 0 or account_equity <= 0:
            return (True, None)
        
        # Notional value of position
        notional_value = qty * entry_price
        
        # Required margin (notional / leverage)
        required_margin = notional_value / leverage
        
        # Check if sufficient equity for margin
        if required_margin > account_equity * 0.95:  # Keep 5% buffer
            reason = (
                f"Position size too large. Required margin: ${required_margin:,.0f}, "
                f"Available equity: ${account_equity * 0.95:,.0f}"
            )
            self.violations.append(reason)
            logger.error(f"[LEVERAGE GUARD] {reason}")
            return (False, reason)
        
        # Check risk per trade
        margin_used_pct = required_margin / account_equity
        if margin_used_pct > max_risk_pct:
            warn_msg = (
                f"Position uses {margin_used_pct:.1%} of equity (max recommended: {max_risk_pct:.1%})"
            )
            self.warnings.append(warn_msg)
            logger.warning(f"[LEVERAGE GUARD] {warn_msg}")
        
        return (True, None)
    
    def get_violations_report(self) -> Dict:
        """Generate report of all violations and warnings."""
        return {
            "total_violations": len(self.violations),
            "total_warnings": len(self.warnings),
            "violations": self.violations,
            "warnings": self.warnings,
            "config": {
                "max_per_trade": self.config.max_per_trade,
                "max_account": self.config.max_account,
                "min_leverage": self.config.min_leverage,
                "warn_threshold": self.config.warn_threshold,
            }
        }
    
    def clear_history(self):
        """Clear violation and warning history."""
        self.violations = []
        self.warnings = []


def validate_leverage_for_entry(
    requested_leverage: float,
    symbol: str,
    max_leverage: float = 10.0,
) -> Tuple[bool, str]:
    """
    Quick validation of leverage for entry signal.
    
    Args:
        requested_leverage: Leverage being used
        symbol: Trading symbol
        max_leverage: Maximum allowed
        
    Returns:
        (is_valid, reason) tuple
    """
    if requested_leverage < 1.0:
        return (False, f"{symbol}: Invalid leverage {requested_leverage}x (must be >= 1.0x)")
    
    if requested_leverage > max_leverage:
        return (
            False,
            f"{symbol}: Leverage {requested_leverage}x exceeds max {max_leverage}x"
        )
    
    return (True, f"{symbol}: Leverage {requested_leverage}x approved")
