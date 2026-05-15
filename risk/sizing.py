# risk/sizing.py

def fixed_fractional_size(
    balance: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    max_position_notional_pct: float = 1.0,
    leverage: float = 1.0,
) -> float:
    """
    Fixed-fractional position sizing with leverage support.

    risk_pct = fraction of balance you are willing to lose
    leverage = position leverage multiplier
    
    NOTE: This function assumes leverage multiplies the final PnL.
    To maintain constant risk_pct even with leverage, we reduce position size by the leverage factor.
    """

    risk_amount = balance * risk_pct
    per_unit_risk = abs(entry_price - stop_price)

    if per_unit_risk <= 0:
        return 0.0

    # Adjust for leverage: if leverage multiplies PnL, reduce qty proportionally
    adjusted_risk_pct = risk_pct / leverage if leverage > 0 else risk_pct
    adjusted_risk_amount = balance * adjusted_risk_pct
    
    qty = adjusted_risk_amount / per_unit_risk

    # Safety caps: avoid sizing above available capital and optional notional cap.
    max_qty_by_balance = balance / entry_price if entry_price > 0 else 0.0
    max_qty_by_notional = (
        (balance * max_position_notional_pct) / entry_price
        if entry_price > 0 and max_position_notional_pct > 0
        else 0.0
    )

    qty = min(qty, max_qty_by_balance, max_qty_by_notional)
    return max(0.0, qty)
