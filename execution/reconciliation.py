# execution/reconciliation.py
"""
Trade Reconciliation Engine

Ensures that local PnL calculations match exchange reality.
Every trade is recorded with full audit trail for later verification.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict
import json
import uuid

from execution.database import get_db


@dataclass
class TradeRecord:
    """
    Complete audit trail for a single trade.
    
    Stores all information needed to:
    - Verify PnL calculation
    - Reconcile against exchange
    - Debug issues
    - Calculate statistics
    """
    # Trade identity
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    symbol: str = ""
    
    # Entry details
    entry_side: str = ""  # "LONG" or "SHORT"
    entry_price_local: float = 0.0  # Our calculated entry
    entry_qty: float = 0.0
    entry_leverage: float = 1.0
    entry_fee_usd: float = 0.0  # Fee paid at entry
    entry_time: str = ""
    
    # Exit details
    exit_price_local: float = 0.0  # Our calculated exit
    exit_qty: float = 0.0  # Usually same as entry_qty unless partial close
    exit_fee_usd: float = 0.0  # Fee paid at exit
    exit_time: str = ""
    exit_reason: str = ""  # "take_profit", "stop_loss", "manual"
    
    # PnL calculation
    pnl_before_fees: float = 0.0  # Raw (exit - entry) * qty * leverage
    slippage_usd: float = 0.0  # Estimated slippage
    pnl_net: float = 0.0  # pnl_before_fees - fees - slippage
    
    # Account state
    balance_before: float = 0.0
    balance_after: float = 0.0
    
    # Trade signal & metadata
    entry_reason: str = ""  # "signal", "breakout", "momentum", etc.
    model_confidence: float = 0.0
    regime: str = ""
    atr: float = 0.0
    adx: float = 0.0
    
    # Validation
    is_reconciled: bool = False
    reconciliation_notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TradeRecord":
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate trade record has all required fields.
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not self.symbol:
            return False, "Missing symbol"
        if not self.entry_side in ["LONG", "SHORT"]:
            return False, f"Invalid side: {self.entry_side}"
        if self.entry_qty <= 0:
            return False, f"Invalid entry qty: {self.entry_qty}"
        if self.entry_price_local <= 0:
            return False, f"Invalid entry price: {self.entry_price_local}"
        if self.exit_price_local <= 0:
            return False, f"Invalid exit price: {self.exit_price_local}"
        if self.entry_leverage < 1.0 or self.entry_leverage > 125.0:
            return False, f"Invalid leverage: {self.entry_leverage}"
        return True, ""
    
    def calculate_pnl(self) -> float:
        """Recalculate PnL from prices (for verification)."""
        if self.entry_side == "LONG":
            raw_pnl = (self.exit_price_local - self.entry_price_local) * self.exit_qty
        else:  # SHORT
            raw_pnl = (self.entry_price_local - self.exit_price_local) * self.exit_qty
        
        # Apply leverage
        pnl_with_leverage = raw_pnl * self.entry_leverage
        
        # Deduct fees and slippage
        total_fees = self.entry_fee_usd + self.exit_fee_usd
        pnl_net = pnl_with_leverage - total_fees - self.slippage_usd
        
        return pnl_net


from typing import Tuple


class ReconciliationEngine:
    """
    Records and validates all trades for auditability.
    """
    
    def __init__(self):
        self.db = get_db()
        self.trades: Dict[str, TradeRecord] = {}
        self._create_trades_table()
    
    def _create_trades_table(self):
        """Ensure trades table exists in database."""
        try:
            # Create trades table
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_side TEXT NOT NULL,
                    entry_price_local REAL NOT NULL,
                    entry_qty REAL NOT NULL,
                    entry_leverage REAL NOT NULL,
                    entry_fee_usd REAL,
                    entry_time TEXT,
                    exit_price_local REAL NOT NULL,
                    exit_qty REAL NOT NULL,
                    exit_fee_usd REAL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    pnl_before_fees REAL NOT NULL,
                    slippage_usd REAL,
                    pnl_net REAL NOT NULL,
                    balance_before REAL NOT NULL,
                    balance_after REAL NOT NULL,
                    entry_reason TEXT,
                    model_confidence REAL,
                    regime TEXT,
                    atr REAL,
                    adx REAL,
                    is_reconciled INTEGER DEFAULT 0,
                    reconciliation_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(trade_id)
                )
            """)
            
            # Create index on symbol
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
            """)
            
            # Create index on timestamp
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)
            """)
            
            # Create index on reconciliation status
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_is_reconciled ON trades(is_reconciled)
            """)
        except Exception as e:
            print(f"[RECONCILIATION] Error creating trades table: {e}")
    
    def record_trade(self, trade: TradeRecord) -> bool:
        """
        Record a completed trade to database.
        
        Args:
            trade: TradeRecord object
            
        Returns:
            True if recorded successfully
        """
        # Validate trade
        is_valid, error = trade.validate()
        if not is_valid:
            print(f"[RECONCILIATION] Invalid trade: {error}")
            return False
        
        # Store in memory cache
        self.trades[trade.trade_id] = trade
        
        # Store in database
        try:
            self.db.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, timestamp, symbol, entry_side, entry_price_local,
                    entry_qty, entry_leverage, entry_fee_usd, entry_time,
                    exit_price_local, exit_qty, exit_fee_usd, exit_time, exit_reason,
                    pnl_before_fees, slippage_usd, pnl_net,
                    balance_before, balance_after,
                    entry_reason, model_confidence, regime, atr, adx,
                    is_reconciled, reconciliation_notes
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                trade.trade_id,
                trade.timestamp,
                trade.symbol,
                trade.entry_side,
                trade.entry_price_local,
                trade.entry_qty,
                trade.entry_leverage,
                trade.entry_fee_usd,
                trade.entry_time,
                trade.exit_price_local,
                trade.exit_qty,
                trade.exit_fee_usd,
                trade.exit_time,
                trade.exit_reason,
                trade.pnl_before_fees,
                trade.slippage_usd,
                trade.pnl_net,
                trade.balance_before,
                trade.balance_after,
                trade.entry_reason,
                trade.model_confidence,
                trade.regime,
                trade.atr,
                trade.adx,
                1 if trade.is_reconciled else 0,
                trade.reconciliation_notes,
            ))
            return True
        except Exception as e:
            print(f"[RECONCILIATION] Error recording trade: {e}")
            return False
    
    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Retrieve a trade record."""
        if trade_id in self.trades:
            return self.trades[trade_id]
        
        try:
            row = self.db.fetchone(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,)
            )
            if row:
                return TradeRecord.from_dict(dict(row))
        except Exception:
            pass
        
        return None
    
    def get_trades_for_symbol(self, symbol: str) -> List[TradeRecord]:
        """Get all trades for a symbol."""
        trades = []
        try:
            rows = self.db.fetchall(
                "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC",
                (symbol,)
            )
            for row in rows:
                trades.append(TradeRecord.from_dict(dict(row)))
        except Exception as e:
            print(f"[RECONCILIATION] Error fetching trades: {e}")
        
        return trades
    
    def get_all_trades(self, limit: int = 1000) -> List[TradeRecord]:
        """Get all trades."""
        trades = []
        try:
            rows = self.db.fetchall(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            for row in rows:
                trades.append(TradeRecord.from_dict(dict(row)))
        except Exception as e:
            print(f"[RECONCILIATION] Error fetching trades: {e}")
        
        return trades
    
    def get_statistics(self) -> Dict:
        """Calculate trading statistics from all records."""
        trades = self.get_all_trades(limit=10000)
        
        if not trades:
            return {"error": "No trades recorded"}
        
        pnl_values = [t.pnl_net for t in trades]
        winning_trades = [p for p in pnl_values if p > 0]
        losing_trades = [p for p in pnl_values if p < 0]
        
        total_pnl = sum(pnl_values)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_trades = len(trades)
        
        return {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": win_count / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "avg_win": sum(winning_trades) / len(winning_trades) if winning_trades else 0,
            "avg_loss": sum(losing_trades) / len(losing_trades) if losing_trades else 0,
            "profit_factor": (
                sum(winning_trades) / abs(sum(losing_trades))
                if losing_trades and sum(losing_trades) != 0
                else 0
            ),
            "expectancy": total_pnl / total_trades if total_trades > 0 else 0,
        }
    
    def export_trades_csv(self, filepath: str) -> bool:
        """Export all trades to CSV for analysis."""
        import csv
        
        trades = self.get_all_trades(limit=10000)
        if not trades:
            return False
        
        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(trades[0]).keys())
                writer.writeheader()
                for trade in trades:
                    writer.writerow(asdict(trade))
            return True
        except Exception as e:
            print(f"[RECONCILIATION] Error exporting CSV: {e}")
            return False


# Global reconciliation engine singleton
_reconciliation_engine = None

def get_reconciliation_engine() -> ReconciliationEngine:
    """Get or create the global reconciliation engine."""
    global _reconciliation_engine
    if _reconciliation_engine is None:
        _reconciliation_engine = ReconciliationEngine()
    return _reconciliation_engine
