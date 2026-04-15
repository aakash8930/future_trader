from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    mode: str
    uptime_seconds: float


class PositionState(BaseModel):
    has_position: bool
    symbol: Optional[str] = None
    side: Optional[str] = None
    entry_price: Optional[float] = None
    avg_entry: Optional[float] = None
    qty: Optional[float] = None
    entry_time: Optional[str] = None
    add_count: int = 0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


class SystemStatus(BaseModel):
    mode: str
    symbols: List[str]
    balance: float
    active_positions: List[PositionState]
    starting_balance: float


class PerformanceStats(BaseModel):
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    total_pnl: float
    sharpe_ratio: Optional[float] = None


class EquityPoint(BaseModel):
    timestamp: str
    balance: float
    pnl: Optional[float] = None


class TradeRow(BaseModel):
    id: int
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    avg_entry: float
    exit_price: Optional[float]
    qty: float
    pnl: Optional[float]
    balance: float
    prob: float
    threshold: float
    atr: float
    atr_pct: float
    adx: float
    regime: str
    stop_loss: float
    take_profit: float
    exit_reason: Optional[str]
    add_count: int


class TradeEvent(BaseModel):
    version: str = "1.0"
    timestamp: str
    type: str  # trade_open, trade_close, skip, system, universe_update
    symbol: Optional[str] = None
    message: str
    # trade_open/close
    side: Optional[str] = None
    entry_price: Optional[float] = None
    qty: Optional[float] = None
    prob: Optional[float] = None
    # trade_close
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    # skip
    reason: Optional[str] = None
    # universe_update
    symbols: Optional[List[str]] = None


class ApiResponse(BaseModel):
    ok: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: str
