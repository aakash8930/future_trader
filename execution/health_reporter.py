import json
import time
from datetime import datetime, timezone
from pathlib import Path


class ServiceHealthReporter:
    def __init__(self, path: str = "runtime/health.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        
        # Metrics tracking
        self.trades_today = 0
        self.trades_long = 0
        self.trades_short = 0
        self.reconnect_count = 0
        self.error_count = 0
        self.last_trade_time = None
        self.session_pnl = 0.0
        self.websocket_connected = False

    def increment_trades(self, side: str = "LONG"):
        """Track trade counts."""
        self.trades_today += 1
        if side.upper() == "LONG":
            self.trades_long += 1
        elif side.upper() == "SHORT":
            self.trades_short += 1
        self.last_trade_time = datetime.now(timezone.utc).isoformat()

    def register_trade_pnl(self, pnl: float):
        """Register trade PnL for session summary."""
        self.session_pnl += pnl

    def increment_reconnects(self):
        """Track reconnection attempts."""
        self.reconnect_count += 1

    def increment_errors(self):
        """Track error count."""
        self.error_count += 1

    def set_websocket_status(self, connected: bool):
        """Update websocket connection status."""
        self.websocket_connected = connected

    def reset_daily(self):
        """Reset daily counters (call once per day)."""
        self.trades_today = 0

    def write(
        self,
        status: str,
        balance: float,
        exchange_connected: bool,
        open_positions: int,
        last_candle_time: str | None,
        extra: dict | None = None,
    ) -> None:
        payload = {
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "uptime_seconds": round(time.time() - self.started_at, 2),
            "balance": float(balance),
            "exchange_connected": bool(exchange_connected),
            "websocket_connected": bool(self.websocket_connected),
            "open_positions": int(open_positions),
            "last_candle_time": last_candle_time,
            "metrics": {
                "trades_today": self.trades_today,
                "trades_long": self.trades_long,
                "trades_short": self.trades_short,
                "session_pnl": round(self.session_pnl, 4),
                "last_trade_time": self.last_trade_time,
                "reconnect_attempts": self.reconnect_count,
                "error_count": self.error_count,
            }
        }
        if extra:
            payload.update(extra)

        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)
