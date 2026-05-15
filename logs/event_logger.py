import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock

EVENT_LOG_PATH = Path(os.getenv("EVENT_LOG_PATH", "logs/structured.jsonl"))


class EventLogger:
    VERSION = "1.0"

    def __init__(self):
        self._lock = Lock()
        self._path = EVENT_LOG_PATH
        self._ensure_dir()

    def _ensure_dir(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event_type: str,
        message: str,
        symbol: str = None,
        **kwargs,
    ):
        """Append a structured event to the configured structured JSONL log."""
        event = {
            "version": self.VERSION,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": event_type,
            "message": message,
            "symbol": symbol,
            **kwargs,
        }
        line = json.dumps(event, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

    def trade_open(self, symbol: str, message: str, **kwargs):
        self.emit("trade_open", message, symbol=symbol, **kwargs)

    def trade_close(self, symbol: str, message: str, **kwargs):
        self.emit("trade_close", message, symbol=symbol, **kwargs)

    def skip(self, symbol: str, message: str, reason: str = None, **kwargs):
        self.emit("skip", message, symbol=symbol, reason=reason, **kwargs)

    def system(self, message: str, **kwargs):
        self.emit("system", message, **kwargs)

    def universe_update(self, message: str, symbols: list = None, **kwargs):
        self.emit("universe_update", message, symbols=symbols, **kwargs)


# Global singleton
_event_logger = None


def get_event_logger() -> EventLogger:
    global _event_logger
    if _event_logger is None:
        _event_logger = EventLogger()
    return _event_logger

