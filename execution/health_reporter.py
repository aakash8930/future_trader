import json
import time
from datetime import datetime, timezone
from pathlib import Path


class ServiceHealthReporter:
    def __init__(self, path: str = "runtime/health.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()

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
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "uptime": round(time.time() - self.started_at, 2),
            "balance": float(balance),
            "exchange_connected": bool(exchange_connected),
            "open_positions": int(open_positions),
            "last_candle_time": last_candle_time,
        }
        if extra:
            payload.update(extra)

        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)
