# config/shared_state.py

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import RLock

STATE_FILE = os.getenv("SHARED_STATE_FILE", "logs/shared_state.json")


@dataclass
class SharedControlState:
    mode: str = "shadow"
    timeframe: str = "15m"
    paused: bool = False
    live_unlock: bool = False
    retrain_on_demand: bool = False
    retrain_on_demand_ts: float = 0.0
    retrain_interval_seconds: int = 900
    last_retrain_timestamp: float = 0.0

    # Strategy config (persistable from dashboard sliders)
    risk_per_trade: float = 0.01        # 1% — stored as fraction (0.01)
    max_active_positions: int = 2
    strategy_min_prob: float = 0.50    # stored as fraction (0.50)
    cooldown_minutes: int = 30

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls) -> "SharedControlState":
        if not os.path.exists(STATE_FILE):
            return cls()
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            return cls(**data)
        except Exception:
            return cls()

    def save(self) -> None:
        path = Path(STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


_global_lock = RLock()
_instance = None


def get_shared_state() -> SharedControlState:
    global _instance
    with _global_lock:
        if _instance is None:
            _instance = SharedControlState.load()
        return _instance


def update_shared_state(**kwargs) -> SharedControlState:
    state = get_shared_state()
    for k, v in kwargs.items():
        if hasattr(state, k):
            setattr(state, k, v)
    state.save()
    return state