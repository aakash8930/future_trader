# config/live.py

import os
from dataclasses import dataclass, field

LIVE_UNLOCK_TOKEN = "YES_I_UNDERSTAND"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


@dataclass(slots=True)
class LiveSettings:
    mode: str = "paper"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "15m"

    starting_balance_usdt: float = 500.0
    cooldown_minutes: int = 30
    risk_per_trade: float = 0.01

    max_active_positions: int = 2
    sleep_seconds: int = 900

    require_model_quality: bool = True
    min_model_val_f1: float = 0.10
    min_model_val_precision: float = 0.10
    min_model_val_recall: float = 0.10

    lookback: int = 300

    @classmethod
    def from_env(cls) -> "LiveSettings":
        raw_symbols = os.getenv("TRADING_SYMBOLS", "BTC/USDT")
        symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]

        # allow the operator to blacklist/skip certain symbols via env var
        # (useful for temporarily disabling a troubled market without
        # editing code or the main list of symbols).
        raw_exclude = os.getenv("EXCLUDED_SYMBOLS", "")
        if raw_exclude:
            excluded = {s.strip().upper() for s in raw_exclude.split(",") if s.strip()}
            symbols = [s for s in symbols if s not in excluded]

        return cls(
            mode=os.getenv("TRADING_MODE", "paper").strip().lower(),
            symbols=symbols,
            timeframe=os.getenv("TRADING_TIMEFRAME", "15m"),
            starting_balance_usdt=_env_float("PAPER_STARTING_BALANCE_USDT", 500.0),
            cooldown_minutes=_env_int("ENTRY_COOLDOWN_MINUTES", 30),
            risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
            max_active_positions=_env_int("MAX_ACTIVE_POSITIONS", 2),
            sleep_seconds=_env_int("LOOP_SLEEP_SECONDS", 900),
            require_model_quality=_env_bool("REQUIRE_MODEL_QUALITY", True),
            min_model_val_f1=_env_float("MIN_MODEL_VAL_F1", 0.10),
            min_model_val_precision=_env_float("MIN_MODEL_VAL_PRECISION", 0.10),
            min_model_val_recall=_env_float("MIN_MODEL_VAL_RECALL", 0.10),
            lookback=_env_int("LOOKBACK_BARS", 300),
        )

    def validate(self) -> None:
        if self.mode not in {"paper", "shadow", "live"}:
            raise ValueError("Invalid TRADING_MODE")

        if not self.symbols:
            raise ValueError("No trading symbols configured")
        # warn if exclusion removed everything
        if os.getenv("EXCLUDED_SYMBOLS") and not self.symbols:
            raise ValueError("EXCLUDED_SYMBOLS filtered out all trading symbols")

        if self.lookback < 220:
            raise ValueError("LOOKBACK_BARS must be >= 220")