#config/live.py

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


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(slots=True)
class LiveSettings:
    mode: str = "paper"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    excluded_symbols: list[str] = field(default_factory=list)
    timeframe: str = "15m"

    starting_balance_usdt: float = 500.0
    cooldown_minutes: int = 30
    risk_per_trade: float = 0.01

    max_active_positions: int = 2
    sleep_seconds: int = 900
    universe_refresh_minutes: int = 60

    require_model_quality: bool = True
    min_model_val_f1: float = 0.10
    min_model_val_precision: float = 0.10
    min_model_val_recall: float = 0.10

    lookback: int = 300

    # Strategy configuration
    strategy_min_prob: float = 0.50
    strategy_min_adx: float = 16.0
    strategy_min_atr_pct: float = 0.0011
    strategy_rsi_long_min: float = 42.0
    strategy_rsi_long_max: float = 66.0
    strategy_fee_pct_per_side: float = 0.0010
    strategy_slippage_pct_per_side: float = 0.0008
    strategy_min_expected_edge: float = 0.00020
    strategy_stop_atr_mult: float = 1.40
    strategy_take_atr_mult: float = 3.40
    strategy_trail_atr_mult: float = 0.95

    # Universe / selector configuration
    selector_top_k_multiplier: int = 2
    selector_min_atr_pct: float = 0.0010
    selector_soft_min_volume_ratio: float = 0.15

    # Exchange configuration
    exchange_name: str = "binance"
    exchange_fallbacks: list[str] = field(default_factory=lambda: ["bybit", "kraken", "okx"])
    exchange_timeout_ms: int = 20000

    @classmethod
    def from_env(cls) -> "LiveSettings":
        raw_symbols = os.getenv("TRADING_SYMBOLS", "BTC/USDT")
        symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]

        raw_excluded = os.getenv("EXCLUDED_SYMBOLS", "")
        excluded = [s.strip().upper() for s in raw_excluded.split(",") if s.strip()]
        if excluded:
            symbols = [s for s in symbols if s not in excluded]

        raw_fallbacks = os.getenv("EXCHANGE_FALLBACKS", "bybit,kraken,okx")
        fallbacks = [s.strip().lower() for s in raw_fallbacks.split(",") if s.strip()]

        return cls(
            mode=os.getenv("TRADING_MODE", "paper").strip().lower(),
            symbols=symbols,
            excluded_symbols=excluded,
            timeframe=os.getenv("TRADING_TIMEFRAME", "15m"),
            starting_balance_usdt=_env_float("PAPER_STARTING_BALANCE_USDT", 500.0),
            cooldown_minutes=_env_int("ENTRY_COOLDOWN_MINUTES", 30),
            risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
            max_active_positions=_env_int("MAX_ACTIVE_POSITIONS", 2),
            sleep_seconds=_env_int("LOOP_SLEEP_SECONDS", 900),
            universe_refresh_minutes=_env_int("UNIVERSE_REFRESH_MINUTES", 60),
            require_model_quality=_env_bool("REQUIRE_MODEL_QUALITY", True),
            min_model_val_f1=_env_float("MIN_MODEL_VAL_F1", 0.10),
            min_model_val_precision=_env_float("MIN_MODEL_VAL_PRECISION", 0.10),
            min_model_val_recall=_env_float("MIN_MODEL_VAL_RECALL", 0.10),
            lookback=_env_int("LOOKBACK_BARS", 300),
            strategy_min_prob=_env_float("STRATEGY_MIN_PROB", 0.50),
            strategy_min_adx=_env_float("STRATEGY_MIN_ADX", 16.0),
            strategy_min_atr_pct=_env_float("STRATEGY_MIN_ATR_PCT", 0.0011),
            strategy_rsi_long_min=_env_float("STRATEGY_RSI_LONG_MIN", 42.0),
            strategy_rsi_long_max=_env_float("STRATEGY_RSI_LONG_MAX", 66.0),
            strategy_fee_pct_per_side=_env_float("STRATEGY_FEE_PCT_PER_SIDE", 0.0010),
            strategy_slippage_pct_per_side=_env_float("STRATEGY_SLIPPAGE_PCT_PER_SIDE", 0.0008),
            strategy_min_expected_edge=_env_float("STRATEGY_MIN_EXPECTED_EDGE", 0.00020),
            strategy_stop_atr_mult=_env_float("STRATEGY_STOP_ATR_MULT", 1.40),
            strategy_take_atr_mult=_env_float("STRATEGY_TAKE_ATR_MULT", 3.40),
            strategy_trail_atr_mult=_env_float("STRATEGY_TRAIL_ATR_MULT", 0.95),
            selector_top_k_multiplier=_env_int("SELECTOR_TOP_K_MULTIPLIER", 2),
            selector_min_atr_pct=_env_float("SELECTOR_MIN_ATR_PCT", 0.0010),
            selector_soft_min_volume_ratio=_env_float("SELECTOR_SOFT_MIN_VOLUME_RATIO", 0.15),
            exchange_name=_env_str("EXCHANGE_NAME", "binance"),
            exchange_fallbacks=fallbacks,
            exchange_timeout_ms=_env_int("EXCHANGE_TIMEOUT_MS", 20000),
        )

    def validate(self) -> None:
        if self.mode not in {"paper", "shadow", "live"}:
            raise ValueError("Invalid TRADING_MODE")

        if not self.symbols:
            raise ValueError("No trading symbols configured")

        if self.lookback < 220:
            raise ValueError("LOOKBACK_BARS must be >= 220")

        if self.strategy_rsi_long_min >= self.strategy_rsi_long_max:
            raise ValueError("STRATEGY_RSI_LONG_MIN must be less than STRATEGY_RSI_LONG_MAX")

        if self.max_active_positions < 1:
            raise ValueError("MAX_ACTIVE_POSITIONS must be >= 1")

        if self.selector_top_k_multiplier < 1:
            raise ValueError("SELECTOR_TOP_K_MULTIPLIER must be >= 1")
