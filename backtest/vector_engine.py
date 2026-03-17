# backtest/vector_engine.py

import numpy as np
import pandas as pd

from data.fetcher import MarketDataFetcher
from models.direction import DirectionModel
from features.technicals import compute_core_features


class VectorBacktestEngine:
    """
        Ultra-fast vectorized backtest (signal-level, approximate).
        This engine is intentionally conservative and not the source of truth.
        Use historical simulator / shadow mode for execution-grade validation.

        Assumptions:
            - One-bar signal delay (enter on next bar)
            - Turnover-based cost model (fees + slippage)
            - No intrabar stop/TP sequencing

    Designed for:
      - strategy validation
      - threshold tuning
      - model comparison
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model_path: str,
        scaler_path: str,
        metadata_path: str,
        lookback: int = 300,
        fee_pct: float = 0.0012,
        slippage_pct: float = 0.0010,
        exchange_name: str = "binance",
        exchange_fallbacks: list = None,
        exchange_timeout_ms: int = 20000,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

        self.data = MarketDataFetcher(
            exchange_name=exchange_name,
            fallback_exchanges=exchange_fallbacks,
            timeout_ms=exchange_timeout_ms,
        )
        self.model = DirectionModel(model_path, scaler_path, metadata_path)

    def run(self, limit: int = 10_000) -> pd.DataFrame:
        df = self.data.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        df = compute_core_features(df)

        # ---- AI inference (vectorized loop, unavoidable) ----
        probs = []
        for i in range(self.lookback, len(df)):
            window = df.iloc[i - self.lookback : i + 1]
            probs.append(self.model.predict_proba(window))

        df = df.iloc[self.lookback :].copy()
        df["prob_up"] = probs

        # ---- Signals ----
        long_th = max(0.45, self.model.long_threshold)

        df["signal"] = 0
        df.loc[
            (df["prob_up"] >= long_th)
            & (df["atr_pct"] > 0.0015)
            & (df["adx"] >= 15)
            & (df["rsi"] >= 48)
            & (df["rsi"] <= 72),
            "signal",
        ] = 1

        # ---- Returns ----
        df["ret"] = df["close"].pct_change().shift(-1)
        df["position"] = df["signal"].shift(1).fillna(0)
        df["strategy_ret"] = df["position"] * df["ret"]

        # ---- Fees + slippage (charged on position changes) ----
        turn = df["position"].diff().abs().fillna(df["position"].abs())
        per_side_cost = self.fee_pct + self.slippage_pct
        df["fees"] = turn * per_side_cost
        df["net_ret"] = (df["strategy_ret"] - df["fees"]).fillna(0.0)

        # ---- Equity ----
        df["equity"] = (1 + df["net_ret"]).cumprod().fillna(1.0)

        return df
