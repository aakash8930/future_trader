import os
import numpy as np

from data.fetcher import MarketDataFetcher
from features.technicals import compute_core_features


def _has_trained_model(symbol: str) -> bool:
    folder = os.path.join("models", symbol.replace("/", "_"))
    return (
        os.path.exists(os.path.join(folder, "model.pt"))
        and os.path.exists(os.path.join(folder, "scaler.save"))
        and os.path.exists(os.path.join(folder, "metadata.json"))
    )


class CoinSelector:
    """
    Scores symbols for live trading.

    Design goals:
    - uses only closed candles
    - avoids selecting symbols the runner will almost certainly reject
    - strongly prefers symbols already aligned with strategy rules
    """

    DEFAULT_SYMBOLS = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "AVAX/USDT",
        "LINK/USDT",
        "DOGE/USDT",
        "BNB/USDT",
    ]

    def __init__(
        self,
        timeframe: str = "15m",
        lookback: int = 240,
        top_k: int = 4,
        min_atr_pct: float = 0.0008,
        soft_min_volume_ratio: float = 0.15,
        rsi_long_min: float
