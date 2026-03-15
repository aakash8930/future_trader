# backtest/engine.py

from typing import Dict, List

import pandas as pd

from data.fetcher import MarketDataFetcher
from models.direction import DirectionModel


class BacktestEngine:
    """
    Signal-level backtest engine.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model_path: str,
        scaler_path: str,
        metadata_path: str,
        lookback: int = 300,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback

        self.data = MarketDataFetcher()
        self.model = DirectionModel(model_path, scaler_path, metadata_path)

    def run(self, limit: int = 2000) -> pd.DataFrame:
        df = self.data.fetch_ohlcv(self.symbol, self.timeframe, limit)

        results: List[Dict] = []

        for i in range(self.lookback, len(df)):
            window = df.iloc[i - self.lookback : i + 1]
            last = window.iloc[-1]

            results.append({
                "time": last["time"],
                "price": last["close"],
                "prob_up": self.model.predict_proba(window),
            })

        return pd.DataFrame(results)
