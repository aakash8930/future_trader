
#features/technicals.py

import ta
import numpy as np
import pandas as pd


def compute_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core indicators used by:
    - models
    - regime detection
    - strategy
    """

    df = df.copy()

    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()
    df["ema_spread"] = (df["ema_fast"] - df["ema_slow"]) / df["close"].replace(0, np.nan)
    df["dist_ema200"] = (df["close"] - df["ema200"]) / df["ema200"].replace(0, np.nan)
    df["ema_fast_slope"] = df["ema_fast"].pct_change(3)

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["rsi_delta"] = df["rsi"].diff(3)

    df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(10).std()

    vol_mean = df["volume"].rolling(30).mean()
    vol_std = df["volume"].rolling(30).std().replace(0, np.nan)
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std

    atr = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]

    adx = ta.trend.ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["adx"] = adx.adx()

    rolling_high = df["high"].rolling(20).max().shift(1)
    df["breakout_strength"] = (df["close"] - rolling_high) / (df["atr"] + 1e-9)

    df.dropna(inplace=True)
    return df
