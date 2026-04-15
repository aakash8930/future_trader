
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

    # --- NEW FEATURES ---

    # OBV (On-Balance Volume)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    # MACD (12, 26, 9)
    macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands (20)
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_width"] = bb.bollinger_wband()
    df["bb_position"] = bb.bollinger_pband()

    # VWAP
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).rolling(14).sum() / df["volume"].rolling(14).sum()

    # Volume momentum
    df["volume_momentum"] = df["volume"].pct_change(5)

    # Price momentum
    df["momentum_5"] = df["close"].pct_change(5)

    # ATR regime (percentile rank of ATR over 50 bars)
    df["atr_regime"] = df["atr"].rolling(50).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)

    df.dropna(inplace=True)
    return df
