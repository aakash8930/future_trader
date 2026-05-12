# train/incremental_trainer.py

"""
Background incremental trainer.
Every N seconds (default 15 minutes), fetches fresh live data from Binance
and fine-tunes existing MLP and XGBoost models.
Supports on-demand trigger via shared_state.
"""

import os
import sys
import time
import json
import torch
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread, Event, RLock
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Module-level trainer state
_trainer: Optional["IncrementalTrainer"] = None
_trainer_lock = RLock()

FEATURE_COLUMNS = [
    "ema_fast", "ema_slow", "ema_spread", "dist_ema200", "ema_fast_slope",
    "rsi", "rsi_delta", "ret", "vol", "volume_zscore", "atr_pct", "adx", "breakout_strength",
    "obv", "macd_line", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "vwap",
    "volume_momentum", "momentum_5", "atr_regime",
]


def _build_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    stop_atr_mult: float = 2.0,
    take_atr_mult: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build binary labels for each bar:
    1 = TP hit before SL, 0 = SL hit first or no hit within horizon.
    Also returns a validity mask (bars where labels are non-NaN).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values

    labels = np.full(len(df), np.nan)
    valid = np.full(len(df), False, dtype=bool)

    for i in range(len(df) - horizon):
        entry = close[i]
        stop = entry - stop_atr_mult * atr[i]
        tp = entry + take_atr_mult * atr[i]

        tp_hit = False
        sl_hit = False

        for j in range(i + 1, min(i + horizon + 1, len(df))):
            h, l = high[j], low[j]
            if h >= tp:
                tp_hit = True
                break
            if l <= stop:
                sl_hit = True
                break

        if tp_hit and not sl_hit:
            labels[i] = 1.0
            valid[i] = True
        elif sl_hit or (not tp_hit and not sl_hit):
            labels[i] = 0.0
            valid[i] = True

    return labels, valid


class IncrementalTrainer:
    """
    Background trainer thread that fine-tunes existing models with live data.
    """

    def __init__(
        self,
        timeframe: str = "15m",
        retrain_interval_seconds: int = 900,
        lookback: int = 800,
        epochs: int = 5,
        lr_mult: float = 0.1,
    ):
        self.timeframe = timeframe
        self.retrain_interval_seconds = retrain_interval_seconds
        self.lookback = lookback
        self.epochs = epochs
        self.lr_mult = lr_mult

        self._stop_event = Event()
        self._trigger_event = Event()
        self._thread: Optional[Thread] = None
        self._active = False
        self._last_symbols: list[str] = []
        self._last_retrain_time: float = 0.0

    def start(self):
        self._stop_event.clear()
        self._trigger_event.clear()
        self._thread = Thread(target=self._train_loop, daemon=True, name="IncrementalTrainer")
        self._thread.start()
        self._active = True
        print(f"[INCREMENTAL_TRAINER] Started (interval={self.retrain_interval_seconds}s, lookback={self.lookback})")

    def stop(self):
        self._stop_event.set()
        self._trigger_event.set()
        self._active = False
        if self._thread:
            self._thread.join(timeout=15)
        print("[INCREMENTAL_TRAINER] Stopped")

    def trigger_retrain(self):
        """Trigger an immediate retrain cycle."""
        self._trigger_event.set()

    def update_interval(self, seconds: int):
        self.retrain_interval_seconds = seconds
        print(f"[INCREMENTAL_TRAINER] Interval updated to {seconds}s")

    def _train_loop(self):
        import time as _time
        self._last_retrain_time = _time.time() + self.retrain_interval_seconds
        # Give the system a moment to settle before first retrain
        _time.sleep(5)

        while not self._stop_event.is_set():
            try:
                # Check for on-demand trigger
                from config.shared_state import get_shared_state, update_shared_state
                state = get_shared_state()

                if state.retrain_on_demand and state.retrain_on_demand_ts > 0:
                    print(f"[INCREMENTAL_TRAINER] On-demand retrain triggered")
                    self._do_retrain()
                    self._last_retrain_time = _time.time()
                    update_shared_state(
                        retrain_on_demand=False,
                        retrain_on_demand_ts=0.0,
                        last_retrain_timestamp=self._last_retrain_time,
                    )
                    continue

                # Interval-based retrain
                if _time.time() - self._last_retrain_time >= self.retrain_interval_seconds:
                    self._do_retrain()
                    self._last_retrain_time = _time.time()
                    update_shared_state(last_retrain_timestamp=self._last_retrain_time)

            except Exception as e:
                print(f"[INCREMENTAL_TRAINER] Loop error: {e}")

            # Wait for trigger or poll interval
            remaining = max(1, self.retrain_interval_seconds - (_time.time() - self._last_retrain_time))
            wait = min(60, remaining)
            self._trigger_event.wait(timeout=wait)
            self._trigger_event.clear()

    def _do_retrain(self):
        """Run incremental training for all symbols that have existing models."""
        print(f"[INCREMENTAL_TRAINER] Retrain cycle started at {datetime.now().isoformat()}")

        from data.fetcher import MarketDataFetcher

        exchange_name = os.getenv("EXCHANGE_NAME", "binance")
        raw_fallbacks = os.getenv("EXCHANGE_FALLBACKS", "bybit,kraken,okx")
        fallbacks = [s.strip().lower() for s in raw_fallbacks.split(",") if s.strip()]
        timeout_ms = int(os.getenv("EXCHANGE_TIMEOUT_MS", "20000"))
        exchange_type = os.getenv("EXCHANGE_TYPE", "cex")

        fetcher = MarketDataFetcher(
            exchange_name=exchange_name,
            fallback_exchanges=fallbacks,
            timeout_ms=timeout_ms,
            exchange_type=exchange_type,
        )

        symbols = self._get_symbols_with_models()
        self._last_symbols = symbols

        trained = 0
        failed = 0

        for symbol in symbols:
            try:
                ok = self._incremental_train_symbol(symbol, fetcher)
                if ok:
                    trained += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[INCREMENTAL_TRAINER] {symbol}: error - {e}")
                failed += 1

        print(f"[INCREMENTAL_TRAINER] Retrain cycle complete (trained={trained}, failed={failed})")

    def _get_symbols_with_models(self) -> list[str]:
        """Find all symbols that have at least one trained model."""
        symbols: list[str] = []
        models_dir = Path("models")
        if not models_dir.exists():
            return symbols

        for item in models_dir.iterdir():
            if not item.is_dir():
                continue
            # Skip LSTM and XGB subdirectories as primary entries
            name = item.name
            if "_lstm" in name or "_xgb" in name:
                continue
            if (item / "metadata.json").exists() and (item / "model.pt").exists():
                symbols.append(name.replace("_", "/"))

        # Also add symbols that only have XGB or LSTM
        for item in models_dir.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            if "_xgb" in name:
                base = name.replace("_xgb", "").replace("_", "/")
            elif "_lstm" in name:
                base = name.replace("_lstm", "").replace("_", "/")
            else:
                continue
            if base not in symbols and (item / "metadata.json").exists():
                symbols.append(base)

        return symbols

    def _incremental_train_symbol(self, symbol: str, fetcher) -> bool:
        """Fine-tune existing MLP/XGBoost models with new data. Returns True if trained."""
        from features.technicals import compute_core_features
        folder = symbol.replace("/", "_")

        # Fetch new data
        df = fetcher.fetch_ohlcv(symbol, self.timeframe, limit=self.lookback)
        if df is None or len(df) < 50:
            print(f"[INCREMENTAL_TRAINER] {symbol}: insufficient data ({len(df) if df else 0} bars)")
            return False

        df = compute_core_features(df)
        labels, valid = _build_triple_barrier_labels(df, horizon=12, stop_atr_mult=2.0, take_atr_mult=3.0)
        df = df.copy()
        df["target"] = labels

        df_valid = df[valid].copy()
        df_valid.dropna(inplace=True)
        if len(df_valid) < 50:
            return False

        # --- MLP incremental ---
        mlp_folder = f"models/{folder}"
        mlp_model_file = os.path.join(mlp_folder, "model.pt")
        if os.path.exists(mlp_model_file):
            # Load feature columns from this specific model's metadata
            mlp_meta_path = os.path.join(mlp_folder, "metadata.json")
            if os.path.exists(mlp_meta_path):
                with open(mlp_meta_path, "r", encoding="utf-8") as f:
                    mlp_meta = json.load(f)
                feature_cols = mlp_meta.get("feature_columns", FEATURE_COLUMNS)
            else:
                feature_cols = FEATURE_COLUMNS

            X = df_valid[feature_cols].values.astype(np.float32)
            y = df_valid["target"].values.astype(np.float32).reshape(-1, 1)

            try:
                self._incremental_train_mlp(symbol, folder, X, y, feature_cols)
            except Exception as e:
                print(f"[INCREMENTAL_TRAINER] {symbol} MLP: {e}")

        # --- XGBoost incremental ---
        xgb_folder = f"models/{folder}_xgb"
        xgb_model_file = os.path.join(xgb_folder, "model.save")
        if os.path.exists(xgb_model_file):
            # Load feature columns from this specific model's metadata
            xgb_meta_path = os.path.join(xgb_folder, "metadata.json")
            if os.path.exists(xgb_meta_path):
                with open(xgb_meta_path, "r", encoding="utf-8") as f:
                    xgb_meta = json.load(f)
                feature_cols = xgb_meta.get("feature_columns", FEATURE_COLUMNS)
            else:
                feature_cols = FEATURE_COLUMNS

            X = df_valid[feature_cols].values.astype(np.float32)
            y = df_valid["target"].values.astype(np.float32).reshape(-1, 1)

            try:
                self._incremental_train_xgb(symbol, folder, X, y, feature_cols)
            except Exception as e:
                print(f"[INCREMENTAL_TRAINER] {symbol} XGBoost: {e}")

        return True

    def _incremental_train_mlp(self, symbol: str, folder: str, X: np.ndarray, y: np.ndarray, feature_cols: list):
        """Fine-tune MLP with lower LR."""
        from sklearn.preprocessing import StandardScaler

        mlp_folder = f"models/{folder}"
        scaler_path = os.path.join(mlp_folder, "scaler.save")
        metadata_path = os.path.join(mlp_folder, "metadata.json")

        n_features = len(feature_cols)

        scaler: StandardScaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

        # Recreate the exact model architecture from training
        class DirectionNet(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        model = DirectionNet(n_features)
        state_dict = torch.load(os.path.join(mlp_folder, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.train()

        # Load metadata to get feature count and create model the same way training did
        metadata = json.load(open(metadata_path, "r"))
        n_features = len(metadata.get("feature_columns", FEATURE_COLUMNS))

        scaler: StandardScaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

        # Recreate the exact model architecture from training
        class DirectionNet(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        model = DirectionNet(n_features)
        state_dict = torch.load(os.path.join(mlp_folder, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.train()

        # Fine-tune with reduced LR
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        original_lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=original_lr * self.lr_mult)
        loss_fn = torch.nn.BCELoss()

        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        torch.save(model.state_dict(), os.path.join(mlp_folder, "model.pt"))

        # Update metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["last_incremental_train_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["incremental_train_count"] = metadata.get("incremental_train_count", 0) + 1
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[INCREMENTAL_TRAINER] {symbol} MLP updated ({self.epochs} epochs @ lr={original_lr * self.lr_mult})")

    def _incremental_train_xgb(self, symbol: str, folder: str, X: np.ndarray, y: np.ndarray, feature_cols: list):
        """Fine-tune XGBoost by adding training rounds."""
        from xgboost import XGBClassifier

        xgb_folder = f"models/{folder}_xgb"
        scaler_path = os.path.join(xgb_folder, "scaler.save")
        metadata_path = os.path.join(xgb_folder, "metadata.json")

        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

        xgb_model = joblib.load(os.path.join(xgb_folder, "model.save"))

        # Add a few more boosting rounds
        xgb_model.n_estimators = (xgb_model.n_estimators or 200) + self.epochs
        xgb_model.fit(X_scaled, y.flatten(), verbose=False)

        joblib.dump(xgb_model, os.path.join(xgb_folder, "model.save"))

        # Update metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["last_incremental_train_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["incremental_train_count"] = metadata.get("incremental_train_count", 0) + 1
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[INCREMENTAL_TRAINER] {symbol} XGBoost updated (n_estimators={xgb_model.n_estimators})")


# ---------------------------------------------------------------------------
# Global trainer singleton management
# ---------------------------------------------------------------------------

def get_trainer() -> IncrementalTrainer:
    global _trainer
    with _trainer_lock:
        if _trainer is None:
            from config.shared_state import get_shared_state
            state = get_shared_state()
            _trainer = IncrementalTrainer(
                timeframe=state.timeframe or "15m",
                retrain_interval_seconds=state.retrain_interval_seconds or 900,
            )
            _trainer.start()
        return _trainer


def stop_trainer():
    global _trainer
    with _trainer_lock:
        if _trainer is not None:
            _trainer.stop()
            _trainer = None


def update_retrain_interval(seconds: int):
    global _trainer
    with _trainer_lock:
        if _trainer is not None:
            _trainer.update_interval(seconds)


def get_trainer_status() -> dict:
    global _trainer
    with _trainer_lock:
        if _trainer is None:
            return {"active": False, "symbols": []}
        return {
            "active": _trainer._active,
            "symbols": _trainer._last_symbols,
            "interval": _trainer.retrain_interval_seconds,
        }
