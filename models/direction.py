# models/direction.py

import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from typing import Optional

from features.technicals import compute_core_features
from models.model_identity import MODEL_NAME, MODEL_VERSION


class DirectionModel:
    """
    Base class for directional AI models.
    Provides common infrastructure; subclasses handle type-specific loading/prediction.
    """

    @classmethod
    def for_symbol(cls, symbol: str, model_type: Optional[str] = None) -> "DirectionModel":
        """
        Load a model for symbol.

        If model_type is specified, load that specific type from its subdirectory.
        If model_type is None, auto-detect by checking XGB -> LSTM -> MLP order.
        """
        folder = symbol.replace('/', '_')

        if model_type:
            # Explicit type: load from specific subdirectory
            subfolder_map = {
                "XGB":  (f"models/{folder}_xgb", "model.save"),
                "LSTM": (f"models/{folder}_lstm", "model.pt"),
                "MLP":  (f"models/{folder}", "model.pt"),
            }
            if model_type not in subfolder_map:
                raise ValueError(f"Unknown model_type: {model_type}")
            subfolder, model_file = subfolder_map[model_type]
            if os.path.exists(os.path.join(subfolder, model_file)):
                return cls._factory(model_type, subfolder, model_file)
            raise FileNotFoundError(f"No {model_type} model found for {symbol}")

        # Auto-detection: try each model type subdirectory in priority order
        for mt, subfolder, model_file in [
            ("XGB",  f"models/{folder}_xgb", "model.save"),
            ("LSTM", f"models/{folder}_lstm", "model.pt"),
            ("MLP",  f"models/{folder}", "model.pt"),
        ]:
            if os.path.exists(os.path.join(subfolder, model_file)):
                return cls._factory(mt, subfolder, model_file)

        raise FileNotFoundError(f"No trained model found for {symbol}")

    @classmethod
    def _factory(cls, model_type: str, subfolder: str, model_file: str) -> "DirectionModel":
        scaler_path = os.path.join(subfolder, "scaler.save")
        metadata_path = os.path.join(subfolder, "metadata.json")

        if model_type == "LSTM":
            return LSTMDirectionModel(
                os.path.join(subfolder, model_file),
                scaler_path,
                metadata_path,
                subfolder,
            )
        elif model_type == "XGB":
            return XGBoostDirectionModel(
                os.path.join(subfolder, model_file),
                scaler_path,
                metadata_path,
                subfolder,
            )
        else:
            return MLPDirectionModel(
                os.path.join(subfolder, model_file),
                scaler_path,
                metadata_path,
                subfolder,
            )

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        metadata_path: str,
        folder: str,
    ):
        self.folder = folder
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.model_name = MODEL_NAME
        self.model_version = MODEL_VERSION

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Use UTF-8 encoding explicitly to avoid Windows cp1252 issues
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata.get(
            "feature_columns",
            [
                "ema_fast", "ema_slow", "ema_spread", "dist_ema200", "ema_fast_slope",
                "rsi", "rsi_delta", "ret", "vol", "volume_zscore", "atr_pct", "adx", "breakout_strength",
            ],
        )
        self.model_name = self.metadata.get("model_name", self.model_name)
        self.model_version = self.metadata.get("model_version", self.model_version)
        self.model_type = self.metadata.get("model_type", "MLP")

        self.scaler = joblib.load(scaler_path)

        # Validate feature count (skip for LSTM: its scaler is on flattened sequences = window*features)
        scaler_features = getattr(self.scaler, "n_features_in_", None)
        feature_count = len(self.feature_columns)
        if scaler_features is not None and self.model_type != "LSTM" and int(scaler_features) != feature_count:
            raise ValueError(
                f"Feature mismatch: metadata={feature_count}, scaler={int(scaler_features)}"
            )

        opt_th = self.metadata.get("optimized_long_threshold")
        if opt_th is not None:
            self.long_threshold = float(opt_th)
            self.short_threshold = 1.0 - self.long_threshold
        else:
            self._init_thresholds()

    def _init_thresholds(self) -> None:
        metrics = self.metadata.get("metrics", {})
        pos_rate = float(metrics.get("val_positive_rate", 0.5))
        f1 = float(metrics.get("val_f1", 0.0))

        if pos_rate < 0.20:
            long_th = 0.45
        elif pos_rate < 0.30:
            long_th = 0.50
        else:
            long_th = 0.50

        if f1 < 0.20:
            long_th += 0.05

        self.long_threshold = float(np.clip(long_th, 0.40, 0.65))
        self.short_threshold = 1.0 - self.long_threshold

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Override in subclass for type-specific behavior."""
        raise NotImplementedError


class MLPDirectionModel(DirectionModel):
    """Standard MLP model (input_dim → 32 → 16 → 1)."""

    def __init__(self, model_path: str, scaler_path: str, metadata_path: str, folder: str):
        super().__init__(model_path, scaler_path, metadata_path, folder)
        self.model = self._load_mlp_model(model_path)

    def _load_mlp_model(self, model_path: str) -> torch.nn.Module:
        state_dict = torch.load(model_path, map_location="cpu")

        w0 = state_dict["net.0.weight"]
        w2 = state_dict["net.2.weight"]

        input_dim = w0.shape[1]
        hidden_1 = w0.shape[0]
        hidden_2 = w2.shape[0]

        class AIModel(torch.nn.Module):
            def __init__(self, in_dim, h1, h2):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, h1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(h1, h2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(h2, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        model = AIModel(input_dim, hidden_1, hidden_2)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict_proba(self, df: pd.DataFrame) -> float:
        if "ema200" not in df.columns or "atr_pct" not in df.columns:
            df = compute_core_features(df)

        if df is None or df.empty:
            return 0.5

        row = df.iloc[-1]

        try:
            features = np.array(
                [[row[col] for col in self.feature_columns]],
                dtype=np.float32,
            )
        except KeyError:
            return 0.5

        features = self.scaler.transform(features)
        tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            prob = float(self.model(tensor).item())

        return prob if 0.0 <= prob <= 1.0 else 0.5


class LSTMDirectionModel(DirectionModel):
    """LSTM sequence model with internal bar buffer."""

    LSTM_WINDOW = 16

    def __init__(self, model_path: str, scaler_path: str, metadata_path: str, folder: str):
        super().__init__(model_path, scaler_path, metadata_path, folder)
        self.window_size = self.metadata.get("window_size", self.LSTM_WINDOW)
        self._bar_buffer: list = []
        self.model = self._load_lstm_model(model_path)

    def _load_lstm_model(self, model_path: str) -> torch.nn.Module:
        state_dict = torch.load(model_path, map_location="cpu")

        # Detect dimensions from LSTM layer weights
        lstm_weight = state_dict["lstm.weight_hh_l0"]
        hidden_size = lstm_weight.shape[0] // 4
        input_size = state_dict["lstm.weight_ih_l0"].shape[1]

        class SeqModel(torch.nn.Module):
            def __init__(self, in_dim, hid):
                super().__init__()
                self.lstm = torch.nn.LSTM(in_dim, hid, batch_first=True, dropout=0.0)
                self.dropout = torch.nn.Dropout(0.2)
                self.fc = torch.nn.Linear(hid, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_out = lstm_out[:, -1, :]
                return self.sigmoid(self.fc(self.dropout(last_out)))

        model = SeqModel(input_size, hidden_size)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict_proba(self, df: pd.DataFrame) -> float:
        if "ema200" not in df.columns or "atr_pct" not in df.columns:
            df = compute_core_features(df)

        if df is None or df.empty:
            return 0.5

        # Pre-fill buffer from historical DataFrame bars (excluding latest)
        # so that fresh-session calls still work correctly
        if len(df) > 1 and len(self._bar_buffer) < self.window_size:
            lookback_bars = min(len(df) - 1, self.window_size)
            for i in range(len(df) - lookback_bars, len(df)):
                bar_features = np.array([[df.iloc[i][col] for col in self.feature_columns]], dtype=np.float32)
                self._bar_buffer.append(bar_features)

        # Get the latest bar's features
        row = df.iloc[-1]
        try:
            bar_features = np.array([[row[col] for col in self.feature_columns]], dtype=np.float32)
        except KeyError:
            return 0.5

        # Accumulate bars in buffer
        self._bar_buffer.append(bar_features)
        if len(self._bar_buffer) > self.window_size:
            self._bar_buffer.pop(0)

        # Need full window to make prediction
        if len(self._bar_buffer) < self.window_size:
            return 0.5

        # Build sequence: (1, window, features)
        seq = np.array(self._bar_buffer, dtype=np.float32)
        seq = seq.reshape(1, self.window_size, -1)

        # Flatten for scaler: (1, window*features)
        seq_flat = seq.reshape(1, -1)
        seq_flat = self.scaler.transform(seq_flat)
        seq = seq_flat.reshape(1, self.window_size, -1)

        tensor = torch.tensor(seq, dtype=torch.float32)
        with torch.no_grad():
            prob = float(self.model(tensor).item())

        return prob if 0.0 <= prob <= 1.0 else 0.5

    def reset_buffer(self):
        """Reset the bar buffer. Call when starting a new session."""
        self._bar_buffer = []


class XGBoostDirectionModel(DirectionModel):
    """XGBoost tabular model."""

    def __init__(self, model_path: str, scaler_path: str, metadata_path: str, folder: str):
        from xgboost import XGBClassifier
        super().__init__(model_path, scaler_path, metadata_path, folder)
        self.model = joblib.load(model_path)

    def predict_proba(self, df: pd.DataFrame) -> float:
        if "ema200" not in df.columns or "atr_pct" not in df.columns:
            df = compute_core_features(df)

        if df is None or df.empty:
            return 0.5

        row = df.iloc[-1]

        try:
            features = np.array(
                [[row[col] for col in self.feature_columns]],
                dtype=np.float32,
            )
        except KeyError:
            return 0.5

        features = self.scaler.transform(features)
        prob = float(self.model.predict_proba(features)[0, 1])

        return prob if 0.0 <= prob <= 1.0 else 0.5