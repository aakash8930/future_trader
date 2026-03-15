# models/direction.py

import os
import json
import torch
import joblib
import numpy as np

from features.technicals import compute_core_features
from models.model_identity import MODEL_NAME, MODEL_VERSION


class DirectionModel:
    """
    Directional AI model.
    """

    @classmethod
    def for_symbol(cls, symbol: str) -> "DirectionModel":
        folder = f"models/{symbol.replace('/', '_')}"
        model_path = f"{folder}/model.pt"
        scaler_path = f"{folder}/scaler.save"
        metadata_path = f"{folder}/metadata.json"

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            raise FileNotFoundError(f"No trained model found for {symbol}")

        return cls(
            model_path=model_path,
            scaler_path=scaler_path,
            metadata_path=metadata_path,
        )

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        metadata_path: str,
    ):
        self.model_name = MODEL_NAME
        self.model_version = MODEL_VERSION

        self.feature_columns = [
            "ema_fast",
            "ema_slow",
            "ema_spread",
            "dist_ema200",
            "ema_fast_slope",
            "rsi",
            "rsi_delta",
            "ret",
            "vol",
            "volume_zscore",
            "atr_pct",
            "adx",
            "breakout_strength",
        ]

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata.get(
            "feature_columns", self.feature_columns
        )
        self.model_name = self.metadata.get("model_name", self.model_name)
        self.model_version = self.metadata.get("model_version", self.model_version)

        self.model = self._load_model(model_path)
        self.scaler = joblib.load(scaler_path)

        scaler_features = getattr(self.scaler, "n_features_in_", None)
        model_input_dim = int(self.model.net[0].in_features)
        feature_count = len(self.feature_columns)

        if scaler_features is not None and int(scaler_features) != feature_count:
            raise ValueError(
                "Feature mismatch between metadata and scaler: "
                f"metadata={feature_count}, scaler={int(scaler_features)}"
            )

        if model_input_dim != feature_count:
            raise ValueError(
                "Feature mismatch between metadata and model: "
                f"metadata={feature_count}, model_input={model_input_dim}"
            )

        opt_th = self.metadata.get("optimized_long_threshold")
        if opt_th is not None:
            self.long_threshold = float(opt_th)
            self.short_threshold = 1.0 - self.long_threshold
        else:
            self._init_thresholds()

    def _load_model(self, model_path: str) -> torch.nn.Module:
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

        print(
            f"[MODEL] {self.model_name} {self.model_version} | "
            f"LONG_TH={self.long_threshold:.2f} "
            f"SHORT_TH={self.short_threshold:.2f}"
        )

    def predict_proba(self, df) -> float:
        # Prevent double indicator computation
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