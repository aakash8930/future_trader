# models/ensemble.py

import os
from typing import List
import numpy as np

from execution.regime_controller import RegimeController, MarketRegime


class EnsembleDirectionModel:
    """
    Combines multiple DirectionModels (MLP, LSTM, XGBoost) using
    regime-aware weighted averaging.

    Models are auto-discovered via for_symbol() factory.
    """

    def __init__(self, models: List, symbol: str = ""):
        if not models:
            raise ValueError("Ensemble requires at least one model")
        self.models = models
        self.symbol = symbol
        self.regime_ctrl = RegimeController()

        n = len(models)
        self.base_weights = np.ones(n) / n

        self.long_threshold = self._aggregate_threshold(self.base_weights, "long_threshold", default=0.55)
        self.short_threshold = self._aggregate_threshold(self.base_weights, "short_threshold", default=0.45)

    @classmethod
    def for_symbol(cls, symbol: str) -> "EnsembleDirectionModel":
        """Auto-detect and load all available model types for a symbol."""
        from models.direction import DirectionModel

        folder = symbol.replace('/', '_')
        discovered = []

        # Check each model type subdirectory
        for model_type, subfolder in [
            ("MLP", f"models/{folder}"),
            ("LSTM", f"models/{folder}_lstm"),
            ("XGB", f"models/{folder}_xgb"),
        ]:
            metadata_path = os.path.join(subfolder, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    # Pass explicit model_type so it loads from the right subdirectory
                    model = DirectionModel.for_symbol(symbol, model_type=model_type)
                    discovered.append(model)
                except Exception:
                    pass

        if not discovered:
            raise FileNotFoundError(f"No trained models found for {symbol}")

        print(f"[ENSEMBLE] {symbol}: discovered {len(discovered)} models: {[m.model_type for m in discovered]}")
        return cls(discovered, symbol=symbol)

    def predict_proba(self, df):
        probs = np.array([float(m.predict_proba(df)) for m in self.models])

        regime = self.regime_ctrl.detect(df)
        weights = self._weights_for_regime(regime)

        long_th = self._aggregate_threshold(weights, "long_threshold", default=0.55)
        short_th = self._aggregate_threshold(weights, "short_threshold", default=0.45)

        if regime == MarketRegime.SIDEWAYS:
            long_th += 0.01
            short_th -= 0.01

        self.long_threshold = float(np.clip(long_th, 0.45, 0.70))
        self.short_threshold = float(np.clip(short_th, 0.30, 0.55))

        prob = float(np.average(probs, weights=weights))
        return prob

    def _aggregate_threshold(self, weights, attr: str, default: float) -> float:
        vals = np.array([
            float(getattr(model, attr, default)) for model in self.models
        ])
        return float(np.average(vals, weights=weights))

    def _weights_for_regime(self, regime: MarketRegime):
        n = len(self.models)

        if n == 1:
            return self.base_weights

        if n == 2:
            # [symbol model, BTC context model] (legacy convention)
            if regime == MarketRegime.TREND_STRONG:
                return np.array([0.7, 0.3])
            elif regime == MarketRegime.TREND_WEAK:
                return np.array([0.85, 0.15])
            else:
                return np.array([0.6, 0.4])

        if n == 3:
            # [MLP, LSTM, XGB]
            if regime == MarketRegime.TREND_STRONG:
                return np.array([0.40, 0.35, 0.25])
            elif regime == MarketRegime.TREND_WEAK:
                return np.array([0.55, 0.25, 0.20])
            else:
                return np.array([0.35, 0.30, 0.35])

        # Fallback: equal weights
        return np.ones(n) / n