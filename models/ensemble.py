# models/ensemble.py

from typing import List
import numpy as np

from execution.regime import detect_regime, MarketRegime


class EnsembleDirectionModel:
    """
    Combines multiple DirectionModels using
    regime-aware weighted averaging.
    """

    def __init__(self, models: List):
        if not models:
            raise ValueError("Ensemble requires at least one model")
        self.models = models

        # Default weights (will be adjusted dynamically)
        self.base_weights = np.ones(len(models)) / len(models)

        # Expose thresholds so StrategyEngine can consume ensemble settings.
        self.long_threshold = self._aggregate_threshold(self.base_weights, "long_threshold", default=0.55)
        self.short_threshold = self._aggregate_threshold(self.base_weights, "short_threshold", default=0.45)

    def predict_proba(self, df):
        probs = np.array([m.predict_proba(df) for m in self.models])

        regime = detect_regime(df)
        weights = self._weights_for_regime(regime)

        # Keep thresholds synchronized with the same regime weighting used for probabilities.
        long_th = self._aggregate_threshold(weights, "long_threshold", default=0.55)
        short_th = self._aggregate_threshold(weights, "short_threshold", default=0.45)

        # Conservative nudge in choppy markets.
        if regime == MarketRegime.CHOPPY:
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

        # Convention:
        # models[0] = symbol model
        # models[1] = BTC context model (if exists)

        if regime == MarketRegime.TRENDING:
            return np.array([0.7, 0.3])[:n]

        if regime == MarketRegime.RANGING:
            return np.array([0.85, 0.15])[:n]

        # CHOPPY → be conservative
        return np.array([0.6, 0.4])[:n]

