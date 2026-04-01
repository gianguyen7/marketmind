"""Hybrid models that combine Polymarket signals with ML features."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class HybridEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble that blends a market price signal with an ML model.

    The final prediction is a weighted average:
        p_hybrid = alpha * p_market + (1 - alpha) * p_ml

    Alpha is learned via logistic regression on validation predictions.
    """

    def __init__(self, ml_model, market_price_index: int = 0, alpha: float = 0.5):
        self.ml_model = ml_model
        self.market_price_index = market_price_index
        self.alpha = alpha
        self._blend_model = None

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.ml_model.fit(X, y)

        # Get ML predictions and market prices
        ml_probs = self.ml_model.predict_proba(X)[:, 1]
        if hasattr(X, "iloc"):
            market_probs = X.iloc[:, self.market_price_index].values
        else:
            market_probs = X[:, self.market_price_index]
        market_probs = np.clip(np.nan_to_num(market_probs, nan=0.5), 0.01, 0.99)

        # Learn blending weights
        blend_X = np.column_stack([ml_probs, market_probs])
        self._blend_model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(random_state=42)),
        ])
        self._blend_model.fit(blend_X, y)
        return self

    def predict_proba(self, X):
        ml_probs = self.ml_model.predict_proba(X)[:, 1]
        if hasattr(X, "iloc"):
            market_probs = X.iloc[:, self.market_price_index].values
        else:
            market_probs = X[:, self.market_price_index]
        market_probs = np.clip(np.nan_to_num(market_probs, nan=0.5), 0.01, 0.99)

        if self._blend_model is not None:
            blend_X = np.column_stack([ml_probs, market_probs])
            probs = self._blend_model.predict_proba(blend_X)[:, 1]
        else:
            probs = self.alpha * market_probs + (1 - self.alpha) * ml_probs

        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
