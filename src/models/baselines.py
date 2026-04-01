"""Baseline probabilistic forecasting models."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseRateModel(BaseEstimator, ClassifierMixin):
    """Predict the training set base rate for all instances."""

    def __init__(self):
        self.base_rate_ = 0.5

    def fit(self, X, y):
        self.base_rate_ = y.mean()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        probs = np.full((n, 2), [1 - self.base_rate_, self.base_rate_])
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RecencyBaselineModel(BaseEstimator, ClassifierMixin):
    """Use the most recent market price as the probability forecast.

    Falls back to base rate if market_price feature is not available.
    """

    def __init__(self, price_col_index: int = 0):
        self.price_col_index = price_col_index
        self.base_rate_ = 0.5

    def fit(self, X, y):
        self.base_rate_ = y.mean()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if hasattr(X, "iloc"):
            prices = X.iloc[:, self.price_col_index].values
        else:
            prices = X[:, self.price_col_index]

        prices = np.clip(np.nan_to_num(prices, nan=self.base_rate_), 0.01, 0.99)
        return np.column_stack([1 - prices, prices])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
