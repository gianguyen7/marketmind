"""Tests for model implementations."""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import BaseRateModel, RecencyBaselineModel
from src.models.logistic_model import build_logistic_model
from src.models.tree_models import build_random_forest, build_xgboost
from src.models.hybrid_models import HybridEnsemble


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "market_price": np.random.uniform(0.1, 0.9, n),
        "volume": np.random.uniform(1000, 100000, n),
        "days_open": np.random.uniform(1, 365, n),
    })
    y = (X["market_price"] + np.random.normal(0, 0.2, n) > 0.5).astype(int)
    return X, y


def test_base_rate_model(sample_data):
    X, y = sample_data
    model = BaseRateModel()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs[:, 1], y.mean())


def test_recency_baseline(sample_data):
    X, y = sample_data
    model = RecencyBaselineModel(price_col_index=0)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_logistic_model(sample_data):
    X, y = sample_data
    model = build_logistic_model()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_random_forest(sample_data):
    X, y = sample_data
    model = build_random_forest(n_estimators=10)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)


def test_xgboost(sample_data):
    X, y = sample_data
    model = build_xgboost(n_estimators=10)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)


def test_hybrid_ensemble(sample_data):
    X, y = sample_data
    base = build_logistic_model()
    hybrid = HybridEnsemble(base, market_price_index=0)
    hybrid.fit(X, y)
    probs = hybrid.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)
