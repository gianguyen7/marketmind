"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.market_features import add_market_features
from src.features.text_features import add_text_features
from src.features.feature_pipeline import run_feature_pipeline, get_feature_matrix


def make_sample_df():
    return pd.DataFrame({
        "condition_id": ["0xa", "0xb", "0xc"],
        "question": ["Will the Fed cut rates?", "Will it rain tomorrow?", "Election outcome by Dec 2024"],
        "price_yes": [0.65, 0.30, 0.80],
        "volume_usd": [50000, 20000, 100000],
        "resolved_yes": [1, 0, 1],
    })


def make_snapshot_df():
    """Sample with snapshot_ts for time-series feature tests."""
    return pd.DataFrame({
        "condition_id": ["0xa"] * 3 + ["0xb"] * 3,
        "question": ["Fed cut?"] * 3 + ["Rain?"] * 3,
        "snapshot_ts": pd.to_datetime(["2024-07-01", "2024-07-15", "2024-08-01"] * 2, utc=True),
        "price_yes": [0.40, 0.55, 0.70, 0.60, 0.45, 0.30],
        "volume_usd": [50000] * 6,
        "resolved_yes": [1] * 3 + [0] * 3,
    })


def test_add_market_features():
    df = add_market_features(make_sample_df())
    assert "implied_prob" in df.columns
    assert "implied_logit" in df.columns
    assert "price_extremity" in df.columns
    assert df["implied_prob"].iloc[0] == 0.65


def test_add_market_features_with_snapshots():
    df = add_market_features(make_snapshot_df())
    assert "extremity_trend" in df.columns


def test_add_text_features():
    df = add_text_features(make_sample_df())
    assert "question_length" in df.columns
    assert "has_temporal_ref" in df.columns
    assert "has_numeric_target" in df.columns
    assert df["has_temporal_ref"].iloc[2] == 1  # "Dec 2024"


def test_run_feature_pipeline():
    df = run_feature_pipeline(make_sample_df())
    assert "implied_prob" in df.columns
    assert "question_length" in df.columns


def test_get_feature_matrix():
    df = run_feature_pipeline(make_sample_df())
    X, y = get_feature_matrix(df, ["implied_prob"], "resolved_yes")
    assert X.shape == (3, 1)
    assert len(y) == 3


def test_get_feature_matrix_missing_features():
    df = run_feature_pipeline(make_sample_df())
    X, y = get_feature_matrix(df, ["implied_prob", "nonexistent_feature"], "resolved_yes")
    assert X.shape[1] == 1
