"""Temporal backtesting to prevent data leakage."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.features.feature_pipeline import get_feature_matrix, run_feature_pipeline


def temporal_backtest(
    df: pd.DataFrame,
    model_builder: callable,
    feature_names: list[str],
    time_col: str = "closed_at",
    n_splits: int = 5,
    min_train_size: int = 50,
    target_col: str = "resolved_yes",
) -> list[dict]:
    """Walk-forward temporal cross-validation.

    At each step, train on all data before the split point,
    predict on the next fold. This prevents temporal leakage.
    """
    df = df.copy()
    df = run_feature_pipeline(df)

    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    fold_size = n // (n_splits + 1)

    results = []
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_end = min((i + 2) * fold_size, n)

        if train_end < min_train_size:
            continue

        train_data = df.iloc[:train_end]
        test_data = df.iloc[train_end:test_end]

        X_train, y_train = get_feature_matrix(train_data, feature_names, target_col)
        X_test, y_test = get_feature_matrix(test_data, feature_names, target_col)

        if len(X_train) < min_train_size or len(X_test) == 0:
            continue

        model = model_builder()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        fold_result = {
            "fold": i,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "brier_score": brier_score_loss(y_test, y_prob),
            "log_loss": log_loss(y_test, y_prob, labels=[0, 1]),
            "mean_pred": float(y_prob.mean()),
            "mean_actual": float(y_test.mean()),
        }
        results.append(fold_result)

    return results


def backtest_summary(results: list[dict]) -> pd.DataFrame:
    """Summarize backtest results across folds."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    summary = pd.DataFrame([{
        "n_folds": len(df),
        "mean_brier": df["brier_score"].mean(),
        "std_brier": df["brier_score"].std(),
        "mean_logloss": df["log_loss"].mean(),
        "std_logloss": df["log_loss"].std(),
        "total_test_samples": df["test_size"].sum(),
    }])
    return summary
