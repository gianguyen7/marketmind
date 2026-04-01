"""Compare models across multiple evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluation.calibration import calibration_error, sharpness


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "",
    split: str = "test",
) -> dict:
    """Compute all evaluation metrics for a set of predictions."""
    if len(y_true) == 0 or len(y_prob) == 0:
        return {"model": model_name, "split": split, "n": 0}

    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    sharp = sharpness(y_prob)

    return {
        "model": model_name,
        "split": split,
        "n": len(y_true),
        "brier_score": brier_score_loss(y_true, y_prob_clipped),
        "log_loss": log_loss(y_true, y_prob_clipped, labels=[0, 1]),
        "calibration_error": calibration_error(y_true, y_prob_clipped),
        "mean_sharpness": sharp["mean_sharpness"],
        "pct_confident": sharp["pct_confident"],
        "base_rate": float(y_true.mean()),
        "mean_prediction": float(y_prob.mean()),
    }


def compare_all_models(
    results: dict,
    split: str = "test",
) -> pd.DataFrame:
    """Compare all trained models on a given split.

    Args:
        results: Dict of {model_name: (model, predictions_dict)} from training.
        split: Which split to evaluate ("train", "val", "test").

    Returns:
        DataFrame with one row per model, sorted by Brier score.
    """
    rows = []
    for model_name, (model, preds) in results.items():
        y_true = preds.get(f"y_{split}", np.array([]))
        y_prob = preds.get(split, np.array([]))
        row = evaluate_predictions(y_true, y_prob, model_name, split)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "brier_score" in df.columns:
        df = df.sort_values("brier_score")
    return df


def subgroup_evaluation(
    df_data: pd.DataFrame,
    y_prob: np.ndarray,
    target_col: str = "resolved_yes",
    group_col: str = "category",
) -> pd.DataFrame:
    """Evaluate model performance by subgroup."""
    df = df_data.copy()
    df["y_prob"] = y_prob

    rows = []
    for group_val, group_df in df.groupby(group_col):
        y_true = group_df[target_col].values
        y_pred = group_df["y_prob"].values
        if len(y_true) < 5:
            continue
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        rows.append({
            "group": group_val,
            "n": len(y_true),
            "brier_score": brier_score_loss(y_true, y_pred_clipped),
            "log_loss": log_loss(y_true, y_pred_clipped, labels=[0, 1]),
            "base_rate": float(y_true.mean()),
        })

    return pd.DataFrame(rows).sort_values("brier_score") if rows else pd.DataFrame()
