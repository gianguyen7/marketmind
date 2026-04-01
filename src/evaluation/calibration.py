"""Calibration analysis for probabilistic forecasts."""

import numpy as np
import pandas as pd


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration curve data.

    Returns DataFrame with columns: bin_mid, mean_predicted, mean_observed, count.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_mid": (bins[i] + bins[i + 1]) / 2,
            "mean_predicted": y_prob[mask].mean(),
            "mean_observed": y_true[mask].mean(),
            "count": int(mask.sum()),
        })

    return pd.DataFrame(rows)


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    cal = calibration_curve(y_true, y_prob, n_bins)
    if len(cal) == 0:
        return 0.0
    total = cal["count"].sum()
    ece = ((cal["count"] / total) * (cal["mean_predicted"] - cal["mean_observed"]).abs()).sum()
    return float(ece)


def sharpness(y_prob: np.ndarray) -> dict:
    """Measure forecast sharpness (how decisive the predictions are).

    Sharper forecasts are further from 0.5.
    """
    deviation = np.abs(y_prob - 0.5)
    return {
        "mean_sharpness": float(deviation.mean()),
        "median_sharpness": float(np.median(deviation)),
        "pct_confident": float((y_prob < 0.2).mean() + (y_prob > 0.8).mean()),
        "pct_uncertain": float(((y_prob >= 0.4) & (y_prob <= 0.6)).mean()),
    }
