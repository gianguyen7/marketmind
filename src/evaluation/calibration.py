"""Calibration analysis for probabilistic forecasts.

Includes:
- Basic calibration curves and ECE
- Brier score decomposition (reliability + resolution + uncertainty)
- Grouped calibration analysis (by theme, horizon, price bucket)
- Favourite-longshot bias detection
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core calibration utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Brier score decomposition
# ---------------------------------------------------------------------------

def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Decompose Brier score into reliability, resolution, and uncertainty.

    Brier = reliability - resolution + uncertainty

    - Reliability (lower = better calibrated): weighted MSE of bin means vs observed freq
    - Resolution (higher = better discrimination): weighted variance of bin observed freq from base rate
    - Uncertainty (fixed for dataset): base_rate * (1 - base_rate)

    Reference: Murphy (1973) "A New Vector Partition of the Probability Score"
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    base_rate = y_true.mean()
    uncertainty = base_rate * (1 - base_rate)
    brier = float(((y_prob - y_true) ** 2).mean())

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    n_total = len(y_true)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        n_k = mask.sum()
        if n_k == 0:
            continue
        weight = n_k / n_total
        mean_pred = y_prob[mask].mean()
        mean_obs = y_true[mask].mean()
        reliability += weight * (mean_pred - mean_obs) ** 2
        resolution += weight * (mean_obs - base_rate) ** 2

    return {
        "brier_score": round(brier, 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "base_rate": round(base_rate, 6),
        "n": len(y_true),
    }


# ---------------------------------------------------------------------------
# Grouped calibration analysis
# ---------------------------------------------------------------------------

def calibration_by_group(
    df: pd.DataFrame,
    group_col: str,
    prob_col: str = "price_yes",
    target_col: str = "resolved_yes",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration metrics per group (e.g., per theme).

    Returns one row per group with ECE, Brier decomposition, and sample size.
    """
    rows = []
    for group_val, group_df in df.groupby(group_col):
        y_true = group_df[target_col].values
        y_prob = np.clip(group_df[prob_col].values, 1e-7, 1 - 1e-7)
        if len(y_true) < 5:
            continue

        decomp = brier_decomposition(y_true, y_prob, n_bins)
        ece = calibration_error(y_true, y_prob, n_bins)
        sharp = sharpness(y_prob)

        rows.append({
            group_col: group_val,
            "n_snapshots": len(y_true),
            "n_markets": group_df["condition_id"].nunique() if "condition_id" in group_df.columns else 0,
            "base_rate": decomp["base_rate"],
            "mean_price": float(y_prob.mean()),
            "brier_score": decomp["brier_score"],
            "ece": ece,
            "reliability": decomp["reliability"],
            "resolution": decomp["resolution"],
            "uncertainty": decomp["uncertainty"],
            "mean_sharpness": sharp["mean_sharpness"],
        })

    return pd.DataFrame(rows).sort_values("brier_score") if rows else pd.DataFrame()


def calibration_by_horizon(
    df: pd.DataFrame,
    horizon_col: str = "days_to_end",
    prob_col: str = "price_yes",
    target_col: str = "resolved_yes",
    bins: list[tuple[float, float, str]] | None = None,
    n_cal_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration metrics at different time-to-resolution horizons.

    Args:
        bins: list of (low, high, label) tuples. Defaults to standard buckets.
    """
    if bins is None:
        bins = [
            (0, 1, "<1d"),
            (1, 7, "1d-1w"),
            (7, 30, "1w-1m"),
            (30, 90, "1m-3m"),
            (90, 365, "3m-1y"),
            (365, float("inf"), ">1y"),
        ]

    rows = []
    for low, high, label in bins:
        mask = (df[horizon_col] >= low) & (df[horizon_col] < high)
        subset = df[mask]
        if len(subset) < 10:
            continue

        y_true = subset[target_col].values
        y_prob = np.clip(subset[prob_col].values, 1e-7, 1 - 1e-7)

        decomp = brier_decomposition(y_true, y_prob, n_cal_bins)
        ece = calibration_error(y_true, y_prob, n_cal_bins)

        rows.append({
            "horizon": label,
            "horizon_low": low,
            "horizon_high": high,
            "n_snapshots": len(y_true),
            "n_markets": subset["condition_id"].nunique() if "condition_id" in subset.columns else 0,
            "base_rate": decomp["base_rate"],
            "mean_price": float(y_prob.mean()),
            "brier_score": decomp["brier_score"],
            "ece": ece,
            "reliability": decomp["reliability"],
            "resolution": decomp["resolution"],
            "uncertainty": decomp["uncertainty"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Favourite-longshot bias analysis
# ---------------------------------------------------------------------------

def favourite_longshot_analysis(
    df: pd.DataFrame,
    prob_col: str = "price_yes",
    target_col: str = "resolved_yes",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Test for favourite-longshot bias: do longshots win more than their price implies?

    For each price bucket, computes:
    - mean_price: average market-implied probability
    - actual_rate: actual resolution rate
    - bias: actual_rate - mean_price (positive = longshot wins more than expected = market underprices longshots)

    Classic F-L bias: positive bias for low-probability events, negative for high-probability.
    """
    y_true = df[target_col].values
    y_prob = np.clip(df[prob_col].values, 1e-7, 1 - 1e-7)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        n_k = int(mask.sum())
        if n_k < 5:
            continue

        mean_price = float(y_prob[mask].mean())
        actual_rate = float(y_true[mask].mean())
        bias = actual_rate - mean_price

        rows.append({
            "price_bucket": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            "bucket_mid": (bins[i] + bins[i + 1]) / 2,
            "mean_price": mean_price,
            "actual_rate": actual_rate,
            "bias": bias,
            "abs_bias": abs(bias),
            "n_snapshots": n_k,
            "n_markets": int(df.loc[mask, "condition_id"].nunique()) if "condition_id" in df.columns else 0,
        })

    return pd.DataFrame(rows)
