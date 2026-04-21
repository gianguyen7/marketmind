"""C1: Calibration correction models.

Trains recalibration models that map market prices to better-calibrated
probabilities. Four approaches, from simplest to most granular:

    C1a. Isotonic regression (global)
    C1b. Platt scaling (global)
    C1c. Per-category recalibration (isotonic + Platt per category)
    C1d. Category × horizon recalibration

All models trained on train split only. Val for model selection. Test touched once.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

from src.evaluation.calibration import (  # noqa: E402
    brier_decomposition,
    calibration_by_group,
    calibration_curve,
    calibration_error,
    favourite_longshot_analysis,
    sharpness,
)


DATE_TAG = datetime.now().strftime("%Y-%m-%d")
OUT_TABLES = Path("outputs/tables")
OUT_FIGS = Path("outputs/figures")
OUT_MODELS = Path("outputs/models")
for d in [OUT_TABLES, OUT_FIGS, OUT_MODELS]:
    d.mkdir(parents=True, exist_ok=True)

with open("configs/modeling.yaml") as f:
    SEED = yaml.safe_load(f).get("random_seed", 42)

GROUP_COL = "category"
MIN_CATEGORY_TRAIN_MARKETS = 50

HORIZON_BINS = [
    (0, 7, "<1w"), (7, 30, "1w-1m"), (30, 90, "1m-3m"),
    (90, 365, "3m-1y"), (365, float("inf"), ">1y"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    return train, val, test


def clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-7, 1 - 1e-7)


def eval_model(
    y_true: np.ndarray, y_prob: np.ndarray, label: str,
) -> dict:
    """Compute standard metrics for a recalibration model."""
    y_prob = clip_probs(y_prob)
    decomp = brier_decomposition(y_true, y_prob)
    ece = calibration_error(y_true, y_prob)
    log_loss = float(-np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    ))
    sharp = sharpness(y_prob)
    return {
        "model": label,
        "brier": decomp["brier_score"],
        "reliability": decomp["reliability"],
        "resolution": decomp["resolution"],
        "ece": ece,
        "log_loss": log_loss,
        "mean_sharpness": sharp["mean_sharpness"],
    }


def horizon_label(days_to_end: float) -> str:
    for low, high, label in HORIZON_BINS:
        if low <= days_to_end < high:
            return label
    return ">1y"


# ---------------------------------------------------------------------------
# C1a: Isotonic regression
# ---------------------------------------------------------------------------

def train_isotonic(train: pd.DataFrame) -> IsotonicRegression:
    """Fit isotonic regression: price_yes → resolved_yes."""
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(train["price_yes"].values, train["resolved_yes"].values)
    return iso


# ---------------------------------------------------------------------------
# C1b: Platt scaling
# ---------------------------------------------------------------------------

def train_platt(train: pd.DataFrame) -> LogisticRegression:
    """Fit logistic regression on logit(price_yes) → resolved_yes."""
    X = logit(clip_probs(train["price_yes"].values)).reshape(-1, 1)
    y = train["resolved_yes"].values
    lr = LogisticRegression(random_state=SEED, max_iter=1000)
    lr.fit(X, y)
    print(f"  Platt: coef={lr.coef_[0][0]:.4f}, intercept={lr.intercept_[0]:.4f}")
    return lr


def predict_platt(lr: LogisticRegression, prices: np.ndarray) -> np.ndarray:
    X = logit(clip_probs(prices)).reshape(-1, 1)
    return lr.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# C1c: Per-category recalibration
# ---------------------------------------------------------------------------

def train_per_category(
    train: pd.DataFrame,
) -> dict[str, dict]:
    """Train isotonic + Platt per category. Pool small categories."""
    cat_counts = train.groupby(GROUP_COL)["condition_id"].nunique()
    large_cats = cat_counts[cat_counts >= MIN_CATEGORY_TRAIN_MARKETS].index.tolist()
    small_cats = cat_counts[cat_counts < MIN_CATEGORY_TRAIN_MARKETS].index.tolist()

    if small_cats:
        n_pooled = cat_counts[cat_counts < MIN_CATEGORY_TRAIN_MARKETS].sum()
        print(f"  Pooling {len(small_cats)} small categories ({n_pooled} markets) "
              f"into '_pooled': {small_cats}")

    models: dict[str, dict] = {}

    # Train per large category
    for cat in large_cats:
        cat_df = train[train[GROUP_COL] == cat]
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(cat_df["price_yes"].values, cat_df["resolved_yes"].values)

        X = logit(clip_probs(cat_df["price_yes"].values)).reshape(-1, 1)
        lr = LogisticRegression(random_state=SEED, max_iter=1000)
        lr.fit(X, cat_df["resolved_yes"].values)

        models[cat] = {"isotonic": iso, "platt": lr}

    # Pooled model for small categories
    if small_cats:
        pooled_df = train[train[GROUP_COL].isin(small_cats)]
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(pooled_df["price_yes"].values, pooled_df["resolved_yes"].values)

        X = logit(clip_probs(pooled_df["price_yes"].values)).reshape(-1, 1)
        lr = LogisticRegression(random_state=SEED, max_iter=1000)
        lr.fit(X, pooled_df["resolved_yes"].values)

        models["_pooled"] = {"isotonic": iso, "platt": lr}

    print(f"  Trained {len(models)} category models "
          f"({len(large_cats)} large + {'1 pooled' if small_cats else '0 pooled'})")
    return models


def predict_per_category(
    models: dict, df: pd.DataFrame, method: str = "isotonic",
) -> np.ndarray:
    """Apply per-category model. Falls back to _pooled for unknown categories."""
    preds = np.zeros(len(df))
    for cat in df[GROUP_COL].unique():
        mask = df[GROUP_COL] == cat
        m = models.get(cat, models.get("_pooled"))
        if m is None:
            preds[mask] = df.loc[mask, "price_yes"].values
            continue
        prices = df.loc[mask, "price_yes"].values
        if method == "isotonic":
            preds[mask] = m["isotonic"].predict(prices)
        else:
            preds[mask] = predict_platt(m["platt"], prices)
    return preds


# ---------------------------------------------------------------------------
# C1d: Category × horizon recalibration
# ---------------------------------------------------------------------------

def train_cat_x_horizon(
    train: pd.DataFrame,
) -> dict[tuple[str, str], IsotonicRegression]:
    """Train isotonic per (category, horizon) bucket. Fall back to category-only
    for buckets with <200 train snapshots."""
    models: dict[tuple[str, str], IsotonicRegression] = {}

    train = train.copy()
    train["_horizon"] = train["days_to_end"].apply(horizon_label)

    n_trained = 0
    n_fallback = 0
    for cat in sorted(train[GROUP_COL].unique()):
        for _, _, hlabel in HORIZON_BINS:
            mask = (train[GROUP_COL] == cat) & (train["_horizon"] == hlabel)
            subset = train[mask]
            if len(subset) >= 200:
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                iso.fit(subset["price_yes"].values, subset["resolved_yes"].values)
                models[(cat, hlabel)] = iso
                n_trained += 1
            else:
                n_fallback += 1

    print(f"  Cat×horizon: {n_trained} bucket models trained, "
          f"{n_fallback} buckets will fall back to category-only")
    return models


def predict_cat_x_horizon(
    cat_x_hor_models: dict,
    cat_models: dict,
    df: pd.DataFrame,
) -> np.ndarray:
    """Predict with cat×horizon model, falling back to per-category isotonic."""
    df = df.copy()
    df["_horizon"] = df["days_to_end"].apply(horizon_label)
    preds = np.zeros(len(df))

    for idx, row in df.iterrows():
        key = (row[GROUP_COL], row["_horizon"])
        if key in cat_x_hor_models:
            preds[idx] = cat_x_hor_models[key].predict([row["price_yes"]])[0]
        else:
            # Fall back to per-category
            cat = row[GROUP_COL]
            m = cat_models.get(cat, cat_models.get("_pooled"))
            if m:
                preds[idx] = m["isotonic"].predict([row["price_yes"]])[0]
            else:
                preds[idx] = row["price_yes"]

    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"C1: Calibration Correction Models ({DATE_TAG})")
    print("=" * 60)

    train, val, test = load_splits()
    print(f"Train: {len(train)} snaps, {train['condition_id'].nunique()} markets")
    print(f"Val:   {len(val)} snaps, {val['condition_id'].nunique()} markets")
    print(f"Test:  {len(test)} snaps, {test['condition_id'].nunique()} markets")

    # --- C1a: Isotonic ---
    print(f"\n{'─'*60}")
    print("C1a: Isotonic regression (global)")
    print(f"{'─'*60}")
    iso_global = train_isotonic(train)

    # --- C1b: Platt ---
    print(f"\n{'─'*60}")
    print("C1b: Platt scaling (global)")
    print(f"{'─'*60}")
    platt_global = train_platt(train)

    # --- C1c: Per-category ---
    print(f"\n{'─'*60}")
    print("C1c: Per-category recalibration")
    print(f"{'─'*60}")
    cat_models = train_per_category(train)

    # --- C1d: Category × horizon ---
    print(f"\n{'─'*60}")
    print("C1d: Category × horizon recalibration")
    print(f"{'─'*60}")
    cat_x_hor_models = train_cat_x_horizon(train)

    # --- C1e: Evaluate all models ---
    print(f"\n{'='*60}")
    print("C1e: Evaluation on val and test")
    print(f"{'='*60}")

    model_preds = {
        "naive (market price)": lambda df: df["price_yes"].values,
        "isotonic_global": lambda df: iso_global.predict(df["price_yes"].values),
        "platt_global": lambda df: predict_platt(platt_global, df["price_yes"].values),
        "isotonic_per_cat": lambda df: predict_per_category(cat_models, df, "isotonic"),
        "platt_per_cat": lambda df: predict_per_category(cat_models, df, "platt"),
        "cat_x_horizon": lambda df: predict_cat_x_horizon(
            cat_x_hor_models, cat_models, df,
        ),
    }

    all_results = []
    for split_name, split_df in [("val", val), ("test", test)]:
        print(f"\n  --- {split_name.upper()} ---")
        print(f"  {'model':<25} {'brier':>8} {'reliab':>8} {'resoln':>8} "
              f"{'ece':>8} {'logloss':>8} {'sharp':>8}")
        print(f"  {'-'*75}")

        for model_name, predict_fn in model_preds.items():
            preds = clip_probs(predict_fn(split_df))
            y_true = split_df["resolved_yes"].values
            metrics = eval_model(y_true, preds, model_name)
            metrics["split"] = split_name
            all_results.append(metrics)

            print(f"  {model_name:<25} {metrics['brier']:>8.4f} "
                  f"{metrics['reliability']:>8.4f} {metrics['resolution']:>8.4f} "
                  f"{metrics['ece']:>8.4f} {metrics['log_loss']:>8.4f} "
                  f"{metrics['mean_sharpness']:>8.4f}")

    results_df = pd.DataFrame(all_results)

    # --- Per-category breakdown on test ---
    print(f"\n{'='*60}")
    print("Per-category Brier on TEST (best model vs naive)")
    print(f"{'='*60}")

    # Find best non-naive model on val
    val_results = results_df[
        (results_df["split"] == "val") & (results_df["model"] != "naive (market price)")
    ]
    best_model = val_results.loc[val_results["brier"].idxmin(), "model"]
    print(f"\n  Best model on val: {best_model}")
    best_predict_fn = model_preds[best_model]

    print(f"\n  {'category':<25} {'naive':>8} {'best':>8} {'Δ':>8} {'n_mkts':>7}")
    print(f"  {'-'*58}")

    cat_rows = []
    for cat in sorted(test[GROUP_COL].unique()):
        cat_df = test[test[GROUP_COL] == cat]
        if len(cat_df) < 20:
            continue
        y_true = cat_df["resolved_yes"].values
        naive_brier = float(((cat_df["price_yes"].values - y_true) ** 2).mean())
        best_preds = clip_probs(best_predict_fn(cat_df))
        best_brier = float(((best_preds - y_true) ** 2).mean())
        delta = best_brier - naive_brier
        n_m = cat_df["condition_id"].nunique()
        print(f"  {cat:<25} {naive_brier:>8.4f} {best_brier:>8.4f} "
              f"{delta:>+8.4f} {n_m:>7}")
        cat_rows.append({
            GROUP_COL: cat, "naive_brier": naive_brier,
            "best_brier": best_brier, "delta": delta, "n_markets": n_m,
        })

    # --- Market-level evaluation ---
    print(f"\n{'='*60}")
    print("Market-level Brier on TEST")
    print(f"{'='*60}")

    for model_name, predict_fn in model_preds.items():
        preds = clip_probs(predict_fn(test))
        test_eval = test.copy()
        test_eval["_pred"] = preds
        market_brier = test_eval.groupby("condition_id").apply(
            lambda g: ((g["_pred"] - g["resolved_yes"]) ** 2).mean(),
            include_groups=False,
        )
        print(f"  {model_name:<25} mean={market_brier.mean():.4f}  "
              f"median={market_brier.median():.4f}")

    # --- F-L bias before/after ---
    print(f"\n{'='*60}")
    print("F-L bias on TEST: naive vs best model")
    print(f"{'='*60}")

    test_copy = test.copy()
    test_copy["best_pred"] = clip_probs(best_predict_fn(test))

    fl_naive = favourite_longshot_analysis(test_copy, "price_yes")
    fl_best = favourite_longshot_analysis(test_copy, "best_pred")

    if len(fl_naive) > 0 and len(fl_best) > 0:
        merged = fl_naive[["price_bucket", "bias"]].rename(
            columns={"bias": "naive_bias"}
        ).merge(
            fl_best[["price_bucket", "bias"]].rename(columns={"bias": "best_bias"}),
            on="price_bucket",
        )
        merged["improvement"] = merged["naive_bias"].abs() - merged["best_bias"].abs()
        print(merged.to_string(index=False, float_format="%.4f"))

    # --- Save ---
    results_df.to_csv(OUT_TABLES / f"c1_recalibration_results_{DATE_TAG}.csv",
                      index=False, float_format="%.6f")
    pd.DataFrame(cat_rows).to_csv(
        OUT_TABLES / f"c1_per_category_test_{DATE_TAG}.csv",
        index=False, float_format="%.4f",
    )

    # Save models
    model_bundle = {
        "isotonic_global": iso_global,
        "platt_global": platt_global,
        "cat_models": cat_models,
        "cat_x_hor_models": cat_x_hor_models,
        "date": DATE_TAG,
        "seed": SEED,
    }
    model_path = OUT_MODELS / f"c1_recalibration_{DATE_TAG}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n  Saved results to {OUT_TABLES}")
    print(f"  Saved models to {model_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("C1 SUMMARY")
    print(f"{'='*60}")
    test_results = results_df[results_df["split"] == "test"].sort_values("brier")
    naive_brier = test_results[test_results["model"] == "naive (market price)"]["brier"].iloc[0]
    best_test = test_results.iloc[0]
    print(f"  Naive (market price) test Brier: {naive_brier:.4f}")
    print(f"  Best model: {best_test['model']} (test Brier: {best_test['brier']:.4f})")
    delta = best_test["brier"] - naive_brier
    print(f"  Improvement: {delta:+.4f} ({delta/naive_brier*100:+.1f}%)")

    if delta < 0:
        print(f"\n  RESULT: Recalibration IMPROVES test Brier by {-delta:.4f}.")
    else:
        print(f"\n  RESULT: Recalibration does NOT improve test Brier.")
        print(f"  The market's reliability of 0.0004 may already be near-optimal.")


if __name__ == "__main__":
    main()
