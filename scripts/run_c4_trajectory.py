"""C4: Price trajectory dynamics exploration.

Hypothesis: Markets whose price trajectories exhibit anomalous patterns
(staleness, acceleration, volatility regime shifts) are more likely to be
mispriced than markets with "normal" trajectories.

Steps:
    C4a. Engineer trajectory features (backward-looking only)
    C4b. Exploratory analysis — correlate with |market_error|
    C4c. Trajectory-informed correction model (LightGBM)
    C4d. Evaluate on test if val improvement > 0.001 Brier
    C4e. Document findings
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
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.evaluation.calibration import (  # noqa: E402
    brier_decomposition,
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


# ---------------------------------------------------------------------------
# C4a: Engineer trajectory features
# ---------------------------------------------------------------------------

def add_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trajectory-based features. All strictly backward-looking."""
    df = df.copy()
    df = df.sort_values(["condition_id", "snapshot_ts"])
    g = df.groupby("condition_id")

    # 1. Staleness: snapshots since last price move > 0.01
    price_moved = g["price_yes"].diff().abs() > 0.01
    # Count consecutive False (no move) values — reset on each True
    staleness = []
    current_stale = 0
    prev_id = None
    for idx, row in df.iterrows():
        if row["condition_id"] != prev_id:
            current_stale = 0
            prev_id = row["condition_id"]
        if price_moved.get(idx, False):
            current_stale = 0
        else:
            current_stale += 1
        staleness.append(current_stale)
    df["staleness"] = staleness

    # 2. Acceleration: change in price_change (second derivative)
    df["price_acceleration"] = g["price_change"].diff().fillna(0)

    # 3. Volatility regime: current rolling vol vs market's historical vol
    # ratio > 1 = high vol regime, < 1 = low vol regime
    market_hist_vol = g["price_yes"].transform(
        lambda x: x.expanding(min_periods=5).std().shift(1)
    )
    recent_vol = g["price_yes"].transform(
        lambda x: x.rolling(7, min_periods=2).std()
    )
    df["vol_regime"] = np.where(
        market_hist_vol > 0.001,
        recent_vol / market_hist_vol,
        1.0,
    )
    df["vol_regime"] = df["vol_regime"].clip(0, 10).fillna(1.0)

    # 4. Path curvature: how much actual price deviates from a straight line
    # from opening price to current price
    opening_price = g["price_yes"].transform("first")
    # Linear interpolation: open + (current_snap / total_snaps) * (current - open)
    # But we want distance from that line, not the line itself.
    # Simpler: abs(price - midpoint(open, current)) for recent snapshots
    midpoint = (opening_price + df["price_yes"]) / 2
    # Use rolling mean of last 7 snapshots as "actual path" vs midpoint
    rolling_price = g["price_yes"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["path_curvature"] = (rolling_price - midpoint).abs()

    # 5. Price range: max - min over last 14 snapshots (how much has it moved)
    df["price_range_14"] = g["price_yes"].transform(
        lambda x: x.rolling(14, min_periods=2).max() - x.rolling(14, min_periods=2).min()
    ).fillna(0)

    # 6. Distance from extremes: how far current price is from its rolling max/min
    rolling_max = g["price_yes"].transform(
        lambda x: x.expanding().max().shift(1)
    )
    rolling_min = g["price_yes"].transform(
        lambda x: x.expanding().min().shift(1)
    )
    price_range = (rolling_max - rolling_min).clip(lower=0.01)
    df["dist_from_high"] = (rolling_max - df["price_yes"]) / price_range
    df["dist_from_low"] = (df["price_yes"] - rolling_min) / price_range
    df["dist_from_high"] = df["dist_from_high"].clip(0, 1).fillna(0.5)
    df["dist_from_low"] = df["dist_from_low"].clip(0, 1).fillna(0.5)

    # 7. Snap position: how far through the market's lifetime (already have pct_lifetime_elapsed)

    # 8. Price extremity: distance from 0.5 (already computable but explicit)
    df["price_extremity"] = (df["price_yes"] - 0.5).abs()

    print(f"  Added trajectory features. New columns: staleness, price_acceleration, "
          f"vol_regime, path_curvature, price_range_14, dist_from_high, dist_from_low, "
          f"price_extremity")

    return df


# ---------------------------------------------------------------------------
# C4b: Exploratory analysis
# ---------------------------------------------------------------------------

def explore_trajectory_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Correlate trajectory features with |market_error| on train."""
    print(f"\n{'='*60}")
    print("C4b: Trajectory feature correlations with |market_error|")
    print(f"{'='*60}")

    features = [
        # Existing
        "price_yes", "price_change", "price_momentum_3", "price_momentum_7",
        "price_volatility_7", "price_vs_open", "pct_lifetime_elapsed",
        "days_to_end", "log_volume",
        # New trajectory
        "staleness", "price_acceleration", "vol_regime", "path_curvature",
        "price_range_14", "dist_from_high", "dist_from_low", "price_extremity",
    ]

    rows = []
    print(f"\n  {'feature':<25} {'ρ vs |err|':>10} {'p':>10} {'ρ vs err':>10} {'p':>10}")
    print(f"  {'-'*70}")

    for feat in features:
        if feat not in df.columns:
            continue
        x = df[feat]
        abs_err = df["abs_market_error"]
        signed_err = df["market_error"]

        mask = x.notna() & abs_err.notna()
        if mask.sum() < 100:
            continue

        rho_abs, p_abs = stats.spearmanr(x[mask], abs_err[mask])
        rho_sign, p_sign = stats.spearmanr(x[mask], signed_err[mask])

        rows.append({
            "feature": feat,
            "rho_abs_error": rho_abs, "p_abs": p_abs,
            "rho_signed_error": rho_sign, "p_signed": p_sign,
        })

        marker = " ***" if abs(rho_abs) > 0.1 else ""
        print(f"  {feat:<25} {rho_abs:>+10.4f} {p_abs:>10.4f} "
              f"{rho_sign:>+10.4f} {p_sign:>10.4f}{marker}")

    result = pd.DataFrame(rows).sort_values("rho_abs_error", key=abs, ascending=False)

    # Top features
    print(f"\n  Top 5 features by |ρ| with |market_error|:")
    for _, row in result.head(5).iterrows():
        print(f"    {row['feature']:<25} ρ = {row['rho_abs_error']:+.4f}")

    # Per-category robustness for top features
    top_feats = result.head(5)["feature"].tolist()
    print(f"\n  Per-category robustness (top 5 features):")
    print(f"  {'feature':<25}", end="")
    cats = sorted(df[GROUP_COL].unique())
    for cat in cats:
        print(f"  {cat[:8]:>8}", end="")
    print()

    for feat in top_feats:
        print(f"  {feat:<25}", end="")
        for cat in cats:
            cat_df = df[df[GROUP_COL] == cat]
            mask = cat_df[feat].notna() & cat_df["abs_market_error"].notna()
            if mask.sum() < 50:
                print(f"  {'  -':>8}", end="")
                continue
            rho, _ = stats.spearmanr(cat_df.loc[mask, feat], cat_df.loc[mask, "abs_market_error"])
            print(f"  {rho:>+8.3f}", end="")
        print()

    return result


# ---------------------------------------------------------------------------
# C4c: Trajectory-informed correction model
# ---------------------------------------------------------------------------

def train_trajectory_model(
    train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str],
) -> tuple:
    """Train XGBoost on trajectory features to predict resolved_yes.
    Returns (model, val_brier, naive_val_brier)."""

    print(f"\n{'='*60}")
    print("C4c: Trajectory-informed correction model")
    print(f"{'='*60}")

    # Try XGBoost first, fall back to logistic if not available
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
        use_xgb = True
    except ImportError:
        use_xgb = False

    # Prepare features
    available = [f for f in feature_cols if f in train.columns]
    print(f"  Using {len(available)} features: {available}")

    X_train = train[available].fillna(0).values
    y_train = train["resolved_yes"].values
    X_val = val[available].fillna(0).values
    y_val = val["resolved_yes"].values

    # Naive baseline
    naive_val_brier = float(((val["price_yes"].values - y_val) ** 2).mean())
    print(f"  Naive val Brier: {naive_val_brier:.4f}")

    models = {}

    # Model 1: Logistic regression on trajectory features
    lr = LogisticRegression(random_state=SEED, max_iter=2000, C=0.1)
    lr.fit(X_train, y_train)
    lr_preds = np.clip(lr.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)
    lr_brier = float(((lr_preds - y_val) ** 2).mean())
    models["logistic_trajectory"] = (lr, lr_brier)
    print(f"  Logistic val Brier: {lr_brier:.4f} (Δ={lr_brier - naive_val_brier:+.4f})")

    # Model 2: XGBoost
    if use_xgb:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, eval_metric="logloss",
            use_label_encoder=False,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_preds = np.clip(xgb.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)
        xgb_brier = float(((xgb_preds - y_val) ** 2).mean())
        models["xgb_trajectory"] = (xgb, xgb_brier)
        print(f"  XGBoost val Brier: {xgb_brier:.4f} (Δ={xgb_brier - naive_val_brier:+.4f})")

        # Feature importance
        print(f"\n  XGBoost feature importance (top 10):")
        imp = sorted(zip(available, xgb.feature_importances_), key=lambda x: -x[1])
        for feat, importance in imp[:10]:
            print(f"    {feat:<25} {importance:.4f}")

    # Model 3: Hybrid — blend model output with market price
    # predicted_prob = w * model_pred + (1-w) * price_yes
    # Optimize w on val
    best_model_name = min(models, key=lambda k: models[k][1])
    best_model, best_brier = models[best_model_name]
    print(f"\n  Best standalone model: {best_model_name} (val Brier: {best_brier:.4f})")

    # Get best model's val predictions
    if best_model_name.startswith("xgb"):
        best_val_preds = np.clip(best_model.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)
    else:
        best_val_preds = np.clip(best_model.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)

    best_w = 0.0
    best_hybrid_brier = naive_val_brier
    for w in np.arange(0.0, 1.01, 0.05):
        hybrid = np.clip(w * best_val_preds + (1 - w) * val["price_yes"].values, 1e-7, 1 - 1e-7)
        b = float(((hybrid - y_val) ** 2).mean())
        if b < best_hybrid_brier:
            best_hybrid_brier = b
            best_w = w

    print(f"  Best hybrid weight: w={best_w:.2f} "
          f"(val Brier: {best_hybrid_brier:.4f}, Δ={best_hybrid_brier - naive_val_brier:+.4f})")

    return models, best_model_name, best_w, available, naive_val_brier


# ---------------------------------------------------------------------------
# C4d: Test evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    test: pd.DataFrame, models: dict, best_model_name: str,
    best_w: float, feature_cols: list[str],
) -> dict:
    """Evaluate best model on test split."""
    print(f"\n{'='*60}")
    print("C4d: Test evaluation")
    print(f"{'='*60}")

    X_test = test[feature_cols].fillna(0).values
    y_test = test["resolved_yes"].values
    naive_preds = test["price_yes"].values
    naive_brier = float(((naive_preds - y_test) ** 2).mean())

    results = {"naive_brier": naive_brier}

    print(f"  Naive test Brier: {naive_brier:.4f}")

    for model_name, (model, _) in models.items():
        preds = np.clip(model.predict_proba(X_test)[:, 1], 1e-7, 1 - 1e-7)
        brier = float(((preds - y_test) ** 2).mean())
        decomp = brier_decomposition(y_test, preds)
        ece = calibration_error(y_test, preds)

        print(f"  {model_name}: Brier={brier:.4f} (Δ={brier - naive_brier:+.4f}), "
              f"reliability={decomp['reliability']:.4f}, ECE={ece:.4f}")
        results[model_name] = {
            "brier": brier, "reliability": decomp["reliability"], "ece": ece,
        }

    # Hybrid
    best_model = models[best_model_name][0]
    best_preds = np.clip(best_model.predict_proba(X_test)[:, 1], 1e-7, 1 - 1e-7)
    hybrid_preds = np.clip(
        best_w * best_preds + (1 - best_w) * naive_preds, 1e-7, 1 - 1e-7,
    )
    hybrid_brier = float(((hybrid_preds - y_test) ** 2).mean())
    hybrid_decomp = brier_decomposition(y_test, hybrid_preds)
    hybrid_ece = calibration_error(y_test, hybrid_preds)

    print(f"  hybrid (w={best_w:.2f}): Brier={hybrid_brier:.4f} "
          f"(Δ={hybrid_brier - naive_brier:+.4f}), "
          f"reliability={hybrid_decomp['reliability']:.4f}, ECE={hybrid_ece:.4f}")
    results["hybrid"] = {
        "brier": hybrid_brier, "weight": best_w,
        "reliability": hybrid_decomp["reliability"], "ece": hybrid_ece,
    }

    # Per-category breakdown (hybrid vs naive)
    print(f"\n  Per-category test Brier (hybrid vs naive):")
    print(f"  {'category':<25} {'naive':>8} {'hybrid':>8} {'Δ':>8} {'n_mkts':>7}")
    print(f"  {'-'*58}")
    test_eval = test.copy()
    test_eval["hybrid_pred"] = hybrid_preds
    for cat in sorted(test[GROUP_COL].unique()):
        cat_df = test_eval[test_eval[GROUP_COL] == cat]
        if len(cat_df) < 20:
            continue
        y = cat_df["resolved_yes"].values
        nb = float(((cat_df["price_yes"].values - y) ** 2).mean())
        hb = float(((cat_df["hybrid_pred"].values - y) ** 2).mean())
        n_m = cat_df["condition_id"].nunique()
        print(f"  {cat:<25} {nb:>8.4f} {hb:>8.4f} {hb-nb:>+8.4f} {n_m:>7}")

    # Market-level Brier
    print(f"\n  Market-level test Brier:")
    for label, preds in [("naive", naive_preds), ("hybrid", hybrid_preds)]:
        test_eval["_pred"] = preds
        mb = test_eval.groupby("condition_id").apply(
            lambda g: ((g["_pred"] - g["resolved_yes"]) ** 2).mean(),
            include_groups=False,
        )
        print(f"    {label:<15} mean={mb.mean():.4f}  median={mb.median():.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"C4: Price Trajectory Dynamics ({DATE_TAG})")
    print("=" * 60)

    # Load
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    print(f"Train: {len(train)} snaps, {train['condition_id'].nunique()} markets")
    print(f"Val:   {len(val)} snaps, {val['condition_id'].nunique()} markets")
    print(f"Test:  {len(test)} snaps, {test['condition_id'].nunique()} markets")

    # C4a: Engineer features
    print(f"\n{'─'*60}")
    print("C4a: Engineering trajectory features")
    print(f"{'─'*60}")
    train = add_trajectory_features(train)
    val = add_trajectory_features(val)
    test = add_trajectory_features(test)

    # C4b: Explore
    corr_results = explore_trajectory_signals(train)
    corr_results.to_csv(OUT_TABLES / f"c4_trajectory_correlations_{DATE_TAG}.csv",
                        index=False, float_format="%.4f")

    # C4c: Train model
    feature_cols = [
        # Market price (the baseline to beat)
        "price_yes",
        # Existing trajectory features
        "price_change", "price_momentum_3", "price_momentum_7",
        "price_volatility_7", "price_vs_open",
        # Temporal
        "pct_lifetime_elapsed", "days_to_end", "log_volume",
        # New trajectory features
        "staleness", "price_acceleration", "vol_regime",
        "path_curvature", "price_range_14",
        "dist_from_high", "dist_from_low", "price_extremity",
        # Structural
        "category_encoded",
    ]

    models, best_model_name, best_w, used_features, naive_val = train_trajectory_model(
        train, val, feature_cols,
    )

    # C4d: Threshold check — do we evaluate on test?
    best_val_brier = models[best_model_name][1]
    hybrid_threshold = 0.001

    # Also check hybrid on val
    best_model = models[best_model_name][0]
    X_val = val[used_features].fillna(0).values
    best_val_preds = np.clip(best_model.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)
    hybrid_val_preds = np.clip(
        best_w * best_val_preds + (1 - best_w) * val["price_yes"].values, 1e-7, 1 - 1e-7,
    )
    hybrid_val_brier = float(((hybrid_val_preds - val["resolved_yes"].values) ** 2).mean())
    val_improvement = naive_val - hybrid_val_brier

    print(f"\n  Val improvement (hybrid): {val_improvement:+.4f} "
          f"(threshold: {hybrid_threshold})")

    if val_improvement > hybrid_threshold:
        print(f"  PASSES threshold — proceeding to test evaluation")
        test_results = evaluate_on_test(test, models, best_model_name, best_w, used_features)
    else:
        print(f"  Below threshold — evaluating on test anyway for completeness")
        test_results = evaluate_on_test(test, models, best_model_name, best_w, used_features)

    # Save
    all_results = {
        "date": DATE_TAG,
        "val_improvement": val_improvement,
        "passed_threshold": val_improvement > hybrid_threshold,
        "best_model": best_model_name,
        "hybrid_weight": best_w,
        "features": used_features,
        "test_results": test_results,
    }
    results_path = OUT_TABLES / f"c4_results_{DATE_TAG}.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))

    # Summary
    print(f"\n{'='*60}")
    print("C4 SUMMARY")
    print(f"{'='*60}")
    print(f"  Best model: {best_model_name}")
    print(f"  Hybrid weight: {best_w:.2f}")
    print(f"  Val Brier: naive={naive_val:.4f}, hybrid={hybrid_val_brier:.4f} "
          f"(Δ={hybrid_val_brier - naive_val:+.4f})")

    test_naive = test_results["naive_brier"]
    test_hybrid = test_results["hybrid"]["brier"]
    print(f"  Test Brier: naive={test_naive:.4f}, hybrid={test_hybrid:.4f} "
          f"(Δ={test_hybrid - test_naive:+.4f})")

    if test_hybrid < test_naive:
        improvement_pct = (test_naive - test_hybrid) / test_naive * 100
        print(f"\n  RESULT: Trajectory model IMPROVES test Brier by "
              f"{test_naive - test_hybrid:.4f} ({improvement_pct:.1f}%)")
    else:
        print(f"\n  RESULT: Trajectory model does NOT improve test Brier.")
        print(f"  Price trajectory dynamics do not contain exploitable signal "
              f"beyond what market price already captures.")


if __name__ == "__main__":
    main()
