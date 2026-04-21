"""Characterize Polymarket calibration by domain, time horizon, and market structure.

This is a research output: answers "where and when are prediction markets miscalibrated?"
Updated 2026-04-12 for the expanded B3 dataset (4,538 markets, 11 categories).

Analyses:
    1. Overall Brier decomposition (all data, and per split)
    2. Calibration by category (snapshot-level and market-level)
    3. Calibration by time horizon
    4. Category × horizon 2D heatmap
    5. Favourite-longshot bias (overall and per category)
    6. Per-split stability check
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.evaluation.calibration import (  # noqa: E402
    brier_decomposition,
    calibration_by_group,
    calibration_by_horizon,
    calibration_curve,
    calibration_error,
    favourite_longshot_analysis,
    sharpness,
)


DATE_TAG = datetime.now().strftime("%Y-%m-%d")
OUT_TABLES = Path("outputs/tables")
OUT_FIGS = Path("outputs/figures")
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# Use 'category' as the grouping column (aliased to 'theme' in B3 splits)
GROUP_COL = "category"


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test and return (all, train, val, test)."""
    dfs = []
    splits = {}
    for split in ["train", "val", "test"]:
        path = f"data/processed/{split}.parquet"
        df = pd.read_parquet(path)
        df["split"] = split
        dfs.append(df)
        splits[split] = df
    all_df = pd.concat(dfs, ignore_index=True)
    n_markets = all_df["condition_id"].nunique()
    n_cats = all_df[GROUP_COL].nunique()
    print(f"Loaded {len(all_df)} snapshots, {n_markets} markets, {n_cats} categories")
    return all_df, splits["train"], splits["val"], splits["test"]


# ---------------------------------------------------------------------------
# 1. Overall Brier decomposition
# ---------------------------------------------------------------------------

def run_overall(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame,
                test: pd.DataFrame) -> pd.DataFrame:
    print_section("1. Overall Brier Decomposition")

    rows = []
    for name, sdf in [("all", df), ("train", train), ("val", val), ("test", test)]:
        decomp = brier_decomposition(sdf["resolved_yes"].values, sdf["price_yes"].values)
        ece = calibration_error(sdf["resolved_yes"].values, sdf["price_yes"].values)
        sharp = sharpness(sdf["price_yes"].values)
        decomp["split"] = name
        decomp["ece"] = ece
        decomp["mean_sharpness"] = sharp["mean_sharpness"]
        decomp["n_markets"] = sdf["condition_id"].nunique()
        rows.append(decomp)

    result = pd.DataFrame(rows)
    cols = ["split", "n", "n_markets", "base_rate", "brier_score", "reliability",
            "resolution", "uncertainty", "ece", "mean_sharpness"]
    result = result[cols]
    print(result.to_string(index=False, float_format="%.4f"))
    return result


# ---------------------------------------------------------------------------
# 2. Calibration by category
# ---------------------------------------------------------------------------

def run_by_category(df: pd.DataFrame) -> pd.DataFrame:
    print_section("2. Calibration by Category")

    cat_cal = calibration_by_group(df, GROUP_COL)
    print(cat_cal.to_string(index=False, float_format="%.4f"))

    # Market-level Brier (one score per market, then average per category)
    print(f"\n  Market-level Brier (averaging per market first, then per category):")
    market_brier = df.groupby("condition_id").apply(
        lambda g: pd.Series({
            "market_brier": ((g["price_yes"] - g["resolved_yes"]) ** 2).mean(),
            GROUP_COL: g[GROUP_COL].iloc[0],
        }),
        include_groups=False,
    )
    cat_market_brier = market_brier.groupby(GROUP_COL)["market_brier"].agg(
        ["mean", "median", "count"]
    ).sort_values("mean")
    print(f"  {GROUP_COL:<25} {'mean':>8} {'median':>8} {'n_mkts':>7}")
    print(f"  {'-'*50}")
    for cat, row in cat_market_brier.iterrows():
        print(f"  {cat:<25} {row['mean']:>8.4f} {row['median']:>8.4f} {int(row['count']):>7}")

    return cat_cal


# ---------------------------------------------------------------------------
# 3. Calibration by horizon
# ---------------------------------------------------------------------------

def run_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    print_section("3. Calibration by Time Horizon")

    horizon_cal = calibration_by_horizon(df)
    print(horizon_cal.to_string(index=False, float_format="%.4f"))
    return horizon_cal


# ---------------------------------------------------------------------------
# 4. Category × Horizon heatmap
# ---------------------------------------------------------------------------

def run_category_x_horizon(df: pd.DataFrame) -> pd.DataFrame:
    print_section("4. Category × Horizon Brier Heatmap")

    horizon_bins = [
        (0, 7, "<1w"), (7, 30, "1w-1m"), (30, 90, "1m-3m"),
        (90, 365, "3m-1y"), (365, float("inf"), ">1y"),
    ]

    rows = []
    for cat in sorted(df[GROUP_COL].unique()):
        cat_df = df[df[GROUP_COL] == cat]
        for low, high, label in horizon_bins:
            mask = (cat_df["days_to_end"] >= low) & (cat_df["days_to_end"] < high)
            subset = cat_df[mask]
            if len(subset) < 20:
                rows.append({
                    GROUP_COL: cat, "horizon": label,
                    "brier": np.nan, "ece": np.nan, "n_snaps": len(subset),
                    "n_markets": subset["condition_id"].nunique() if len(subset) > 0 else 0,
                    "base_rate": np.nan,
                })
                continue
            decomp = brier_decomposition(
                subset["resolved_yes"].values, subset["price_yes"].values,
            )
            ece = calibration_error(
                subset["resolved_yes"].values, subset["price_yes"].values,
            )
            rows.append({
                GROUP_COL: cat, "horizon": label,
                "brier": decomp["brier_score"], "ece": ece,
                "n_snaps": len(subset),
                "n_markets": subset["condition_id"].nunique(),
                "base_rate": decomp["base_rate"],
            })

    result = pd.DataFrame(rows)

    # Print as pivot
    pivot = result.pivot(index=GROUP_COL, columns="horizon", values="brier")
    pivot = pivot[["<1w", "1w-1m", "1m-3m", "3m-1y", ">1y"]]
    print(pivot.to_string(float_format="%.4f", na_rep="  -  "))

    return result


# ---------------------------------------------------------------------------
# 5. Favourite-longshot bias
# ---------------------------------------------------------------------------

def run_fl_bias(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("5. Favourite-Longshot Bias")

    fl_overall = favourite_longshot_analysis(df)
    print("Overall:")
    print(fl_overall.to_string(index=False, float_format="%.4f"))

    # Interpret
    if len(fl_overall) > 0:
        longshot = fl_overall[fl_overall["bucket_mid"] < 0.3]
        fav = fl_overall[fl_overall["bucket_mid"] > 0.7]
        if len(longshot) > 0:
            lb = longshot["bias"].mean()
            print(f"\n  Longshot bias (price < 0.3): {lb:+.4f}", end="")
            if lb > 0.01:
                print("  → market underprices longshots (classic F-L)")
            elif lb < -0.01:
                print("  → market overprices longshots (reverse F-L)")
            else:
                print("  → roughly calibrated")
        if len(fav) > 0:
            fb = fav["bias"].mean()
            print(f"  Favourite bias (price > 0.7): {fb:+.4f}", end="")
            if fb < -0.01:
                print("  → market overprices favourites (classic F-L)")
            elif fb > 0.01:
                print("  → market underprices favourites (reverse F-L)")
            else:
                print("  → roughly calibrated")

    # Per-category F-L
    print(f"\n  Per-category longshot bias (price < 0.3):")
    print(f"  {'category':<25} {'bias':>8} {'n_snaps':>8} {'n_mkts':>7}")
    print(f"  {'-'*50}")

    fl_per_cat_rows = []
    for cat in sorted(df[GROUP_COL].unique()):
        cat_df = df[df[GROUP_COL] == cat]
        fl_cat = favourite_longshot_analysis(cat_df)
        if len(fl_cat) == 0:
            continue
        ls = fl_cat[fl_cat["bucket_mid"] < 0.3]
        if len(ls) > 0:
            bias = ls["bias"].mean()
            n_s = ls["n_snapshots"].sum()
            n_m = ls["n_markets"].sum()
            print(f"  {cat:<25} {bias:>+8.4f} {n_s:>8} {n_m:>7}")
            fl_per_cat_rows.append({
                GROUP_COL: cat, "longshot_bias": bias,
                "n_snapshots": n_s, "n_markets": n_m,
            })

    fl_per_cat = pd.DataFrame(fl_per_cat_rows)
    return fl_overall, fl_per_cat


# ---------------------------------------------------------------------------
# 6. Per-split stability
# ---------------------------------------------------------------------------

def run_split_stability(train: pd.DataFrame, val: pd.DataFrame,
                        test: pd.DataFrame) -> pd.DataFrame:
    print_section("6. Per-Split Stability Check")
    print("  (Do calibration patterns hold across temporal splits?)\n")

    rows = []
    for sname, sdf in [("train", train), ("val", val), ("test", test)]:
        for cat in sorted(sdf[GROUP_COL].unique()):
            subset = sdf[sdf[GROUP_COL] == cat]
            if len(subset) < 20:
                continue
            decomp = brier_decomposition(
                subset["resolved_yes"].values, subset["price_yes"].values,
            )
            rows.append({
                "split": sname, GROUP_COL: cat,
                "brier": decomp["brier_score"],
                "reliability": decomp["reliability"],
                "base_rate": decomp["base_rate"],
                "n_markets": subset["condition_id"].nunique(),
            })

    result = pd.DataFrame(rows)
    pivot = result.pivot(index=GROUP_COL, columns="split", values="brier")
    pivot = pivot[["train", "val", "test"]]
    print(pivot.to_string(float_format="%.4f", na_rep="  -  "))

    # Flag categories where Brier shifts >0.05 between splits
    print(f"\n  Categories with Brier shift > 0.05 between train and test:")
    for cat in pivot.index:
        t = pivot.loc[cat, "train"]
        te = pivot.loc[cat, "test"]
        if pd.notna(t) and pd.notna(te) and abs(te - t) > 0.05:
            print(f"    {cat}: train={t:.4f} → test={te:.4f} (Δ={te-t:+.4f})")

    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_calibration_by_category(df: pd.DataFrame) -> None:
    """Reliability diagram per category."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(dash="dash", color="gray")))

    for cat in sorted(df[GROUP_COL].unique()):
        subset = df[df[GROUP_COL] == cat]
        cal = calibration_curve(subset["resolved_yes"].values, subset["price_yes"].values)
        if len(cal) > 0:
            n_m = subset["condition_id"].nunique()
            fig.add_trace(go.Scatter(
                x=cal["mean_predicted"], y=cal["mean_observed"],
                mode="lines+markers",
                name=f"{cat} ({n_m} mkts)",
                text=cal["count"],
                hovertemplate="%{text} snaps<br>predicted: %{x:.2f}<br>observed: %{y:.2f}",
            ))

    fig.update_layout(
        title=f"Polymarket Calibration by Category ({DATE_TAG}, n={df['condition_id'].nunique()} markets)",
        xaxis_title="Market Price (predicted probability)",
        yaxis_title="Actual Resolution Rate",
        width=900, height=650,
    )
    path = OUT_FIGS / "calibration_by_category.html"
    fig.write_html(str(path))
    print(f"  Saved: {path}")


def plot_calibration_by_horizon(df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(dash="dash", color="gray")))

    horizon_bins = [
        (0, 7, "<1w"), (7, 30, "1w-1m"), (30, 90, "1m-3m"),
        (90, 365, "3m-1y"), (365, float("inf"), ">1y"),
    ]
    for low, high, label in horizon_bins:
        mask = (df["days_to_end"] >= low) & (df["days_to_end"] < high)
        subset = df[mask]
        if len(subset) < 20:
            continue
        cal = calibration_curve(subset["resolved_yes"].values, subset["price_yes"].values)
        if len(cal) > 0:
            n_m = subset["condition_id"].nunique()
            fig.add_trace(go.Scatter(
                x=cal["mean_predicted"], y=cal["mean_observed"],
                mode="lines+markers", name=f"{label} ({n_m} mkts)",
            ))

    fig.update_layout(
        title=f"Polymarket Calibration by Time Horizon ({DATE_TAG})",
        xaxis_title="Market Price", yaxis_title="Actual Resolution Rate",
        width=900, height=650,
    )
    path = OUT_FIGS / "calibration_by_horizon.html"
    fig.write_html(str(path))
    print(f"  Saved: {path}")


def plot_fl_bias(fl: pd.DataFrame) -> None:
    if len(fl) == 0:
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fl["price_bucket"], y=fl["bias"],
        marker_color=["#e74c3c" if b < 0 else "#2ecc71" for b in fl["bias"]],
        text=[f"n={n}" for n in fl["n_snapshots"]],
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Favourite-Longshot Bias ({DATE_TAG}, positive = underpriced)",
        xaxis_title="Price Bucket", yaxis_title="Bias (actual - predicted)",
        width=800, height=400,
    )
    path = OUT_FIGS / "favourite_longshot_bias.html"
    fig.write_html(str(path))
    print(f"  Saved: {path}")


def plot_category_x_horizon_heatmap(heatmap_df: pd.DataFrame) -> None:
    pivot = heatmap_df.pivot(index=GROUP_COL, columns="horizon", values="brier")
    pivot = pivot[["<1w", "1w-1m", "1m-3m", "3m-1y", ">1y"]]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        text=np.where(np.isnan(pivot.values), "",
                      np.vectorize(lambda x: f"{x:.3f}")(pivot.values)),
        texttemplate="%{text}",
        colorbar_title="Brier",
    ))
    fig.update_layout(
        title=f"Category × Horizon Brier Score ({DATE_TAG})",
        xaxis_title="Time to Resolution", yaxis_title="Category",
        width=800, height=500,
    )
    path = OUT_FIGS / "calibration_category_x_horizon.html"
    fig.write_html(str(path))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"MarketMind: Polymarket Calibration Analysis ({DATE_TAG})")
    print("=" * 60)

    all_df, train, val, test = load_splits()

    # 1. Overall
    overall = run_overall(all_df, train, val, test)

    # 2. By category
    cat_cal = run_by_category(all_df)

    # 3. By horizon
    horizon_cal = run_by_horizon(all_df)

    # 4. Category × horizon
    cat_x_hor = run_category_x_horizon(all_df)

    # 5. F-L bias
    fl_overall, fl_per_cat = run_fl_bias(all_df)

    # 6. Split stability
    split_stab = run_split_stability(train, val, test)

    # Save tables
    overall.to_csv(OUT_TABLES / "brier_decomposition.csv", index=False, float_format="%.6f")
    cat_cal.to_csv(OUT_TABLES / "calibration_by_category.csv", index=False, float_format="%.4f")
    horizon_cal.to_csv(OUT_TABLES / "calibration_by_horizon.csv", index=False, float_format="%.4f")
    cat_x_hor.to_csv(OUT_TABLES / "calibration_category_x_horizon.csv", index=False, float_format="%.4f")
    fl_overall.to_csv(OUT_TABLES / "favourite_longshot_bias.csv", index=False, float_format="%.4f")
    if len(fl_per_cat) > 0:
        fl_per_cat.to_csv(OUT_TABLES / "fl_bias_per_category.csv", index=False, float_format="%.4f")
    split_stab.to_csv(OUT_TABLES / "calibration_split_stability.csv", index=False, float_format="%.4f")
    print(f"\n  Saved all tables to {OUT_TABLES}")

    # Plots
    print("\nGenerating plots...")
    plot_calibration_by_category(all_df)
    plot_calibration_by_horizon(all_df)
    plot_fl_bias(fl_overall)
    plot_category_x_horizon_heatmap(cat_x_hor)

    # Also keep the old 'theme' versions for backward compat
    # (the B3 data has theme=category so these are identical)
    (OUT_FIGS / "calibration_by_theme.html").unlink(missing_ok=True)

    print(f"\nCalibration analysis complete. {DATE_TAG}")


if __name__ == "__main__":
    main()
