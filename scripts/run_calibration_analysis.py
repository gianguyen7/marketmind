"""Characterize Polymarket calibration by domain, time horizon, and market structure.

This is a research output: answers "where and when are prediction markets miscalibrated?"
Run this before ML experiments to motivate where correction models should focus.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.evaluation.calibration import (
    brier_decomposition,
    calibration_by_group,
    calibration_by_horizon,
    calibration_curve,
    favourite_longshot_analysis,
)


def load_all_snapshots() -> pd.DataFrame:
    """Load all splits and tag them."""
    dfs = []
    for split in ["train", "val", "test"]:
        path = f"data/processed/{split}.parquet"
        if Path(path).exists():
            df = pd.read_parquet(path)
            df["split"] = split
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_overall_brier(df: pd.DataFrame) -> None:
    """Overall Brier decomposition for the market price."""
    print_section("Overall Market Calibration (Brier Decomposition)")

    decomp = brier_decomposition(df["resolved_yes"].values, df["price_yes"].values)
    print(f"  Brier score:  {decomp['brier_score']:.4f}")
    print(f"  Reliability:  {decomp['reliability']:.4f}  (lower = better calibrated)")
    print(f"  Resolution:   {decomp['resolution']:.4f}  (higher = better discrimination)")
    print(f"  Uncertainty:  {decomp['uncertainty']:.4f}  (fixed: base_rate * (1 - base_rate))")
    print(f"  Base rate:    {decomp['base_rate']:.4f}")
    print(f"  n snapshots:  {decomp['n']}")
    print(f"  n markets:    {df['condition_id'].nunique()}")
    return decomp


def run_by_theme(df: pd.DataFrame) -> pd.DataFrame:
    """Calibration analysis by theme/domain."""
    print_section("Calibration by Theme (Domain)")

    theme_cal = calibration_by_group(df, "theme")
    print(theme_cal.to_string(index=False, float_format="%.4f"))
    return theme_cal


def run_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Calibration analysis by time-to-resolution."""
    print_section("Calibration by Time Horizon")

    horizon_cal = calibration_by_horizon(df)
    print(horizon_cal.to_string(index=False, float_format="%.4f"))
    return horizon_cal


def run_fl_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Favourite-longshot bias analysis."""
    print_section("Favourite-Longshot Bias Analysis")

    fl = favourite_longshot_analysis(df)
    print(fl.to_string(index=False, float_format="%.4f"))

    # Interpret
    if len(fl) > 0:
        longshot_bins = fl[fl["bucket_mid"] < 0.3]
        favourite_bins = fl[fl["bucket_mid"] > 0.7]

        if len(longshot_bins) > 0:
            avg_longshot_bias = longshot_bins["bias"].mean()
            print(f"\n  Longshot bias (price < 0.3): {avg_longshot_bias:+.4f}")
            if avg_longshot_bias > 0.01:
                print("  -> Longshots win MORE often than prices imply (classic F-L bias)")
            elif avg_longshot_bias < -0.01:
                print("  -> Longshots win LESS often than prices imply (reverse F-L bias)")
            else:
                print("  -> Longshots roughly well-calibrated")

        if len(favourite_bins) > 0:
            avg_fav_bias = favourite_bins["bias"].mean()
            print(f"  Favourite bias (price > 0.7): {avg_fav_bias:+.4f}")
            if avg_fav_bias < -0.01:
                print("  -> Favourites win LESS often than prices imply (classic F-L bias)")
            elif avg_fav_bias > 0.01:
                print("  -> Favourites win MORE often than prices imply (reverse F-L bias)")
            else:
                print("  -> Favourites roughly well-calibrated")

    return fl


def plot_calibration_by_theme(df: pd.DataFrame, output_path: str) -> None:
    """Plot calibration curves for each theme."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(dash="dash", color="gray")))

    for theme in sorted(df["theme"].unique()):
        subset = df[df["theme"] == theme]
        cal = calibration_curve(subset["resolved_yes"].values, subset["price_yes"].values)
        if len(cal) > 0:
            fig.add_trace(go.Scatter(
                x=cal["mean_predicted"], y=cal["mean_observed"],
                mode="lines+markers", name=f"{theme} (n={len(subset)})",
                text=cal["count"],
                hovertemplate="%{text} samples<br>predicted: %{x:.2f}<br>observed: %{y:.2f}",
            ))

    fig.update_layout(title="Polymarket Calibration by Theme",
                      xaxis_title="Market Price", yaxis_title="Actual Resolution Rate",
                      width=800, height=600)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def plot_calibration_by_horizon(df: pd.DataFrame, output_path: str) -> None:
    """Plot calibration curves for each time horizon bucket."""
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
            fig.add_trace(go.Scatter(
                x=cal["mean_predicted"], y=cal["mean_observed"],
                mode="lines+markers", name=f"{label} (n={len(subset)})",
            ))

    fig.update_layout(title="Polymarket Calibration by Time to Resolution",
                      xaxis_title="Market Price", yaxis_title="Actual Resolution Rate",
                      width=800, height=600)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def plot_fl_bias(fl: pd.DataFrame, output_path: str) -> None:
    """Plot favourite-longshot bias."""
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
        title="Favourite-Longshot Bias (positive = market underprices this outcome)",
        xaxis_title="Price Bucket", yaxis_title="Bias (actual - predicted)",
        width=800, height=400,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("MarketMind: Polymarket Calibration Analysis")
    print("=" * 60)

    df = load_all_snapshots()
    print(f"\nLoaded {len(df)} snapshots across {df['condition_id'].nunique()} markets")

    # 1. Overall
    decomp = run_overall_brier(df)

    # 2. By theme
    theme_cal = run_by_theme(df)

    # 3. By horizon
    horizon_cal = run_by_horizon(df)

    # 4. Favourite-longshot bias
    fl = run_fl_bias(df)

    # 5. Save tables
    out_tables = Path("outputs/tables")
    out_tables.mkdir(parents=True, exist_ok=True)

    theme_cal.to_csv(out_tables / "calibration_by_theme.csv", index=False, float_format="%.4f")
    horizon_cal.to_csv(out_tables / "calibration_by_horizon.csv", index=False, float_format="%.4f")
    fl.to_csv(out_tables / "favourite_longshot_bias.csv", index=False, float_format="%.4f")

    decomp_df = pd.DataFrame([decomp])
    decomp_df.to_csv(out_tables / "brier_decomposition.csv", index=False, float_format="%.6f")
    print(f"\n  Saved tables to {out_tables}")

    # 6. Plots
    print("\nGenerating plots...")
    out_figs = "outputs/figures"
    plot_calibration_by_theme(df, f"{out_figs}/calibration_by_theme.html")
    plot_calibration_by_horizon(df, f"{out_figs}/calibration_by_horizon.html")
    plot_fl_bias(fl, f"{out_figs}/favourite_longshot_bias.html")

    print("\nCalibration analysis complete.")


if __name__ == "__main__":
    main()
