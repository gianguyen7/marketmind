"""Generate evaluation reports and figures."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.evaluation.calibration import calibration_curve


def plot_calibration_curves(
    model_results: dict,
    split: str = "test",
    n_bins: int = 10,
    output_path: str | None = None,
) -> go.Figure:
    """Plot calibration curves for multiple models."""
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Perfect",
        line=dict(dash="dash", color="gray"),
    ))

    for model_name, (model, preds) in model_results.items():
        y_true = preds.get(f"y_{split}", np.array([]))
        y_prob = preds.get(split, np.array([]))
        if len(y_true) == 0:
            continue

        cal = calibration_curve(y_true, y_prob, n_bins)
        fig.add_trace(go.Scatter(
            x=cal["mean_predicted"],
            y=cal["mean_observed"],
            mode="lines+markers",
            name=model_name,
            text=cal["count"],
            hovertemplate="%{text} samples<br>predicted: %{x:.2f}<br>observed: %{y:.2f}",
        ))

    fig.update_layout(
        title="Calibration Curves",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        width=700, height=500,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "brier_score",
    output_path: str | None = None,
) -> go.Figure:
    """Bar chart comparing models on a metric."""
    fig = px.bar(
        comparison_df.sort_values(metric),
        x="model", y=metric,
        title=f"Model Comparison: {metric}",
        color="model",
    )
    fig.update_layout(showlegend=False, width=700, height=400)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)

    return fig


def plot_prediction_distributions(
    model_results: dict,
    split: str = "test",
    output_path: str | None = None,
) -> go.Figure:
    """Histogram of prediction distributions per model."""
    fig = go.Figure()

    for model_name, (model, preds) in model_results.items():
        y_prob = preds.get(split, np.array([]))
        if len(y_prob) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=y_prob, name=model_name,
            opacity=0.6, nbinsx=30,
        ))

    fig.update_layout(
        title="Prediction Distributions",
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        barmode="overlay",
        width=700, height=400,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)

    return fig


def save_comparison_table(
    comparison_df: pd.DataFrame,
    output_path: str = "outputs/tables/model_comparison.csv",
) -> None:
    """Save model comparison table to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved comparison table to {output_path}")
