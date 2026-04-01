"""Streamlit dashboard for MarketMind model comparison."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluation.calibration import calibration_curve, calibration_error, sharpness
from src.evaluation.compare_models import evaluate_predictions

st.set_page_config(page_title="MarketMind", layout="wide")
st.title("MarketMind: Forecasting Model Comparison")


@st.cache_data
def load_comparison_table():
    path = "outputs/tables/model_comparison.csv"
    if Path(path).exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_data_splits():
    splits = {}
    for name in ["train", "val", "test"]:
        path = f"data/processed/{name}.parquet"
        if Path(path).exists():
            splits[name] = pd.read_parquet(path)
    return splits


# --- Sidebar ---
st.sidebar.header("Settings")
split_choice = st.sidebar.selectbox("Evaluation split", ["test", "val", "train"])
n_bins = st.sidebar.slider("Calibration bins", 5, 20, 10)

# --- Main content ---
comparison = load_comparison_table()
splits = load_data_splits()

if comparison is not None:
    st.header("Model Comparison")

    # Filter to selected split
    if "split" in comparison.columns:
        comp_split = comparison[comparison["split"] == split_choice]
    else:
        comp_split = comparison

    if len(comp_split) > 0:
        # Metrics table
        display_cols = [c for c in ["model", "brier_score", "log_loss", "calibration_error",
                                     "mean_sharpness", "pct_confident", "n"] if c in comp_split.columns]
        st.dataframe(comp_split[display_cols].sort_values("brier_score"), use_container_width=True)

        # Bar chart
        col1, col2 = st.columns(2)
        with col1:
            metric = st.selectbox("Metric", ["brier_score", "log_loss", "calibration_error"])
            if metric in comp_split.columns:
                fig = px.bar(
                    comp_split.sort_values(metric),
                    x="model", y=metric,
                    title=f"{metric} by Model ({split_choice})",
                    color="model",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "mean_sharpness" in comp_split.columns:
                fig2 = px.bar(
                    comp_split.sort_values("mean_sharpness", ascending=False),
                    x="model", y="mean_sharpness",
                    title=f"Forecast Sharpness ({split_choice})",
                    color="model",
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info(f"No results for split '{split_choice}'")

elif splits:
    st.info("No model comparison table found. Run the training and evaluation pipeline first.")
    st.header("Data Summary")
    for name, df in splits.items():
        st.subheader(f"{name} set: {len(df)} rows")
        if "resolved_yes" in df.columns:
            st.metric("Base rate", f"{df['resolved_yes'].mean():.3f}")
        if "category" in df.columns:
            st.write(df["category"].value_counts().head(10))

else:
    st.warning("No data found. Run the data pipeline first: `python scripts/run_data_pipeline.py`")

st.markdown("---")
st.caption("MarketMind - Probabilistic Forecasting Research Pipeline")
