"""MarketMind — Polymarket Calibration Research Dashboard.

Run with: streamlit run src/dashboard/app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MarketMind",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

TABLES = Path("outputs/tables")
FIGURES = Path("outputs/figures")
DATA = Path("data/processed")
RAW = Path("data/raw")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict | None:
    path = TABLES / name
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_splits() -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ["train", "val", "test"]:
        p = DATA / f"{name}.parquet"
        if p.exists():
            splits[name] = pd.read_parquet(p)
    return splits


@st.cache_data
def load_raw_markets() -> pd.DataFrame | None:
    p = RAW / "gamma_markets.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["start_date"] = pd.to_datetime(df["start_date"], format="ISO8601", utc=True)
        return df
    return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Overview",
    "Calibration Explorer",
    "Favourite-Longshot Bias",
    "Exploitation Attempts",
    "Data Deep Dive",
]

st.sidebar.title("MarketMind")
st.sidebar.caption("Polymarket Calibration Research")
page = st.sidebar.radio("Navigate", PAGES)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------


def page_overview():
    st.title("Is Polymarket Efficient?")
    st.markdown(
        "This research asked whether binary prediction markets on Polymarket have "
        "structural inefficiencies that a model can exploit. After four independent "
        "exploitation attempts on 4,538 markets, the answer is **no** — the market "
        "is remarkably well-calibrated."
    )

    # Headline metrics
    brier_df = load_csv("brier_decomposition.csv")
    if brier_df is not None:
        overall = brier_df[brier_df["split"] == "all"].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brier Score", f"{overall['brier_score']:.4f}")
        c2.metric("Reliability", f"{overall['reliability']:.4f}", help="How well prices match resolution rates. Lower is better. 0 = perfect.")
        c3.metric("ECE", f"{overall['ece']:.1%}", help="Expected Calibration Error")
        c4.metric("Markets", f"{int(overall['n_markets']):,}")

    st.markdown("---")

    # Research arc table
    st.subheader("Research Arc")
    arc_data = [
        ["B5: Structural alpha-shift", "Long-horizon markets under-price YES → bet YES", "+0.1495", "Much worse", "Rule over-corrects the 75-81% that resolve NO"],
        ["C1: Calibration correction", "Platt/isotonic recalibration of F-L bias", "-0.0001", "Negligible", "Reliability already 0.0004 — nothing to fix"],
        ["C4: Trajectory dynamics", "XGB on price trajectory features, blended with market", "-0.0006", "0.7% better", "Standalone model worse than naive; only works as blend"],
        ["Global ML correction", "RF/XGB regressor on market_error", "Positive", "Worse", "Overfits badly; train MAE 0.016 → test 0.408"],
    ]
    arc_df = pd.DataFrame(arc_data, columns=["Attempt", "Approach", "Test Brier Delta", "Verdict", "Why"])
    st.dataframe(arc_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Brier decomposition by split
    if brier_df is not None:
        st.subheader("Brier Decomposition by Split")
        split_df = brier_df[brier_df["split"] != "all"].copy()
        split_df["split"] = pd.Categorical(split_df["split"], ["train", "val", "test"])
        split_df = split_df.sort_values("split")

        cols_show = ["split", "n_markets", "base_rate", "brier_score", "reliability", "resolution", "uncertainty", "ece"]
        display = split_df[cols_show].copy()
        display.columns = ["Split", "Markets", "Base Rate", "Brier", "Reliability", "Resolution", "Uncertainty", "ECE"]
        st.dataframe(
            display.style.format({
                "Base Rate": "{:.3f}", "Brier": "{:.4f}", "Reliability": "{:.4f}",
                "Resolution": "{:.4f}", "Uncertainty": "{:.4f}", "ECE": "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

    # Key conclusion
    st.info(
        "**Bottom line:** Polymarket's crowd aggregates information well, calibrates "
        "probabilities accurately (reliability 0.0004), and leaves very little room "
        "for systematic improvement. The favourite-longshot bias is real but too small "
        "to exploit. The most valuable output of this project is the *characterization* "
        "of where and how the market works, not a model that beats it."
    )


# ---------------------------------------------------------------------------
# Page: Calibration Explorer
# ---------------------------------------------------------------------------


def page_calibration():
    st.title("Calibration Explorer")

    tab_cat, tab_horizon, tab_heatmap = st.tabs(["By Category", "By Horizon", "Category x Horizon"])

    # --- By Category ---
    with tab_cat:
        cat_df = load_csv("calibration_by_category.csv")
        if cat_df is None:
            st.warning("calibration_by_category.csv not found")
            return

        sort_col = st.selectbox("Sort by", ["brier_score", "ece", "reliability", "n_markets"], key="cat_sort")
        cat_sorted = cat_df.sort_values(sort_col, ascending=(sort_col != "n_markets"))

        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cat_sorted["category"], y=cat_sorted["brier_score"],
            name="Brier", marker_color="#636EFA",
        ))
        fig.add_trace(go.Bar(
            x=cat_sorted["category"], y=cat_sorted["ece"],
            name="ECE", marker_color="#EF553B",
        ))
        fig.update_layout(
            barmode="group", title="Calibration by Category",
            yaxis_title="Score (lower is better)", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        display_cols = ["category", "n_markets", "n_snapshots", "base_rate", "brier_score", "ece", "reliability", "resolution"]
        st.dataframe(
            cat_sorted[display_cols].style.format({
                "base_rate": "{:.3f}", "brier_score": "{:.4f}", "ece": "{:.4f}",
                "reliability": "{:.4f}", "resolution": "{:.4f}", "n_snapshots": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown(
            "**Key patterns:** Sports is the best-calibrated large category (Brier 0.065, 2,099 markets). "
            "Geopolitics has the worst Brier (0.124) but excellent reliability — the high score comes from "
            "base-rate uncertainty, not miscalibration. Government policy has the worst reliability (0.039)."
        )

    # --- By Horizon ---
    with tab_horizon:
        hor_df = load_csv("calibration_by_horizon.csv")
        if hor_df is None:
            st.warning("calibration_by_horizon.csv not found")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hor_df["horizon"], y=hor_df["brier_score"],
            mode="lines+markers", name="Brier", line=dict(width=3),
        ))
        fig.add_trace(go.Scatter(
            x=hor_df["horizon"], y=hor_df["ece"],
            mode="lines+markers", name="ECE", line=dict(width=3),
        ))
        fig.add_trace(go.Scatter(
            x=hor_df["horizon"], y=hor_df["reliability"],
            mode="lines+markers", name="Reliability", line=dict(width=3),
        ))
        fig.update_layout(
            title="Calibration by Time Horizon", height=400,
            xaxis_title="Time to Resolution", yaxis_title="Score",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            hor_df[["horizon", "n_markets", "n_snapshots", "base_rate", "brier_score", "ece", "reliability"]].style.format({
                "base_rate": "{:.3f}", "brier_score": "{:.4f}", "ece": "{:.4f}", "reliability": "{:.4f}", "n_snapshots": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown(
            "**Non-monotonic pattern.** Calibration is best at <1d (resolution imminent) and worst at "
            "1d-1w and >1y. The >1y degradation (ECE 0.101, reliability 0.028) reflects genuine "
            "miscalibration on very long-horizon markets."
        )

    # --- Category x Horizon ---
    with tab_heatmap:
        heatmap_df = load_csv("calibration_category_x_horizon.csv")
        if heatmap_df is None:
            st.warning("calibration_category_x_horizon.csv not found")
            return

        metric = st.selectbox("Heatmap metric", ["brier", "ece"], key="heatmap_metric")
        pivot = heatmap_df.pivot_table(index="category", columns="horizon", values=metric, aggfunc="first")

        # Order horizons sensibly
        horizon_order = ["<1w", "1w-1m", "1m-3m", "3m-1y", ">1y"]
        ordered_cols = [h for h in horizon_order if h in pivot.columns]
        pivot = pivot[ordered_cols]

        fig = px.imshow(
            pivot, text_auto=".3f", aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title=f"{metric.upper()} by Category x Horizon",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Worst pockets:** geopolitics >1y (Brier 0.574), science_tech >1y (0.422), "
            "government_policy 3m-1y (0.337). **Best:** entertainment <1w (0.029), "
            "recession_economy 1m-3m (0.007)."
        )


# ---------------------------------------------------------------------------
# Page: Favourite-Longshot Bias
# ---------------------------------------------------------------------------


def page_fl_bias():
    st.title("Favourite-Longshot Bias")

    st.markdown(
        "The classic favourite-longshot bias: longshots (low-probability events) win "
        "more often than their prices imply, while favourites win less often. "
        "Polymarket exhibits this pattern clearly."
    )

    tab_overall, tab_category = st.tabs(["Overall F-L Bias", "Per Category"])

    with tab_overall:
        fl_df = load_csv("favourite_longshot_bias.csv")
        if fl_df is None:
            st.warning("favourite_longshot_bias.csv not found")
            return

        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=fl_df["mean_price"], y=fl_df["mean_price"],
            mode="lines", name="Perfect calibration",
            line=dict(dash="dash", color="gray"),
        ))

        # Actual resolution rates
        fig.add_trace(go.Scatter(
            x=fl_df["mean_price"], y=fl_df["actual_rate"],
            mode="lines+markers", name="Actual resolution rate",
            line=dict(width=3, color="#636EFA"),
            marker=dict(size=10),
        ))

        # Bias bars
        fig.add_trace(go.Bar(
            x=fl_df["mean_price"], y=fl_df["bias"],
            name="Bias (actual - predicted)",
            marker_color=fl_df["bias"].apply(lambda x: "#2CA02C" if x > 0 else "#D62728"),
            opacity=0.6, yaxis="y2",
        ))

        fig.update_layout(
            title="Favourite-Longshot Bias Across All Markets",
            xaxis_title="Market Price (predicted probability)",
            yaxis_title="Resolution Rate",
            yaxis2=dict(title="Bias", overlaying="y", side="right", range=[-0.1, 0.1]),
            height=500, legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        col1, col2, col3 = st.columns(3)
        longshot_bias = fl_df[fl_df["bucket_mid"] < 0.3]["bias"].mean()
        fav_bias = fl_df[fl_df["bucket_mid"] > 0.7]["bias"].mean()
        col1.metric("Longshot Bias (price < 0.3)", f"+{longshot_bias:.1%}")
        col2.metric("Favourite Bias (price > 0.7)", f"{fav_bias:.1%}")
        col3.metric("Crossover Point", "~0.45")

        st.dataframe(
            fl_df[["price_bucket", "mean_price", "actual_rate", "bias", "n_snapshots", "n_markets"]].style.format({
                "mean_price": "{:.4f}", "actual_rate": "{:.4f}", "bias": "{:+.4f}", "n_snapshots": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

    with tab_category:
        fl_cat_df = load_csv("fl_bias_per_category.csv")
        if fl_cat_df is None:
            st.warning("fl_bias_per_category.csv not found")
            return

        fl_cat_sorted = fl_cat_df.sort_values("longshot_bias", ascending=False)

        fig = px.bar(
            fl_cat_sorted, x="category", y="longshot_bias",
            color="longshot_bias",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Longshot Bias by Category",
            text=fl_cat_sorted["longshot_bias"].apply(lambda x: f"{x:+.1%}"),
        )
        fig.update_layout(height=450, yaxis_title="Longshot Bias (positive = underpriced)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            fl_cat_sorted.style.format({
                "longshot_bias": "{:+.4f}", "n_snapshots": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown(
            "**Government policy** has a massive +17.4pp longshot bias — the market badly "
            "underprices low-probability government events. **Crypto** and **social media** show "
            "reverse bias (longshots are *overpriced*). Note low-n categories are fragile."
        )


# ---------------------------------------------------------------------------
# Page: Exploitation Attempts
# ---------------------------------------------------------------------------


def page_exploitation():
    st.title("Four Attempts to Beat the Market")

    st.markdown(
        "We tried four independent approaches to systematically outperform Polymarket's "
        "crowd-sourced probabilities. All failed."
    )

    tab_b5, tab_c1, tab_c4, tab_stability = st.tabs([
        "B5: Structural Rule", "C1: Recalibration", "C4: Trajectory Dynamics", "Split Stability",
    ])

    # --- B5 ---
    with tab_b5:
        st.subheader("B5: Walk-Forward Structural Hypothesis")
        st.markdown(
            "**Hypothesis:** Long-horizon, low-liquidity binary markets systematically "
            "under-price YES outcomes. Apply a blanket alpha-shift to correct."
        )

        b5 = load_json("b5_results_2026-04-12.json")
        if b5 and "phase2" in b5:
            phase2 = b5["phase2"]
            splits_data = phase2.get("splits", phase2)

            rows = []
            for split_name in ["train", "val", "test"]:
                if split_name in splits_data:
                    s = splits_data[split_name]
                    rows.append({
                        "Split": split_name,
                        "Struct Markets": s.get("n_struct", s.get("n_bets", "?")),
                        "Naive Brier": s.get("brier_naive_struct", s.get("brier_naive", 0)),
                        "Rule Brier": s.get("brier_rule_struct", s.get("brier_rule", 0)),
                        "Delta": s.get("brier_rule_struct", s.get("brier_rule", 0)) - s.get("brier_naive_struct", s.get("brier_naive", 0)),
                        "P&L ($100/bet)": s.get("pnl_total", 0),
                        "Win Rate": s.get("win_rate", 0),
                    })

            if rows:
                results_df = pd.DataFrame(rows)
                st.dataframe(
                    results_df.style.format({
                        "Naive Brier": "{:.4f}", "Rule Brier": "{:.4f}", "Delta": "{:+.4f}",
                        "P&L ($100/bet)": "${:,.0f}", "Win Rate": "{:.1%}",
                    }),
                    use_container_width=True, hide_index=True,
                )

        st.error(
            "**Result: FAILED.** The rule makes Brier worse on every split and loses $464 on 61 test bets. "
            "YES under-pricing is real (+0.48 mean error on YES markets) but only 19-25% of structural-pop "
            "markets resolve YES. A blanket alpha-shift over-corrects the majority."
        )

        # Show the B5 figure if available
        b5_fig = FIGURES / "b5_rule_evaluation_2026-04-12.png"
        if b5_fig.exists():
            st.image(str(b5_fig), caption="B5 Rule Evaluation: Brier and P&L by split")

    # --- C1 ---
    with tab_c1:
        st.subheader("C1: Calibration Correction Models")
        st.markdown(
            "**Approach:** Train recalibration models (isotonic, Platt, per-category) to correct "
            "the documented favourite-longshot bias."
        )

        c1_df = load_csv("c1_recalibration_results_2026-04-12.csv")
        if c1_df is not None:
            test_df = c1_df[c1_df["split"] == "test"].copy()
            if len(test_df) > 0:
                test_df = test_df.sort_values("brier")
                naive_brier = test_df[test_df["model"] == "naive (market price)"]["brier"].values
                if len(naive_brier) > 0:
                    test_df["delta_vs_naive"] = test_df["brier"] - naive_brier[0]

                fig = px.bar(
                    test_df, x="model", y="brier",
                    color="brier", color_continuous_scale="RdYlGn_r",
                    title="Test Brier by Recalibration Model",
                    text=test_df["brier"].apply(lambda x: f"{x:.4f}"),
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                display_cols = [c for c in ["model", "brier", "reliability", "ece", "log_loss", "delta_vs_naive"] if c in test_df.columns]
                st.dataframe(
                    test_df[display_cols].style.format({
                        "brier": "{:.4f}", "reliability": "{:.4f}", "ece": "{:.4f}",
                        "log_loss": "{:.4f}", "delta_vs_naive": "{:+.4f}",
                    }),
                    use_container_width=True, hide_index=True,
                )

        st.error(
            "**Result: NEGLIGIBLE.** Best model (Platt global) improves Brier by 0.0001 (0.1%). "
            "The market's reliability of 0.0004 is already near-optimal. More granular models "
            "(per-category, category x horizon) overfit and make things *worse*."
        )

    # --- C4 ---
    with tab_c4:
        st.subheader("C4: Price Trajectory Dynamics")
        st.markdown(
            "**Approach:** Engineer features from *how* prices evolve over time (staleness, "
            "volatility regime, path curvature) and blend an XGBoost model with the market price."
        )

        c4 = load_json("c4_results_2026-04-12.json")
        if c4:
            test_results = c4.get("test_results", c4.get("test", {}))
            naive_brier = test_results.get("naive_brier", c4.get("naive_brier", 0))
            models = [{"Model": "naive (market price)", "Brier": naive_brier, "Reliability": 0.0004, "ECE": 0.0135}]
            for model_name in ["logistic_trajectory", "xgb_trajectory", "hybrid"]:
                if model_name in test_results and isinstance(test_results[model_name], dict):
                    m = test_results[model_name]
                    models.append({"Model": model_name, "Brier": m["brier"],
                                   "Reliability": m.get("reliability", 0),
                                   "ECE": m.get("ece", 0)})
            if models:
                m_df = pd.DataFrame(models).sort_values("Brier")
                st.dataframe(
                    m_df.style.format({"Brier": "{:.4f}", "Reliability": "{:.4f}", "ECE": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )

        # Trajectory feature correlations
        corr_df = load_csv("c4_trajectory_correlations_2026-04-12.csv")
        if corr_df is not None:
            top = corr_df.head(10).copy()
            fig = px.bar(
                top, x="feature", y="rho_abs_error",
                color="rho_abs_error", color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                title="Top Trajectory Features vs |Market Error|",
                text=top["rho_abs_error"].apply(lambda x: f"{x:+.3f}"),
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "**Result: SMALL IMPROVEMENT (0.7%).** The hybrid model (45% XGB + 55% market price) "
            "improves test Brier by 0.0006. But the standalone XGB is *worse* than naive — it only "
            "works when heavily anchored to the market price. Key insight: staleness predicts "
            "accuracy (stale markets are confident and correct), not mispricing."
        )

    # --- Split Stability ---
    with tab_stability:
        st.subheader("Split Stability: Which Findings Generalize?")
        stab_df = load_csv("calibration_split_stability.csv")
        if stab_df is None:
            st.warning("calibration_split_stability.csv not found")
            return

        # Pivot to show train vs test
        train_df = stab_df[stab_df["split"] == "train"][["category", "brier"]].rename(columns={"brier": "train_brier"})
        test_df = stab_df[stab_df["split"] == "test"][["category", "brier"]].rename(columns={"brier": "test_brier"})
        merged = train_df.merge(test_df, on="category", how="outer")
        merged["shift"] = merged["test_brier"] - merged["train_brier"]
        merged = merged.sort_values("shift", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=merged["category"], y=merged["train_brier"], name="Train Brier", marker_color="#636EFA"))
        fig.add_trace(go.Bar(x=merged["category"], y=merged["test_brier"], name="Test Brier", marker_color="#EF553B"))
        fig.update_layout(barmode="group", title="Brier Score: Train vs Test by Category", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            merged.style.format({
                "train_brier": "{:.4f}", "test_brier": "{:.4f}", "shift": "{:+.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown(
            "**Stable:** sports, politics, recession_economy, crypto (shift < 0.02). "
            "**Fragile:** entertainment (+0.178, only 9 test markets), geopolitics (+0.100), "
            "science_tech (+0.096). Findings on fragile categories should not be over-interpreted."
        )


# ---------------------------------------------------------------------------
# Page: Data Deep Dive
# ---------------------------------------------------------------------------


def page_data():
    st.title("Data Deep Dive")

    tab_overview, tab_volume, tab_splits = st.tabs(["Dataset Overview", "Volume Analysis", "Train/Val/Test Splits"])

    with tab_overview:
        markets = load_raw_markets()
        if markets is None:
            st.warning("gamma_markets.parquet not found")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Markets Fetched", f"{len(markets):,}")
        c2.metric("After Filtering", "4,538")
        c3.metric("Categories", "11")
        c4.metric("Total Snapshots", "640,450")

        st.subheader("Markets by Category")
        cat_counts = markets["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.bar(
            cat_counts, x="category", y="count",
            color="count", color_continuous_scale="Blues",
            title="Market Count by Category",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Markets by Start Year")
        markets["start_year"] = markets["start_date"].dt.year
        by_year = markets.groupby("start_year").agg(
            n_markets=("condition_id", "count"),
            total_volume=("volume_usd", "sum"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=by_year["start_year"], y=by_year["n_markets"],
            name="Markets", marker_color="#636EFA",
        ))
        fig.add_trace(go.Scatter(
            x=by_year["start_year"], y=by_year["total_volume"] / 1e9,
            name="Volume ($B)", yaxis="y2",
            line=dict(width=3, color="#EF553B"),
            mode="lines+markers",
        ))
        fig.update_layout(
            title="Markets and Volume by Year",
            yaxis=dict(title="Number of Markets"),
            yaxis2=dict(title="Total Volume ($B)", overlaying="y", side="right"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_volume:
        markets = load_raw_markets()
        if markets is None:
            st.warning("gamma_markets.parquet not found")
            return

        markets["start_year"] = markets["start_date"].dt.year

        st.subheader("Volume by Category and Year")
        vol_pivot = markets.pivot_table(
            values="volume_usd", index="category", columns="start_year",
            aggfunc="sum", fill_value=0,
        )
        vol_display = (vol_pivot / 1e6).round(0).astype(int)
        vol_display["Total"] = vol_display.sum(axis=1)
        vol_display = vol_display.sort_values("Total", ascending=False)
        st.dataframe(vol_display, use_container_width=True)

        st.subheader("Volume Distribution")
        fig = px.histogram(
            markets, x="volume_usd", nbins=50,
            title="Volume Distribution (log scale)",
            log_x=True, log_y=True,
        )
        fig.update_layout(height=400, xaxis_title="Volume (USD)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Category Volume Over Time")
        markets["start_quarter"] = markets["start_date"].dt.to_period("Q").astype(str)
        vol_q = markets.groupby(["start_quarter", "category"])["volume_usd"].sum().reset_index()
        fig = px.area(
            vol_q, x="start_quarter", y="volume_usd", color="category",
            title="Volume by Category and Quarter",
        )
        fig.update_layout(height=450, yaxis_title="Volume (USD)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_splits:
        splits = load_splits()
        if not splits:
            st.warning("No split data found")
            return

        st.subheader("Split Summary")
        rows = []
        for name, df in splits.items():
            n_markets = df["condition_id"].nunique() if "condition_id" in df.columns else len(df)
            base_rate = df["resolved_yes"].mean() if "resolved_yes" in df.columns else None
            rows.append({
                "Split": name,
                "Snapshots": len(df),
                "Markets": n_markets,
                "Base Rate": base_rate,
            })

        split_summary = pd.DataFrame(rows)
        st.dataframe(
            split_summary.style.format({"Base Rate": "{:.3f}", "Snapshots": "{:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Category Distribution by Split")
        all_cats = []
        for name, df in splits.items():
            if "category" in df.columns:
                cat_counts = df.groupby("category")["condition_id"].nunique().reset_index()
                cat_counts.columns = ["category", "markets"]
                cat_counts["split"] = name
                all_cats.append(cat_counts)

        if all_cats:
            cat_all = pd.concat(all_cats)
            fig = px.bar(
                cat_all, x="category", y="markets", color="split",
                barmode="group", title="Markets per Category by Split",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGE_MAP = {
    "Overview": page_overview,
    "Calibration Explorer": page_calibration,
    "Favourite-Longshot Bias": page_fl_bias,
    "Exploitation Attempts": page_exploitation,
    "Data Deep Dive": page_data,
}

PAGE_MAP[page]()

# Footer
st.markdown("---")
st.caption("MarketMind — Polymarket Calibration Research | Data: 4,538 markets, 639K snapshots (Jan 2022 - Apr 2026)")
