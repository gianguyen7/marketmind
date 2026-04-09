"""Feature pipeline: compose all feature engineering steps."""

import pandas as pd

from src.features.market_features import add_cross_market_features, add_market_features
from src.features.text_features import add_text_features


# Feature sets for different experiment types
FEATURE_SETS = {
    # Baseline: market price signals only (for recency model)
    "market_only": [
        "implied_prob", "implied_logit", "log_volume", "volume_rank",
        "price_extremity",
        "price_change", "price_momentum_3", "price_momentum_7",
        "price_volatility_7", "price_vs_open", "extremity_trend",
        "pct_lifetime_elapsed",
    ],

    # Market correction: cross-market structural signals (no price — that's what we're correcting)
    "correction": [
        # Cross-market features
        "theme_historical_base_rate",
        "price_bucket_historical_accuracy",
        "price_vs_historical_accuracy",
        "event_group_n_markets",
        "event_group_price_sum",
        "event_group_price_deviation",
        "price_vs_theme_mean",
        # Temporal context
        "days_to_end",
        "pct_lifetime_elapsed",
        # Market structure
        "theme_encoded",
        "price_extremity",
        "price_volatility_7",
    ],

    # Ensemble: market price + cross-market correction signals
    "ensemble": [
        # Market price (the base signal)
        "implied_prob", "implied_logit",
        # Cross-market correction signals
        "theme_historical_base_rate",
        "price_bucket_historical_accuracy",
        "price_vs_historical_accuracy",
        "event_group_n_markets",
        "event_group_price_sum",
        "event_group_price_deviation",
        "price_vs_theme_mean",
        # Temporal + momentum
        "days_to_end", "pct_lifetime_elapsed",
        "price_change", "price_momentum_3", "price_volatility_7",
        "extremity_trend",
        # Structure
        "theme_encoded",
        "price_extremity",
    ],
}


def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = add_market_features(df)
    df = add_text_features(df)
    df = add_cross_market_features(df)
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str = "resolved_yes",
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X, y from dataframe given feature names.

    Drops rows with NaN in features and fills remaining with 0.
    """
    available = [f for f in feature_names if f in df.columns]
    missing = set(feature_names) - set(available)
    if missing:
        print(f"  Warning: missing features {missing}, using available: {available}")

    X = df[available].copy()
    y = df[target_col].copy()

    # Drop rows where target is missing
    mask = y.notna()
    X, y = X[mask], y[mask]

    # Log and fill NaNs per feature
    nan_counts = X.isna().sum()
    if nan_counts.any():
        for col in nan_counts[nan_counts > 0].index:
            print(f"  Filling {nan_counts[col]} NaNs in '{col}' with 0 ({nan_counts[col]/len(X):.1%} of rows)")
    X = X.fillna(0)

    return X, y
