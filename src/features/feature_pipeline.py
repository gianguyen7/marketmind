"""Feature pipeline: compose all feature engineering steps."""

import pandas as pd

from src.features.market_features import add_market_features
from src.features.text_features import add_text_features


# Feature sets for different experiment types
FEATURE_SETS = {
    "market_only": [
        "implied_prob", "implied_logit", "log_volume", "volume_rank",
        "price_extremity",
        # Snapshot time-series features
        "price_change", "price_momentum_3", "price_momentum_7",
        "price_volatility_7", "price_vs_open", "extremity_trend",
        "pct_lifetime_elapsed",
    ],
    "ml_only": [
        "days_to_end", "days_to_meeting", "pct_lifetime_elapsed",
        "snapshot_num", "theme_encoded", "volume_rank", "num_outcomes",
        "question_length", "question_word_count", "has_temporal_ref",
        "has_numeric_target", "is_will_question",
    ],
    "hybrid": [
        # Market features
        "implied_prob", "implied_logit", "log_volume", "volume_rank",
        "price_extremity",
        "price_change", "price_momentum_3", "price_momentum_7",
        "price_volatility_7", "price_vs_open", "extremity_trend",
        # Temporal features
        "days_to_end", "days_to_meeting", "pct_lifetime_elapsed",
        "snapshot_num",
        # Categorical + text features
        "theme_encoded", "num_outcomes",
        "question_length", "has_temporal_ref", "has_numeric_target",
    ],
}


def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = add_market_features(df)
    df = add_text_features(df)
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
        print(f"Warning: missing features {missing}, using available: {available}")

    X = df[available].fillna(0).copy()
    y = df[target_col].copy()

    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]
