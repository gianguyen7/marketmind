"""Tree-based models for probability forecasting."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int = 8,
    min_samples_leaf: int = 10,
) -> RandomForestClassifier:
    """Create a random forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
) -> XGBClassifier:
    """Create an XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
        eval_metric="logloss",
    )
