"""Logistic regression model for probability forecasting."""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_logistic_model(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """Create a logistic regression pipeline with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        )),
    ])
