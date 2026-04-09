"""Model training orchestration.

Supports both:
- Classification experiments (target: resolved_yes)
- Regression experiments (target: market_error)
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_pipeline import FEATURE_SETS, get_feature_matrix, run_feature_pipeline
from src.models.baselines import BaseRateModel, RecencyBaselineModel
from src.models.logistic_model import build_logistic_model
from src.models.tree_models import (
    build_random_forest, build_xgboost,
    build_random_forest_regressor, build_xgboost_regressor,
)


def load_modeling_config(path: str = "configs/modeling.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(name: str, config: dict) -> Any:
    """Instantiate a model by name from config."""
    model_cfg = config.get("models", {}).get(name, {})

    if name == "base_rate":
        return BaseRateModel()
    elif name == "recency":
        return RecencyBaselineModel(price_col_index=0)
    elif name == "logistic":
        return build_logistic_model(
            C=model_cfg.get("C", 1.0),
            max_iter=model_cfg.get("max_iter", 1000),
        )
    elif name == "random_forest":
        return build_random_forest(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 8),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 10),
        )
    elif name == "xgboost":
        return build_xgboost(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 6),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            subsample=model_cfg.get("subsample", 0.8),
        )
    elif name == "linear_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
    elif name == "random_forest_regressor":
        return build_random_forest_regressor(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 8),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 10),
        )
    elif name == "xgboost_regressor":
        return build_xgboost_regressor(
            n_estimators=model_cfg.get("n_estimators", 200),
            max_depth=model_cfg.get("max_depth", 6),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            subsample=model_cfg.get("subsample", 0.8),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def train_experiment(
    experiment_config_path: str,
    modeling_config_path: str = "configs/modeling.yaml",
    data_dir: str = "data/processed",
    output_dir: str = "outputs/models",
) -> dict[str, Any]:
    """Train all models specified in an experiment config.

    Supports both classification (predict_proba) and regression (predict) tasks.
    Returns dict of {model_name: (fitted_model, predictions_dict)}.
    """
    with open(experiment_config_path) as f:
        exp_cfg = yaml.safe_load(f)
    model_cfg = load_modeling_config(modeling_config_path)

    exp = exp_cfg["experiment"]
    task = exp.get("task", "classification")
    target_col = exp.get("target", "resolved_yes")

    print(f"\n{'='*60}")
    print(f"Experiment: {exp['name']} - {exp['description']}")
    print(f"Task: {task}, Target: {target_col}")
    print(f"{'='*60}")

    # Load data
    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    val_df = pd.read_parquet(f"{data_dir}/val.parquet")
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")

    # Run feature pipeline
    train_df = run_feature_pipeline(train_df)
    val_df = run_feature_pipeline(val_df)
    test_df = run_feature_pipeline(test_df)

    # Determine features from feature_set name or explicit list
    feature_set_name = exp.get("feature_set", "")
    feature_names = FEATURE_SETS.get(feature_set_name, [])
    if not feature_names:
        feature_names = exp.get("features", [])
    if not feature_names:
        feature_names = FEATURE_SETS.get("market_only", [])

    np.random.seed(model_cfg.get("random_seed", 42))

    results = {}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for model_name in exp["models"]:
        print(f"\nTraining: {model_name}")
        model = build_model(model_name, model_cfg)

        # Baselines need market_only features (implied_prob first for recency)
        if model_name in ("base_rate", "recency"):
            model_features = FEATURE_SETS["market_only"]
            model_target = "resolved_yes"  # baselines always predict outcome
        else:
            model_features = feature_names
            model_target = target_col

        X_train, y_train = get_feature_matrix(train_df, model_features, target_col=model_target)
        X_val, y_val = get_feature_matrix(val_df, model_features, target_col=model_target)
        X_test, y_test = get_feature_matrix(test_df, model_features, target_col=model_target)

        if len(X_train) == 0:
            print(f"  Skipping {model_name}: no training data after feature extraction")
            continue

        model.fit(X_train, y_train)

        if task == "regression" and model_name not in ("base_rate", "recency"):
            # Regression: predict continuous market error
            preds = {
                "train": model.predict(X_train),
                "val": model.predict(X_val) if len(X_val) > 0 else np.array([]),
                "test": model.predict(X_test) if len(X_test) > 0 else np.array([]),
                "y_train": y_train.values,
                "y_val": y_val.values if len(y_val) > 0 else np.array([]),
                "y_test": y_test.values if len(y_test) > 0 else np.array([]),
                "task": "regression",
            }
            # Print regression metrics
            for split_name, X_s, y_s in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
                if len(X_s) > 0:
                    pred = preds[split_name]
                    mae = mean_absolute_error(y_s, pred)
                    rmse = mean_squared_error(y_s, pred) ** 0.5
                    print(f"  {split_name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        else:
            # Classification: predict probabilities
            preds = {
                "train": model.predict_proba(X_train)[:, 1],
                "val": model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([]),
                "test": model.predict_proba(X_test)[:, 1] if len(X_test) > 0 else np.array([]),
                "y_train": y_train.values,
                "y_val": y_val.values if len(y_val) > 0 else np.array([]),
                "y_test": y_test.values if len(y_test) > 0 else np.array([]),
                "task": "classification",
            }

        # Also store market IDs for market-level evaluation
        preds["market_ids_train"] = train_df.loc[X_train.index, "condition_id"].values
        preds["market_ids_val"] = val_df.loc[X_val.index, "condition_id"].values if len(X_val) > 0 else np.array([])
        preds["market_ids_test"] = test_df.loc[X_test.index, "condition_id"].values if len(X_test) > 0 else np.array([])

        # Save model
        model_file = out_path / f"{exp['name']}_{model_name}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved to {model_file}")

        results[f"{exp['name']}_{model_name}"] = (model, preds)

    return results
