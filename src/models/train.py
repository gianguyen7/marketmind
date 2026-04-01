"""Model training orchestration."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.features.feature_pipeline import FEATURE_SETS, get_feature_matrix, run_feature_pipeline
from src.models.baselines import BaseRateModel, RecencyBaselineModel
from src.models.logistic_model import build_logistic_model
from src.models.tree_models import build_random_forest, build_xgboost
from src.models.hybrid_models import HybridEnsemble


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
    else:
        raise ValueError(f"Unknown model: {name}")


def train_experiment(
    experiment_config_path: str,
    modeling_config_path: str = "configs/modeling.yaml",
    data_dir: str = "data/processed",
    output_dir: str = "outputs/models",
) -> dict[str, Any]:
    """Train all models specified in an experiment config.

    Returns dict of {model_name: (fitted_model, predictions_dict)}.
    """
    with open(experiment_config_path) as f:
        exp_cfg = yaml.safe_load(f)
    model_cfg = load_modeling_config(modeling_config_path)

    exp = exp_cfg["experiment"]
    print(f"\n{'='*60}")
    print(f"Experiment: {exp['name']} - {exp['description']}")
    print(f"{'='*60}")

    # Load data
    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    val_df = pd.read_parquet(f"{data_dir}/val.parquet")
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")

    # Run feature pipeline
    train_df = run_feature_pipeline(train_df)
    val_df = run_feature_pipeline(val_df)
    test_df = run_feature_pipeline(test_df)

    # Determine features
    feature_names = exp.get("features", [])
    if not feature_names:
        feature_names = FEATURE_SETS.get("ml_only", [])

    # If using market price, prepend it
    if exp.get("use_market_price") and "implied_prob" not in feature_names:
        feature_names = ["implied_prob"] + feature_names

    np.random.seed(model_cfg.get("random_seed", 42))

    results = {}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for model_name in exp["models"]:
        print(f"\nTraining: {model_name}")
        model = build_model(model_name, model_cfg)

        X_train, y_train = get_feature_matrix(train_df, feature_names)
        X_val, y_val = get_feature_matrix(val_df, feature_names)
        X_test, y_test = get_feature_matrix(test_df, feature_names)

        if len(X_train) == 0:
            print(f"  Skipping {model_name}: no training data after feature extraction")
            continue

        model.fit(X_train, y_train)

        preds = {
            "train": model.predict_proba(X_train)[:, 1],
            "val": model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([]),
            "test": model.predict_proba(X_test)[:, 1] if len(X_test) > 0 else np.array([]),
            "y_train": y_train.values,
            "y_val": y_val.values if len(y_val) > 0 else np.array([]),
            "y_test": y_test.values if len(y_test) > 0 else np.array([]),
        }

        # Save model
        model_file = out_path / f"{exp['name']}_{model_name}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved to {model_file}")

        results[model_name] = (model, preds)

    return results


def train_hybrid_experiment(
    modeling_config_path: str = "configs/modeling.yaml",
    data_dir: str = "data/processed",
    output_dir: str = "outputs/models",
) -> dict[str, Any]:
    """Train hybrid models that blend market price with ML."""
    model_cfg = load_modeling_config(modeling_config_path)
    feature_names = FEATURE_SETS["hybrid"]

    train_df = run_feature_pipeline(pd.read_parquet(f"{data_dir}/train.parquet"))
    val_df = run_feature_pipeline(pd.read_parquet(f"{data_dir}/val.parquet"))
    test_df = run_feature_pipeline(pd.read_parquet(f"{data_dir}/test.parquet"))

    X_train, y_train = get_feature_matrix(train_df, feature_names)
    X_val, y_val = get_feature_matrix(val_df, feature_names)
    X_test, y_test = get_feature_matrix(test_df, feature_names)

    results = {}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find the index of implied_prob in feature_names
    price_idx = feature_names.index("implied_prob") if "implied_prob" in feature_names else 0

    for base_name in ["logistic", "random_forest", "xgboost"]:
        hybrid_name = f"hybrid_{base_name}"
        print(f"\nTraining: {hybrid_name}")

        base_model = build_model(base_name, model_cfg)
        hybrid = HybridEnsemble(base_model, market_price_index=price_idx)
        hybrid.fit(X_train, y_train)

        preds = {
            "train": hybrid.predict_proba(X_train)[:, 1],
            "val": hybrid.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([]),
            "test": hybrid.predict_proba(X_test)[:, 1] if len(X_test) > 0 else np.array([]),
            "y_train": y_train.values,
            "y_val": y_val.values if len(y_val) > 0 else np.array([]),
            "y_test": y_test.values if len(y_test) > 0 else np.array([]),
        }

        model_file = out_path / f"hybrid_{base_name}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(hybrid, f)

        results[hybrid_name] = (hybrid, preds)

    return results
