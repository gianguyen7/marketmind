"""Run full evaluation: metrics, calibration plots, comparison charts."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.evaluation.compare_models import compare_all_models, evaluate_predictions
from src.evaluation.reporting import (
    plot_calibration_curves,
    plot_model_comparison,
    plot_prediction_distributions,
    save_comparison_table,
)
from src.features.feature_pipeline import FEATURE_SETS, get_feature_matrix, run_feature_pipeline


def load_all_models(model_dir: str = "outputs/models") -> dict:
    """Load all saved model artifacts."""
    models = {}
    for pkl_file in Path(model_dir).glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            models[pkl_file.stem] = pickle.load(f)
    return models


def main():
    print("=" * 60)
    print("MarketMind Evaluation")
    print("=" * 60)

    # Load test data
    test_path = "data/processed/test.parquet"
    if not Path(test_path).exists():
        print("No test data found. Run data pipeline first.")
        return

    test_df = run_feature_pipeline(pd.read_parquet(test_path))

    # Load models and generate predictions
    saved_models = load_all_models()
    if not saved_models:
        print("No trained models found. Run training first.")
        return

    # Use hybrid features (superset)
    feature_names = FEATURE_SETS["hybrid"]
    X_test, y_test = get_feature_matrix(test_df, feature_names)

    results = {}
    for name, model in saved_models.items():
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            preds = {
                "test": y_prob,
                "y_test": y_test.values,
            }
            results[name] = (model, preds)
            print(f"  Loaded and predicted: {name}")
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    if not results:
        print("No models could generate predictions.")
        return

    # Compare
    comparison = compare_all_models(results, split="test")
    print("\n" + comparison.to_string(index=False))
    save_comparison_table(comparison)

    # Generate plots
    print("\nGenerating plots...")
    plot_calibration_curves(results, "test", output_path="outputs/figures/calibration.html")
    plot_model_comparison(comparison, "brier_score", output_path="outputs/figures/brier_comparison.html")
    plot_prediction_distributions(results, "test", output_path="outputs/figures/pred_distributions.html")

    print("\nEvaluation complete. See outputs/figures/ and outputs/tables/")


if __name__ == "__main__":
    main()
