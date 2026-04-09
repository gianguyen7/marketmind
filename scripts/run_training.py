"""Run model training across all experiment configs.

Handles both classification (resolved_yes) and regression (market_error) experiments.
"""

import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train import train_experiment
from src.evaluation.compare_models import compare_all_models, evaluate_market_level
from src.evaluation.reporting import save_comparison_table


def save_predictions(all_results: dict, output_dir: str = "outputs/predictions") -> None:
    """Save predictions from all models so evaluation doesn't need to re-predict."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for model_name, (model, preds) in all_results.items():
        pred_data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in preds.items()}
        (out / f"{model_name}.json").write_text(json.dumps(pred_data))
    print(f"\nSaved predictions for {len(all_results)} models to {out}")


def main():
    print("=" * 60)
    print("MarketMind Model Training")
    print("=" * 60)

    all_results = {}
    exp_dir = Path("configs/experiments")

    for exp_file in sorted(exp_dir.glob("*.yaml")):
        try:
            results = train_experiment(str(exp_file))
            all_results.update(results)
        except Exception as e:
            print(f"Error in {exp_file.stem}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("No models trained successfully.")
        return

    # Save predictions for evaluation
    save_predictions(all_results)

    # Separate classification and regression results
    clf_results = {k: v for k, v in all_results.items() if v[1].get("task") == "classification"}
    reg_results = {k: v for k, v in all_results.items() if v[1].get("task") == "regression"}

    # Classification model comparison (Brier, log loss, calibration)
    if clf_results:
        for split in ["val", "test"]:
            print(f"\n{'='*60}")
            print(f"Classification Models — {split} set (snapshot-level)")
            print(f"{'='*60}")
            comparison = compare_all_models(clf_results, split=split)
            print(comparison.to_string(index=False, float_format="%.4f"))
            if split == "test":
                save_comparison_table(comparison)

        # Market-level evaluation
        print(f"\n{'='*60}")
        print("Classification Models — test set (MARKET-level, n=markets)")
        print(f"{'='*60}")
        mkt_rows = []
        for name, (model, preds) in clf_results.items():
            market_ids = preds.get("market_ids_test", np.array([]))
            y_true = preds.get("y_test", np.array([]))
            y_prob = preds.get("test", np.array([]))
            if len(market_ids) > 0:
                row = evaluate_market_level(y_true, y_prob, market_ids, name, "test")
                mkt_rows.append(row)
        if mkt_rows:
            import pandas as pd
            mkt_df = pd.DataFrame(mkt_rows).sort_values("brier_score")
            print(mkt_df.to_string(index=False, float_format="%.4f"))

    # Regression model comparison (MAE, RMSE)
    if reg_results:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        print(f"\n{'='*60}")
        print("Regression Models — Market Error Prediction")
        print(f"{'='*60}")
        rows = []
        for name, (model, preds) in reg_results.items():
            for split in ["val", "test"]:
                y_true = preds.get(f"y_{split}", np.array([]))
                y_pred = preds.get(split, np.array([]))
                if len(y_true) == 0:
                    continue
                naive_mae = np.abs(y_true).mean()  # naive baseline: predict 0 (market is correct)
                rows.append({
                    "model": name, "split": split,
                    "n": len(y_true),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
                    "naive_mae": naive_mae,
                    "improvement_vs_naive": 1 - mean_absolute_error(y_true, y_pred) / naive_mae if naive_mae > 0 else 0,
                })
        if rows:
            import pandas as pd
            reg_df = pd.DataFrame(rows).sort_values(["split", "mae"])
            print(reg_df.to_string(index=False, float_format="%.4f"))

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
