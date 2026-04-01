"""Run model training across all experiment configs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train import train_experiment, train_hybrid_experiment
from src.evaluation.compare_models import compare_all_models
from src.evaluation.reporting import save_comparison_table


def main():
    print("=" * 60)
    print("MarketMind Model Training")
    print("=" * 60)

    all_results = {}
    exp_dir = Path("configs/experiments")

    for exp_file in sorted(exp_dir.glob("*.yaml")):
        if exp_file.stem == "hybrid":
            continue  # handled separately
        try:
            results = train_experiment(str(exp_file))
            all_results.update(results)
        except Exception as e:
            print(f"Error in {exp_file.stem}: {e}")

    # Hybrid experiments
    try:
        print("\n--- Hybrid Models ---")
        hybrid_results = train_hybrid_experiment()
        all_results.update(hybrid_results)
    except Exception as e:
        print(f"Error in hybrid training: {e}")

    # Compare all models
    if all_results:
        print("\n" + "=" * 60)
        print("Model Comparison (test set)")
        print("=" * 60)
        comparison = compare_all_models(all_results, split="test")
        print(comparison.to_string(index=False))
        save_comparison_table(comparison)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
