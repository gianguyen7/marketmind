---
name: run-forecast-experiment
description: Train and evaluate models for a named experiment config
---

# /run-forecast-experiment

Run a complete experiment: load config, train models, evaluate, save results.

## Arguments
- `experiment_name` — matches a file in `configs/experiments/{name}.yaml`

## Steps

1. **Load experiment config** from `configs/experiments/{experiment_name}.yaml`
   - Validate all referenced models exist in `configs/modeling.yaml`

2. **Load processed data** from `data/processed/`
   - Verify train/test splits exist; if not, prompt to run `/build-polymarket-dataset`

3. **Train each model** listed in the experiment config
   - Use hyperparams from `configs/modeling.yaml`
   - Use `random_seed` for reproducibility
   - If `use_market_price: true`, include Polymarket price as feature (hybrid mode)
   - Save models to `outputs/models/{experiment}_{model}_{date}.pkl`

4. **Evaluate on test set**
   - Compute: Brier score, log loss, calibration ECE, sharpness
   - Generate comparison table
   - Save to `outputs/tables/{experiment}_results.csv`

5. **Generate figures**
   - Calibration curves overlay
   - Brier score comparison bar chart
   - Save to `outputs/figures/{experiment}_*.png`

6. **Print summary** — top-line results for each model

## Usage
```
/run-forecast-experiment baseline
/run-forecast-experiment hybrid_v1
```

## Output
- Trained models in `outputs/models/`
- Results table in `outputs/tables/`
- Figures in `outputs/figures/`
