---
name: analyze-calibration
description: Deep calibration analysis across all trained models
---

# /analyze-calibration

Perform detailed calibration analysis comparing all trained models.

## Steps

1. **Load all saved models** from `outputs/models/`
2. **Load test set** from `data/processed/test.parquet`
3. **Generate predictions** from each model on test set

4. **Calibration analysis**
   - Compute ECE (Expected Calibration Error) per model
   - Build reliability diagrams with `calibration_bins` from config
   - Compute calibration by probability bucket (are 70% predictions right 70% of the time?)
   - Compare: which model is most reliable at each confidence level?

5. **Subgroup calibration**
   - Break down calibration by `category` and `days_to_resolution_bucket`
   - Identify where Polymarket is well-calibrated vs where ML models are better

6. **Overconfidence/underconfidence analysis**
   - Flag probability ranges where models are systematically biased
   - Compare Polymarket vs ML vs hybrid in each range

7. **Output**
   - `outputs/figures/calibration_overlay.png` — all models on one plot
   - `outputs/figures/calibration_by_subgroup.png` — faceted by subgroup
   - `outputs/tables/calibration_summary.csv` — ECE and bias by model
   - Print top-3 findings

## Usage
```
/analyze-calibration
/analyze-calibration --subgroup category
```
