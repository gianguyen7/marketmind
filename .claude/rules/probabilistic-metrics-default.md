---
name: probabilistic-metrics-default
description: Always use probabilistic metrics (Brier, log loss, calibration) as primary evaluation
---

# Probabilistic Metrics Default

This is a probability forecasting project, not a classification project.

## Primary Metrics (always report)
1. **Brier score** — `mean((predicted_prob - actual_outcome)^2)`
2. **Log loss** — `mean(-(y*log(p) + (1-y)*log(1-p)))`
3. **ECE** — Expected Calibration Error from reliability diagram

## Secondary Metrics (report when relevant)
- **Sharpness** — std of predicted probabilities (higher = more decisive)
- **Resolution** — Brier decomposition: does the model distinguish events?
- **AUC-ROC** — acceptable as secondary, never primary

## Never Use as Primary
- Accuracy (useless for probability evaluation)
- F1 score (requires arbitrary threshold)
- Precision/recall (same issue)

## Why This Matters
A model that predicts 0.51 for everything has ~50% accuracy but terrible Brier score. We care about the quality of probability estimates, not binary classifications.

## Implementation
- All models must implement `predict_proba()` or equivalent
- Never call `predict()` for evaluation — always `predict_proba()`
- Calibration curves: use `sklearn.calibration.calibration_curve` with `n_bins` from config
