# Research Plan

## Objective

Evaluate Polymarket as a forecasting signal and compare against statistical/ML models on binary event prediction.

## Approach

### Phase 1: Data Collection & Exploration
- Fetch resolved binary markets from Polymarket CLOB API
- Build resolved outcome labels
- Explore base rates, volume distributions, category breakdowns
- Assess data quality and coverage

### Phase 2: Feature Engineering
- Market features: implied probability, volume, liquidity, spread, momentum
- Text features: question length, temporal references, numeric targets
- Temporal features: days open, time-to-resolution

### Phase 3: Modeling
- **Baselines**: base rate, recency (market price at snapshot)
- **ML models**: logistic regression, random forest, XGBoost
- **Hybrid models**: ML + Polymarket features via learned blending

### Phase 4: Evaluation
- Brier score and log loss for accuracy
- Calibration curves and ECE for reliability
- Sharpness for decisiveness
- Subgroup analysis by category and time horizon
- Temporal backtesting to validate robustness

### Phase 5: Analysis & Reporting
- When does the market outperform ML?
- When does ML add value over market price alone?
- What hybrid configurations work best?
- Conditions where each approach fails

## Key Decisions
- Focus on resolved binary markets (clear ground truth)
- Temporal train/val/test split to prevent leakage
- Evaluate probability quality, not just accuracy
