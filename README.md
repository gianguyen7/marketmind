# MarketMind

Probabilistic forecasting research pipeline: **human market forecasts vs machine models vs hybrid models**.

## Research Questions

1. How accurate and calibrated are Polymarket implied probabilities?
2. How do traditional ML models compare on binary event forecasting?
3. Does a hybrid model (Polymarket + ML) outperform either alone?
4. Under what conditions does each approach perform best?

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys

# Run end-to-end pipeline
python scripts/run_data_pipeline.py
python scripts/run_training.py
python scripts/run_evaluation.py

# Launch dashboard
streamlit run src/dashboard/app.py

# Tests
pytest tests/
```

## Project Structure

```
configs/          # YAML configs for data, modeling, experiments
data/             # raw → interim → processed pipeline stages
src/
  data/           # API fetching, outcome resolution, dataset building
  features/       # Market, text, and pipeline feature engineering
  models/         # Baselines, logistic, tree, hybrid models + training
  evaluation/     # Backtesting, calibration, model comparison, reporting
  dashboard/      # Streamlit app for interactive comparison
scripts/          # CLI entry points for pipeline stages
tests/            # pytest suite
outputs/          # Figures, tables, saved models, reports
```

## Models

| Model | Description |
|-------|-------------|
| Base rate | Historical resolution rate by category |
| Recency baseline | Weighted average of recent market prices |
| Logistic regression | L2-regularized on engineered features |
| Random forest | Sklearn RF with default-ish hyperparams |
| XGBoost | Gradient boosted trees |
| Hybrid | ML models augmented with Polymarket features |

## Evaluation

- **Brier score** and **log loss** for probability accuracy
- **Calibration curves** for reliability
- **Sharpness** for decisiveness of forecasts
- **Subgroup analysis** by category and time-to-resolution
- **Temporal backtesting** to prevent leakage

## Agent Workflow (optional)

For agent-assisted development, recommended responsibilities:

- **data-agent**: Fetch markets, resolve outcomes, build datasets
- **feature-agent**: Engineer and validate features, run feature pipeline
- **model-agent**: Train models, tune hyperparameters, save artifacts
- **eval-agent**: Run backtests, compute metrics, generate comparison tables
- **report-agent**: Produce figures, update results summary, refresh dashboard
