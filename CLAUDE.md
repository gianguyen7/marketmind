# MarketMind — Claude Code Project Instructions

## What This Is
Probabilistic forecasting research comparing Polymarket crowd probabilities against ML and hybrid models. Focus: calibration, backtesting, temporal leakage prevention, and forecast comparison.

## Key Conventions
- Python 3.10+, use type hints on public functions
- All configs in `configs/*.yaml` — never hardcode hyperparams or paths
- Data flows: `data/raw/` → `data/interim/` → `data/processed/`
- Models saved to `outputs/models/`, figures to `outputs/figures/`
- Random seed: always use `configs/modeling.yaml:random_seed`
- Tests: `pytest tests/` — match `test_*.py`

## Critical Research Rules
1. **Temporal leakage is the #1 failure mode.** Never let future data leak into training. All splits must be time-based. Never shuffle time series data randomly.
2. **Default to probabilistic metrics** (Brier score, log loss, calibration curves). Accuracy/F1 are secondary for probability forecasting.
3. **Log every assumption.** If you choose a default, a threshold, or a heuristic, print or log it. No silent decisions.
4. **Reproducibility.** Set seeds, log configs, version datasets. Every experiment must be re-runnable.

## Agents
Agents live in `.claude/agents/`. Use them for scoped, parallelizable work:
- `data-agent` — fetch, clean, resolve Polymarket data
- `feature-agent` — engineer and validate features
- `model-agent` — train, tune, save models
- `eval-agent` — backtest, compute metrics, compare models
- `report-agent` — generate figures, tables, dashboard updates

## Skills (Slash Commands)
Skills live in `.claude/skills/`. Invoke with `/skill-name`:
- `/build-polymarket-dataset` — end-to-end dataset construction
- `/run-forecast-experiment` — train + evaluate a named experiment
- `/analyze-calibration` — calibration analysis across models
- `/summarize-results` — generate results summary table
- `/build-streamlit-dashboard` — scaffold or update the Streamlit app

## Hooks
Configured in `.claude/settings.local.json`. Lightweight checks:
- **pre-commit**: lint with ruff, block temporal leakage patterns
- **post-file-write**: warn if writing to `data/` directly (use pipeline)

## File Layout Reference
```
configs/           YAML configs (data, modeling, experiments)
data/raw/          Raw API responses
data/interim/      Cleaned, joined intermediate data
data/processed/    Final train/test datasets
src/data/          Fetching, resolution, dataset building
src/features/      Feature engineering
src/models/        Model definitions + training
src/evaluation/    Backtesting, calibration, metrics
src/dashboard/     Streamlit app
scripts/           CLI entry points
tests/             pytest suite
outputs/           Figures, tables, models, reports
```
