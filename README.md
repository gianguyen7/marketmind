# MarketMind

Research pipeline investigating **where and when prediction markets are miscalibrated**, and whether ML models can exploit those systematic errors.

## Research Question

> No published work trains ML models to detect and exploit systematic Polymarket miscalibration patterns. We aim to fill that gap.

**Core question**: Where and when are prediction markets miscalibrated, and can ML models exploit those systematic errors?

**Research flow**:
1. **Characterize** Polymarket calibration by domain, time horizon, and market structure
2. **Identify** systematic biases (favourite-longshot patterns, domain-specific miscalibration)
3. **Build ML models** that predict market errors — correcting the market, not replacing it
4. **Evaluate** via Brier decomposition: does ML correction improve calibration, resolution, or sharpness?

## Key Findings (In Progress)

| Finding | Detail |
|---|---|
| Government shutdown mispriced | Brier 0.48 — market prices ~26% but actual resolution rate 74% |
| Calibration degrades at longer horizons | Brier 0.013 (<1 day) → 0.094 (3-12 months), ~5x worse |
| Reverse favourite-longshot bias | Favourites win MORE than prices imply (+7.7% bias for price > 0.7) |
| Market well-calibrated overall | Reliability = 0.003 (very low), but systematic domain-specific errors |

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add API keys (FRED_API_KEY for economic data)

# Run calibration analysis (research output)
python scripts/run_calibration_analysis.py

# Run model training + evaluation
python scripts/run_training.py
python scripts/run_evaluation.py

# Launch dashboard
streamlit run src/dashboard/app.py

# Tests
pytest tests/
```

## Project Structure

```
configs/
  data.yaml              # API endpoints, pipeline settings, split strategy
  modeling.yaml           # Model hyperparameters, evaluation config
  calendar.yaml           # FOMC meeting dates, CPI release schedule
  experiments/            # Experiment configs (baseline, market_correction, ensemble)
data/
  raw/                    # API responses (market metadata + price snapshots)
  interim/                # Enriched snapshots with temporal features
  processed/              # Train/val/test splits with market error targets
  external/               # FRED, futures, GDELT cached data
src/
  data/                   # Polymarket API fetching, outcome resolution, dataset building
  features/               # Market, cross-market, economic, calendar, text features
  models/                 # Baselines, classifiers, regressors, hybrid models
  evaluation/             # Calibration analysis, Brier decomposition, model comparison
  dashboard/              # Streamlit app
scripts/                  # CLI entry points for each pipeline stage
tests/                    # pytest suite
outputs/                  # Figures, tables, saved models, predictions
docs/                     # Research plans and documentation
memory/                   # Persistent context for Claude Code sessions
```

## Data

**Current**: 95 resolved binary markets from Polymarket (Fed rate decisions, government shutdown, leadership, recession). 25K snapshots at 12-hour intervals.

**Expanding to**: ~5,000 resolved markets across politics, sports, crypto, geopolitics, entertainment, science — all with >$1M volume. See `docs/data_expansion_plan.md`.

**External data sources** (planned/in progress):
- FRED economic data (Treasury yields, VIX, CPI, unemployment)
- Fed Funds futures (institutional rate expectations via yfinance)
- FOMC/economic calendar
- GDELT news sentiment

## Experiments

| Experiment | Question | Target | Models |
|---|---|---|---|
| `baseline` | How good is the market alone? | `resolved_yes` | base_rate, recency (market price) |
| `market_correction` | Can ML predict where the market is wrong? | `market_error` (regression) | linear regression, RF regressor, XGBoost regressor |
| `ensemble` | Does blending ML correction with price improve calibration? | `resolved_yes` | logistic, RF, XGBoost (with cross-market features) |

## Evaluation

- **Brier decomposition**: reliability + resolution + uncertainty (not just aggregate Brier)
- **Market-level metrics**: one prediction per market (n=95) alongside snapshot-level (n=25K)
- **Calibration curves**: sliced by domain, time horizon, and price bucket
- **Favourite-longshot bias analysis**: systematic over/under-pricing detection
- **Temporal integrity**: event-group splits prevent correlated market leakage

## Related Work

| Study | Contribution | Gap we address |
|---|---|---|
| Le (2026) "Decomposing Crowd Wisdom" | Calibration varies by domain | Nobody built ML to exploit these patterns |
| Reichenbach & Walther (SSRN) | 124M trades analyzed, prices track probabilities | Descriptive only, no ML modeling |
| Page & Clemen (2013) | Favourite-longshot bias at long horizons | No ML correction attempted on Polymarket |
| ForecastBench | LLMs vs superforecasters | Different domain, not market correction |

## Agent Workflow

For agent-assisted development:

- **data-agent**: Fetch markets, resolve outcomes, build datasets
- **feature-agent**: Engineer and validate features
- **model-agent**: Train models, tune hyperparameters
- **eval-agent**: Run backtests, compute metrics
- **report-agent**: Generate figures, update dashboard
