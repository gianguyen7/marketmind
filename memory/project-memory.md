# Project Memory

> Active project state, architecture snapshots, and work-in-progress tracking.
> Updated after significant milestones. Source of truth for "where are we?"

---

## Last Updated: 2026-04-11

## Project: MarketMind

**One-liner:** Research on where and when Polymarket is systematically miscalibrated, and whether ML can exploit those specific mispricings.

**Framing (as of 2026-04-11):** Local / selection, not global. The market is well-calibrated on average; the question is "where is it wrong?" not "can ML beat the market?"

## Phase Status

| Phase | Status | Key Artifact |
|-------|--------|-------------|
| Scaffold & Setup | Complete | Full pipeline, configs, agents, skills, hooks, rules |
| Data Collection (95-market curated) | Complete | `data/raw/markets_registry.parquet` (95 markets, 6 themes), `data/raw/market_snapshots.parquet` (25,083 snapshots) |
| Feature Engineering v1 | Complete | `src/features/feature_pipeline.py` with `market_only`, `correction`, `ensemble` feature sets; 6 cross-market features added |
| Calibration Characterization | Complete | `scripts/run_calibration_analysis.py` → figures + tables for by-theme, by-horizon, F-L bias |
| Model Training v1 | Complete | `baseline`, `ensemble`, `market_correction` experiments run; **all underperform naive for correction target** |
| Track A — Shutdown Drill-Down | **Active, not started** | Next action: create `scripts/analyze_government_shutdown.py` for step A1 |
| Track B — Data Expansion (Gamma) | **Active, not started** | Next action: create `src/data/fetch_gamma.py` for step B1 |
| Theme-Stratified Split | Blocked on Track B | Deferred until expanded dataset lands |
| Per-Category Micro-Models | Blocked on Track B | Deferred until expanded dataset + stratified split |
| Dashboard & Reporting | Scaffolded, not refreshed | Streamlit app exists; needs update after Track A findings |

## Active Work (Dual Track)

### Track A — Government Shutdown Drill-Down
- **Data:** 10 markets, 717 snapshots, mean |market_error| ≈ 0.62 (the most mispriced theme)
- **Split coverage:** train 2 / val 5 / test 3 — the only theme with usable cross-split coverage
- **Form:** exploratory script `scripts/analyze_government_shutdown.py` (to create)
- **Sequence:** A1 characterize → A2 find signals → A3 write rule → A4 micro-model only if rule fails
- **Stopping rule:** if A1 shows no coherent pattern, stop and reconsider

### Track B — Data Expansion (Gamma API)
- **Plan source:** `docs/data_expansion_plan.md`
- **Target:** ~5,000 binary markets with volume > $1M across ~9 categories (Politics, Fed, Sports, Crypto, Geopolitics, Entertainment, Government, Science, Other)
- **First step:** `src/data/fetch_gamma.py` — paginate Gamma API, regex classify, resolve from `outcomePrices`, fetch price history via CLOB with checkpoint/resume
- **Delegatable:** natural fit for `data-agent` subagent
- **Estimated fetch time:** ~35 minutes wall clock
- **Follow-on:** FRED + FOMC calendar features, Fed Funds futures (yfinance), then optional GDELT/Trends/Metaculus

## Known Blockers

- **Train/test theme overlap is near-zero on current split.** `fed_rate_decisions` is 82% of train / 8% of test; `fed_leadership` is 5% of train / 76% of test. Structural — cannot be fixed with current data, must wait for expansion.
- **Global correction models overfit badly and fail on test.** Multiple rounds confirmed — this is not a feature-engineering problem on 95 markets.
- **Cross-market features have high NaN rates** (44% on `theme_historical_base_rate`). Dissolves at scale; do not patch on 95 markets.

## Architecture Snapshot

```
configs/*.yaml
  ↓
scripts/run_data_pipeline.py → data/raw/ → data/interim/ → data/processed/{train,val,test}.parquet
scripts/run_calibration_analysis.py → outputs/tables/, outputs/figures/
scripts/run_training.py → outputs/models/, outputs/predictions/
scripts/run_evaluation.py → outputs/tables/, outputs/figures/
src/dashboard/app.py → Streamlit UI (scaffolded, stale)
```

## Experiment Configs (live)

| Experiment | Config | Task | Status |
|---|---|---|---|
| baseline | `configs/experiments/baseline.yaml` | classification | baseline for recency/base_rate |
| ensemble | `configs/experiments/ensemble.yaml` | classification | doesn't beat market |
| market_correction | `configs/experiments/market_correction.yaml` | regression | underperforms naive (kept for comparison) |

**Deleted 2026-04-11:** `market_correction_long_horizon.yaml` and its three pickles — falsified by population shift between train/test long-horizon subsets.

## Models Defined

| Model | Type | File |
|-------|------|------|
| Base rate / Recency | Baselines | `src/models/baselines.py` |
| Logistic regression | ML classifier | `src/models/logistic_model.py` |
| Random Forest / XGBoost (cls) | Tree-based ML | `src/models/tree_models.py` |
| Random Forest / XGBoost (reg) | Regressors for market_error | `src/models/tree_models.py` |
| Linear regression (reg) | Regressor for market_error | wired in `src/models/train.py` |
| Hybrid ensemble | ML + market blending | `src/models/hybrid_models.py` |

## Key Files (for next session)

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project rules and memory protocol |
| `docs/research_plan.md` | **Current research framing (rewritten 2026-04-11)** |
| `docs/execution_plan.md` | **Concrete next-action plan (created 2026-04-11)** |
| `docs/data_expansion_plan.md` | Track B source of truth — unchanged |
| `docs/results_summary.md` | Results template — to be appended by Track A |
| `memory/recent-memory.md` | Rolling context + next-session starting instructions |
| `configs/markets.yaml` | Hand-curated 95-market registry (Track A only) |
| `configs/modeling.yaml` | Hyperparameters + random seed |
| `configs/data.yaml` | Data pipeline config — needs `gamma_fetch` section for Track B |
