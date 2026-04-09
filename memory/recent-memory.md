# Recent Memory

> Rolling 48-hour context window. Updated after each conversation.
> Important items get promoted to long-term memory nightly via `/consolidate-memory`.
> Loaded inline at conversation startup via CLAUDE.md.

---

## Last Updated: 2026-04-09

## Current Focus

- Research reframed: "Where and when are prediction markets miscalibrated, and can ML exploit systematic errors?"
- Full pipeline runs end-to-end: calibration analysis → training (baseline, ensemble, market_correction) → evaluation
- Round 2 results show market correction regressors underperform naive baseline — features need work

## Recent Decisions (Last 48hr)

- Replaced 4 unfocused experiments with 3 targeted ones: baseline, market_correction (regression), ensemble
- Added Brier decomposition (reliability + resolution + uncertainty) to all evaluation
- Added market-level evaluation (n=95 markets) alongside snapshot-level (n=25K)
- Added market_error target column (resolved_yes - price_yes) for regression experiments
- Added 6 cross-market features: theme_historical_base_rate, price_bucket_historical_accuracy, event_group_n_markets, event_group_price_sum, event_group_price_deviation, price_vs_theme_mean
- Created calibration analysis script producing calibration by theme, by horizon, and F-L bias analysis

## Key Findings

### Calibration Analysis
- Overall Brier: 0.0756, very low reliability (0.0029) = well-calibrated overall
- By theme: fed_leadership best (Brier 0.038), government_shutdown worst (0.482 — market systematically underprices shutdown risk)
- By horizon: <1d Brier=0.013 → 3m-1y Brier=0.094 (5x degradation)
- F-L bias: REVERSE pattern — favourites win MORE than prices imply (bias +0.077 for price>0.7)

### Model Results
- Market price (recency) still dominates: Brier 0.052 (test), 0.0017 (market-level)
- Ensemble models don't beat market: best ensemble Brier 0.087 market-level vs 0.0017 recency
- Market correction regressors WORSE than naive (predicting 0): RF MAE=0.216 vs naive MAE=0.103
- Correction models overfit heavily: RF train MAE=0.016 but test MAE=0.216

## Open Threads

- Cross-market features are too weak / possibly leaking information in unexpected ways
- NaN fill rate for theme_historical_base_rate is 44% on val set — needs better imputation
- Government shutdown mispricing (Brier 0.48) is the clearest exploitation opportunity
- Need to try: predicting market error ONLY for long-horizon snapshots (3m+) where market is weakest

## Blockers / Watch Items

- Only 95 markets (26 event groups) limits what models can learn
- Test set is 83% fed_leadership — very different from training distribution
- Feature NaN fill with 0 is suboptimal for base rate features (0 ≠ "no data")
