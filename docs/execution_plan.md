# Execution Plan — Option 3 (dual track)

> Created: 2026-04-11. This is the concrete "what to do next" document.
> For the research framing behind it, see `docs/research_plan.md`.
> For the data expansion details referenced by Track B, see `docs/data_expansion_plan.md`.

## TL;DR for the next session

1. **Start with Track A, step A1** — open the shutdown drill-down on the 10 markets we already have. Fire up `scripts/analyze_government_shutdown.py` (needs to be created) and produce the characterization figures and stats.
2. **In parallel, kick off Track B, step B1a** — implement `src/data/fetch_gamma.py` and start the bulk Gamma API fetch. This can run in the background or via the `data-agent` subagent while Track A proceeds.
3. **Do not touch the current train/val/test split yet.** Both tracks leave it alone. The split gets rebuilt only after Track B produces the expanded dataset (step B3).

## Why Option 3

- Shutdown is the one theme with cross-split coverage on current data. It can yield real findings right now, independent of dataset size.
- Data expansion is a big enough job that it deserves its own workstream but it doesn't block anything — it's additive.
- Per-theme micro-models and the theme-stratified split only become meaningful after expansion. Trying to build them on 95 markets across 6 themes is premature.

## Track A — Government Shutdown Drill-Down

**Data:** 10 markets, 717 snapshots, mean `|market_error|` ≈ 0.62. Splits today: train 2 markets / val 5 / test 3.

**Form:** `scripts/analyze_government_shutdown.py` — one script, exploratory, prints findings + saves figures. No new configs, no new feature sets until the drill-down justifies them. Findings get written to `docs/results_summary.md`.

### A1. Characterize the mispricing — **start here**

Questions to answer, each with a figure and a number:

1. **Price trajectories.** For each of the 10 markets, plot `price_yes` over time with the resolution outcome annotated. Save as `outputs/figures/shutdown_trajectories_2026-04-11.png`. Group by resolved_yes so the two populations are visually separable.
2. **Error decomposition.** Compute per-market: final price, resolution, `|market_error|` mean, `|market_error|` at start vs end. Does the error shrink as resolution approaches (good market, slow convergence) or stay constant (persistent mispricing)?
3. **Directional bias.** Is `market_error = resolved_yes - price_yes` systematically positive or negative? Report mean and 95% bootstrap CI across snapshots and across markets.
4. **Outlier check.** Are the 0.62 mean errors driven by a few catastrophic markets, or is it uniform across the 10?

**Acceptance:** a 5-line finding in `docs/results_summary.md` that says what kind of mispricing this is (under-pricing, over-pricing, late convergence, catastrophic misses).

### A2. Search for mispricing signals

**Only if A1 shows a consistent pattern.** Otherwise stop and reconsider.

Candidate features from existing feature pipeline:
- `days_to_end`, `pct_lifetime_elapsed`
- `price_extremity`, `price_volatility_7`, `price_momentum_3/7`
- `event_group_price_deviation`, `event_group_n_markets`
- `log_volume`, `volume_rank`
- Optional new: `days_to_next_fiscal_deadline` (manual lookup — federal funding deadlines are public)

For each candidate:
- Scatter plot vs `market_error` (with LOESS / rolling mean)
- Spearman correlation with `market_error` on train+val (hold test out)
- Note effect size, not just p-value

**Acceptance:** 1–2 features with clear monotone relationships to `market_error`, documented in the findings doc.

### A3. Write a bet rule

Translate A2's findings into a human-readable threshold rule. Example shape: `IF theme == shutdown AND days_to_end > 60 AND price_yes < 0.4 THEN bet YES (predicted_prob = max(price_yes, 0.6))`.

Backtest:
- On train (2 markets) — sanity check
- On val (5 markets) — did it work
- On test (3 markets) — does it generalize

Metrics per split: MAE vs naive (predict price_yes), Brier vs naive, and **dollar P&L if we had bet $100 per trigger** (interpretability — real units matter more than tiny Brier differences on n=10).

**Acceptance:** rule must beat naive on both val and test. If it only beats naive on val, it's noise.

### A4. Micro-model (only if A3 fails)

If a threshold rule can't capture the effect, fit a single-theme model:
- `configs/experiments/shutdown_micro.yaml` (new)
- Target: regression on `market_error`
- Features: whatever A2 identified
- CV: leave-one-market-out on train+val combined (8 markets)
- Final eval: test (3 markets)

Compare head-to-head with A3's rule. Prefer the rule on ties.

## Track B — Data Expansion (parallel)

**Source of truth:** `docs/data_expansion_plan.md`. This section is the execution order.

### B1. Gamma API bulk fetch

**Status (updated 2026-04-11 session 2 — supersedes earlier table):** most of B1 was already implemented in a prior session and was wrongly listed as "CREATE" in the original plan. A subagent audit caught the discrepancy; the updated status below reflects the real codebase. See `memory/feedback_verify_codebase_before_planning.md` for the lesson.

| # | Action | File | Status |
|---|---|---|---|
| B1a | Verify | `src/data/fetch_gamma.py` — 463 lines; paginator, binary filter, regex classifier, resolution extractor, price-history fetcher with checkpoint/resume, `run()` entry point | **Already existed** before this session. No changes made. |
| B1b | Modify | `configs/data.yaml` — add `gamma_fetch` section (`min_volume_usd`, `binary_only`, `closed_only`, `max_markets`, `checkpoint_path`, `dry_run_dir`) | **Done 2026-04-11 s2.** |
| B1c | ~~Create~~ | ~~`configs/categories.yaml`~~ | **Deferred.** Classification already works via hardcoded `CATEGORY_PATTERNS` in `fetch_gamma.py:40`. Moving to YAML is cosmetic; revisit only if categories need to be tuned without a code change. |
| B1d | Modify | `src/data/resolve_outcomes.py` — handle markets without `meeting_date` | **Mostly pre-existing.** Module is already gated on `if "meeting_date" in df.columns`. Remaining work (per-source logging) deferred until the first fetch reveals actual non-fed markets. |
| B1e | Modify | `src/data/build_dataset.py` — Gamma event-id grouping; `category_encoded` alongside `theme_encoded` | **Done 2026-04-11 s2.** Added `_assign_event_groups_gamma()` and dispatch in `assign_event_groups()`. Both paths coexist; Track A data is untouched. |
| B1f | Modify | `scripts/run_data_pipeline.py` — `--gamma-fetch`, `--dry-run`, `--max-markets` flags; Gamma step runs before the curated pipeline | **Done 2026-04-11 s2.** No-flag invocation is unchanged. |
| B1g | Run | `python3 scripts/run_data_pipeline.py --gamma-fetch --dry-run` (50 markets) for smoke verification, then `--gamma-fetch` for the real ~5,000-market fetch | **Dry-run: pending.** Full fetch: pending. |
| B1h | Verify | `data/raw/gamma_markets.parquet` ≥ 5,000 markets, ≥ 5 categories with ≥ 50 markets each, resolution rate > 95%, **and** an empirically-defined long-horizon low-liquidity population of ≥ 50 markets. Population definition (revised 2026-04-11 s2 after the $100k absolute threshold was invalidated by the dry-run, which showed min_volume_usd in the top-100 fetch was $53M): take the **bottom 25% of fetched markets by `volume_usd`** AND markets whose **max `days_to_end` across snapshots is > 90 days**. Report the count, median volume, median max-`days_to_end`, and category breakdown of that cohort. If fewer than 50 markets qualify, flag loudly — Track B5 walk-forward cannot adjudicate the structural hypothesis without this population, and `gamma_fetch.min_volume_usd` may need to be lowered below $1M to capture more of the long tail. | **Pending** — runs after B1g completes. |

**Skipped deliverables (versus original plan):**

- `configs/categories.yaml` (B1c) — redundant given the working hardcoded patterns.
- `tests/test_fetch_gamma.py` — the dry-run smoke fetch is a better functional test; pure unit test deferred.

**Delegation note:** this was launched to the `data-agent` subagent, which was permission-blocked on Write for new files and produced the audit that caught the plan staleness. Remaining work is being executed in the main session instead. For future large data tasks, subagent permissions need to be fixed before delegation will work; B1's remaining scope was too small to be worth fixing that first.

**Estimated fetch time (from expansion plan):** ~35 minutes wall-clock for metadata + prices on the full ~5,000-market fetch. Dry-run is ~30 seconds for 50 markets. Run with checkpoint/resume so it can be interrupted.

### B2. External features — after B1 lands

Order:
1. FRED series + cache (`src/data/fetch_fred.py`, `src/features/economic_features.py`)
2. FOMC/economic calendar (`configs/calendar.yaml`, `src/features/calendar_features.py`)
3. Fed Funds futures via yfinance (`src/data/fetch_market_data.py`)
4. Optional: GDELT, Trends, Metaculus, on-chain

Register new feature sets in `src/features/feature_pipeline.py`. Keep the `correction` feature set backward-compatible for anything Track A produced.

### B3. Theme-stratified temporal split — after B1 lands

Once there's enough data per category:

1. Group markets by `category` (from regex classifier).
2. For each category with ≥10 resolved markets, sort by `end_date` and take 60/20/20.
3. Drop or pool categories with <10 markets into `other`.
4. Concatenate per-category splits into global `train.parquet` / `val.parquet` / `test.parquet`.
5. Write `data/processed/split_manifest.json`: `{category: {train: [condition_ids], val: [...], test: [...]}}`.
6. **Leakage audit:** verify `theme_historical_base_rate`, `price_bucket_historical_accuracy`, and any other cross-market aggregates use `expanding().shift(1)` *over a global time axis*, not a per-split one. With per-category cutoffs, a "training data only" aggregate silently becomes a future-leaking aggregate. This is the highest-risk step.

### B4. Re-establish baselines on expanded split

Expect all current numbers to move. The following are invalid once B3 lands and must be re-computed before any new comparative claims are made:

- Recency market-level Brier 0.0017
- Ensemble best Brier 0.087 market-level
- market_correction RF test MAE 0.216
- Today's falsified long-horizon numbers (already deleted)
- Calibration analysis by theme / by horizon / F-L bias analysis

Re-run:
- `configs/experiments/baseline.yaml`
- `scripts/run_calibration_analysis.py`
- Any shutdown rule / model from Track A (verify it still holds on the bigger shutdown universe, which should now be ~300 markets per the expansion plan's estimate)

## Track C — Calibration Correction & Deliverables (2026-04-12)

**Context:** B5 walk-forward showed the structural hypothesis (long-horizon low-liquidity YES under-pricing) describes a real phenomenon but does not produce a profitable rule — the identification problem (which markets resolve YES?) remains unsolved. The calibration analysis on the expanded dataset surfaced a cleaner, more actionable finding: classic favourite-longshot bias across 4,538 markets, with per-category variation. The project pivots from "exploit structural mispricing" to "characterize and correct systematic calibration errors."

### C1. Calibration correction models — **start here**

Build recalibration models that correct the documented F-L bias and per-category miscalibration. Unlike the structural hypothesis, this is a **mapping correction** (price → better probability) not an **outcome prediction** (will this market resolve YES?).

| # | Action | Details | Status |
|---|--------|---------|--------|
| C1a | Isotonic regression | Train `sklearn.isotonic.IsotonicRegression` on train prices vs resolutions. Evaluate Brier on val and test. This is the simplest recalibration baseline. | Pending |
| C1b | Platt scaling | Train logistic regression on `logit(price_yes)` → `resolved_yes`. Compare to isotonic. | Pending |
| C1c | Per-category recalibration | Train separate isotonic/Platt models per category (government_policy has 100x worse reliability than sports — one model won't fit both). Pool categories with <50 train markets. | Pending |
| C1d | Category × horizon recalibration | The heatmap shows calibration failures are localized (geopolitics >1y, government_policy 3m-1y). Train recalibration on category × horizon buckets where train reliability > 0.005. | Pending |
| C1e | Evaluate all models | Brier, log loss, ECE, reliability on val and test. Market-level Brier (one score per market). Per-category breakdown. Compare to naive (raw market price). | Pending |
| C1f | Document findings | Append to `docs/results_summary.md`. Key question: does recalibration improve test Brier, or is the market's 0.0004 reliability already optimal? | Pending |

**Acceptance criteria:** At least one recalibration model must beat raw market price on test-split Brier and ECE. If none do, the F-L bias is real but too small to exploit at the snapshot level — document as a negative result.

**Leakage guard:** Recalibration models are trained on train split only. Val is for model selection. Test is touched once for final evaluation.

### C2. Streamlit dashboard

Visualize the calibration findings interactively. Skeleton exists at `src/dashboard/app.py`.

| # | Action | Details | Status |
|---|--------|---------|--------|
| C2a | Calibration overview page | Overall Brier decomposition, reliability diagram, F-L bias chart | Pending |
| C2b | Category explorer | Select category → see calibration curve, Brier, ECE, F-L bias, n_markets | Pending |
| C2c | Category × horizon heatmap | Interactive version of the static heatmap | Pending |
| C2d | Recalibration comparison | If C1 produces a model that beats naive, show before/after calibration curves | Pending |
| C2e | Split stability view | Train vs val vs test Brier per category — highlights fragile categories | Pending |

### C3. Research write-up

Package findings as a standalone research report.

| # | Action | Details | Status |
|---|--------|---------|--------|
| C3a | Draft report | Structure: Introduction, Data, Methods, Results (calibration characterization + F-L bias + recalibration + structural hypothesis negative result), Discussion | Pending |
| C3b | Key figures | Export 4-6 publication-quality figures (reliability diagram, F-L bias, heatmap, recalibration before/after, B5 signal scatter) | Pending |
| C3c | Review and polish | Ensure all numbers trace to code, all caveats attached, reproducibility checklist passes | Pending |

### C1 Status: COMPLETE (2026-04-12)

**Result: Negative.** Best model (Platt global) improves test Brier by 0.0001 (0.1%). Market reliability of 0.0004 is already near-optimal. More granular models overfit and make things worse. See `docs/results_summary.md` §C1 for full write-up.

### C4. Price trajectory dynamics — **start here**

**Context:** Three approaches to beating the market have failed (structural rules, calibration correction, global ML). All used point-in-time price + market metadata. This track exploits the *shape* of the price trajectory — information contained in how prices evolve over time, not just where they are at a snapshot.

**Hypothesis:** Markets whose price trajectories exhibit specific dynamic patterns (staleness, abnormal volatility, momentum/reversion) are more likely to be mispriced at any given snapshot than markets with "normal" trajectories. Unlike the structural hypothesis (which needed outcome prediction), this is about identifying *when the crowd is processing information poorly*, which should manifest in trajectory anomalies.

| # | Action | Details | Status |
|---|--------|---------|--------|
| C4a | Engineer trajectory features | staleness, acceleration, vol_regime, path_curvature, price_range_14, dist_from_high/low, price_extremity | **Done 2026-04-12** |
| C4b | Exploratory analysis | Top signals: price_extremity (ρ=-0.974), staleness (ρ=-0.743), price_range_14 (ρ=+0.693). All robust across 11 categories. Key finding: staleness predicts accuracy, not mispricing. | **Done 2026-04-12** |
| C4c | Trajectory-informed correction model | XGB standalone worse than naive. Hybrid (45% XGB + 55% market) val Brier 0.0890 vs naive 0.0909 (Δ=-0.0018). Passes 0.001 threshold. | **Done 2026-04-12** |
| C4d | Evaluate on test | Hybrid test Brier 0.0790 vs naive 0.0795 (Δ=-0.0006, 0.7%). Helps science_tech (-0.040), hurts geopolitics (+0.021). | **Done 2026-04-12** |
| C4e | Document findings | Append to results_summary.md. Whether positive or negative, this is the final exploitation attempt. | Pending |

**Leakage guard:** All trajectory features must be strictly backward-looking. No feature can use future price information. Rolling windows use `.shift(1)` convention.

**Acceptance criteria:** Improvement of > 0.001 Brier on val (10x what Platt achieved). If not, trajectory dynamics do not contain exploitable signal beyond what market price already captures.

### Execution order (updated)

```
C1 (DONE) → C4a → C4b → C4c → C4d → C4e  (trajectory dynamics)
                                       ↓
                                      C2    (dashboard)
                                       ↓
                                      C3    (write-up)
```

C2/C3 proceed after C4 regardless of result — findings are publishable either way.

### Research backlog

Ideas to explore if C4 yields a positive signal, or to note as future work in the write-up:

1. **Event group dynamics** — Within-group information flow: when market A in an event group resolves, does market B's price adjust immediately or with lag? Cross-market information lag could be exploitable.
2. **Volume-price divergence** — Markets where volume spikes but price doesn't move (or vice versa). May signal informed/uninformed disagreement.
3. **Resolution surprise prediction** — Predict `|market_error|` at final snapshot rather than `resolved_yes`. Different target, different use case ("which markets should I not trust?").
4. **Convergence speed prediction** — Predict how quickly a market will converge to its final price. Useful for knowing when to act on market signals.

## Decision Points (things we commit to decide later)

- **Category grouping** — resolved 2026-04-12. Using 11 categories from updated regex classifier.
- **Selection target** — resolved 2026-04-12. C1 used `resolved_yes` for recalibration. C4 will use both `resolved_yes` (for correction model) and `|market_error|` (for exploratory analysis).
- **Cross-market feature redesign** — deferred to backlog item 1 (event group dynamics).

## Falsified / dead-letter

| Item | Status | Notes |
|---|---|---|
| Long-horizon-only correction regressor | **Falsified 2026-04-11** | Train and test long-horizon populations have different theme composition. |
| Global market_correction regressor | **Underperforms naive** | Keep config for comparison. |
| Generic "ML beats market globally" framing | **Retired** | Market is well-calibrated (reliability 0.0004). |
| Structural alpha-shift rule (B5) | **Failed 2026-04-12** | Time-horizon signals confirmed but rule loses money. Identification problem unsolved. |
| Calibration correction (C1) | **Near-zero improvement 2026-04-12** | Best model (Platt) improves Brier by 0.0001. Market already near-optimal. |
| Trajectory dynamics (C4) | **Small improvement 2026-04-12** | Hybrid (45% XGB + 55% market) improves test Brier by 0.0006 (0.7%). Standalone model worse than naive. Not enough to justify a correction model. |

## Next-session starter prompt

> "Resume from `docs/execution_plan.md`. C1 and C4 are complete — both show Polymarket is efficient. Next: C2 (Streamlit dashboard) and C3 (research write-up). The research backlog has 4 ideas if we want to explore further before writing up."
