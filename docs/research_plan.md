# Research Plan

> Last revised: 2026-04-11. Supersedes earlier "baselines vs ML vs hybrid" framing.

## Objective

Identify **structural conditions** under which binary event markets on Polymarket predictably mis-weight outcomes, and determine whether rules built on those conditions **retain their edge when applied to events drawn from a different political/economic regime than the one used to construct them**. The research question is about *regime-invariant structural inefficiency*, not about beating the market on a single backtest.

This reframing (2026-04-11, session 2) supersedes the earlier "find where the market is wrong and exploit it locally" framing. The earlier version was vulnerable to the devil's advocate critique that financial time series are non-stationary: any in-sample edge could be an artifact of the specific regime the training data was drawn from. This version commits the project to producing claims that are structural (about mechanisms) rather than parametric (about levels), and to validating those claims on genuinely out-of-sample events, not held-out splits of the same regime.

## Core Research Questions

1. **Where is the market wrong?** Which themes, horizons, and market structures show systematic reliability error — and is the pattern consistent with a structural mechanism (liquidity starvation, attention limits, tail-aversion), or does it look idiosyncratic to the specific events in the sample?
2. **Can a model *identify* mispriced markets before resolution**, using only features available at the snapshot time, with a rule that is motivated by a structural prior rather than discovered by searching the data?
3. **Does the rule survive regime shift?** When applied to new markets resolving *after* the rule is frozen — events drawn from a different political/economic regime than the training data — does the rule still fire profitably, or does its edge collapse?
4. **Does the picture hold when we expand the market universe** from 95 curated Fed/macro markets to ~5,000 diverse binary markets spanning multiple categories and time periods?

Anything that doesn't feed one of those four questions is out of scope.

## Regime-Invariance Discipline

Financial time series are non-stationary. A result can be real in-sample and meaningless out-of-sample. This project guards against that in four specific ways:

1. **Claim structure, not levels.** The output of this research is a falsifiable *structural hypothesis* (e.g., "long-horizon binary tail-risk markets under-price YES when liquidity is below $X") with a pre-committed list of observations that would falsify it. Not a backtest point estimate.
2. **Prefer relative, cross-sectional signals over absolute levels.** Cross-sectional claims ("among simultaneously-open shutdown markets, the ones with highest `event_group_price_deviation` are most mispriced") survive regime shift better than level claims ("shutdown markets have Brier 0.48"). When two rules tie in-sample, prefer the cross-sectional one.
3. **Economically-motivated priors beat data-mined patterns.** A finding is more trustworthy if we had a structural reason to expect it before looking at the data (liquidity constraints, attention constraints, retail tail-aversion). Rules that fall out of an exhaustive feature search without a prior are suspect — one-off degrees of freedom are cheap.
4. **Walk-forward is the only honest evaluation.** Historical test splits are for *rejecting* hypotheses, not confirming them. The real validation is paper-trading a frozen rule on markets that resolve after the rule is written down. Track B4 adds a walk-forward milestone for every rule that survives its Track A-style in-sample analysis.

### Always report findings with their failure modes

Any backtest number in this project's outputs must appear in the same sentence as its structural failure modes: effective sample size (not snapshot count), selection on the dependent variable, regime window, and the feature-search degrees of freedom. A Brier improvement without those caveats is misinformation.

## What We've Learned So Far (carry-forward facts)

- **Market is well-calibrated overall** on the 95-market dataset: Brier 0.076, reliability 0.0029.
- **Miscalibration is concentrated by theme.** `government_shutdown` (|market_error| ≈ 0.62, 10 markets) is dramatically underpriced; `tariffs_trade` (|err| 0.47, 1 market) also looks mispriced. `fed_leadership` is already well-calibrated (0.077) — nothing to exploit.
- **Calibration degrades with horizon.** Brier goes from 0.013 (<1d) to 0.094 (3m-1y). But this is partly confounded with theme — long-horizon markets over-represent non-fed themes.
- **Reverse favorite-longshot bias.** Favorites with price > 0.7 win *more* often than the price implies (bias +0.077).
- **Global correction models fail.** Three rounds of regressors (linear, RF, XGBoost) on `market_error` are worse than predicting zero. The regressors overfit badly (RF: train MAE 0.016 → test 0.408 on the long-horizon subset).
- **Structural blocker: train/test theme overlap is near-zero.** Time-based split put ~82% of `fed_rate_decisions` in train and ~76% of `fed_leadership` in test. Per-theme micro-models can't be validated on the current split; only `government_shutdown` has usable coverage across all three splits.
- **Long-horizon hypothesis falsified (2026-04-11).** Training a correction regressor only on `days_to_end >= 90` snapshots made things worse, because the long-horizon train population is dominated by mispriced non-fed themes while the long-horizon test population is dominated by well-calibrated `fed_leadership`. Population shift, not features.

## Approach (Dual Track)

The remaining work runs on two parallel tracks. Track A produces immediate research findings on current data. Track B enlarges the dataset so the per-theme framework can be validated properly.

### Track A — Government shutdown drill-down (immediate)

The only theme where we have train/val/test coverage (2/5/3 markets) *and* large market errors. Exploratory analysis, then a rule, then a model — in that order, and only if each stage justifies the next.

**A1. Characterize the mispricing.**
- Plot all 10 shutdown markets: price trajectory vs. time, annotated with resolution outcome.
- Decompose: are errors from (a) wrong final verdict, (b) correct verdict but late confidence, or (c) a few catastrophic markets dominating the mean?
- Directional: does the market systematically under- or over-price shutdown risk?

**A2. Search for mispricing signals.**
- Candidate features: `days_to_end`, `price_extremity`, `event_group_price_deviation`, `price_volatility_7`, proximity to fiscal-calendar dates.
- For each candidate, plot against `market_error` and compute correlation on train+val.
- Goal: find 1–2 features that separate mispriced snapshots from well-priced ones.

**A3. Write a bet rule.**
- Express the mispricing as a human-readable threshold rule (e.g., "bet YES on shutdown markets when days_to_end > X and price_yes < Y").
- Backtest on train → val → test. Report MAE and Brier vs. naive.
- Accept only if the rule beats naive on **both** val and test.

**A4. Only if the rule isn't enough, fit a micro-model.**
- Single-theme regressor on the 10 markets. Cross-validate by market, not snapshot.
- Compare directly to the rule from A3 — simpler wins ties.

**Deliverables:** `scripts/analyze_government_shutdown.py` (or a notebook) + `outputs/figures/shutdown_*.png` + a short findings note appended to `docs/results_summary.md`.

### Track B — Data expansion (parallel, larger scope)

See `docs/data_expansion_plan.md` for the full plan. Summary:

**B1. Gamma API bulk fetch (Phase 1 of expansion plan).**
- Paginate `gamma-api.polymarket.com/markets?closed=true&order=volumeNum`, filter to binary markets with volume > $1M, target ~5,000 markets.
- Regex-classify each market into categories (Politics, Fed, Sports, Crypto, Geopolitics, Entertainment, Government, Science, Other).
- Fetch price history for each via CLOB API (checkpoint/resume).
- Generalize `resolve_outcomes.py` for non-Fed markets and update `build_dataset.py` to use Gamma event IDs for grouping.

**B2. External features (Phase 2+).**
- FRED economic series + FOMC/economic calendar features (highest priority after expansion).
- Fed Funds futures via yfinance (gold-standard miscalibration signal for Fed markets).
- GDELT, Trends, Metaculus, on-chain — stretch.

**B3. Theme-stratified temporal split.**
- Once the expanded dataset exists, redo the split: sort each category's markets by `end_date`, take 60/20/20 *within category*. No shuffling, no cross-category randomization.
- Drop single-market categories or pool into `other`.
- Write `data/processed/split_manifest.json` with category → ordered market list → assignment for reproducibility.
- Audit cross-market features for backward-looking-ness under the new split (critical: per-category cutoffs mean a naive "all training data" aggregate can leak the future).

**B4. Re-establish baselines on the new split.**
- All current numbers (recency 0.0017 market-level, ensemble 0.087, etc.) become non-comparable on expanded data — expected, not a regression.

**B5. Walk-forward validation (new milestone, 2026-04-11).**
- Any rule or model that survives Track A-style in-sample validation must be **frozen in a versioned config** and then evaluated on markets that resolved *after* the freeze date.
- The expanded Gamma dataset is not the out-of-sample set — it is a larger in-sample set. The out-of-sample set is the set of markets that will resolve in the weeks following rule freeze.
- Report: hit rate, P&L per trigger, Brier vs. naive — all computed only on post-freeze resolutions.
- A rule that beats naive in-sample and fails on post-freeze resolutions is a negative result worth publishing, not a failed experiment worth hiding.

### Hand-off: Track A informs Track B

Findings from the shutdown drill-down (what features capture mispricing on a known-mispriced theme) become candidate features for the post-expansion per-category models. The drill-down is the prototype for the per-theme/per-category workflow that runs at scale in Track B.

## Methodology Rules (unchanged, non-negotiable)

- **Temporal leakage is the #1 failure mode.** All splits are time-based, within-group. Features are strictly backward-looking.
- **Probabilistic metrics are primary.** Brier, log loss, ECE. Accuracy/F1 never primary.
- **No silent assumptions.** Every fill, drop, threshold, and default is logged.
- **Reproducibility.** Seeds from config, dataset hashes logged, outputs named with experiment + date.
- **Lean implementation.** Functions over frameworks; pandas + sklearn + xgboost directly.
- **Research first.** Correctness over elegance, interpretability over performance, simple rules over complex models when they tie.

## What's Explicitly Out of Scope

- A global "beat the market" model. The market is well-calibrated on average; the research question is about structural inefficiency, not global accuracy.
- Backtest-only claims. Any finding that lives only in the historical test split, without a walk-forward follow-up, is not a finding — it is a hypothesis awaiting test.
- Deep learning. Dataset is too small and the interpretability bar is too high. Interpretability is load-bearing here: we need to be able to tell a structural story about *why* a rule should survive regime change, and opaque models make that impossible.
- Real-time trading infrastructure. Research only. Walk-forward evaluation is paper-traded.
- Non-binary or unresolved markets.
- Any approach that requires randomly shuffling time series.
- Theme-specific claims that don't generalize to a structural mechanism. "Shutdown markets are mispriced" is not an acceptable terminal finding — it must be reframed as "low-liquidity long-horizon binary tail-risk markets are mispriced" (or similar structural form) before it can leave the project as a result.
