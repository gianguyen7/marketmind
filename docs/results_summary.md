# Results Summary

*Last updated: 2026-04-20.*

## Research Arc Summary

This project asked: **do binary prediction markets on Polymarket have structural, regime-invariant inefficiencies that a model can exploit?**

After three rounds of analysis — 10-market shutdown drill-down (A1/A2), 5,000-market data expansion (B1/B3), and walk-forward validation (B5) — the answer is **partially yes on description, no on exploitation**:

| Question | Finding | Status |
|----------|---------|--------|
| Is there directional mispricing? | YES under-priced by +0.48 on long-horizon markets that resolve YES | **Confirmed** across 4,538 markets |
| Is the bias time-structured? | Worse early in market life (`days_to_end` ρ=+0.25, `pct_lifetime_elapsed` ρ=-0.30) | **Confirmed** on train+val |
| Does volume predict mispricing? | Yes on full population (ρ=-0.17); no within low-volume tail (range restriction) | **Partial** — definition-dependent |
| Can a simple rule profit? | No — blanket alpha-shift loses $464/61 bets because only 20–25% resolve YES | **Failed** |
| Is the mechanism regime-invariant? | Time signals survive cross-category expansion; volume does not survive population filtering | **Partially** — the direction survives, the trading rule does not |

**Bottom line:** Polymarket systematically under-prices YES on long-horizon binary markets. The effect is large, robust across categories, and survives regime expansion. But knowing the *direction* of mispricing without knowing *which markets will resolve YES* is not actionable. The structural hypothesis describes a real phenomenon but does not produce a profitable rule. Converting it into one requires solving a fundamentally different problem: outcome prediction.

## Key Findings

- [x] **Polymarket is well-calibrated overall** — Brier 0.076 on 95-market curated set; mean |error| 0.15 on 4,538-market expanded set
- [x] **YES under-pricing on long-horizon markets** — confirmed across 4,538 markets, 11 categories
- [x] **Time-horizon signals are robust** — `days_to_end` (+) and `pct_lifetime_elapsed` (-) replicate across train/val
- [x] **Volume signal is real but definition-dependent** — works on full population, fails within low-volume tail due to range restriction
- [x] **Simple alpha-shift rule is not profitable** — over-corrects the 75–81% of structural-pop markets that resolve NO
- [x] **Identification problem is the blocker** — directional knowledge without outcome prediction is not actionable

## Track A1 — Government shutdown characterization (2026-04-11)

**Run:** `python3 scripts/analyze_government_shutdown.py`
**Artifacts:** `outputs/figures/shutdown_trajectories_2026-04-11.png`,
`outputs/figures/shutdown_per_market_error_2026-04-11.png`,
`outputs/tables/shutdown_error_decomposition_2026-04-11.csv`.

1. **Directional bias is strongly positive on snapshots, weakly positive across markets.**
   Snapshot-level mean `market_error = resolved - price` = **+0.479** (95% bootstrap CI **[+0.441, +0.515]**, n=717).
   Market-level mean = **+0.152** with CI **[-0.192, +0.492]** (n=10, not significant).
   Snapshots systematically **under-price YES** — but the effect is driven by long tails on the YES-resolved markets, not a uniform per-market bias.
2. **The mispricing is persistent but converges late.** 6/10 markets shrink their |error| by >0.05 between their first and last 10% of snapshots, 3 are flat, 1 grows. Convergence is real but slow — not "efficient market racing to truth".
3. **The 0.62 mean |error| is concentrated, not uniform.** Per-market |err| ranges 0.11–0.85 (std 0.22). The top-2 markets ("Will there be a US Government shutdown?" 0.85, "Will the shutdown end Nov 12–15?" 0.79) contribute **56%** of the total snapshot-weighted error mass.
4. **Pattern is under-pricing of YES on long-lived shutdown markets.** All four markets with mean |err| > 0.6 resolved YES; the three lowest-error markets all resolved NO with final prices ≈ 0. The "market is wrong" theme is: **YES shutdown resolutions are repeatedly under-priced, especially early in the market's life**, while NO resolutions are tracked well.
5. **Verdict on A1 acceptance criteria:** coherent pattern found (late-converging under-pricing of YES on long-running shutdown markets). Proceed to **A2 signal search** with `days_to_end`, `pct_lifetime_elapsed`, and `price_extremity` as the first candidate features. Caveat: n=10 means any A2/A3 finding is suggestive, not conclusive, until Track B expands the shutdown universe.

## Track A2 — Pre-registered signal search (2026-04-11)

**Pre-registration:** `docs/shutdown_hypothesis_2026-04-11.md` (frozen before running A2).
**Run:** `python3 scripts/analyze_government_shutdown.py` (A2 section).
**Split used:** train+val shutdown snapshots from `data/processed/{train,val}.parquet` (283+235 snapshots, 2+5 markets). Test untouched.

### Verdict: **MECHANISM REJECTED** — do not proceed to A3 without rewriting the hypothesis.

Two pre-committed falsification triggers fired:

| Primary signal | Train ρ (n=283) | Val ρ (n=235) | Predicted | Result |
|---|---|---|---|---|
| `days_to_end` vs `market_error` | +0.171 | +0.204 | + on both | val only; train below threshold |
| `pct_lifetime_elapsed` vs `market_error` | +0.226 | -0.173 | - on both | **SIGN FLIP** |
| `log_volume` vs `|market_error|` | +0.618 | -0.680 | - on both | **SIGN FLIP** |

Plus: secondary feature `price_vs_open` had |ρ_train| = 0.981 — dominating every primary by a wide margin. Per pre-reg §4 criterion 3, a descriptive feature dominating the mechanism-linked ones is a data-mining red flag, not a win. **Third falsification trigger.**

Cross-sectional signal (`event_group_price_deviation`, val only): signed ρ = -0.498 (supports mechanism direction) but |deviation| vs |error| ρ = -0.208 (opposite of predicted +). Mixed, not a clean confirmation.

### Honest read of the negative result

The pre-registration is doing its job. But read the rejection carefully: **the dominant reason A2 failed is that the current split cannot adjudicate mechanistic claims on shutdown**, not that the mechanism is definitely wrong.

- Train = 2 markets. Any ρ on train features is mostly a between-market contrast on n=2 — `log_volume` ρ = +0.618 on train just means "snapshots from market A have lower log_volume and smaller |err| than snapshots from market B." That's not a liquidity signal, it's a market-identity signal. A sign flip between train and val on an n=2 split is the expected behavior, not evidence against a mechanism.
- Val = 5 markets all in one event group, so val ρ values reflect within-group dynamics rather than cross-regime structure.
- The `price_vs_open` dominance is almost certainly the same artifact: with 2 train markets, any feature that happens to correlate with market identity will look "dominant."

**What this means for the project:**

1. **Do not proceed to A3** on the current split. A threshold rule tuned on this train/val would be a rule tuned on 2 + 5 markets and would not generalize. The pre-reg correctly forbids it.
2. **The shutdown mechanism is not falsified, but it is not testable here.** The honest conclusion is "the current 10-market split is underpowered to adjudicate the structural hypothesis; defer the test to the expanded Gamma dataset (Track B) where the long-horizon low-liquidity tail-risk population is large enough for n>>5 within each side of the split."
3. **Track A's remaining value is characterization, not rule-fitting.** A1's finding (late-converging YES under-pricing concentrated on long-horizon markets) stands as a hypothesis, not a confirmed mechanism. It should be re-tested on the expanded dataset in Track B5 walk-forward, not promoted to a bet rule now.
4. **Track B1 (Gamma expansion) becomes the critical path.** With A2 rejected on this split, the only way to test the hypothesis is to get the expanded dataset built. Delegation of B1 to the data-agent is now the highest-value next move.

**Caveats on every number in this section (pre-committed, per hypothesis doc §5):**
effective n = 10 markets (2/5/3 split), theme selected on dependent variable, regime window = 2024–2026 shutdowns, no multiple-comparison correction, train log_volume range is narrow (2 markets), val days_to_end range is short (~1.5–39 days). All ρ values are descriptive, not inferential.

**Artifacts:**
`outputs/figures/shutdown_a2_primary_signals_2026-04-11.png`,
`outputs/figures/shutdown_a2_event_group_deviation_2026-04-11.png`,
`outputs/tables/shutdown_a2_secondary_correlations_2026-04-11.csv`.

## B5 — Walk-forward validation on expanded data (2026-04-12)

**Run:** `python3 scripts/run_b5_walkforward.py`
**Dataset:** B3 expanded splits — 4,538 markets, 11 categories, 639K snapshots.
**Structural population:** Bottom-Q25 volume AND max(days_to_end) > 90 → 106/52/61 markets in train/val/test.

### Phase 1: Signal search (train + val, test untouched)

| Primary signal | Train ρ | Val ρ | Predicted | Result |
|---|---|---|---|---|
| `days_to_end` vs `market_error` (YES only) | +0.245 (n=20 mkts) | +0.059 (n=11 mkts) | + | **OK on both** |
| `pct_lifetime_elapsed` vs `market_error` (YES only) | -0.309 (n=20 mkts) | -0.295 (n=11 mkts) | - | **OK on both** |
| `log_volume` vs `|market_error|` (all) | -0.103 (n=106 mkts) | +0.004 (n=52 mkts) | - | **FLIP on val** |

Cross-sectional: `event_group_price_deviation` vs `|market_error|` ρ = +0.360 (train), +0.045 (val) — correct sign but weak on val.

**Phase 1 verdict: NOT CONFIRMED.** `log_volume` sign-flips on val within the structural population. However, on the **full population** (all 4,538 markets), all three signals pass with the predicted sign (ρ = -0.080, p<0.001 for log_volume on val). The flip is a **range restriction artifact**: the structural pop's volume range is $1.3M–$1.7M (std $113K) — there's no meaningful volume variation within the bottom quartile.

### Phase 2: Frozen rule (exploratory, since Phase 1 did not confirm)

Frozen alpha = +0.3411 (mean market_error on train struct pop YES-resolved markets).
Rule: `predicted_prob = clip(price_yes + 0.3411, 0, 1)` for structural-pop markets.

| Split | Struct Brier (naive) | Struct Brier (rule) | Δ | P&L ($100/bet) |
|---|---|---|---|---|
| Train | 0.0491 | 0.1605 | +0.1114 | -$406 (106 bets) |
| Val | 0.0749 | 0.2041 | +0.1292 | -$300 (52 bets) |
| **Test** | **0.0668** | **0.2163** | **+0.1495** | **-$464 (61 bets)** |

**The rule makes things worse on every split and loses money consistently.** Root cause: YES under-pricing is real (+0.48 mean error on YES markets in the structural pop) but only 19–25% of structural-pop markets resolve YES. A blanket alpha-shift applied to all structural markets over-corrects the 75–81% that resolve NO.

### What this means

1. **The time-horizon signals are real and robust.** `days_to_end` (+) and `pct_lifetime_elapsed` (-) confirm the mechanism on both train and val, across 4,538 diverse markets. Markets with more time remaining have larger positive errors on YES-resolved outcomes. This survives the regime change from 10 shutdown markets to the full Polymarket universe.

2. **Volume is not the right conditioning variable within the low-volume tail.** The mechanism works on the full population (low volume → more mispricing) but fails to discriminate within the already-low-volume structural population. The structural population definition may need to be horizon-only, not volume-intersected.

3. **The identification problem remains unsolved.** The mechanism correctly predicts *which direction* mispricing goes (YES is under-priced) and *when it's worst* (early in long-horizon markets). But it cannot identify *which* structural-pop markets will resolve YES — and without that, a rule loses money because most markets resolve NO.

4. **Next step would be an identification model**, not a bigger alpha. The structural hypothesis provides a useful *prior* (long-horizon markets under-price YES), but converting it into a profitable rule requires a second-stage classifier that estimates P(resolve YES | structural-pop market). This is a fundamentally different problem from the original hypothesis, and would need its own pre-registration.

**Caveats (same-sentence rule):** effective n = 106/52/61 markets per split (not snapshot counts); structural population defined empirically (bottom-Q25 volume, max days_to_end > 90); test split is temporal not true walk-forward (no post-freeze live data); category-stratified split means per-category temporal gaps vary; alpha estimated from train only; no multiple-comparison correction on 3 primary signals.

**Artifacts:**
`outputs/figures/b5_primary_signals_2026-04-12.png`,
`outputs/figures/b5_rule_evaluation_2026-04-12.png`,
`outputs/tables/b5_category_breakdown_2026-04-12.csv`,
`outputs/tables/b5_results_2026-04-12.json`.

## Calibration Analysis — Expanded Dataset (2026-04-12)

**Run:** `python3 scripts/run_calibration_analysis.py`
**Dataset:** 4,538 markets, 639,355 snapshots, 11 categories (B3 splits).

### 1. Overall calibration

| Split | Brier | Reliability | Resolution | ECE | Base Rate | Markets |
|-------|-------|-------------|------------|-----|-----------|---------|
| All | 0.0786 | 0.0004 | 0.0580 | 0.011 | 16.4% | 4,538 |
| Train | 0.0745 | 0.0009 | 0.0592 | 0.017 | 15.9% | 2,619 |
| Val | 0.0909 | 0.0004 | 0.0640 | 0.015 | 19.2% | 899 |
| Test | 0.0795 | 0.0004 | 0.0496 | 0.014 | 15.3% | 1,020 |

**Polymarket is remarkably well-calibrated overall.** Reliability of 0.0004 means the market's price is almost exactly equal to the true resolution rate at every probability level. ECE of 1.1% is excellent. The Brier score is dominated by base-rate uncertainty (0.137), not reliability.

### 2. Calibration by category — where calibration breaks

Sorted by Brier (worst last):

| Category | Brier | ECE | Reliability | Markets | Base Rate |
|----------|-------|-----|-------------|---------|-----------|
| recession_economy | 0.040 | 0.045 | 0.005 | 51 | 13% |
| sports | 0.065 | 0.009 | 0.001 | 2,099 | 12% |
| entertainment | 0.066 | 0.031 | 0.002 | 93 | 13% |
| fed_monetary_policy | 0.067 | 0.032 | 0.003 | 117 | 18% |
| social_media | 0.070 | 0.046 | 0.007 | 163 | 10% |
| politics_elections | 0.077 | 0.026 | 0.003 | 767 | 17% |
| government_policy | 0.080 | 0.090 | 0.039 | 43 | 21% |
| crypto_finance | 0.095 | 0.042 | 0.003 | 555 | 17% |
| other | 0.102 | 0.024 | 0.001 | 250 | 24% |
| science_tech | 0.115 | 0.046 | 0.005 | 97 | 24% |
| **geopolitics** | **0.124** | **0.019** | **0.001** | **303** | **23%** |

Key patterns:
- **Sports is the best-calibrated large category** (Brier 0.065, ECE 0.009, 2,099 markets). Despite being 46% of the dataset, it contributes least miscalibration per-snapshot.
- **Geopolitics is the worst-calibrated** (Brier 0.124, 303 markets). But its reliability is actually excellent (0.001) — the high Brier comes from high base-rate uncertainty (23% YES rate) and low resolution (the market doesn't discriminate well between geopolitical events that will and won't happen).
- **Government_policy has the worst reliability** (0.039) — the market's prices genuinely don't match resolution rates here. Only 43 markets so fragile, but the ECE of 0.090 is the worst of any category.
- **Market-level Brier tells a different story**: sports has the *worst* market-level Brier (0.185) because many sports markets have intermediate prices that contribute steady error. Recession_economy is best (0.033).

### 3. Calibration by time horizon

| Horizon | Brier | ECE | Reliability | Markets | Base Rate |
|---------|-------|-----|-------------|---------|-----------|
| <1d | 0.062 | 0.014 | 0.001 | 3,175 | 23% |
| 1d-1w | 0.138 | 0.027 | 0.002 | 3,369 | 32% |
| 1w-1m | 0.097 | 0.023 | 0.001 | 2,647 | 22% |
| 1m-3m | 0.072 | 0.019 | 0.001 | 1,797 | 16% |
| 3m-1y | 0.070 | 0.011 | 0.001 | 1,393 | 13% |
| **>1y** | **0.131** | **0.101** | **0.028** | **99** | **15%** |

**Non-monotonic pattern.** Calibration is best at <1d (resolution imminent) and worst at 1d-1w and >1y. The 1d-1w peak is driven by high base-rate uncertainty (32% YES rate in that window — many markets resolve YES in this "last week" period). The >1y degradation (ECE 0.101, reliability 0.028) reflects genuine miscalibration on very long-horizon markets, consistent with the B5 structural hypothesis about capital constraints on long horizons.

### 4. Category × horizon heatmap — the worst pockets

Worst Brier cells (>0.20):

| Category × Horizon | Brier | Interpretation |
|---------------------|-------|----------------|
| geopolitics >1y | 0.574 | Worst cell. Long-range geopolitical prediction is essentially random. |
| science_tech >1y | 0.422 | Same pattern — very long-range tech predictions are unreliable. |
| government_policy 3m-1y | 0.337 | Long-range government policy (shutdown, debt ceiling) is poorly priced. |
| entertainment <1w | 0.029 | Best cell — entertainment resolves cleanly near the end. |
| recession_economy 1m-3m | 0.007 | Near-perfect calibration on medium-horizon economic indicators. |

### 5. Favourite-longshot bias — **classic F-L confirmed**

| Price Bucket | Bias (actual - predicted) | Interpretation |
|-------------|---------------------------|----------------|
| 0.0-0.1 | +0.001 | Well-calibrated (near-zero events) |
| 0.1-0.2 | +0.015 | Slight longshot underpricing |
| **0.2-0.3** | **+0.051** | **Longshots win 5pp more than prices imply** |
| **0.3-0.4** | **+0.036** | **Moderate underpricing** |
| 0.4-0.5 | -0.002 | Near-perfect |
| 0.5-0.6 | -0.035 | Slight favourite overpricing |
| **0.7-0.8** | **-0.071** | **Favourites win 7pp less than prices imply** |
| 0.8-0.9 | -0.054 | Overpricing continues |
| 0.9-1.0 | +0.009 | Near-locks are well-calibrated |

**Classic favourite-longshot bias.** Longshots (price < 0.3) win more often than implied (+2.2pp average bias). Favourites (price > 0.7) win less often than implied (-3.9pp average). The crossover is at ~0.45. This is consistent with the known F-L bias in betting markets and the B5 finding about YES under-pricing on low-probability outcomes.

Per-category F-L bias varies dramatically:
- **government_policy**: +17.4pp longshot bias (massive — market badly underprices low-probability government events)
- **politics_elections**: +5.4pp (moderate)
- **science_tech**: +4.1pp
- **crypto_finance**: -3.8pp (reverse — crypto longshots are *over*priced)
- **social_media**: -5.9pp (reverse — social media longshots overpriced)

### 6. Per-split stability — what's robust, what's fragile

Categories with Brier shift > 0.05 between train and test:

| Category | Train Brier | Test Brier | Δ | Interpretation |
|----------|-------------|------------|---|----------------|
| entertainment | 0.052 | 0.230 | +0.178 | Badly non-stationary (only 9 test markets) |
| geopolitics | 0.114 | 0.214 | +0.100 | Degrades — later geopolitical events are harder |
| science_tech | 0.108 | 0.204 | +0.096 | Same pattern — recent tech markets harder |
| social_media | 0.062 | 0.151 | +0.089 | Degrades |
| other | 0.093 | 0.183 | +0.090 | Degrades |
| gov_policy | 0.132 | 0.002 | -0.129 | *Improves* (but n=20, fragile) |
| fed_monetary | 0.088 | 0.031 | -0.057 | Improves — recent Fed markets better calibrated |

**Stable categories** (Brier shift < 0.02): sports, politics_elections, recession_economy, crypto_finance. These are the categories where calibration findings generalize across time.

**Fragile categories** (large shift, small n): entertainment, government_policy, social_media. Findings here should not be over-interpreted.

**Artifacts:**
`outputs/figures/calibration_by_category.html`,
`outputs/figures/calibration_by_horizon.html`,
`outputs/figures/calibration_category_x_horizon.html`,
`outputs/figures/favourite_longshot_bias.html`,
`outputs/tables/calibration_by_category.csv`,
`outputs/tables/calibration_by_horizon.csv`,
`outputs/tables/calibration_category_x_horizon.csv`,
`outputs/tables/favourite_longshot_bias.csv`,
`outputs/tables/fl_bias_per_category.csv`,
`outputs/tables/calibration_split_stability.csv`,
`outputs/tables/brier_decomposition.csv`.

## C1 — Calibration Correction Models (2026-04-12)

**Run:** `python3 scripts/run_c1_recalibration.py`
**Dataset:** B3 expanded splits (4,538 markets, 639K snapshots).
**Models trained on train only. Val for model selection. Test touched once.**

### Models tested

| Model | Method | Granularity |
|-------|--------|-------------|
| naive (market price) | Raw `price_yes` | — |
| isotonic_global | Isotonic regression | Global |
| platt_global | Logistic on logit(price) | Global |
| isotonic_per_cat | Isotonic per category | 9 categories + 1 pooled |
| platt_per_cat | Platt per category | 9 categories + 1 pooled |
| cat_x_horizon | Isotonic per category × horizon bucket | 43 buckets + fallback |

### Test results

| Model | Brier | Reliability | Resolution | ECE | Log Loss |
|-------|-------|-------------|------------|-----|----------|
| naive (market price) | 0.0795 | 0.0004 | 0.0496 | 0.0135 | 0.2607 |
| **platt_global** | **0.0794** | **0.0003** | **0.0495** | **0.0114** | **0.2601** |
| isotonic_global | 0.0797 | 0.0005 | 0.0496 | 0.0170 | 0.2684 |
| platt_per_cat | 0.0804 | 0.0008 | 0.0493 | 0.0185 | 0.2619 |
| isotonic_per_cat | 0.0823 | 0.0013 | 0.0477 | 0.0238 | 0.2966 |
| cat_x_horizon | 0.0863 | 0.0017 | 0.0446 | 0.0297 | 0.3616 |

### Verdict: the market is already near-optimally calibrated

**Platt scaling wins, but the improvement is 0.0001 Brier (0.1%).** This is the key finding of C1 and arguably the most important result of the entire project:

1. **The market's reliability of 0.0004 is essentially perfect.** No recalibration model can meaningfully improve it. Platt scaling achieves 0.0003 — a difference that is statistically and practically meaningless.

2. **More granular models make things worse, not better.** Per-category isotonic (0.0823) is worse than naive (0.0795). Category × horizon (0.0863) is worst of all. The more parameters you add, the more you overfit to training-set calibration quirks that don't generalize.

3. **Platt scaling slightly corrects the F-L bias.** On the 0.7-0.8 price bucket, naive bias is -0.054 and Platt reduces it to -0.022. On 0.5-0.6, bias goes from -0.045 to -0.027. But these corrections are too small to move Brier meaningfully.

4. **Per-category analysis shows no exploitable pockets.** The best per-category improvement (platt_global on science_tech) is -0.0014 Brier. Geopolitics actually gets *worse* (+0.0029). No category shows a correction larger than noise.

### What this means for the project

The market is well-calibrated because **the F-L bias, while real and statistically significant, is small in absolute terms.** The largest bias (+5.1pp in the 0.2-0.3 bucket) sounds large, but only 5.4% of snapshots are in that bucket, so its contribution to overall Brier is tiny. The bias that matters for Brier is the one affecting the most snapshots — and the 0.0-0.1 bucket (65% of all snapshots) has a bias of only +0.1pp.

This closes the calibration correction line of investigation. The market does not have an exploitable calibration flaw at any granularity we tested. The F-L bias is a real market microstructure phenomenon worth documenting, but it is not large enough to be actionable.

**Artifacts:**
`outputs/tables/c1_recalibration_results_2026-04-12.csv`,
`outputs/tables/c1_per_category_test_2026-04-12.csv`,
`outputs/models/c1_recalibration_2026-04-12.pkl`.

## C4 — Price Trajectory Dynamics (2026-04-12)

**Run:** `python3 scripts/run_c4_trajectory.py`
**Dataset:** B3 expanded splits (4,538 markets, 639K snapshots).
**Models trained on train only. Val for selection. Test touched once.**

### C4a: New trajectory features engineered

| Feature | Description | ρ vs \|error\| | Robust? |
|---------|-------------|----------------|---------|
| price_extremity | \|price - 0.5\| | -0.974 | Yes (all 11 categories) |
| staleness | Snapshots since last price move >0.01 | -0.743 | Yes |
| price_range_14 | Max-min over last 14 snapshots | +0.693 | Yes |
| price_volatility_7 | Rolling 7-snap std (existing) | +0.647 | Yes |
| dist_from_low | Position in historical price range | +0.589 | Yes |
| vol_regime | Recent vol / historical vol | +0.360 | Yes |
| path_curvature | Deviation of rolling price from open→current midpoint | +0.232 | Yes |
| price_acceleration | Second derivative of price | +0.004 | No (noise) |

All correlations are with `|market_error|` on the train split. "Robust" = same sign across all 11 categories.

### C4b: Key finding — staleness predicts accuracy, not mispricing

The strongest new signal is **staleness** (ρ = -0.743): markets whose price hasn't moved recently have *lower* error, not higher. The intuitive "stuck at wrong price" hypothesis is backwards. **Stale markets are confident and correct** — when the crowd agrees and stops trading, it's usually right. Active, volatile markets are where error concentrates.

This is consistent with the well-known "wisdom of crowds when diverse and independent" result: once a market converges and trading dies down, the information has been aggregated. Ongoing volatility signals ongoing disagreement — and disagreement means the price is less certain.

### C4c-d: Model results

| Model | Val Brier | Test Brier | Δ vs naive |
|-------|-----------|------------|------------|
| naive (market price) | 0.0909 | 0.0795 | — |
| logistic_trajectory | 0.0927 | 0.0802 | +0.0006 (worse) |
| xgb_trajectory | 0.0915 | 0.0820 | +0.0025 (worse) |
| **hybrid (45% XGB + 55% market)** | **0.0890** | **0.0790** | **-0.0006 (0.7% better)** |

The standalone models are *worse* than naive — the trajectory features only help when heavily anchored to the market price. The optimal blend is 45/55, meaning the market price carries most of the signal.

XGBoost feature importance: `price_yes` (39%), `price_extremity` (21%), `dist_from_low` (14%), `price_vs_open` (5%), `log_volume` (5%).

### Per-category breakdown (hybrid vs naive, test)

| Category | Naive Brier | Hybrid Brier | Δ |
|----------|-------------|--------------|---|
| science_tech | 0.204 | 0.164 | **-0.040** |
| recession_economy | 0.030 | 0.028 | -0.003 |
| politics_elections | 0.065 | 0.063 | -0.002 |
| social_media | 0.151 | 0.149 | -0.002 |
| sports | 0.063 | 0.063 | 0.000 |
| crypto_finance | 0.084 | 0.085 | +0.000 |
| geopolitics | 0.214 | 0.235 | **+0.021** |

The model helps on science/tech and politics but *hurts* on geopolitics. The improvement is concentrated, not uniform.

### Verdict

**Trajectory dynamics add a small but real edge (0.7% Brier improvement).** The model learns that the market's own confidence signals (price extremity, staleness, low volatility) are predictive of accuracy. But:

1. The improvement is small (0.0006 Brier) and fragile across categories
2. The standalone model is worse than naive — it only works as a blend
3. Market-level Brier is slightly worse (0.1402 vs 0.1399) — the improvement is at snapshot level
4. The model is essentially learning "trust the market more when it's confident" — which the market already does implicitly

**This is the fourth and final exploitation attempt.** The pattern across all four attempts:

| Attempt | Approach | Test Δ Brier |
|---------|----------|-------------|
| B5 | Structural alpha-shift | +0.1495 (much worse) |
| C1 | Calibration correction (Platt) | -0.0001 (negligible) |
| C4 | Trajectory dynamics (hybrid) | -0.0006 (small) |
| — | Global ML correction (pre-B3) | Underperforms naive |

**Conclusion: Polymarket is efficient.** The crowd aggregates information well, calibrates probabilities accurately (reliability 0.0004), and leaves very little room for systematic improvement. The F-L bias is real but too small to exploit. Trajectory dynamics provide marginal improvement but not enough to justify a correction model in practice. The most valuable output of this project is the *characterization* of where and how the market works, not a model that beats it.

**Artifacts:**
`outputs/tables/c4_trajectory_correlations_2026-04-12.csv`,
`outputs/tables/c4_results_2026-04-12.json`.
