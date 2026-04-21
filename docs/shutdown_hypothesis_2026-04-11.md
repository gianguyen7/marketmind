# Shutdown Mispricing — Pre-Registered Hypothesis

> **Written: 2026-04-11, before running A2 feature search.**
> This document exists to prevent the A2 signal search from becoming a degrees-of-freedom exercise. The hypothesis, the features that will test it, and the caveats that will be attached to every resulting number are all committed *before* looking at any feature/error relationships in the data.
>
> A2's job is to answer "did the pre-registered primary signals behave as the mechanism predicts, yes or no?" — not "which feature correlates best with market_error?"

---

## 1. Structural mechanism (the prior)

**Claim:** On long-horizon binary tail-risk markets, the market systematically under-prices YES resolutions because the marginal trader is capital-constrained and attention-limited. YES probabilities bleed downward over time — not toward a revised estimate of the true probability, but toward zero — because holding a YES position through a weeks-long resolution window ties up capital on a low-probability outcome that retail traders and small LPs are structurally unwilling to sit on. The arbitrage that would push YES back up does not happen because the participants who could run it are not present in sufficient size.

**Grounding in A1 findings (do not circular-reason past this point):**

- A1 showed that 4/4 shutdown markets with mean |err| > 0.6 resolved YES, while the 3 lowest-error markets all resolved NO with final price ≈ 0.
- Snapshot-level directional bias was strongly positive (mean `resolved - price` = +0.479, 95% CI [+0.441, +0.515]) but market-level bias was weakly positive and not significant (n=10). The mechanism predicts this asymmetry: the bias lives in the *time a market spends mispriced*, not in every individual market equally.
- 56% of total snapshot-weighted error mass sat on the top-2 markets by |err|. The mechanism predicts concentration: markets with the longest horizons and lowest liquidity should carry disproportionate error mass, while short-horizon high-volume markets should track truth.

**What the mechanism is *not*:**

- Not a claim that shutdown markets as a theme are mispriced. That framing is theme-specific and unlikely to survive regime shift per `docs/research_plan.md` §Regime-Invariance Discipline. If the shutdown data ends up supporting the mechanism, the claim that leaves the project is the structural one (long-horizon low-liquidity binary tail markets under-price YES), not the thematic one.
- Not a claim about shutdown politics, fiscal calendars, or the specific 2024–2025 political environment. Those would be regime-specific and non-transferable.

## 2. Features that test the mechanism (pre-committed)

**Primary signals** — chosen because they directly probe the mechanism. These are the only features A2 is allowed to use to reach a confirm/falsify verdict. All three must point in the mechanism-predicted direction on both train and val for the hypothesis to be considered supported.

| Feature | Mechanism prediction | Direction |
|---|---|---|
| `days_to_end` | Longer horizons → more time for YES to bleed down, more room for under-pricing | Larger `days_to_end` → larger positive `market_error` (on markets that will resolve YES) |
| `pct_lifetime_elapsed` | Early in market life → more time remaining for capital-constrained under-pricing to persist | Smaller `pct_lifetime_elapsed` → larger positive `market_error` (on markets that will resolve YES) |
| `log_volume` / `volume_rank` | Low liquidity → no arbitrage pressure → under-pricing persists | Smaller `log_volume` → larger `|market_error|` |

**Cross-sectional secondary signal** — the research plan prefers cross-sectional features because they survive regime shift better. This one is pre-committed as well but is tested separately from the primaries.

| Feature | Mechanism prediction |
|---|---|
| `event_group_price_deviation` | Within an event group of simultaneously-open markets, the markets that deviate most from their group's implied consensus should be the most mispriced |

**Explicitly demoted to secondary / descriptive only:**

- `price_volatility_7`, `price_momentum_3`, `price_momentum_7`, `price_extremity`, `price_change`, `price_vs_open`.
- These features *describe* the price path but are not mechanism-linked. If any of them dominates the primary signals in A2, that is a red flag for data mining, not a win. They will be plotted for visual sanity-check but will not be allowed to drive the confirm/falsify verdict.

## 3. Confirmation criteria (pre-committed)

The mechanism is considered **supported by A2** if and only if all of the following hold:

1. `days_to_end` shows a monotone positive relationship with signed `market_error` on markets that resolved YES, on both train and val splits, with the same sign. (Spearman ρ > 0.2 on both splits is the minimum threshold.)
2. `log_volume` shows a monotone negative relationship with `|market_error|` on both train and val, same sign.
3. `event_group_price_deviation` (where defined — not all 10 markets will have event group peers) shows a monotone positive relationship with `|market_error|` on train+val pooled.
4. No secondary feature beats all three primaries on both splits. (If a secondary feature dominates, the mechanism is not the real driver, and any A3 rule built on it would be data-mined.)

## 4. Falsification criteria (pre-committed)

The mechanism is considered **rejected by A2**, and Track A stops for reconsideration, if any of the following hold:

1. Either primary signal (`days_to_end`, `log_volume`) has sign flip between train and val. Flip = noise = no real signal.
2. `|market_error|` is uncorrelated (|Spearman ρ| < 0.1) with both `days_to_end` and `log_volume` on train+val pooled.
3. A secondary descriptive feature (e.g. `price_volatility_7`) has materially stronger monotone relationship with `market_error` than both primaries on both splits. This would indicate the real driver is a price-path property, not a structural property, and the mechanism needs to be rewritten before proceeding.
4. The directional bias disappears once we condition on `days_to_end`. If bias is flat across horizon buckets, the "long horizon bleeds down" story is wrong.

If A2 lands in the rejection region, **do not proceed to A3.** Document the negative result in `docs/results_summary.md` and stop Track A until the mechanism is rewritten or abandoned.

## 5. Caveats that attach to every A2 number (pre-committed)

These go in the same sentence as any number A2 produces. Not a footnote, not an appendix — the same sentence. They exist now, before the numbers, so they cannot be stripped later when a result looks exciting.

- **Effective n is 10 markets, not 717 snapshots.** Snapshots within a market are highly correlated; bootstrap CIs over snapshots lie.
- **Theme was selected on the dependent variable.** Shutdown was picked because it had the worst in-sample Brier (0.482), so any finding here is conditional on "themes where the market was already most wrong in-sample."
- **Regime window is narrow.** All 10 markets are from the 2024–2026 shutdown cycles; any finding is a statement about that specific political and liquidity regime until proven otherwise on the expanded Gamma dataset and on walk-forward post-freeze resolutions.
- **"Long-horizon low-liquidity" is defined empirically, not by absolute thresholds.** (Revised 2026-04-11 session 2.) An earlier version of this doc implied specific numeric cutoffs (`days_to_end > 90`, `volume_usd < $100k`). The Gamma dry-run showed the min volume at $1M `min_volume_usd` filter was $53M, making any absolute-dollar threshold meaningless. The structural population under test is now defined **relative to the fetched Gamma universe**: bottom quartile by `volume_usd`, intersected with markets whose max `days_to_end` across snapshots exceeds 90 days. Any rule or claim built on this population must state both the quartile cutoff and the absolute dollar value of that cutoff in the same sentence, so the result can be retested when the universe changes.
- **No multiple-comparison correction.** Three primary features tested; p-values and CIs are reported uncorrected and should be read as descriptive, not inferential.
- **Split is structurally weak.** train = 2 markets, val = 5 markets, test = 3 markets. A "rule that works on val" is a rule that works on five markets. Test is held out of A2 entirely — it is only touched in A3.

## 6. What A2 will produce

1. A `a2_signal_search()` function in `scripts/analyze_government_shutdown.py` that:
   - Operates on train+val snapshots only (test is held out).
   - For each primary signal: scatter plot vs `market_error` (and vs `|market_error|` for volume), LOESS or rolling-mean overlay, Spearman ρ on train and val *separately*.
   - For the cross-sectional signal: same, pooled.
   - For the secondary signals: one summary figure, correlations only, no per-feature deep dive.
   - Prints the pre-committed caveats (§5) before and after the result block.
2. A short findings note appended to `docs/results_summary.md` with the confirm/falsify verdict and, if confirmed, the structural claim as it will leave the project (not the theme-specific version).
3. Figures saved with the dated naming convention: `outputs/figures/shutdown_a2_<signal>_2026-04-11.png`.

---

## Commitment

This document is the version of the hypothesis that gets compared against A2's results. It does not get edited after A2 runs. If A2's findings suggest a better mechanism, that is a new hypothesis for a new document, not a retroactive rewrite of this one.
