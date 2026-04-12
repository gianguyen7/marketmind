# Recent Memory

> Rolling 48-hour context window. Updated after each conversation.
> Important items get promoted to long-term memory nightly via `/consolidate-memory`.
> Loaded inline at conversation startup via CLAUDE.md.

---

## Last Updated: 2026-04-12

## START-OF-SESSION INSTRUCTIONS (pick up immediately)

**Read `docs/execution_plan.md` first**, then `docs/research_plan.md` §Regime-Invariance Discipline. The research framing changed this session — all claims must now be structural (about mechanisms) not parametric (about levels), and walk-forward validation on post-freeze resolutions is mandatory before reporting a rule as "working."

**Next concrete action: Track B3 — build theme-stratified temporal split on the expanded 5,000-market Gamma dataset.** Data is at `data/raw/gamma_markets.parquet` (5,000 markets) and `data/raw/gamma_snapshots.parquet` (640,450 snapshots). The current `data/processed/{train,val,test}.parquet` is from the old 95-market curated registry and must NOT be overwritten until B3 produces the new split. B3 requires: enrichment of Gamma snapshots, Gamma event-group assignment (already wired in `build_dataset.py`), per-category temporal split (60/20/20), leakage audit on cross-market aggregates, and a split manifest. See `docs/execution_plan.md` §B3 for details.

**Do not re-run `scripts/run_data_pipeline.py` without flags** — that would overwrite the curated splits. Use `--gamma-fetch` for Gamma-only operations.

## Current Focus

- **Track B3** is the critical path. Nothing else (B5 walk-forward, A2 retest on expanded data) can proceed until the new split exists.
- Track A is **stopped at A1**. A2 mechanism was rejected by pre-registered criteria (sign flip on primaries, secondary domination). The mechanism is not falsified — the 2/5/3 market split was underpowered — but the pre-reg correctly forbids A3. The test defers to Track B5 on expanded data.
- Research framing is **regime-invariant structural hypotheses**, not "find where the market is wrong." See `docs/research_plan.md` §Regime-Invariance Discipline and `docs/shutdown_hypothesis_2026-04-11.md` for the pre-registration template.

## Recent Decisions (Last 48hr)

### 2026-04-12

- **Regime-invariance discipline adopted.** User played devil's advocate: "with financial data, can we assume past patterns repeat?" This killed the earlier "local/selection" framing. Reframed to: structural hypotheses about mechanisms (capital constraints, attention limits) that should persist across regimes, validated by walk-forward on post-freeze resolutions. Supersedes the 2026-04-11 local/selection framing. Codified in `docs/research_plan.md` §Regime-Invariance Discipline and a feedback memory at `~/.claude/projects/.../memory/feedback_regime_invariance.md`.
- **A1 completed.** Shutdown characterization found: (1) systematic YES under-pricing (snapshot bias +0.479, CI [+0.441, +0.515]), (2) late convergence (6/10 markets shrink |err| >0.05), (3) concentrated errors (top-2 markets = 56% of error mass), (4) all 4 high-error markets resolved YES, 3 lowest-error resolved NO. Pattern: long-horizon shutdown markets under-price YES.
- **A2 pre-registered and rejected.** Pre-registration at `docs/shutdown_hypothesis_2026-04-11.md` committed the mechanism, 3 primary signals, confirmation/falsification criteria, and mandatory caveats BEFORE running any correlations. Result: `pct_lifetime_elapsed` sign flip, `log_volume` sign flip, `price_vs_open` (secondary) dominated all primaries. Root cause: train = 2 markets, too thin to adjudicate. Track A stopped per pre-reg rules; mechanism deferred to B5.
- **A2 population threshold changed to empirical.** The absolute $100k volume cutoff was invalidated by the dry-run (min volume at $1M floor was $53M). Now: bottom 25% of fetched universe by volume AND max days_to_end > 90. On the actual fetch: Q25 volume = $1.74M, 203 markets qualify.
- **B1 scope trimmed.** `src/data/fetch_gamma.py` already existed (462 lines) — the execution plan was stale. Real work was only: add `gamma_fetch` config section, add Gamma event-group dispatch + `category_encoded` to `build_dataset.py`, add CLI flags to `run_data_pipeline.py`.
- **B1 completed.** Full Gamma fetch: 5,000 markets, 640,450 snapshots, 11 categories, 100% resolution rate. All B1h criteria pass including the A2 empirical population check (203 markets).
- **Execution plan stale lesson saved as feedback memory.** Plans that say "CREATE file X" must be verified against the codebase before acting. Memory at `~/.claude/projects/.../memory/feedback_verify_codebase_before_planning.md`.

### 2026-04-11

- (Carried forward from prior session — see `.claude/sessions/` logs for details)
- Falsified long-horizon correction hypothesis. Structural blocker: near-zero train/test theme overlap.
- Chose Option 3 dual-track. Rewrote `docs/research_plan.md` and created `docs/execution_plan.md`.

## Data State

- `data/raw/markets_registry.parquet`: 95 curated markets, 6 themes (unchanged)
- `data/raw/market_snapshots.parquet`: 25,083 curated snapshots (unchanged)
- `data/raw/gamma_markets.parquet`: **5,000 Gamma markets**, 11 categories (NEW)
- `data/raw/gamma_snapshots.parquet`: **640,450 Gamma snapshots** across 4,911 markets (NEW)
- `data/interim/snapshots_enriched.parquet`: 25,083 enriched curated snapshots (unchanged)
- `data/processed/{train,val,test}.parquet`: curated split (unchanged, FROZEN until B3)

## Open Threads

- **B3 stratified split on expanded data** — critical path, next action
- B2 external features (FRED, calendar, futures) — independent of B3, lower priority
- B5 walk-forward validation — depends on B3
- `other` category = 17% of Gamma markets — regex classifier needs improvement
- 373 sparse markets (<5 snapshots) — filtering decision needed in B3
- `US-current-affairs` category (3 markets) — leaked from Gamma event field, needs normalization
- Subagent Write permissions blocked — needs settings fix before future delegation

## Blockers / Watch Items

- All prior 95-market numeric results become non-comparable once B3 lands. Expect every metric to shift.
- Per-category temporal cutoffs in B3 will create cross-market aggregate leakage risk. Audit mandatory.
- The A2 empirical population (203 markets) may be thin for within-category statistical power. Consider lowering `gamma_fetch.min_volume_usd` below $1M if B5 needs more long-tail markets.
