# Recent Memory

> Rolling 48-hour context window. Updated after each conversation.
> Important items get promoted to long-term memory nightly via `/consolidate-memory`.
> Loaded inline at conversation startup via CLAUDE.md.

---

## Last Updated: 2026-04-15

## START-OF-SESSION INSTRUCTIONS (pick up immediately)

**Next action: polish the Streamlit dashboard for a non-technical audience.** The app at `src/dashboard/app.py` is functional but assumes the viewer knows what Brier scores, ECE, and reliability decomposition mean. Rewrite with:
- A narrative story: "We asked if prediction markets can be beaten → here's what we found"
- Plain-English explanations for every metric (tooltips, expandable sections)
- Intuitive visuals before tables
- A guided intro/walkthrough page
- Analogies and everyday language instead of jargon

**Run:** `streamlit run src/dashboard/app.py`

**After dashboard polish: C3 research write-up.**

## Current Focus

- **Dashboard polish (C2)** — critical path. Make it accessible to someone with no forecasting background.
- After C2: **C3** (research write-up as standalone report).
- All research tracks (A, B, C1, C4) are **complete**. The "market is efficient" conclusion is definitive.

## Recent Decisions (Last 48hr)

### 2026-04-15

- **CLOB v2 migration caused price history truncation.** The CLOB API now only serves 31 days of price history for active markets. Closed markets return empty. Root cause: Polymarket's exchange overhaul started April 6, 2026. No official documentation of this change.
- **Our dataset is irreplaceable through current APIs.** 640K snapshots spanning Nov 2022 – Apr 2026. 1,879 markets have >31-day spans. Cannot be re-fetched.
- **Dashboard built (C2 first pass).** 5 pages: Overview, Calibration Explorer, F-L Bias, Exploitation Attempts, Data Deep Dive. All data loads from existing CSV/JSON artifacts.
- **Next session: narrative polish for non-technical audience.** Decided after reviewing the current dashboard — it's functional but jargon-heavy.
- **Documented API data availability.** `docs/polymarket_data_availability.md` — comprehensive reference on what the APIs serve now vs. what we captured.

### 2026-04-12

- (Carried forward) Polymarket is efficient. Four exploitation attempts failed. F-L bias is real but too small. Pivot to characterization/packaging.

## Data State

- `data/raw/gamma_markets.parquet`: 5,000 markets, 11 categories (irreplaceable — Gamma API purged older data)
- `data/raw/gamma_snapshots.parquet`: 640,450 snapshots across 4,911 markets (irreplaceable — CLOB now 31-day window only)
- `data/processed/{train,val,test}.parquet`: B3 expanded split (4,538 markets, 639K snapshots)
- All output artifacts in `outputs/tables/` and `outputs/figures/` are intact and loaded by the dashboard.

## Open Threads

- **Dashboard narrative polish** — next action
- C3 research write-up — after dashboard
- Adapt fetch pipeline for CLOB v2 if future data needed (use startTs/endTs with 15-day chunks, active markets only)

## Blockers / Watch Items

- Will CLOB v2 restore historical data access? Unknown. No official word.
- `streamlit` and `plotly` are installed but not in `requirements.txt` yet.
- Entertainment has only 9 test markets — per-category findings there remain fragile.
