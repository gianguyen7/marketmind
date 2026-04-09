# MarketMind: Data Expansion & External Feature Engineering Plan

## 1. Context & Motivation

### The Problem
We have **95 markets** across **6 Fed/macro themes** — too narrow to draw generalizable conclusions about prediction market calibration. Our market correction models overfit (RF: train MAE 0.016 → test MAE 0.216) because there's not enough diversity or volume to learn robust patterns. The research question — "where and when are prediction markets miscalibrated?" — requires breadth across domains, not just depth in Fed markets.

### What's Available
Polymarket has **~150,000+ resolved binary markets** across politics, sports, crypto, entertainment, geopolitics, science, and more:

| Offset (by volume) | Volume | Example Market |
|---|---|---|
| 0 | $1,531M | Trump wins 2024 election |
| 500 | $11.7M | NBA Finals, recession, Biden resignation |
| 1,000 | $6.1M | Soccer, elections |
| 2,000 | $3.2M | Sports, entertainment |
| 5,000 | $1.3M | Entertainment, predictions |
| 10,000 | $0.66M | Political events |
| 20,000 | $0.32M | AI, crypto |
| 50,000 | $0.12M | Crypto price predictions |

**Recommended scope**: Top ~5,000 markets by volume (all with >$1M volume). This gives us **50x more markets** with meaningful liquidity, across diverse categories.

### External Data Opportunity
Current features are all derived from Polymarket price data. External data sources can provide signals the crowd doesn't fully price in:

| Source | Effort | Impact | What it gives |
|---|---|---|---|
| **FRED** (Fed economic data) | Low | Very High | Treasury yields, VIX, CPI, unemployment — directly predicts Fed markets |
| **FOMC/Economic calendar** | Very Low | High | `days_to_next_fomc`, data release schedule — explains horizon calibration |
| **Fed Funds futures** (yfinance) | Low | Very High | Institutional rate expectations vs crowd — gold-standard miscalibration signal |
| **GDELT** (news events) | Medium | Medium-High | News tone/volume spikes the crowd may underweight |
| **Google Trends** | Low-Medium | Medium | Attention surges not yet reflected in price |
| **Metaculus** | Medium | Medium-High | Superforecaster vs crowd disagreement |
| **Polymarket on-chain** (The Graph) | Medium | High | Whale trades, volume spikes, bid-ask spreads |

---

## 2. Track A: Expand Market Universe

### 2.1 Data Ingestion Strategy

**Source**: Gamma API (`https://gamma-api.polymarket.com/markets`)

**Approach**: Paginate through all closed markets sorted by volume descending. Filter to binary (Yes/No) markets with volume > $1M.

**API details discovered**:
- Pagination: `limit=100` (max), `offset=N`
- Sort: `order=volumeNum&ascending=false`
- Filter: `closed=true`
- No rate limit encountered in testing
- No `category` field on newer markets — must classify from question text
- Resolution from `outcomePrices`: `["1","0"]` = Yes won, `["0","1"]` = No won
- `clobTokenIds` gives token IDs for price history fetches
- `endDate`, `closedTime`, `startDate` available

**Fields to extract**:
```
conditionId, question, slug, description, outcomes, outcomePrices,
volumeNum, liquidityNum, endDate, closedTime, startDate,
clobTokenIds, negRisk, groupItemTitle, events[0].category,
events[0].title, lastTradePrice
```

**Estimated fetch time**:
- Market metadata: 5,000 markets ÷ 100/page = 50 API calls (~30 seconds)
- Price history: 5,000 markets × 1 call each × 0.4s delay = ~33 minutes
- Total: ~35 minutes for full dataset

### 2.2 Market Classification

Since newer markets lack `category`, we need text-based classification from the `question` and `description` fields.

**Categories** (derived from actual top-volume markets):

| Category | Pattern Examples | Est. Count (>$1M vol) |
|---|---|---|
| **Politics/Elections** | "Will X win", "presidential election", "nominee" | ~800 |
| **Fed/Monetary Policy** | "Fed rate", "FOMC", "interest rate", "bps" | ~200 |
| **Sports** | "NBA", "NFL", "win the", "Super Bowl", "Champions" | ~1,500 |
| **Crypto/Finance** | "Bitcoin", "ETH", "price of", "market cap" | ~500 |
| **Geopolitics** | "war", "invasion", "ceasefire", "sanctions" | ~300 |
| **Entertainment** | "Oscar", "Grammy", "box office", "movie" | ~200 |
| **Government/Policy** | "shutdown", "bill", "executive order", "tariff" | ~300 |
| **Science/Tech** | "AI", "FDA approval", "climate", "space" | ~200 |
| **Other** | Misc predictions | ~900 |

**Implementation**: Regex-based classifier (fast, transparent, auditable) with keyword patterns per category. No ML classification needed for this — the categories are clearly signaled in question text.

### 2.3 Resolution Determination

From Gamma API `outcomePrices` field:
- `["1", "0"]` → resolved_yes = 1 (Yes won)
- `["0", "1"]` → resolved_yes = 0 (No won)
- All zeros or non-binary → skip (voided/ambiguous)

Cross-validate with CLOB API `tokens[].winner` field for a random sample.

### 2.4 Price History Fetching

Use existing CLOB API endpoint (already implemented in `fetch_markets.py`):
```
GET /prices-history?market={yes_token_id}&interval=max&fidelity=720
```

Returns array of `{t: unix_timestamp, p: price}` — same format as current pipeline.

**Key change**: Current code reads token IDs from `configs/markets.yaml`. New code reads from Gamma API's `clobTokenIds` field.

### 2.5 Event Group Assignment

Current approach: hand-curated theme/meeting grouping in `configs/markets.yaml`.

**New approach for 5,000 markets**: Automatic event grouping.
- Markets sharing the same `events[0].id` from Gamma API = same event group
- Fallback: markets with very similar questions and same `endDate` = same group
- Each standalone market = its own group

### 2.6 Pipeline Changes

**New file**: `src/data/fetch_gamma.py`
- `fetch_all_closed_markets()` → paginate Gamma API, filter binary, volume > $1M
- `classify_market(question, description)` → regex category assignment
- `determine_resolution(outcomePrices)` → resolved_yes from prices
- `fetch_price_histories(markets)` → CLOB price history with checkpoint/resume

**Modified files**:
- `configs/data.yaml` — add `gamma_fetch` section with volume threshold, category filters
- `src/data/build_dataset.py` — update `assign_event_groups()` to use Gamma event IDs
- `scripts/run_data_pipeline.py` — add Gamma fetch step before existing pipeline
- `src/data/resolve_outcomes.py` — handle markets without `meeting_date` (non-Fed markets)

**Data flow**:
```
Gamma API (all closed markets)
  → Filter: binary, volume > $1M
  → Classify: regex category from question text
  → data/raw/gamma_markets.parquet (~5,000 markets)

CLOB API (price history per market)
  → Checkpoint/resume (reuse existing pattern)
  → data/raw/gamma_snapshots.parquet (~500K-1M snapshots)

Merge with existing curated registry data
  → data/interim/all_snapshots_enriched.parquet
  → data/processed/train.parquet, val.parquet, test.parquet
```

---

## 3. Track B: External Feature Engineering

### 3.1 FRED Economic Data (Priority 1)

**Package**: `fredapi` (needs install)
**API key**: Free, instant registration at fred.stlouisfed.org
**Rate limit**: 120 requests/minute (more than enough)

**Series to fetch**:

| Series ID | Name | Frequency | Feature Purpose |
|---|---|---|---|
| `DGS2` | 2-Year Treasury Yield | Daily | Near-term rate expectations |
| `DGS10` | 10-Year Treasury Yield | Daily | Long-term rate outlook |
| `T10Y2Y` | 10Y-2Y Spread | Daily | Yield curve slope (recession signal) |
| `VIXCLS` | VIX Index | Daily | Market uncertainty/fear |
| `DFEDTARU` | Fed Funds Target Upper | Daily | Current rate |
| `ICSA` | Initial Unemployment Claims | Weekly | Labor market health |
| `CPIAUCSL` | CPI All Urban Consumers | Monthly | Inflation (Fed's primary focus) |
| `UNRATE` | Unemployment Rate | Monthly | Fed dual mandate |
| `UMCSENT` | Consumer Sentiment | Monthly | Consumer outlook |

**Derived features**:
- `yield_curve_slope`: T10Y2Y value at snapshot date (negative = inverted = recession risk)
- `vix_level`: VIX at snapshot date
- `vix_change_5d`: 5-day VIX momentum (rising fear)
- `rate_level`: Current DFEDTARU (context for rate decision markets)
- `cpi_yoy`: CPI year-over-year change (inflation trajectory)
- `claims_vs_trend`: ICSA vs 4-week moving average (labor market surprise)
- `real_rate`: DFEDTARU - CPI_YoY (real interest rate)

**Implementation**:
- New file: `src/data/fetch_fred.py` — fetch and cache series to `data/external/fred/`
- New file: `src/features/economic_features.py` — join FRED data to snapshots by date, compute derived features
- All features are backward-looking by construction (they use data published before the snapshot date)

### 3.2 FOMC & Economic Calendar (Priority 2)

**Source**: Hardcoded list (8 meetings/year) + FRED release calendar
**No API needed** — just a static list of dates

**Features**:
- `days_to_next_fomc`: Days until next FOMC meeting from snapshot date
- `is_fomc_week`: Binary — snapshot is within 7 days of FOMC meeting
- `fomc_meetings_remaining_year`: How many FOMC meetings left in calendar year
- `days_since_last_fomc`: Days since most recent FOMC decision
- `days_to_next_cpi`: Days until next CPI release
- `is_data_release_week`: Binary — major economic data releasing this week
- `data_releases_before_resolution`: Count of major releases between snapshot and market end date

**Implementation**:
- New file: `src/features/calendar_features.py` — static date lists + lookup functions
- Reference data in `configs/calendar.yaml` (FOMC dates 2020-2027, CPI release dates)

### 3.3 Fed Funds Futures (Priority 3)

**Package**: `yfinance` (already installed)
**No API key needed**

**Tickers**:
- `ZQ=F` — 30-Day Fed Funds Futures (front month implied rate)
- `^IRX` — 13-Week T-Bill Rate (short-term rate proxy)

**Features**:
- `fed_funds_futures_implied_rate`: 100 - ZQ=F price = market-implied Fed funds rate
- `futures_vs_current_rate`: Implied rate - current DFEDTARU (market expects cut/hike?)
- `futures_vs_polymarket`: Compare futures-implied probability to Polymarket price for same FOMC meeting — **this is the gold-standard miscalibration signal**. When institutional money (futures) disagrees with retail crowd (Polymarket), someone is wrong.
- `tbill_momentum_5d`: 5-day change in 13-week T-bill rate

**Implementation**:
- Add to `src/data/fetch_fred.py` or new `src/data/fetch_market_data.py`
- Cache daily data to `data/external/futures/`
- Join to snapshots by date

### 3.4 GDELT News Events (Priority 4)

**Package**: `gdelt-doc-api` (needs install)
**No API key needed**
**Rate limit**: ~1 request/second

**Approach**: For each market category, define keyword queries. Fetch article volume and average tone over trailing 7-day and 30-day windows.

**Features**:
- `news_volume_7d`: Article count matching market keywords in last 7 days
- `news_volume_spike`: Z-score of 7d volume vs 30d trailing average
- `news_tone_7d`: Average GDELT tone score for matching articles (-10 to +10)
- `news_tone_change`: Tone shift (7d avg vs 30d avg) — sentiment momentum
- `news_tone_vs_price`: Divergence between sentiment direction and price movement

**Implementation**:
- New file: `src/data/fetch_gdelt.py` — keyword queries per category, cache results
- New file: `src/features/news_features.py` — compute volume/tone features
- Cache to `data/external/gdelt/`

**Keyword mapping** (per category):
- Fed rate: "federal reserve" OR "FOMC" OR "interest rate"
- Government shutdown: "government shutdown" OR "continuing resolution"
- Geopolitics: market-specific keywords from question text
- Elections: candidate names + "election"

### 3.5 Google Trends (Priority 5)

**Package**: `pytrends` (needs install)
**No API key needed**
**Rate limit**: Aggressive — 10-60s delays needed between requests

**Features**:
- `search_interest_7d`: Google Trends interest index for market keywords
- `search_interest_spike`: Z-score vs 30-day trailing baseline
- `search_vs_price_divergence`: When public attention surges but market price doesn't move

**Implementation**:
- New file: `src/data/fetch_trends.py` — batch keyword queries, cache results
- Cache to `data/external/trends/`
- Run as separate batch job (slow due to rate limits)

### 3.6 Metaculus Superforecaster Data (Priority 6)

**Package**: `forecasting-tools` (needs install)
**API token**: Free from metaculus.com/aib

**Features**:
- `metaculus_prediction`: Community probability for matching question
- `metaculus_vs_polymarket`: Divergence between superforecaster aggregate and crowd price — strong miscalibration signal
- `metaculus_n_forecasters`: Number of forecasters (confidence in Metaculus signal)

**Implementation**:
- New file: `src/data/fetch_metaculus.py` — search for matching questions, fetch predictions
- Matching: fuzzy text matching between Polymarket questions and Metaculus questions
- Cache to `data/external/metaculus/`
- This is the hardest integration — question matching is imperfect

### 3.7 Polymarket On-Chain Data (Priority 7)

**Source**: The Graph GraphQL subgraph
**No API key needed**

**Features**:
- `whale_trade_imbalance_24h`: Net direction of trades > $10K in last 24h
- `volume_spike_zscore`: Unusual volume vs trailing 30-day average
- `bid_ask_spread`: Current spread (wide = low confidence)
- `retail_vs_institutional_ratio`: Many small trades vs few large trades

**Implementation**:
- New file: `src/data/fetch_onchain.py` — GraphQL queries
- Cache to `data/external/onchain/`
- This requires understanding the subgraph schema — medium effort

---

## 4. Implementation Plan

### Phase 1: Market Universe Expansion (highest priority)
**Goal**: Go from 95 → ~5,000 markets across all categories.

| Step | File | Description |
|---|---|---|
| 1.1 | `src/data/fetch_gamma.py` (new) | Gamma API paginator + market classifier + resolution extractor |
| 1.2 | `configs/data.yaml` | Add `gamma_fetch` config section |
| 1.3 | `src/data/fetch_gamma.py` | Price history fetcher with checkpoint/resume (reuse pattern from `fetch_markets.py`) |
| 1.4 | `src/data/resolve_outcomes.py` | Generalize to handle non-Fed markets (no meeting_date) |
| 1.5 | `src/data/build_dataset.py` | Update event group assignment to use Gamma event IDs |
| 1.6 | `scripts/run_data_pipeline.py` | Add Gamma fetch + merge step |
| 1.7 | Run & validate | Fetch markets, build dataset, verify split integrity |

### Phase 2: FRED + Calendar Features (quick wins)
**Goal**: Add economic context features that the crowd may underweight.

| Step | File | Description |
|---|---|---|
| 2.1 | `pip install fredapi` | Install FRED API client |
| 2.2 | `src/data/fetch_fred.py` (new) | Fetch and cache FRED series |
| 2.3 | `src/features/economic_features.py` (new) | Yield curve, VIX, CPI, claims features |
| 2.4 | `configs/calendar.yaml` (new) | FOMC dates 2020-2027, CPI release dates |
| 2.5 | `src/features/calendar_features.py` (new) | days_to_next_fomc, is_fomc_week, etc. |
| 2.6 | `src/features/feature_pipeline.py` | Register new feature sets |
| 2.7 | Run & validate | Retrain models with new features |

### Phase 3: Fed Funds Futures (high-value, low-effort)
**Goal**: Add institutional vs crowd divergence signal.

| Step | File | Description |
|---|---|---|
| 3.1 | `src/data/fetch_market_data.py` (new) | Fetch ZQ=F, ^IRX from yfinance |
| 3.2 | `src/features/economic_features.py` | Add futures-derived features |
| 3.3 | Run & validate | Focus on Fed rate markets — does divergence predict error? |

### Phase 4: GDELT News Sentiment (medium effort)
**Goal**: Add news volume/tone features.

| Step | File | Description |
|---|---|---|
| 4.1 | `pip install gdelt-doc-api` | Install GDELT client |
| 4.2 | `src/data/fetch_gdelt.py` (new) | Keyword-based article queries per category |
| 4.3 | `src/features/news_features.py` (new) | Volume spike, tone, divergence features |
| 4.4 | Run & validate | Test on government shutdown markets (clearest mispricing) |

### Phase 5: Google Trends + Metaculus (stretch goals)

These are lower priority — implement only if Phases 1-4 show promise.

---

## 5. Config Changes

### `configs/data.yaml` additions:
```yaml
gamma_fetch:
  min_volume_usd: 1000000
  binary_only: true
  closed_only: true
  max_markets: 5000
  checkpoint_path: "data/raw/.checkpoint_gamma.json"

fred:
  api_key_env: "FRED_API_KEY"
  series:
    - DGS2
    - DGS10
    - T10Y2Y
    - VIXCLS
    - DFEDTARU
    - ICSA
    - CPIAUCSL
    - UNRATE
    - UMCSENT
  cache_dir: "data/external/fred"

futures:
  tickers:
    - "ZQ=F"
    - "^IRX"
  cache_dir: "data/external/futures"
```

### New `configs/calendar.yaml`:
```yaml
fomc_meetings:
  2024: ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
         "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"]
  2025: ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
         "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"]
  2026: ["2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
         "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"]
```

---

## 6. Updated Feature Sets

```python
FEATURE_SETS = {
    "market_only": [  # existing - unchanged
        "implied_prob", "implied_logit", ...
    ],
    "correction": [  # updated with external features
        # Cross-market structural
        "theme_historical_base_rate", "price_bucket_historical_accuracy",
        "event_group_n_markets", "event_group_price_deviation",
        # Economic context (FRED)
        "yield_curve_slope", "vix_level", "vix_change_5d",
        "rate_level", "cpi_yoy", "claims_vs_trend",
        # Calendar
        "days_to_next_fomc", "is_fomc_week",
        "data_releases_before_resolution",
        # Futures divergence
        "futures_vs_polymarket",
        # Temporal
        "days_to_end", "pct_lifetime_elapsed",
        # Structure
        "category_encoded", "price_extremity", "price_volatility_7",
    ],
    "ensemble": [  # market price + correction signals
        "implied_prob", "implied_logit",
        # All correction features
        ...
    ],
}
```

---

## 7. Files Summary

### New Files (11)
| File | Purpose |
|---|---|
| `src/data/fetch_gamma.py` | Bulk Polymarket market fetcher + classifier |
| `src/data/fetch_fred.py` | FRED economic data fetcher |
| `src/data/fetch_market_data.py` | yfinance futures data fetcher |
| `src/data/fetch_gdelt.py` | GDELT news event fetcher |
| `src/features/economic_features.py` | FRED-derived features |
| `src/features/calendar_features.py` | FOMC/economic calendar features |
| `src/features/news_features.py` | GDELT-derived features |
| `configs/calendar.yaml` | Static FOMC + CPI release dates |
| `configs/categories.yaml` | Market classification regex patterns |
| `data/external/fred/` | FRED data cache (directory) |
| `data/external/futures/` | Futures data cache (directory) |

### Modified Files (7)
| File | Changes |
|---|---|
| `configs/data.yaml` | Add gamma_fetch, fred, futures config sections |
| `requirements.txt` | Add fredapi, gdelt-doc-api |
| `src/data/resolve_outcomes.py` | Generalize for non-Fed markets |
| `src/data/build_dataset.py` | Auto event grouping from Gamma event IDs; category encoding |
| `src/features/feature_pipeline.py` | Register new feature sets, add external feature pipeline step |
| `src/features/market_features.py` | Add category-level features (replacing theme-only) |
| `scripts/run_data_pipeline.py` | Add Gamma fetch + external data steps |

---

## 8. Verification Checklist

### Phase 1 (Market Expansion)
- [ ] Gamma API fetches ≥5,000 closed binary markets with volume > $1M
- [ ] Market classifier assigns category to >95% of markets
- [ ] Resolution determined from outcomePrices for all fetched markets
- [ ] Price history fetched for all markets (with checkpoint/resume)
- [ ] Event group assignment works for non-Fed markets
- [ ] Temporal split integrity passes (no group in multiple splits)
- [ ] Dataset has markets from ≥5 different categories in each split

### Phase 2 (FRED + Calendar)
- [ ] FRED series fetched and cached to `data/external/fred/`
- [ ] Economic features joined to snapshots by date (no future data)
- [ ] Calendar features computed correctly (days_to_next_fomc matches known dates)
- [ ] Market correction model MAE improves vs naive on Fed rate markets

### Phase 3 (Futures)
- [ ] Fed Funds futures data fetched via yfinance
- [ ] `futures_vs_polymarket` feature computed for FOMC markets
- [ ] Feature shows meaningful correlation with market error

### Phase 4 (GDELT)
- [ ] GDELT queries return results for each category's keywords
- [ ] News volume/tone features cached and joined by date
- [ ] Government shutdown markets show improved prediction with news features

---

## 9. Risk Assessment

| Risk | Mitigation |
|---|---|
| Gamma API changes or rate-limits during bulk fetch | Checkpoint/resume; polite delay; can spread across sessions |
| Many sports markets dilute signal | Filter by category or analyze categories separately |
| FRED API key requires registration | Free, instant — just need `.env` setup |
| Price history unavailable for some markets | Track fetch failures; exclude markets with <5 snapshots |
| Category classification errors | Manual review of edge cases; log unclassified markets |
| External features introduce subtle temporal leakage | All features use data available before snapshot date; FRED/yfinance data has publication lag built in |
| Google Trends rate limiting too aggressive | Make it a stretch goal; batch process offline |
