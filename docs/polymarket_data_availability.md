# Polymarket API Data Availability & Retention

> Investigated 2026-04-15. Documents what the Polymarket APIs serve today,
> what we captured before the CLOB v2 migration, and why the two differ.

---

## 1. Our Dataset (captured ~April 9, 2026)

### Market Metadata (`data/raw/gamma_markets.parquet`)

| Field | Value |
|-------|-------|
| Total markets | 5,000 (top by volume, closed, binary) |
| Volume floor | $1M USD |
| Earliest market start | 2022-01-11 |
| Latest market start | 2026-04-08 |
| Categories | 11 (sports 45%, other 5%, politics 14%, crypto 11%, ...) |

**Markets by start year:**

| Year | Markets | Total Volume | Median Volume | Notes |
|------|---------|-------------|---------------|-------|
| 2022 | 2 | $15M | $7.3M | Only 2 survive the $1M floor |
| 2023 | 13 | $70M | $3.6M | Pre-CLOB era, very thin |
| 2024 | 1,101 | $17.4B | $4.8M | US election year explosion |
| 2025 | 2,551 | $14.1B | $2.3M | Sustained growth, lower per-market volume |
| 2026 | 1,323 | $5.4B | $2.3M | Partial year (Q1 + early Q2) |

**Markets by quarter:**

| Quarter | Markets | Total Volume |
|---------|---------|-------------|
| 2022Q1 | 2 | $15M |
| 2023Q1 | 3 | $17M |
| 2023Q2 | 2 | $6M |
| 2023Q3 | 3 | $28M |
| 2023Q4 | 5 | $19M |
| 2024Q1 | 180 | $5,487M |
| 2024Q2 | 58 | $448M |
| 2024Q3 | 390 | $7,107M |
| 2024Q4 | 473 | $4,320M |
| 2025Q1 | 407 | $2,240M |
| 2025Q2 | 440 | $3,665M |
| 2025Q3 | 458 | $3,677M |
| 2025Q4 | 1,246 | $4,476M |
| 2026Q1 | 1,283 | $5,268M |
| 2026Q2 | 40 | $90M |

### Price History Snapshots (`data/raw/gamma_snapshots.parquet`)

| Field | Value |
|-------|-------|
| Total snapshots | 640,450 |
| Markets with data | 4,911 (of 5,000; 89 returned no history) |
| Earliest snapshot | 2022-11-18 |
| Latest snapshot | 2026-04-09 |
| Median snapshots/market | ~130 |
| Fetch method | CLOB `/prices-history`, `interval=max`, `fidelity=720` |
| Source field | All rows: `snapshot_source = "api"` |

**Snapshot span statistics:**

| Metric | Value |
|--------|-------|
| Markets with span > 31 days | 1,879 / 4,911 (38%) |
| Markets with span > 365 days | 43 / 4,911 |
| Maximum span | 674 days |
| Median span | 15 days |
| Longest market | "Which party wins 2024 US Presidential Election?" (Jan 2023 – Nov 2024) |

The `fidelity=720` parameter returned up to 720 evenly-spaced price points spanning
each market's full lifetime — approximately 2 points/day for long-running markets.

### Volume by Category and Year ($M)

| Category | 2022 | 2023 | 2024 | 2025 | 2026 |
|----------|------|------|------|------|------|
| sports | 0 | 0 | 6,339 | 4,589 | 2,586 |
| politics_elections | 15 | 43 | 8,639 | 2,228 | 196 |
| crypto_finance | 0 | 18 | 666 | 1,124 | 482 |
| geopolitics | 0 | 0 | 311 | 961 | 1,619 |
| fed_monetary_policy | 0 | 4 | 715 | 2,781 | 0 |
| entertainment | 0 | 0 | 252 | 450 | 26 |
| science_tech | 0 | 4 | 147 | 576 | 62 |
| recession_economy | 0 | 0 | 68 | 89 | 95 |
| other | 0 | 2 | 205 | 699 | 71 |
| social_media | 0 | 0 | 12 | 240 | 188 |
| government_policy | 0 | 0 | 7 | 320 | 32 |

---

## 2. Current API State (tested 2026-04-15)

### Gamma API (`gamma-api.polymarket.com/markets`)

| Query | Earliest Available |
|-------|--------------------|
| Closed markets, sorted by startDate asc | **2021-01-18** (full historical index) |
| Active markets, sorted by startDate asc | **2025-05-02** |
| Active markets, sorted by createdAt asc | **2025-05-02** |

- Market metadata (condition IDs, questions, volumes, dates, clobTokenIds) is intact for closed markets going back to January 2021.
- Offset-based pagination caps at ~1,000 results. New `GET /markets/keyset` endpoint uses cursor-based pagination (released April 10, 2026).
- As of April 9, 2026, the `closed` query parameter defaults to `false` — closed markets are excluded unless explicitly requested.

### CLOB API (`clob.polymarket.com/prices-history`)

| Query | Result |
|-------|--------|
| Active market, `interval=max`, `fidelity=100` | Returns data for **last ~31 days only** |
| Active market, `interval=1w`, `fidelity=10` | Returns last 7 days |
| Closed market (any params) | **Empty** `{"history": []}` |
| Recently closed market (hours old) | Brief data availability, then purged |
| `startTs`/`endTs` with range > ~30 days | Error: `"interval is too long"` |
| `interval=max` without `startTs`/`endTs` | Returns last ~31 days (no error, but truncated) |

**Confirmed behavior on April 15, 2026:**
- Every active market tested, regardless of start date, shows a data start of **2026-03-15** (~31 days ago).
- Closed/resolved markets return zero price history.
- The `interval=max` parameter still works syntactically but is silently capped at ~31 days.

### Supported Parameters (from official docs)

| Parameter | Description |
|-----------|-------------|
| `market` (required) | Token ID (asset ID) to query |
| `startTs` | Filter by items after this unix timestamp |
| `endTs` | Filter by items before this unix timestamp |
| `interval` | Time aggregation: `max`, `all`, `1m`, `1w`, `1d`, `6h`, `1h` |
| `fidelity` | Accuracy in minutes (default: 1 minute) |

The official docs do **not** document:
- The 31-day retention limit
- Behavior for closed/resolved markets
- Any deprecation of `interval=max`

---

## 3. What Changed and Why

### Timeline

| Date | Event |
|------|-------|
| 2025-12-22 | GitHub issue [#216](https://github.com/Polymarket/py-clob-client/issues/216) filed: `/prices-history` returns empty for resolved markets at sub-12h granularity. Workaround: use `startTs`/`endTs` in 15-day chunks. |
| 2026-03-30 | Fee structure V2 implementation |
| 2026-04-06 | **Polymarket announces full exchange overhaul**: CTF Exchange V2, CLOB v2, new collateral token (Polymarket USD replacing USDC.e). Rollout over 2-3 weeks. |
| 2026-04-07 | Press coverage of the overhaul ([Blockhead](https://www.blockhead.co/2026/04/07/polymarket-overhauls-exchange-stack-with-new-contracts-order-book-collateral-token/), [CoinDesk](https://www.coindesk.com/markets/2026/04/06/polymarket-reveals-a-full-exchange-upgrade-to-take-control-of-its-own-trading-and-truth/), [Bitcoin Magazine](https://bitcoinmagazine.com/news/polymarket-unveils-exchange-overhaul)) |
| 2026-04-09 | Gamma API changelog: `closed` defaults to `false`. Our last snapshot captured this date. |
| ~2026-04-09 | **Our fetch completed.** 640K snapshots with full historical depth via CLOB v1. |
| 2026-04-10 | Keyset pagination endpoints released (replacing offset-based) |
| 2026-04-15 | **Tested today.** CLOB returns max 31 days, closed markets return empty. CLOB v2 appears to be live. |

### Root Cause (assessed, not officially confirmed)

The CLOB v1-to-v2 migration that began April 6, 2026 is the most likely cause. The new CLOB v2 backend either:

1. Did not migrate historical price data from v1, or
2. Implemented a stricter retention policy (31-day rolling window), or
3. Both — historical data was not carried over, and new data is retained for only ~31 days.

No official announcement, press release, or changelog entry documents this change.
The December 2025 GitHub issue already showed signs of data fragility for resolved
markets, suggesting retention was tightening before the v2 migration.

---

## 4. Implications for This Project

### What we have is irreplaceable through these APIs

Our `gamma_snapshots.parquet` contains 640K price points spanning Nov 2022 – Apr 2026
with full market lifetimes. This data cannot be re-fetched from the CLOB API today.
A fresh fetch would yield at most 31 days of data for active markets and nothing for
the 5,000 closed markets in our dataset.

### Alternative sources for historical data (not yet explored)

- **On-chain data**: Polymarket runs on Polygon. Trade-level data exists in blockchain
  transactions and could be reconstructed from event logs on the CTF Exchange contract.
- **Third-party archives**: Services like [PolymarketData](https://www.polymarketdata.co/)
  may maintain independent historical snapshots.
- **Community datasets**: Kaggle, HuggingFace, or academic repositories may have
  historical Polymarket data exports.

### Data freshness risk

If the project needs to expand the dataset with new markets in the future, the fetch
pipeline (`src/data/fetch_gamma.py`) will need to be adapted:
- Use `startTs`/`endTs` with chunked requests (15-day windows) instead of `interval=max`
- Accept that only active markets will return price history
- Consider capturing data continuously rather than in batch to avoid losing history
  when markets close

---

## 5. Sources

- [Polymarket Official Changelog](https://docs.polymarket.com/changelog)
- [Polymarket CLOB Timeseries Documentation](https://docs.polymarket.com/developers/CLOB/timeseries)
- [py-clob-client Issue #216: Empty data for resolved markets](https://github.com/Polymarket/py-clob-client/issues/216)
- [Blockhead: Polymarket Overhauls Exchange Stack (2026-04-07)](https://www.blockhead.co/2026/04/07/polymarket-overhauls-exchange-stack-with-new-contracts-order-book-collateral-token/)
- [CoinDesk: Polymarket Full Exchange Upgrade (2026-04-06)](https://www.coindesk.com/markets/2026/04/06/polymarket-reveals-a-full-exchange-upgrade-to-take-control-of-its-own-trading-and-truth/)
- [Bitcoin News: Polymarket April 2026 Upgrade](https://news.bitcoin.com/polymarkets-april-2026-upgrade-new-stablecoin-faster-order-matching-smart-contract-wallet-support/)
- [The Block: Polymarket Trading Engine Overhaul](https://www.theblock.co/post/396450/polymarket-unveils-plans-trading-engine-overhaul-native-stablecoin)
