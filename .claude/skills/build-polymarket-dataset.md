---
name: build-polymarket-dataset
description: End-to-end dataset construction from Polymarket API to processed train/test splits
---

# /build-polymarket-dataset

Build or refresh the MarketMind dataset from scratch.

## Steps

1. **Fetch raw data** from Polymarket CLOB API per `configs/data.yaml`
   - Pull resolved binary markets with volume > `min_volume`
   - Save raw JSON responses to `data/raw/markets_{date}.json`

2. **Resolve outcomes**
   - Map each market to binary YES/NO outcome
   - Log any markets with ambiguous resolution
   - Save to `data/interim/resolved_markets.parquet`

3. **Build snapshots**
   - Create time series of price snapshots per market
   - Interval: `snapshot_interval_hours` from config
   - Save to `data/interim/snapshots.parquet`

4. **Engineer features** (call feature pipeline)
   - Run `src/features/` pipeline on interim data
   - Apply temporal train/test split per config cutoffs
   - Save to `data/processed/train.parquet` and `data/processed/test.parquet`

5. **Validate**
   - Assert no future leakage: all train timestamps < `train_cutoff`
   - Assert no duplicate market_ids within splits
   - Print summary: n_markets, n_features, date_range, resolution_rate

## Usage
```
/build-polymarket-dataset              # full rebuild
/build-polymarket-dataset --refresh    # fetch new markets, append
```

## Output
- `data/processed/train.parquet`
- `data/processed/test.parquet`
- `data/interim/dataset_summary.json`
