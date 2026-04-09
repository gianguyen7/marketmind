"""Build the final modeling dataset from enriched market snapshots.

Each row = one market at one point in time.
Target = eventual binary resolution of that market.

Split strategy: event-group temporal split.
- Correlated markets are clustered into event groups (e.g., all markets
  for a single FOMC meeting, all Fed chair nominees).
- All markets in an event group go to the same split — no exceptions.
- Event groups are sorted by resolution date and split 60/20/20.
- Class weights are computed per split to handle imbalance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/data.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich snapshot rows with modeling features."""
    df = df.copy()

    # Ensure numeric types
    numeric_cols = ["volume_usd", "price_yes", "days_to_end", "days_to_meeting",
                    "pct_lifetime_elapsed"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop snapshots missing critical fields
    required = ["condition_id", "resolved_yes", "price_yes", "snapshot_ts"]
    n_before = len(df)
    df = df.dropna(subset=[c for c in required if c in df.columns])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows missing required fields ({n_before} -> {len(df)})")

    # Theme encoding
    if "theme" in df.columns:
        df["theme_encoded"] = df["theme"].astype("category").cat.codes

    # Volume rank (percentile across markets)
    if "volume_usd" in df.columns:
        df["volume_rank"] = df["volume_usd"].rank(pct=True)
        df["log_volume"] = np.log1p(df["volume_usd"])

    # Days-to-end buckets
    if "days_to_end" in df.columns:
        df["days_to_end_bucket"] = pd.cut(
            df["days_to_end"],
            bins=[0, 1, 7, 30, 90, 365, np.inf],
            labels=["<1d", "1d-1w", "1w-1m", "1m-3m", "3m-1y", ">1y"],
        )

    # Market price as explicit feature name
    if "price_yes" in df.columns:
        df["market_price"] = df["price_yes"]

    # Market error targets: where is the market wrong?
    if "price_yes" in df.columns and "resolved_yes" in df.columns:
        df["market_error"] = df["resolved_yes"] - df["price_yes"]
        df["abs_market_error"] = df["market_error"].abs()
        mean_err = df["market_error"].mean()
        mean_abs = df["abs_market_error"].mean()
        print(f"  Market error: mean={mean_err:+.4f}, mean_abs={mean_abs:.4f}")

    # Binary: number of outcomes always 2
    df["num_outcomes"] = 2

    return df


# ---------------------------------------------------------------------------
# Event group assignment
# ---------------------------------------------------------------------------

_PROXIMITY_DAYS = 30  # markets within this window = same event


def assign_event_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each market to an event group based on theme and correlation structure.

    Rules:
    - fed_rate_decisions: group by meeting_date (mutually exclusive outcomes)
    - fed_leadership: nominees sharing end_date are one group; others standalone
    - government_shutdown: cluster by end_date proximity (same shutdown event)
    - fed_rate_annual: group by end_date (same-year rate count markets)
    - Single-market themes: each market = own group

    Adds 'event_group' and 'event_group_end' columns.
    """
    df = df.copy()

    # Build market-level metadata (one row per market)
    market_meta = (
        df.groupby("condition_id")
        .agg(
            theme=("theme", "first"),
            end_date=("end_date", "first"),
            meeting_date=("meeting_date", "first"),
            question=("question", "first"),
        )
        .reset_index()
    )

    # Parse dates
    for col in ["end_date", "meeting_date"]:
        market_meta[col] = pd.to_datetime(market_meta[col], errors="coerce", utc=True)

    group_map = {}  # condition_id -> event_group name

    for theme in market_meta["theme"].unique():
        theme_markets = market_meta[market_meta["theme"] == theme].copy()

        if theme == "fed_rate_decisions":
            # Group by FOMC meeting date
            for mtg_date, group in theme_markets.groupby("meeting_date"):
                label = f"fomc_{mtg_date.strftime('%Y_%m_%d')}"
                for cid in group["condition_id"]:
                    group_map[cid] = label

        elif theme == "fed_leadership":
            # Nominees sharing end_date = one correlated event
            for end_dt, group in theme_markets.groupby("end_date"):
                if len(group) > 1:
                    label = f"fed_chair_nominees_{end_dt.strftime('%Y')}"
                else:
                    # Standalone (Lisa Cook, Powell out)
                    q = group["question"].iloc[0][:40].replace(" ", "_").lower()
                    label = f"fed_leadership_{q}"
                for cid in group["condition_id"]:
                    group_map[cid] = label

        elif theme in ("government_shutdown", "fed_rate_annual"):
            # Cluster by end_date proximity
            _assign_proximity_groups(theme_markets, group_map, theme)

        else:
            # Single-market themes: each market is its own group
            for _, row in theme_markets.iterrows():
                q = row["question"][:30].replace(" ", "_").lower()
                group_map[row["condition_id"]] = f"{theme}_{q}"

    # Map back to full dataframe
    df["event_group"] = df["condition_id"].map(group_map)

    # Event group resolution date = max end_date of markets in the group
    group_end = (
        df.groupby("event_group")["end_date"]
        .first()  # all markets in group share similar end_date
        .reset_index()
        .rename(columns={"end_date": "event_group_end"})
    )
    # Use max end_date per group for proper ordering
    group_end = (
        df.groupby("event_group")
        .agg(event_group_end=("end_date", "max"))
        .reset_index()
    )
    df = df.merge(group_end, on="event_group", how="left")

    n_groups = df["event_group"].nunique()
    n_markets = df["condition_id"].nunique()
    print(f"  Assigned {n_markets} markets to {n_groups} event groups")

    return df


def _assign_proximity_groups(
    theme_markets: pd.DataFrame, group_map: dict, theme: str,
) -> None:
    """Cluster markets by end_date proximity within a theme."""
    sorted_markets = theme_markets.sort_values("end_date").reset_index(drop=True)
    cluster_id = 0
    cluster_start = sorted_markets["end_date"].iloc[0]
    current_cluster = []

    for _, row in sorted_markets.iterrows():
        if (row["end_date"] - cluster_start).days > _PROXIMITY_DAYS and current_cluster:
            # Finish current cluster
            label = f"{theme}_c{cluster_id}_{cluster_start.strftime('%Y_%m')}"
            for cid in current_cluster:
                group_map[cid] = label
            cluster_id += 1
            cluster_start = row["end_date"]
            current_cluster = [row["condition_id"]]
        else:
            current_cluster.append(row["condition_id"])

    # Last cluster
    if current_cluster:
        label = f"{theme}_c{cluster_id}_{cluster_start.strftime('%Y_%m')}"
        for cid in current_cluster:
            group_map[cid] = label


# ---------------------------------------------------------------------------
# Event-group temporal split
# ---------------------------------------------------------------------------

def split_event_group_temporal(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by event group, ordered by resolution date.

    All markets in an event group go to the same split. Groups are sorted
    by their resolution date (max end_date), then allocated by group count
    (not snapshot count) to ensure each split gets meaningful groups even
    when group sizes are highly unequal.
    """
    # Build group-level summary
    group_summary = (
        df.groupby("event_group")
        .agg(
            event_group_end=("event_group_end", "first"),
            n_snapshots=("snapshot_ts", "count"),
            n_markets=("condition_id", "nunique"),
            themes=("theme", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values("event_group_end")
    )

    n_groups = len(group_summary)
    n_train = max(1, round(n_groups * train_frac))
    n_val = max(1, round(n_groups * val_frac))
    # Ensure at least 1 group in test
    if n_train + n_val >= n_groups:
        n_val = max(1, n_groups - n_train - 1)

    group_splits = {}
    for i, (_, row) in enumerate(group_summary.iterrows()):
        if i < n_train:
            group_splits[row["event_group"]] = "train"
        elif i < n_train + n_val:
            group_splits[row["event_group"]] = "val"
        else:
            group_splits[row["event_group"]] = "test"

    print(f"\n  Group allocation: {n_train} train / {n_val} val / "
          f"{n_groups - n_train - n_val} test (of {n_groups} total)")

    df["_split"] = df["event_group"].map(group_splits)

    train = df[df["_split"] == "train"].drop(columns=["_split"])
    val = df[df["_split"] == "val"].drop(columns=["_split"])
    test = df[df["_split"] == "test"].drop(columns=["_split"])

    # Report group assignments
    print(f"\n  Event group assignments (sorted by resolution date):")
    print(f"  {'group':<45} {'split':<6} {'markets':>7} {'snaps':>7} {'end_date':>12}")
    print(f"  {'-'*80}")
    for _, row in group_summary.iterrows():
        s = group_splits[row["event_group"]]
        end = str(row["event_group_end"])[:10]
        print(f"  {row['event_group']:<45} {s:<6} {row['n_markets']:>7} "
              f"{row['n_snapshots']:>7} {end:>12}")

    # Split summary
    print(f"\n  {'Split':<7} {'Rows':>7} {'Markets':>8} {'Groups':>7} {'Themes':>7} {'Base Rate':>10}")
    print(f"  {'-'*50}")
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        n_markets = split_df["condition_id"].nunique()
        n_groups = split_df["event_group"].nunique()
        n_themes = split_df["theme"].nunique() if "theme" in split_df.columns else 0
        base_rate = split_df["resolved_yes"].mean() if len(split_df) > 0 else 0
        print(f"  {name:<7} {len(split_df):>7} {n_markets:>8} {n_groups:>7} "
              f"{n_themes:>7} {base_rate:>10.3f}")

    # Theme breakdown
    print(f"\n  Theme representation per split:")
    all_themes = sorted(df["theme"].unique())
    print(f"  {'theme':<25} {'train':>7} {'val':>7} {'test':>7}")
    print(f"  {'-'*49}")
    for theme in all_themes:
        t = len(train[train["theme"] == theme])
        v = len(val[val["theme"] == theme])
        te = len(test[test["theme"] == theme])
        marker = " ***" if (t == 0 or te == 0) else ""
        print(f"  {theme:<25} {t:>7} {v:>7} {te:>7}{marker}")

    # Temporal integrity check
    _check_temporal_integrity(train, val, test)

    return train, val, test


def _check_temporal_integrity(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
) -> None:
    """Verify no event group is split across boundaries."""
    all_df = pd.concat([
        train.assign(_split="train"),
        val.assign(_split="val"),
        test.assign(_split="test"),
    ])

    splits_per_group = all_df.groupby("event_group")["_split"].nunique()
    leaked = splits_per_group[splits_per_group > 1]
    if len(leaked):
        print(f"\n  INTEGRITY FAILURE: {len(leaked)} event groups split across boundaries!")
        for g in leaked.index:
            splits = all_df[all_df["event_group"] == g]["_split"].unique()
            print(f"    {g}: {splits}")
    else:
        print(f"\n  Temporal integrity: PASSED (all {splits_per_group.shape[0]} "
              f"event groups contained within single splits)")

    # Check ordering
    if len(train) > 0 and len(test) > 0:
        train_max_end = train["event_group_end"].max()
        test_min_end = test["event_group_end"].min()
        if train_max_end > test_min_end:
            print(f"  WARNING: train group end ({train_max_end}) > test group start ({test_min_end})")
            print(f"  (This can happen when snapshot collection periods overlap resolution dates)")
        else:
            print(f"  Temporal order: PASSED (train ends {str(train_max_end)[:10]} "
                  f"< test starts {str(test_min_end)[:10]})")


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    target_col: str = "resolved_yes",
) -> dict:
    """Compute inverse-frequency class weights per split and per theme.

    Returns a dict with:
    - global: {0: weight, 1: weight} based on train set
    - per_theme: {theme: {0: weight, 1: weight}} based on train set
    - split_base_rates: {train: rate, val: rate, test: rate}
    """
    weights = {"split_base_rates": {}}

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        if len(split_df) > 0:
            weights["split_base_rates"][name] = float(split_df[target_col].mean())

    # Global weights from train set
    train_counts = train[target_col].value_counts()
    n_total = len(train)
    n_classes = len(train_counts)
    weights["global"] = {}
    for cls, count in train_counts.items():
        weights["global"][int(cls)] = round(n_total / (n_classes * count), 4)
    print(f"\n  Class weights (from train): {weights['global']}")
    print(f"  Interpretation: class 1 (yes) gets {weights['global'].get(1, 0):.2f}x "
          f"weight vs class 0 (no)")

    # Per-theme weights from train set
    weights["per_theme"] = {}
    for theme in sorted(train["theme"].unique()):
        theme_df = train[train["theme"] == theme]
        theme_counts = theme_df[target_col].value_counts()
        n_theme = len(theme_df)
        n_cls = len(theme_counts)
        theme_weights = {}
        for cls, count in theme_counts.items():
            theme_weights[int(cls)] = round(n_theme / (n_cls * count), 4)
        weights["per_theme"][theme] = theme_weights

    return weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    input_path: str = "data/interim/snapshots_enriched.parquet",
    config_path: str = "configs/data.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and save the modeling dataset with event-group temporal splits."""
    cfg = load_config(config_path)
    ds = cfg["dataset"]

    df = pd.read_parquet(input_path)
    n_markets = df["condition_id"].nunique() if "condition_id" in df.columns else "?"
    print(f"Loaded {len(df)} snapshots across {n_markets} markets")

    df = build_modeling_dataset(df)
    df = assign_event_groups(df)

    split_cfg = ds.get("split", {})
    train_frac = split_cfg.get("train_frac", 0.6)
    val_frac = split_cfg.get("val_frac", 0.2)

    train, val, test = split_event_group_temporal(df, train_frac, val_frac)

    # Compute and save class weights
    weights = compute_class_weights(train, val, test, ds.get("target_col", "resolved_yes"))

    out_dir = Path(cfg["data_pipeline"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    weights_path = out_dir / "class_weights.json"
    weights_path.write_text(json.dumps(weights, indent=2))
    print(f"\nSaved splits to {out_dir}")
    print(f"Saved class weights to {weights_path}")

    return train, val, test


if __name__ == "__main__":
    run()
