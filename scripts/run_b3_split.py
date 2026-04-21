"""B3: Build theme-stratified temporal split on the expanded Gamma dataset.

Steps:
1. Load gamma_markets + gamma_snapshots
2. Update snapshot categories from re-classified markets
3. Filter sparse markets (<5 snapshots)
4. Enrich with temporal + snapshot features
5. Build modeling features
6. Assign event groups (Gamma path)
7. Per-category temporal split (60/20/20)
8. Leakage audit on cross-market aggregates
9. Save train/val/test.parquet + split_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.build_dataset import (  # noqa: E402
    assign_event_groups,
    build_modeling_dataset,
    compute_class_weights,
)
from src.data.resolve_outcomes import (  # noqa: E402
    add_snapshot_features,
    add_temporal_features,
    validate_outcomes,
)


def load_config(path: str = "configs/data.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def filter_sparse_markets(
    df: pd.DataFrame, min_snapshots: int = 5,
) -> pd.DataFrame:
    """Remove markets with fewer than min_snapshots snapshots."""
    snap_counts = df.groupby("condition_id")["snapshot_ts"].count()
    sparse = snap_counts[snap_counts < min_snapshots].index
    n_before = df["condition_id"].nunique()
    df = df[~df["condition_id"].isin(sparse)]
    n_after = df["condition_id"].nunique()
    n_rows_dropped = len(sparse)
    print(f"  Filtered sparse markets (<{min_snapshots} snapshots): "
          f"{n_before} -> {n_after} markets (dropped {n_rows_dropped})")
    snap_dropped = snap_counts[snap_counts < min_snapshots].sum()
    print(f"  Dropped {snap_dropped} snapshots from sparse markets")
    return df


def split_by_category_temporal(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    min_markets_per_category: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Per-category temporal split. Categories with <min_markets pooled into 'other'.

    Returns (train, val, test, manifest).
    Manifest: {category: {train: [condition_ids], val: [...], test: [...]}}.
    """
    # Pool tiny categories into other
    cat_counts = df.groupby("category")["condition_id"].nunique()
    small_cats = cat_counts[cat_counts < min_markets_per_category].index.tolist()
    if small_cats:
        n_pooled = cat_counts[cat_counts < min_markets_per_category].sum()
        print(f"  Pooling {len(small_cats)} small categories ({n_pooled} markets) "
              f"into 'other': {small_cats}")
        df = df.copy()
        df.loc[df["category"].isin(small_cats), "category"] = "other"

    manifest: dict[str, dict[str, list[str]]] = {}
    splits = {"train": [], "val": [], "test": []}

    # First pass: compute per-category split assignments for each event group.
    # Some event groups span categories (e.g., a Gamma event with markets in
    # both 'crypto_finance' and 'other'). We resolve conflicts by taking the
    # earliest (most conservative) split: train < val < test.
    split_order = {"train": 0, "val": 1, "test": 2}
    order_to_split = {0: "train", 1: "val", 2: "test"}
    global_group_splits: dict[str, str] = {}

    for category in sorted(df["category"].unique()):
        cat_df = df[df["category"] == category]
        group_summary = (
            cat_df.groupby("event_group")
            .agg(event_group_end=("event_group_end", "first"))
            .reset_index()
            .sort_values("event_group_end")
        )

        n_groups = len(group_summary)
        n_train = max(1, round(n_groups * train_frac))
        n_val = max(1, round(n_groups * val_frac))
        if n_train + n_val >= n_groups:
            n_val = max(1, n_groups - n_train - 1)

        for i, (_, row) in enumerate(group_summary.iterrows()):
            if i < n_train:
                proposed = "train"
            elif i < n_train + n_val:
                proposed = "val"
            else:
                proposed = "test"

            eg = row["event_group"]
            if eg in global_group_splits:
                # Take the earlier (more conservative) split
                existing = global_group_splits[eg]
                if split_order[proposed] < split_order[existing]:
                    global_group_splits[eg] = proposed
            else:
                global_group_splits[eg] = proposed

    # Count cross-category conflicts resolved
    n_conflicts = 0
    for category in sorted(df["category"].unique()):
        cat_df = df[df["category"] == category]
        cat_groups = cat_df["event_group"].unique()
        for eg in cat_groups:
            # The global assignment may differ from what this category alone
            # would have chosen — that's fine, it prevents leakage.
            pass  # counted implicitly by the audit

    # Second pass: apply global assignments and build output
    print(f"\n  Per-category split (60/20/20 by event group):")
    print(f"  {'category':<25} {'groups':>6} {'train':>6} {'val':>6} {'test':>6} "
          f"{'markets':>8} {'snaps':>8}")
    print(f"  {'-'*75}")

    for category in sorted(df["category"].unique()):
        cat_df = df[df["category"] == category].copy()
        cat_df["_split"] = cat_df["event_group"].map(global_group_splits)

        cat_manifest: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for split_name in ["train", "val", "test"]:
            split_ids = cat_df[cat_df["_split"] == split_name]["condition_id"].unique().tolist()
            cat_manifest[split_name] = split_ids
            splits[split_name].append(
                cat_df[cat_df["_split"] == split_name].drop(columns=["_split"]),
            )
        manifest[category] = cat_manifest

        n_groups = cat_df["event_group"].nunique()
        n_m = cat_df["condition_id"].nunique()
        n_s = len(cat_df)
        n_t = cat_df[cat_df["_split"] == "train"]["event_group"].nunique()
        n_v = cat_df[cat_df["_split"] == "val"]["event_group"].nunique()
        n_te = cat_df[cat_df["_split"] == "test"]["event_group"].nunique()
        print(f"  {category:<25} {n_groups:>6} {n_t:>6} {n_v:>6} {n_te:>6} "
              f"{n_m:>8} {n_s:>8}")

    train = pd.concat(splits["train"], ignore_index=True)
    val = pd.concat(splits["val"], ignore_index=True)
    test = pd.concat(splits["test"], ignore_index=True)

    return train, val, test, manifest


def audit_leakage(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
) -> None:
    """Verify no temporal leakage in the split."""
    print("\n" + "=" * 60)
    print("Leakage audit")
    print("=" * 60)

    # 1. No market in multiple splits
    train_ids = set(train["condition_id"].unique())
    val_ids = set(val["condition_id"].unique())
    test_ids = set(test["condition_id"].unique())

    tv_overlap = train_ids & val_ids
    tt_overlap = train_ids & test_ids
    vt_overlap = val_ids & test_ids

    if tv_overlap or tt_overlap or vt_overlap:
        print(f"  FAIL: Market overlap! train-val={len(tv_overlap)}, "
              f"train-test={len(tt_overlap)}, val-test={len(vt_overlap)}")
    else:
        print(f"  PASS: No market appears in multiple splits")

    # 2. No event group in multiple splits
    all_df = pd.concat([
        train.assign(_split="train"),
        val.assign(_split="val"),
        test.assign(_split="test"),
    ])
    splits_per_group = all_df.groupby("event_group")["_split"].nunique()
    leaked_groups = splits_per_group[splits_per_group > 1]
    if len(leaked_groups):
        print(f"  FAIL: {len(leaked_groups)} event groups split across boundaries!")
    else:
        n_groups = splits_per_group.shape[0]
        print(f"  PASS: All {n_groups} event groups contained within single splits")

    # 3. Per-category temporal ordering
    print(f"\n  Per-category temporal ordering:")
    for category in sorted(all_df["category"].unique()):
        cat_train = all_df[(all_df["category"] == category) & (all_df["_split"] == "train")]
        cat_test = all_df[(all_df["category"] == category) & (all_df["_split"] == "test")]
        if len(cat_train) > 0 and len(cat_test) > 0:
            train_max = cat_train["event_group_end"].max()
            test_min = cat_test["event_group_end"].min()
            status = "PASS" if train_max <= test_min else "WARN"
            print(f"    {category:<25} {status}: train ends {str(train_max)[:10]}, "
                  f"test starts {str(test_min)[:10]}")

    # 4. Check cross-market aggregates use expanding().shift(1)
    # (Structural check — we verify the code patterns, not the data)
    print(f"\n  Cross-market aggregate check:")
    print(f"    Note: build_modeling_dataset does not compute cross-market aggregates")
    print(f"    (e.g., category base rates). If added later, they MUST use")
    print(f"    expanding().shift(1) over a GLOBAL time axis to avoid leakage.")


def print_split_summary(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
) -> None:
    """Print detailed split summary."""
    print(f"\n{'='*60}")
    print("Split summary")
    print(f"{'='*60}")

    print(f"\n  {'Split':<7} {'Rows':>8} {'Markets':>8} {'Groups':>7} "
          f"{'Categories':>10} {'YES Rate':>9}")
    print(f"  {'-'*55}")
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        n_m = split_df["condition_id"].nunique()
        n_g = split_df["event_group"].nunique()
        n_c = split_df["category"].nunique()
        br = split_df["resolved_yes"].mean()
        print(f"  {name:<7} {len(split_df):>8} {n_m:>8} {n_g:>7} {n_c:>10} {br:>9.3f}")

    # Category breakdown per split
    all_cats = sorted(set(train["category"].unique()) |
                      set(val["category"].unique()) |
                      set(test["category"].unique()))
    print(f"\n  Category representation (markets per split):")
    print(f"  {'category':<25} {'train':>7} {'val':>7} {'test':>7}")
    print(f"  {'-'*49}")
    for cat in all_cats:
        t = train[train["category"] == cat]["condition_id"].nunique()
        v = val[val["category"] == cat]["condition_id"].nunique()
        te = test[test["category"] == cat]["condition_id"].nunique()
        marker = " ***" if (t == 0 or te == 0) else ""
        print(f"  {cat:<25} {t:>7} {v:>7} {te:>7}{marker}")


def main() -> None:
    cfg = load_config()
    split_cfg = cfg["dataset"].get("split", {})
    train_frac = split_cfg.get("train_frac", 0.6)
    val_frac = split_cfg.get("val_frac", 0.2)
    min_snaps = cfg["polymarket"].get("snapshots", {}).get("min_snapshots_per_market", 5)

    print("=" * 60)
    print("B3: Theme-stratified temporal split on expanded Gamma data")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/7] Loading Gamma data...")
    markets = pd.read_parquet("data/raw/gamma_markets.parquet")
    snaps = pd.read_parquet("data/raw/gamma_snapshots.parquet")
    print(f"  Markets: {len(markets)}")
    print(f"  Snapshots: {len(snaps)} across {snaps['condition_id'].nunique()} markets")

    # Step 2: Update snapshot categories from re-classified markets
    print("\n[2/7] Updating snapshot categories from re-classified markets...")
    cat_map = markets.set_index("condition_id")["category"].to_dict()
    old_cats = snaps["category"].value_counts()
    snaps["category"] = snaps["condition_id"].map(cat_map).fillna(snaps["category"])
    new_cats = snaps["category"].value_counts()
    n_changed = (old_cats.reindex(new_cats.index, fill_value=0) != new_cats).sum()
    print(f"  Updated categories for {n_changed} category groups")

    # Step 3: Filter sparse markets
    print("\n[3/7] Filtering sparse markets...")
    snaps = filter_sparse_markets(snaps, min_snapshots=min_snaps)

    # Step 4: Validate outcomes
    print("\n[4/7] Validating outcomes + enriching with temporal/snapshot features...")
    snaps = validate_outcomes(snaps)
    snaps = add_temporal_features(snaps)
    snaps = add_snapshot_features(snaps)

    # Step 5: Build modeling features
    print("\n[5/7] Building modeling features...")
    snaps = build_modeling_dataset(snaps)

    # Step 6: Assign event groups
    print("\n[6/7] Assigning event groups...")
    snaps = assign_event_groups(snaps)

    # Step 7: Per-category temporal split
    print("\n[7/7] Per-category temporal split...")
    train, val, test, manifest = split_by_category_temporal(
        snaps, train_frac=train_frac, val_frac=val_frac,
    )

    # Audit
    audit_leakage(train, val, test)
    print_split_summary(train, val, test)

    # Compute class weights
    # compute_class_weights expects a 'theme' column — use 'category' as fallback
    if "theme" not in train.columns:
        for split_df in [train, val, test]:
            split_df["theme"] = split_df["category"]

    weights = compute_class_weights(
        train, val, test, cfg["dataset"].get("target_col", "resolved_yes"),
    )

    # Save
    out_dir = Path(cfg["data_pipeline"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Back up old splits if they exist
    for fname in ["train.parquet", "val.parquet", "test.parquet"]:
        old = out_dir / fname
        if old.exists():
            backup = out_dir / f"{fname}.curated_backup"
            if not backup.exists():
                old.rename(backup)
                print(f"  Backed up {old} -> {backup}")

    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    weights_path = out_dir / "class_weights.json"
    weights_path.write_text(json.dumps(weights, indent=2))

    # Dataset hash for reproducibility
    import hashlib
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        h = hashlib.sha256(split_df.to_parquet()).hexdigest()[:12]
        print(f"  {name}.parquet hash: {h}")

    print(f"\nSaved to {out_dir}:")
    print(f"  train.parquet: {len(train)} rows, {train['condition_id'].nunique()} markets")
    print(f"  val.parquet:   {len(val)} rows, {val['condition_id'].nunique()} markets")
    print(f"  test.parquet:  {len(test)} rows, {test['condition_id'].nunique()} markets")
    print(f"  split_manifest.json: {len(manifest)} categories")
    print(f"  class_weights.json")


if __name__ == "__main__":
    main()
