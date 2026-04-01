"""Build the final modeling dataset from enriched market snapshots.

Each row = one market at one point in time.
Target = eventual binary resolution of that market.
"""

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
    df = df.dropna(subset=[c for c in required if c in df.columns])

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

    # Binary: number of outcomes always 2
    df["num_outcomes"] = 2

    return df


def split_temporal(
    df: pd.DataFrame,
    train_cutoff: str,
    test_cutoff: str,
    time_col: str = "snapshot_ts",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split snapshot data temporally to prevent leakage.

    Splits on snapshot timestamp. A single market can have snapshots
    in both train and test — correct for point-in-time forecasting.
    """
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        train_cut = pd.to_datetime(train_cutoff, utc=True)
        test_cut = pd.to_datetime(test_cutoff, utc=True)

        train = df[df[time_col] < train_cut]
        val = df[(df[time_col] >= train_cut) & (df[time_col] < test_cut)]
        test = df[df[time_col] >= test_cut]
    else:
        print("WARNING: no time column found, using random split (not recommended)")
        n = len(df)
        idx = np.random.permutation(n)
        train = df.iloc[idx[: int(0.6 * n)]]
        val = df.iloc[idx[int(0.6 * n): int(0.8 * n)]]
        test = df.iloc[idx[int(0.8 * n):]]

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        n_markets = split_df["condition_id"].nunique() if "condition_id" in split_df.columns else "?"
        print(f"  {name}: {len(split_df)} snapshots across {n_markets} markets")

    return train, val, test


def run(
    input_path: str = "data/interim/snapshots_enriched.parquet",
    config_path: str = "configs/data.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and save the modeling dataset with temporal splits."""
    cfg = load_config(config_path)
    ds = cfg["dataset"]

    df = pd.read_parquet(input_path)
    n_markets = df["condition_id"].nunique() if "condition_id" in df.columns else "?"
    print(f"Loaded {len(df)} snapshots across {n_markets} markets")

    df = build_modeling_dataset(df)

    train, val, test = split_temporal(
        df, ds["train_cutoff"], ds["test_cutoff"], ds.get("time_col", "snapshot_ts"),
    )

    out_dir = Path(cfg["data_pipeline"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    print(f"Saved splits to {out_dir}")

    return train, val, test


if __name__ == "__main__":
    run()
