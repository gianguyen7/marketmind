"""Enrich snapshot time series with temporal and rolling features.

Registry data already has resolved_yes as a boolean — no need to infer
from string resolution fields. This module focuses on time-series enrichment.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def validate_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure resolved_yes is clean int (0/1). Drop rows without resolution."""
    df = df.copy()

    if "resolved_yes" not in df.columns:
        raise ValueError("resolved_yes column missing — check registry or fetch step")

    before = len(df)
    df = df.dropna(subset=["resolved_yes"])
    df["resolved_yes"] = df["resolved_yes"].astype(int)

    id_col = "condition_id" if "condition_id" in df.columns else "market_id"
    n_markets = df[id_col].nunique() if id_col in df.columns else "?"
    print(f"Validated outcomes: {before} -> {len(df)} rows ({n_markets} markets)")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features relative to each snapshot.

    For snapshot data, computes how far along the market's lifetime
    each snapshot is — key for point-in-time analysis.
    """
    df = df.copy()

    # Parse date columns
    for col in ["end_date", "snapshot_ts", "meeting_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Snapshot-level: days until market end/resolution
    if "snapshot_ts" in df.columns and "end_date" in df.columns:
        df["days_to_end"] = (
            (df["end_date"] - df["snapshot_ts"]).dt.total_seconds() / 86400
        ).clip(lower=0).fillna(0)

    # Days until meeting (for FOMC markets)
    if "snapshot_ts" in df.columns and "meeting_date" in df.columns:
        df["days_to_meeting"] = (
            (df["meeting_date"] - df["snapshot_ts"]).dt.total_seconds() / 86400
        ).clip(lower=0).fillna(0)

    # Snapshot position: fraction of time elapsed from first snapshot to end_date
    if "snapshot_ts" in df.columns and "end_date" in df.columns:
        id_col = "condition_id" if "condition_id" in df.columns else "market_id"
        first_snap = df.groupby(id_col)["snapshot_ts"].transform("min")
        total_span = (df["end_date"] - first_snap).dt.total_seconds()
        elapsed = (df["snapshot_ts"] - first_snap).dt.total_seconds()
        df["pct_lifetime_elapsed"] = np.where(
            total_span > 0,
            (elapsed / total_span).clip(0, 1),
            1.0,
        )

    return df


def add_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-market time-series features to snapshot data.

    Computes momentum, volatility, and position relative to price history.
    """
    df = df.copy()

    if "snapshot_ts" not in df.columns or "price_yes" not in df.columns:
        return df

    id_col = "condition_id" if "condition_id" in df.columns else "market_id"
    df = df.sort_values([id_col, "snapshot_ts"])

    grouped = df.groupby(id_col)

    # Price change from previous snapshot
    df["price_change"] = grouped["price_yes"].diff().fillna(0)

    # Rolling momentum (3 and 7 snapshot windows)
    df["price_momentum_3"] = grouped["price_yes"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    ) - df["price_yes"]

    df["price_momentum_7"] = grouped["price_yes"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ) - df["price_yes"]

    # Rolling volatility
    df["price_volatility_7"] = grouped["price_yes"].transform(
        lambda x: x.rolling(7, min_periods=2).std()
    ).fillna(0)

    # Snapshot number within market (1-indexed)
    df["snapshot_num"] = grouped.cumcount() + 1
    df["total_snapshots"] = grouped["snapshot_ts"].transform("count")

    # Distance from opening price
    df["price_vs_open"] = df["price_yes"] - grouped["price_yes"].transform("first")

    return df


def run(
    input_path: str = "data/raw/market_snapshots.parquet",
    output_path: str = "data/interim/snapshots_enriched.parquet",
) -> pd.DataFrame:
    """Load raw snapshots, validate outcomes, add features."""
    snap_df = pd.read_parquet(input_path)
    print(f"Loaded {len(snap_df)} snapshots")

    snap_df = validate_outcomes(snap_df)
    snap_df = add_temporal_features(snap_df)
    snap_df = add_snapshot_features(snap_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    snap_df.to_parquet(output_path, index=False)
    print(f"Saved enriched snapshots to {output_path}")
    return snap_df


if __name__ == "__main__":
    run()
