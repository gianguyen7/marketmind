"""Features derived from Polymarket snapshot data."""

import numpy as np
import pandas as pd


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features from market price and volume data.

    Works on snapshot-level data (one row per market per time point).
    """
    df = df.copy()

    # Implied probability from market price
    if "price_yes" in df.columns:
        df["implied_prob"] = df["price_yes"].clip(0.01, 0.99)
        df["implied_logit"] = np.log(df["implied_prob"] / (1 - df["implied_prob"]))

    # Volume features
    if "volume" in df.columns:
        df["log_volume"] = np.log1p(df["volume"])
        df["volume_rank"] = df["volume"].rank(pct=True)

    # Liquidity features
    if "liquidity" in df.columns:
        df["log_liquidity"] = np.log1p(df["liquidity"])

    # Price extremity: how far from 0.5
    if "price_yes" in df.columns:
        df["price_extremity"] = (df["price_yes"] - 0.5).abs()

    # Snapshot-derived features (from resolve_outcomes.add_snapshot_features)
    # These already exist if the pipeline ran: price_change, price_momentum_3/7,
    # price_volatility_7, price_vs_open, snapshot_num, pct_lifetime_elapsed

    # Confidence trend: is the market becoming more or less decisive?
    if "price_extremity" in df.columns and "condition_id" in df.columns and "snapshot_ts" in df.columns:
        df = df.sort_values(["condition_id", "snapshot_ts"])
        df["extremity_trend"] = df.groupby("condition_id")["price_extremity"].diff().fillna(0)

    return df
