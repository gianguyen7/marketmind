"""Features derived from Polymarket snapshot data.

Two categories:
1. Single-market features: price, volume, momentum (already partly priced in)
2. Cross-market features: structural signals the crowd might miss (theme base rates,
   event group consistency, historical calibration). All backward-looking only.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-market features (from price/volume data)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cross-market features (structural signals the crowd might miss)
# All use backward-looking only: expanding + shift to prevent leakage.
# ---------------------------------------------------------------------------

def add_cross_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-market structural features.

    These capture information that individual market prices may not reflect:
    - Historical base rates by theme
    - Historical calibration accuracy at similar price levels
    - Event group structural signals
    """
    df = df.copy()

    if "condition_id" not in df.columns or "resolved_yes" not in df.columns:
        return df

    df = _add_theme_base_rate(df)
    df = _add_price_bucket_accuracy(df)
    df = _add_event_group_features(df)
    df = _add_price_vs_theme_mean(df)

    return df


def _add_theme_base_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Historical base rate for this theme: what % of past markets in this theme resolved Yes?

    Uses market-level resolution (one value per market), then expanding mean shifted by 1.
    This prevents any within-market leakage.

    Hypothesis: crowds may not anchor to domain-specific base rates.
    """
    if "theme" not in df.columns or "event_group_end" not in df.columns:
        return df

    # Build market-level data: one row per market
    market_meta = (
        df.groupby("condition_id")
        .agg(theme=("theme", "first"), resolved_yes=("resolved_yes", "first"),
             event_group_end=("event_group_end", "first"))
        .reset_index()
        .sort_values("event_group_end")
    )

    # Within each theme, compute expanding mean of resolved_yes, shifted by 1 market
    theme_rates = {}
    for theme, group in market_meta.groupby("theme"):
        rates = group["resolved_yes"].expanding().mean().shift(1)
        for cid, rate in zip(group["condition_id"], rates):
            theme_rates[cid] = rate

    df["theme_historical_base_rate"] = df["condition_id"].map(theme_rates)
    n_filled = df["theme_historical_base_rate"].isna().sum()
    print(f"  theme_historical_base_rate: {n_filled} NaNs (first market per theme)")
    return df


def _add_price_bucket_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """For markets at similar price levels historically, what was the actual resolution rate?

    Detects favourite-longshot bias: if markets priced at 0.1 historically resolve at 0.15,
    the market systematically underprices longshots.

    Uses market-level data to avoid within-market leakage.
    """
    if "event_group_end" not in df.columns:
        return df

    # Build market-level: use mean price across snapshots as the market's "typical" price
    market_meta = (
        df.groupby("condition_id")
        .agg(mean_price=("price_yes", "mean"), resolved_yes=("resolved_yes", "first"),
             event_group_end=("event_group_end", "first"))
        .reset_index()
        .sort_values("event_group_end")
    )

    # Bucket by price decile, compute expanding accuracy per bucket
    market_meta["price_bucket"] = pd.cut(market_meta["mean_price"], bins=5, labels=False)

    bucket_accuracy = {}
    for bucket, group in market_meta.groupby("price_bucket"):
        rates = group["resolved_yes"].expanding().mean().shift(1)
        for cid, rate in zip(group["condition_id"], rates):
            bucket_accuracy[cid] = rate

    df["price_bucket_historical_accuracy"] = df["condition_id"].map(bucket_accuracy)

    # Also compute: snapshot-level deviation from historical accuracy
    # (how far is this snapshot's price from the historical accuracy for this price bucket?)
    if "price_yes" in df.columns:
        df["price_vs_historical_accuracy"] = df["price_yes"] - df["price_bucket_historical_accuracy"]

    n_filled = df["price_bucket_historical_accuracy"].isna().sum()
    print(f"  price_bucket_historical_accuracy: {n_filled} NaNs (first market per bucket)")
    return df


def _add_event_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Event group structural features.

    - n_markets in event group: more correlated markets = different dynamics
    - price_sum: for mutually exclusive markets, sum should be ~1.0. Deviation = mispricing.
    """
    if "event_group" not in df.columns:
        return df

    # Number of markets per event group (static per group)
    group_sizes = df.groupby("event_group")["condition_id"].nunique()
    df["event_group_n_markets"] = df["event_group"].map(group_sizes)

    # Sum of Yes prices across markets in same event group at each snapshot time
    # For FOMC meetings: "cut 25bp" + "cut 50bp" + "no change" + "hike" should ≈ 1.0
    if "snapshot_ts" in df.columns and "price_yes" in df.columns:
        group_price_sum = (
            df.groupby(["event_group", "snapshot_ts"])["price_yes"]
            .transform("sum")
        )
        df["event_group_price_sum"] = group_price_sum
        # Deviation from 1.0 = potential arbitrage / mispricing signal
        df["event_group_price_deviation"] = (group_price_sum - 1.0).abs()

    return df


def _add_price_vs_theme_mean(df: pd.DataFrame) -> pd.DataFrame:
    """How far is this market's price from the theme average at this timepoint?

    Outlier markets within a theme may be mispriced.
    """
    if "theme" not in df.columns or "price_yes" not in df.columns:
        return df

    if "snapshot_ts" in df.columns:
        theme_mean_price = df.groupby(["theme", "snapshot_ts"])["price_yes"].transform("mean")
    else:
        theme_mean_price = df.groupby("theme")["price_yes"].transform("mean")

    df["price_vs_theme_mean"] = df["price_yes"] - theme_mean_price
    return df
