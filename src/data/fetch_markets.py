"""Load curated markets from registry and fetch price history from Polymarket."""

import time
from pathlib import Path

import pandas as pd
import requests
import yaml


def load_data_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_registry(registry_path: str = "configs/markets.yaml") -> dict:
    with open(registry_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Registry parsing
# ---------------------------------------------------------------------------

def parse_registry(registry: dict) -> pd.DataFrame:
    """Flatten the nested registry YAML into one row per market.

    Carries theme, meeting_date, and actual_outcome metadata through.
    """
    rows = []
    themes = registry.get("themes", {})

    for theme_key, theme_data in themes.items():
        theme_label = theme_data.get("description", theme_key)

        # Theme with meetings (FOMC-style)
        if "meetings" in theme_data:
            for meeting in theme_data["meetings"]:
                meeting_date = meeting.get("meeting_date")
                actual_outcome = meeting.get("actual_outcome")
                for m in meeting.get("markets", []):
                    rows.append(_market_row(m, theme_key, theme_label,
                                           meeting_date, actual_outcome))

        # Theme with flat market list
        if "markets" in theme_data:
            for m in theme_data["markets"]:
                rows.append(_market_row(m, theme_key, theme_label))

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} markets across {df['theme'].nunique()} themes from registry")
    return df


def _market_row(
    m: dict,
    theme_key: str,
    theme_label: str,
    meeting_date: str | None = None,
    actual_outcome: str | None = None,
) -> dict:
    return {
        "condition_id": m["condition_id"],
        "question": m["question"],
        "slug": m.get("slug", ""),
        "yes_token": m.get("yes_token", ""),
        "volume_usd": m.get("volume_usd", 0),
        "resolved_yes": m.get("resolved_yes"),
        "end_date": m.get("end_date", ""),
        "theme": theme_key,
        "theme_label": theme_label,
        "meeting_date": meeting_date,
        "actual_outcome": actual_outcome,
    }


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

def fetch_price_history(
    token_id: str,
    api_url: str = "https://clob.polymarket.com",
    fidelity: int = 720,
    interval: str = "max",
) -> list[dict]:
    """Fetch price history for a single YES token.

    Uses /prices-history endpoint. For resolved markets,
    fidelity must be >= 720 (12-hour intervals).
    """
    endpoint = f"{api_url}/prices-history"
    params = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        history = data.get("history", data) if isinstance(data, dict) else data
        if isinstance(history, list):
            return history
        return []
    except Exception as e:
        print(f"  Warning: price history failed for token {token_id[:20]}...: {e}")
        return []


def build_snapshots(
    markets_df: pd.DataFrame,
    api_url: str,
    fidelity: int = 720,
    interval: str = "max",
    min_snapshots: int = 5,
) -> pd.DataFrame:
    """Fetch price history for all markets and build snapshot time series.

    Returns one row per market per timestamp.
    """
    all_snapshots = []

    for idx, row in markets_df.iterrows():
        q = row["question"][:55]
        token_id = row["yes_token"]

        if not token_id:
            print(f"  [{idx+1}/{len(markets_df)}] SKIP (no token): {q}")
            continue

        print(f"  [{idx+1}/{len(markets_df)}] {q}...")
        history = fetch_price_history(token_id, api_url, fidelity, interval)

        if not history:
            print(f"    -> 0 snapshots (will create single row from registry)")
            # Fallback: single row from registry data
            all_snapshots.append({
                "condition_id": row["condition_id"],
                "question": row["question"],
                "slug": row["slug"],
                "theme": row["theme"],
                "theme_label": row["theme_label"],
                "meeting_date": row.get("meeting_date"),
                "actual_outcome": row.get("actual_outcome"),
                "volume_usd": row["volume_usd"],
                "resolved_yes": int(row["resolved_yes"]) if row["resolved_yes"] is not None else None,
                "end_date": row["end_date"],
                "snapshot_ts": pd.to_datetime(row["end_date"], errors="coerce", utc=True),
                "price_yes": None,
                "is_final_snapshot": True,
                "snapshot_source": "registry_fallback",
            })
            time.sleep(0.3)
            continue

        for i, point in enumerate(history):
            ts = point.get("t", point.get("timestamp"))
            price = point.get("p", point.get("price"))
            if ts is None or price is None:
                continue

            if isinstance(ts, (int, float)):
                snapshot_ts = pd.to_datetime(ts, unit="s", utc=True)
            else:
                snapshot_ts = pd.to_datetime(ts, errors="coerce", utc=True)

            all_snapshots.append({
                "condition_id": row["condition_id"],
                "question": row["question"],
                "slug": row["slug"],
                "theme": row["theme"],
                "theme_label": row["theme_label"],
                "meeting_date": row.get("meeting_date"),
                "actual_outcome": row.get("actual_outcome"),
                "volume_usd": row["volume_usd"],
                "resolved_yes": int(row["resolved_yes"]) if row["resolved_yes"] is not None else None,
                "end_date": row["end_date"],
                "snapshot_ts": snapshot_ts,
                "price_yes": float(price),
                "is_final_snapshot": (i == len(history) - 1),
                "snapshot_source": "api",
            })

        print(f"    -> {len(history)} snapshots")
        time.sleep(0.3)

    snapshots_df = pd.DataFrame(all_snapshots)

    # Filter markets with too few snapshots
    if min_snapshots > 1 and len(snapshots_df) > 0:
        counts = snapshots_df.groupby("condition_id").size()
        sparse = counts[counts < min_snapshots].index
        if len(sparse) > 0:
            print(f"\nNote: {len(sparse)} markets have <{min_snapshots} snapshots")

    return snapshots_df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_raw(df: pd.DataFrame, filename: str, output_dir: str = "data/raw") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / filename
    df.to_parquet(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config_path: str = "configs/data.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load registry, fetch price history, save raw data.

    Returns (markets_df, snapshots_df).
    """
    cfg = load_data_config(config_path)
    pm = cfg["polymarket"]

    # Step 1: Parse registry
    print("=" * 60)
    print("Step 1: Loading curated market registry")
    print("=" * 60)
    registry = load_registry(pm["registry"])
    markets_df = parse_registry(registry)

    save_raw(markets_df, "markets_registry.parquet", cfg["data_pipeline"]["raw_dir"])

    # Print summary
    print(f"\nThemes:")
    for theme, count in markets_df["theme"].value_counts().items():
        print(f"  {theme}: {count} markets")
    resolved_count = markets_df["resolved_yes"].notna().sum()
    print(f"\nResolved: {resolved_count}/{len(markets_df)}")
    print(f"Total volume: ${markets_df['volume_usd'].sum():,.0f}")

    # Step 2: Fetch price history
    ph = pm.get("price_history", {})
    fidelity = ph.get("fidelity", 720)
    interval = ph.get("interval", "max")
    min_snaps = pm.get("snapshots", {}).get("min_snapshots_per_market", 5)

    print(f"\n{'=' * 60}")
    print(f"Step 2: Fetching price history (fidelity={fidelity}, interval={interval})")
    print(f"{'=' * 60}")

    snapshots_df = build_snapshots(
        markets_df, pm["api_url"], fidelity, interval, min_snaps
    )

    save_raw(snapshots_df, "market_snapshots.parquet", cfg["data_pipeline"]["raw_dir"])

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Fetch complete:")
    print(f"  Markets: {len(markets_df)}")
    print(f"  Total snapshots: {len(snapshots_df)}")
    if len(snapshots_df) > 0:
        api_snaps = snapshots_df[snapshots_df["snapshot_source"] == "api"]
        if len(api_snaps) > 0:
            snaps_per = api_snaps.groupby("condition_id").size()
            print(f"  Snapshots per market (API): median={snaps_per.median():.0f}, "
                  f"min={snaps_per.min()}, max={snaps_per.max()}")
        fallback_count = (snapshots_df["snapshot_source"] == "registry_fallback").sum()
        if fallback_count:
            print(f"  Markets with fallback only: {fallback_count}")
    print("=" * 60)

    return markets_df, snapshots_df


if __name__ == "__main__":
    run()
