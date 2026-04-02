"""Load curated markets from registry and fetch price history from Polymarket.

Features:
- Exponential backoff with jitter on retryable HTTP errors (429, 5xx)
- Checkpoint/resume: saves progress per-market so crashes don't restart from scratch
- Configurable rate limiting between requests
- Schema validation after fetch
"""

import json
import random
import time
from pathlib import Path

import pandas as pd
import requests
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

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
# HTTP with retry + exponential backoff
# ---------------------------------------------------------------------------

def _request_with_retry(
    url: str,
    params: dict,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on_status: list[int] | None = None,
    timeout: int = 30,
) -> requests.Response | None:
    """GET request with exponential backoff + jitter.

    Returns the Response on success, or None after all retries exhausted.
    """
    if retry_on_status is None:
        retry_on_status = [429, 500, 502, 503, 504]

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code == 200:
                return resp

            if resp.status_code in retry_on_status and attempt < max_retries:
                delay = _backoff_delay(attempt, base_delay, max_delay, resp)
                print(f"    HTTP {resp.status_code} — retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue

            # Non-retryable error
            resp.raise_for_status()

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                delay = _backoff_delay(attempt, base_delay, max_delay)
                print(f"    Timeout — retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            print(f"    Timeout after {max_retries} retries")
            return None

        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                delay = _backoff_delay(attempt, base_delay, max_delay)
                print(f"    Connection error — retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            print(f"    Connection error after {max_retries} retries")
            return None

        except requests.exceptions.HTTPError as e:
            print(f"    HTTP error (non-retryable): {e}")
            return None

    return None


def _backoff_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    resp: requests.Response | None = None,
) -> float:
    """Exponential backoff with jitter. Respects Retry-After header on 429."""
    if resp is not None and resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), max_delay)
            except ValueError:
                pass

    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, delay * 0.25)
    return min(delay + jitter, max_delay)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str) -> dict:
    """Load checkpoint: {condition_id: list_of_snapshot_dicts}."""
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text())
        print(f"Resuming from checkpoint: {len(data)} markets already fetched")
        return data
    return {}


def _save_checkpoint(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _clear_checkpoint(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
        print(f"Cleared checkpoint: {path}")


# ---------------------------------------------------------------------------
# Price history fetching
# ---------------------------------------------------------------------------

def fetch_price_history(
    token_id: str,
    api_url: str = "https://clob.polymarket.com",
    fidelity: int = 720,
    interval: str = "max",
    retry_cfg: dict | None = None,
) -> list[dict]:
    """Fetch price history for a single YES token with retry support."""
    endpoint = f"{api_url}/prices-history"
    params = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }

    cfg = retry_cfg or {}
    resp = _request_with_retry(
        endpoint,
        params,
        max_retries=cfg.get("max_retries", 3),
        base_delay=cfg.get("base_delay_sec", 1.0),
        max_delay=cfg.get("max_delay_sec", 30.0),
        retry_on_status=cfg.get("retry_on_status", [429, 500, 502, 503, 504]),
    )

    if resp is None:
        return []

    try:
        data = resp.json()
        history = data.get("history", data) if isinstance(data, dict) else data
        if isinstance(history, list):
            return history
        return []
    except Exception as e:
        print(f"    Warning: failed to parse response for token {token_id[:20]}...: {e}")
        return []


# ---------------------------------------------------------------------------
# Snapshot building with checkpoint/resume
# ---------------------------------------------------------------------------

def _make_snapshot_row(row: pd.Series, ts, price: float, is_final: bool, source: str) -> dict:
    """Build a single snapshot dict from a market row + price point."""
    if isinstance(ts, (int, float)):
        snapshot_ts = pd.to_datetime(ts, unit="s", utc=True).isoformat()
    else:
        snapshot_ts = pd.to_datetime(ts, errors="coerce", utc=True).isoformat()

    return {
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
        "price_yes": price,
        "is_final_snapshot": is_final,
        "snapshot_source": source,
    }


def build_snapshots(
    markets_df: pd.DataFrame,
    api_url: str,
    fidelity: int = 720,
    interval: str = "max",
    min_snapshots: int = 5,
    retry_cfg: dict | None = None,
    rate_limit_delay: float = 0.4,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """Fetch price history for all markets and build snapshot time series.

    Supports checkpoint/resume: if checkpoint_path is set and a previous run
    was interrupted, already-fetched markets are loaded from the checkpoint.
    """
    checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    total = len(markets_df)
    fetched_from_checkpoint = 0
    fetched_from_api = 0

    for idx, row in markets_df.iterrows():
        cid = row["condition_id"]
        q = row["question"][:55]
        token_id = row["yes_token"]
        market_num = idx + 1

        # Already in checkpoint — skip API call
        if cid in checkpoint:
            fetched_from_checkpoint += 1
            continue

        if not token_id:
            print(f"  [{market_num}/{total}] SKIP (no token): {q}")
            # Save fallback row to checkpoint
            fallback_ts = pd.to_datetime(row["end_date"], errors="coerce", utc=True)
            checkpoint[cid] = [_make_snapshot_row(
                row, fallback_ts.isoformat() if pd.notna(fallback_ts) else None,
                None, True, "registry_fallback",
            )]
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, checkpoint)
            continue

        print(f"  [{market_num}/{total}] {q}...")
        history = fetch_price_history(token_id, api_url, fidelity, interval, retry_cfg)

        if not history:
            print(f"    -> 0 snapshots (single fallback row from registry)")
            fallback_ts = pd.to_datetime(row["end_date"], errors="coerce", utc=True)
            checkpoint[cid] = [_make_snapshot_row(
                row, fallback_ts.isoformat() if pd.notna(fallback_ts) else None,
                None, True, "registry_fallback",
            )]
        else:
            snapshots = []
            for i, point in enumerate(history):
                ts = point.get("t", point.get("timestamp"))
                price = point.get("p", point.get("price"))
                if ts is None or price is None:
                    continue
                snapshots.append(_make_snapshot_row(
                    row, ts, float(price), (i == len(history) - 1), "api",
                ))
            checkpoint[cid] = snapshots
            print(f"    -> {len(snapshots)} snapshots")
            fetched_from_api += 1

        # Save checkpoint after each market
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, checkpoint)

        # Rate limiting
        time.sleep(rate_limit_delay)

    # Summary
    if fetched_from_checkpoint:
        print(f"\nResumed {fetched_from_checkpoint} markets from checkpoint")
    print(f"Fetched {fetched_from_api} markets from API")

    # Flatten checkpoint into DataFrame
    all_snapshots = []
    for cid, snaps in checkpoint.items():
        all_snapshots.extend(snaps)

    snapshots_df = pd.DataFrame(all_snapshots)

    # Convert snapshot_ts back to datetime
    if len(snapshots_df) > 0 and "snapshot_ts" in snapshots_df.columns:
        snapshots_df["snapshot_ts"] = pd.to_datetime(
            snapshots_df["snapshot_ts"], errors="coerce", utc=True
        )

    # Log markets with sparse data
    if min_snapshots > 1 and len(snapshots_df) > 0:
        counts = snapshots_df.groupby("condition_id").size()
        sparse = counts[counts < min_snapshots].index
        if len(sparse) > 0:
            print(f"\nNote: {len(sparse)} markets have <{min_snapshots} snapshots")

    return snapshots_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "condition_id", "question", "theme", "resolved_yes",
    "snapshot_ts", "price_yes", "snapshot_source",
]


def validate_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and log data quality metrics."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    n_markets = df["condition_id"].nunique()
    n_rows = len(df)
    n_api = (df["snapshot_source"] == "api").sum()
    n_fallback = (df["snapshot_source"] == "registry_fallback").sum()
    n_null_price = df["price_yes"].isna().sum()
    n_null_resolution = df["resolved_yes"].isna().sum()

    print(f"\n--- Snapshot validation ---")
    print(f"  Total rows:        {n_rows}")
    print(f"  Unique markets:    {n_markets}")
    print(f"  From API:          {n_api} rows")
    print(f"  Fallback only:     {n_fallback} rows")
    print(f"  Null price_yes:    {n_null_price} rows")
    print(f"  Null resolved_yes: {n_null_resolution} rows")

    if n_api > 0:
        api_df = df[df["snapshot_source"] == "api"]
        snaps_per = api_df.groupby("condition_id").size()
        print(f"  Snapshots/market (API): median={snaps_per.median():.0f}, "
              f"min={snaps_per.min()}, max={snaps_per.max()}")

    if n_rows == 0:
        print("  WARNING: empty dataset!")
    print(f"----------------------------")

    return df


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
    """Load registry, fetch price history, validate, save raw data.

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

    # Step 2: Fetch price history with retry + checkpoint
    ph = pm.get("price_history", {})
    fidelity = ph.get("fidelity", 720)
    interval = ph.get("interval", "max")
    min_snaps = pm.get("snapshots", {}).get("min_snapshots_per_market", 5)
    retry_cfg = pm.get("retry", {})
    rate_delay = pm.get("rate_limit", {}).get("delay_between_requests_sec", 0.4)
    ckpt_cfg = pm.get("checkpoint", {})
    ckpt_path = ckpt_cfg.get("path") if ckpt_cfg.get("enabled", False) else None

    print(f"\n{'=' * 60}")
    print(f"Step 2: Fetching price history (fidelity={fidelity}, interval={interval})")
    print(f"  Retry: max_retries={retry_cfg.get('max_retries', 3)}, "
          f"base_delay={retry_cfg.get('base_delay_sec', 1.0)}s")
    print(f"  Rate limit: {rate_delay}s between requests")
    print(f"  Checkpoint: {'enabled' if ckpt_path else 'disabled'}")
    print(f"{'=' * 60}")

    snapshots_df = build_snapshots(
        markets_df, pm["api_url"], fidelity, interval, min_snaps,
        retry_cfg=retry_cfg,
        rate_limit_delay=rate_delay,
        checkpoint_path=ckpt_path,
    )

    # Step 3: Validate
    snapshots_df = validate_snapshots(snapshots_df)

    # Step 4: Save
    save_raw(snapshots_df, "market_snapshots.parquet", cfg["data_pipeline"]["raw_dir"])

    # Clear checkpoint on successful completion
    if ckpt_path:
        _clear_checkpoint(ckpt_path)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Fetch complete:")
    print(f"  Markets: {len(markets_df)}")
    print(f"  Total snapshots: {len(snapshots_df)}")
    print("=" * 60)

    return markets_df, snapshots_df


if __name__ == "__main__":
    run()
