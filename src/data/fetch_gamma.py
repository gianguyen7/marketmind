"""Bulk fetch resolved markets from Polymarket Gamma API.

Fetches all closed binary markets above a volume threshold, classifies them
by category using regex patterns, determines resolution from outcomePrices,
and fetches price histories with checkpoint/resume.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import yaml

from src.data.fetch_markets import (
    _backoff_delay,
    _clear_checkpoint,
    _load_checkpoint,
    _request_with_retry,
    _save_checkpoint,
    fetch_price_history,
    save_raw,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_data_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Market classification (regex-based)
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS = {
    "fed_monetary_policy": [
        r"\bfed\b.*\b(rate|cut|hike|bps|basis point|interest)\b",
        r"\bfomc\b", r"\bfederal reserve\b", r"\bfed funds\b",
        r"\bfed (increase|decrease|raise|lower)s?\b",
    ],
    "politics_elections": [
        r"\b(win|won)\b.*\b(election|presidential|primary|nominee|nomination)\b",
        r"\b(democrat|republican|gop|dnc|rnc)\b",
        r"\b(president|governor|senator|congress|mayor)\b.*\b(win|elected|nominee)\b",
        r"\b(electoral|popular vote|swing state)\b",
        r"\bpresident(ial)?\b",
    ],
    "sports": [
        r"\b(nba|nfl|mlb|nhl|mls|epl|ucl|serie a|la liga|bundesliga|ligue 1)\b",
        r"\b(super bowl|world series|stanley cup|champions league)\b",
        r"\b(win|beat|defeat)\b.*\b(game|match|series|finals|championship)\b",
        r"\b(mvp|ballon d'or|heisman|cy young)\b",
        r"\bwin on \d{4}-\d{2}-\d{2}\b",  # "Will X win on 2026-02-17?"
    ],
    "crypto_finance": [
        r"\b(bitcoin|btc|ethereum|eth|solana|sol|crypto)\b",
        r"\b(price of|market cap)\b.*\b(above|below|between)\b",
        r"\b(etf|sec|approve|approval)\b.*\b(crypto|bitcoin|ethereum)\b",
        r"\btrade (above|below)\b",
    ],
    "geopolitics": [
        r"\b(war|invasion|ceasefire|peace|sanctions|troops|military)\b",
        r"\b(ukraine|russia|china|taiwan|iran|israel|gaza|nato)\b",
        r"\b(nuclear|missile|drone strike|airstrike)\b",
    ],
    "government_policy": [
        r"\b(government shutdown|debt ceiling|continuing resolution)\b",
        r"\b(executive order|tariff|trade war|bill pass)\b",
        r"\b(impeach|resign|fired|cabinet)\b",
    ],
    "entertainment": [
        r"\b(oscar|grammy|emmy|golden globe|academy award)\b",
        r"\b(box office|grossing movie|billboard|streaming)\b",
        r"\b(album|song|movie|film|tv show|netflix|disney)\b.*\b(#1|top|win|nominated)\b",
    ],
    "science_tech": [
        r"\b(ai|artificial intelligence|gpt|llm|openai|anthropic|google ai)\b",
        r"\b(fda|clinical trial|vaccine|drug approval)\b",
        r"\b(spacex|nasa|launch|mars|moon)\b",
        r"\b(climate|carbon|temperature|weather)\b",
    ],
    "recession_economy": [
        r"\b(recession|gdp|unemployment rate|inflation rate|cpi)\b",
        r"\b(jobs report|payroll|consumer (confidence|sentiment))\b",
    ],
}


def classify_market(question: str, description: str = "") -> str:
    """Classify a market into a category based on question text.

    Returns the first matching category, or 'other' if none match.
    Patterns are checked in priority order.
    """
    text = f"{question} {description}".lower()

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return category

    return "other"


# ---------------------------------------------------------------------------
# Resolution determination
# ---------------------------------------------------------------------------

def determine_resolution(outcome_prices: str, outcomes: str) -> int | None:
    """Determine if a market resolved Yes or No from outcomePrices.

    Returns: 1 (Yes won), 0 (No won), or None (ambiguous/voided).
    """
    try:
        prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
        outcome_labels = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
    except (json.JSONDecodeError, TypeError):
        return None

    if not prices or len(prices) < 2:
        return None

    # Standard binary: ["Yes", "No"] with prices ["1", "0"] or ["0", "1"]
    if len(prices) == 2:
        p0, p1 = float(prices[0]), float(prices[1])

        # Clear resolution
        if p0 == 1.0 and p1 == 0.0:
            # First outcome won
            if outcome_labels and outcome_labels[0].lower() == "yes":
                return 1
            return 1  # assume first=Yes for binary
        elif p0 == 0.0 and p1 == 1.0:
            # Second outcome won
            if outcome_labels and outcome_labels[0].lower() == "yes":
                return 0
            return 0

        # Both zero = voided
        if p0 == 0.0 and p1 == 0.0:
            return None

    return None


# ---------------------------------------------------------------------------
# Gamma API fetcher
# ---------------------------------------------------------------------------

def fetch_closed_markets(
    gamma_url: str = "https://gamma-api.polymarket.com",
    min_volume: float = 1_000_000,
    max_markets: int = 5000,
    rate_delay: float = 0.3,
) -> pd.DataFrame:
    """Fetch all closed binary markets from Gamma API above volume threshold.

    Paginates through results sorted by volume descending.
    """
    endpoint = f"{gamma_url}/markets"
    page_size = 100
    offset = 0
    all_markets = []

    print(f"Fetching closed markets from Gamma API (min volume: ${min_volume/1e6:.1f}M)...")

    while len(all_markets) < max_markets:
        resp = _request_with_retry(
            endpoint,
            params={
                "closed": "true",
                "limit": page_size,
                "offset": offset,
                "order": "volumeNum",
                "ascending": "false",
            },
        )

        if resp is None:
            print(f"  API request failed at offset {offset}")
            break

        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            # First page may return dict instead of list
            if isinstance(data, dict):
                offset += page_size
                time.sleep(rate_delay)
                continue
            break

        # Filter and process
        page_markets = []
        for m in data:
            vol = m.get("volumeNum", 0)
            if vol < min_volume:
                # Since sorted by volume desc, all remaining markets are below threshold
                print(f"  Reached volume floor (${vol/1e6:.2f}M < ${min_volume/1e6:.1f}M) at offset {offset}")
                all_markets.extend(page_markets)
                # Done
                return _build_market_df(all_markets)

            outcomes = m.get("outcomes", "[]")
            try:
                outcome_list = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
            except json.JSONDecodeError:
                continue

            # Only binary Yes/No markets
            if len(outcome_list) != 2:
                continue

            resolution = determine_resolution(m.get("outcomePrices", "[]"), outcomes)
            if resolution is None:
                continue  # Skip voided/ambiguous markets

            # Extract yes token ID for price history
            clob_tokens = m.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(clob_tokens) if isinstance(clob_tokens, str) else clob_tokens
            except json.JSONDecodeError:
                token_ids = []

            yes_token = token_ids[0] if token_ids else ""

            # Extract event info
            events = m.get("events", [])
            event_id = str(events[0]["id"]) if events else ""
            event_title = events[0].get("title", "") if events else ""
            event_category = events[0].get("category", "") if events else ""

            question = m.get("question", "")
            description = m.get("description", "")
            category = event_category if event_category else classify_market(question, description)

            page_markets.append({
                "condition_id": m.get("conditionId", ""),
                "question": question,
                "slug": m.get("slug", ""),
                "description": description[:500],  # truncate long descriptions
                "yes_token": yes_token,
                "volume_usd": vol,
                "resolved_yes": resolution,
                "end_date": m.get("endDate", ""),
                "start_date": m.get("startDate", ""),
                "closed_time": m.get("closedTime", ""),
                "category": category,
                "event_id": event_id,
                "event_title": event_title,
                "neg_risk": m.get("negRisk", False),
                "group_item_title": m.get("groupItemTitle", ""),
            })

        all_markets.extend(page_markets)
        offset += page_size

        if len(all_markets) % 500 < page_size:
            print(f"  Fetched {len(all_markets)} markets so far (offset {offset})...")

        time.sleep(rate_delay)

    return _build_market_df(all_markets)


def _build_market_df(markets: list[dict]) -> pd.DataFrame:
    """Convert list of market dicts to DataFrame with summary stats."""
    df = pd.DataFrame(markets)
    if len(df) == 0:
        print("WARNING: no markets fetched!")
        return df

    n = len(df)
    n_yes = (df["resolved_yes"] == 1).sum()
    n_no = (df["resolved_yes"] == 0).sum()
    print(f"\nFetched {n} binary resolved markets")
    print(f"  Resolved Yes: {n_yes} ({n_yes/n:.1%})")
    print(f"  Resolved No:  {n_no} ({n_no/n:.1%})")
    print(f"  Volume range: ${df['volume_usd'].min()/1e6:.2f}M - ${df['volume_usd'].max()/1e6:.1f}M")
    print(f"  Total volume: ${df['volume_usd'].sum()/1e9:.1f}B")
    print(f"\n  Categories:")
    for cat, count in df["category"].value_counts().items():
        print(f"    {cat}: {count}")

    return df


# ---------------------------------------------------------------------------
# Price history fetching with checkpoint/resume
# ---------------------------------------------------------------------------

def fetch_all_price_histories(
    markets_df: pd.DataFrame,
    api_url: str = "https://clob.polymarket.com",
    fidelity: int = 720,
    interval: str = "max",
    retry_cfg: dict | None = None,
    rate_delay: float = 0.4,
    checkpoint_path: str = "data/raw/.checkpoint_gamma_snapshots.json",
    min_snapshots: int = 5,
) -> pd.DataFrame:
    """Fetch price history for all markets with checkpoint/resume.

    Returns a DataFrame with one row per market-snapshot.
    """
    checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    total = len(markets_df)
    fetched_api = 0
    fetched_ckpt = 0
    skipped = 0

    for i, (_, row) in enumerate(markets_df.iterrows()):
        cid = row["condition_id"]
        market_num = i + 1

        if cid in checkpoint:
            fetched_ckpt += 1
            continue

        token_id = row["yes_token"]
        if not token_id:
            checkpoint[cid] = []
            skipped += 1
            continue

        if market_num % 100 == 0 or market_num == 1:
            print(f"  [{market_num}/{total}] Fetching price history...")

        history = fetch_price_history(token_id, api_url, fidelity, interval, retry_cfg)

        snapshots = []
        for j, point in enumerate(history):
            ts = point.get("t", point.get("timestamp"))
            price = point.get("p", point.get("price"))
            if ts is None or price is None:
                continue
            snapshots.append({
                "condition_id": cid,
                "question": row["question"],
                "slug": row["slug"],
                "category": row["category"],
                "event_id": row["event_id"],
                "volume_usd": row["volume_usd"],
                "resolved_yes": int(row["resolved_yes"]),
                "end_date": row["end_date"],
                "start_date": row.get("start_date", ""),
                "snapshot_ts": pd.to_datetime(ts, unit="s", utc=True).isoformat() if isinstance(ts, (int, float)) else ts,
                "price_yes": float(price),
                "is_final_snapshot": (j == len(history) - 1),
                "snapshot_source": "api",
            })

        checkpoint[cid] = snapshots
        fetched_api += 1

        if checkpoint_path and fetched_api % 50 == 0:
            _save_checkpoint(checkpoint_path, checkpoint)

        time.sleep(rate_delay)

    # Final checkpoint save
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, checkpoint)

    print(f"\nPrice history fetch complete:")
    print(f"  From checkpoint: {fetched_ckpt}")
    print(f"  From API: {fetched_api}")
    print(f"  Skipped (no token): {skipped}")

    # Flatten
    all_snapshots = []
    for cid, snaps in checkpoint.items():
        all_snapshots.extend(snaps)

    df = pd.DataFrame(all_snapshots)
    if len(df) > 0:
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)

        # Log sparse markets
        counts = df.groupby("condition_id").size()
        n_sparse = (counts < min_snapshots).sum()
        if n_sparse:
            print(f"  Markets with <{min_snapshots} snapshots: {n_sparse}")
        print(f"  Total snapshots: {len(df)} across {df['condition_id'].nunique()} markets")
        print(f"  Median snapshots/market: {counts.median():.0f}")

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config_path: str = "configs/data.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all closed binary markets from Gamma API and their price histories.

    Returns (markets_df, snapshots_df).
    """
    cfg = load_data_config(config_path)
    pm = cfg["polymarket"]
    gamma_cfg = cfg.get("gamma_fetch", {})

    min_volume = gamma_cfg.get("min_volume_usd", 1_000_000)
    max_markets = gamma_cfg.get("max_markets", 5000)

    # Step 1: Fetch market metadata
    print("=" * 60)
    print("Gamma API: Fetching resolved binary markets")
    print("=" * 60)

    markets_df = fetch_closed_markets(
        gamma_url=pm.get("gamma_api_url", "https://gamma-api.polymarket.com"),
        min_volume=min_volume,
        max_markets=max_markets,
    )

    if len(markets_df) == 0:
        print("No markets fetched!")
        return markets_df, pd.DataFrame()

    save_raw(markets_df, "gamma_markets.parquet", cfg["data_pipeline"]["raw_dir"])

    # Step 2: Fetch price histories
    print(f"\n{'=' * 60}")
    print("Gamma API: Fetching price histories")
    print("=" * 60)

    ph = pm.get("price_history", {})
    retry_cfg = pm.get("retry", {})
    rate_delay = pm.get("rate_limit", {}).get("delay_between_requests_sec", 0.4)
    ckpt_path = gamma_cfg.get("checkpoint_path", "data/raw/.checkpoint_gamma_snapshots.json")

    snapshots_df = fetch_all_price_histories(
        markets_df,
        api_url=pm.get("api_url", "https://clob.polymarket.com"),
        fidelity=ph.get("fidelity", 720),
        interval=ph.get("interval", "max"),
        retry_cfg=retry_cfg,
        rate_delay=rate_delay,
        checkpoint_path=ckpt_path,
    )

    if len(snapshots_df) > 0:
        save_raw(snapshots_df, "gamma_snapshots.parquet", cfg["data_pipeline"]["raw_dir"])

    # Clear checkpoint on success
    if ckpt_path:
        _clear_checkpoint(ckpt_path)

    print(f"\n{'=' * 60}")
    print(f"Gamma fetch complete: {len(markets_df)} markets, {len(snapshots_df)} snapshots")
    print("=" * 60)

    return markets_df, snapshots_df


if __name__ == "__main__":
    run()
