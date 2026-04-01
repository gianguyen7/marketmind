"""Tests for data pipeline components."""

import pandas as pd
import pytest

from src.data.resolve_outcomes import validate_outcomes, add_temporal_features, add_snapshot_features
from src.data.build_dataset import build_modeling_dataset, split_temporal
from src.data.fetch_markets import parse_registry


# --- Fixtures ---

def make_mini_registry():
    """Minimal registry structure matching configs/markets.yaml format."""
    return {
        "themes": {
            "fed_rate_decisions": {
                "description": "FOMC meeting outcome markets",
                "meetings": [
                    {
                        "meeting_date": "2024-09-18",
                        "actual_outcome": "cut_50bp",
                        "markets": [
                            {
                                "question": "Fed rate cut by September 18?",
                                "condition_id": "0xabc123",
                                "slug": "fed-rate-cut-sep-18",
                                "yes_token": "token_abc",
                                "volume_usd": 20000000,
                                "resolved_yes": True,
                                "end_date": "2024-09-18",
                            },
                            {
                                "question": "No change in Fed rates after September 2024?",
                                "condition_id": "0xdef456",
                                "slug": "no-change-fed-sep-24",
                                "yes_token": "token_def",
                                "volume_usd": 23000000,
                                "resolved_yes": False,
                                "end_date": "2024-09-18",
                            },
                        ],
                    },
                ],
            },
            "government_shutdown": {
                "description": "US government shutdown markets",
                "markets": [
                    {
                        "question": "US government shutdown Saturday?",
                        "condition_id": "0xghi789",
                        "slug": "us-gov-shutdown",
                        "yes_token": "token_ghi",
                        "volume_usd": 157000000,
                        "resolved_yes": True,
                        "end_date": "2026-01-31",
                    },
                ],
            },
        },
    }


def make_sample_snapshots():
    """Create snapshot time-series data for two markets."""
    dates = [
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-07-15", tz="UTC"),
        pd.Timestamp("2024-08-01", tz="UTC"),
        pd.Timestamp("2024-08-15", tz="UTC"),
        pd.Timestamp("2024-09-01", tz="UTC"),
    ]
    prices = [0.40, 0.45, 0.55, 0.60, 0.70]

    rows = []
    for cond_id, question, resolved, theme, end_date in [
        ("0xabc", "Fed rate cut by Sep 18?", 1, "fed_rate_decisions", "2024-09-18"),
        ("0xdef", "No change after Sep meeting?", 0, "fed_rate_decisions", "2024-09-18"),
    ]:
        for i, (dt, price) in enumerate(zip(dates, prices)):
            rows.append({
                "condition_id": cond_id,
                "question": question,
                "theme": theme,
                "theme_label": "FOMC meetings",
                "meeting_date": "2024-09-18",
                "actual_outcome": "cut_50bp",
                "volume_usd": 20000000,
                "resolved_yes": resolved,
                "end_date": end_date,
                "snapshot_ts": dt,
                "price_yes": price if cond_id == "0xabc" else 1 - price,
                "is_final_snapshot": i == len(dates) - 1,
                "snapshot_source": "api",
            })
    return pd.DataFrame(rows)


# --- Tests ---

def test_parse_registry():
    registry = make_mini_registry()
    df = parse_registry(registry)
    assert len(df) == 3
    assert "condition_id" in df.columns
    assert "theme" in df.columns
    assert "meeting_date" in df.columns
    # Fed markets should have meeting_date
    fed = df[df["theme"] == "fed_rate_decisions"]
    assert (fed["meeting_date"] == "2024-09-18").all()
    # Shutdown market should have no meeting_date
    shutdown = df[df["theme"] == "government_shutdown"]
    assert shutdown["meeting_date"].isna().all()


def test_validate_outcomes():
    snap_df = make_sample_snapshots()
    result = validate_outcomes(snap_df)
    assert "resolved_yes" in result.columns
    assert result["resolved_yes"].dtype == int
    m1 = result[result["condition_id"] == "0xabc"]
    assert (m1["resolved_yes"] == 1).all()
    m2 = result[result["condition_id"] == "0xdef"]
    assert (m2["resolved_yes"] == 0).all()


def test_add_temporal_features():
    snap_df = make_sample_snapshots()
    result = add_temporal_features(snap_df)
    assert "days_to_end" in result.columns
    assert "days_to_meeting" in result.columns
    assert "pct_lifetime_elapsed" in result.columns
    # Earlier snapshots should have lower pct_lifetime_elapsed
    m1 = result[result["condition_id"] == "0xabc"].sort_values("snapshot_ts")
    assert m1["pct_lifetime_elapsed"].iloc[0] < m1["pct_lifetime_elapsed"].iloc[-1]


def test_add_snapshot_features():
    snap_df = make_sample_snapshots()
    result = add_snapshot_features(snap_df)
    assert "price_change" in result.columns
    assert "price_momentum_3" in result.columns
    assert "price_volatility_7" in result.columns
    assert "snapshot_num" in result.columns
    assert "price_vs_open" in result.columns


def test_build_modeling_dataset():
    snap_df = make_sample_snapshots()
    snap_df = validate_outcomes(snap_df)
    snap_df = add_temporal_features(snap_df)
    snap_df = add_snapshot_features(snap_df)
    result = build_modeling_dataset(snap_df)
    assert "market_price" in result.columns
    assert "theme_encoded" in result.columns
    assert "volume_rank" in result.columns
    assert len(result) == 10  # 5 snapshots x 2 markets


def test_split_temporal():
    snap_df = make_sample_snapshots()
    snap_df = validate_outcomes(snap_df)
    snap_df = add_temporal_features(snap_df)
    snap_df = build_modeling_dataset(snap_df)
    train, val, test = split_temporal(snap_df, "2024-08-01", "2024-09-01", "snapshot_ts")
    assert len(train) + len(val) + len(test) == len(snap_df)
    if len(train) > 0 and len(test) > 0:
        assert train["snapshot_ts"].max() < test["snapshot_ts"].min()
