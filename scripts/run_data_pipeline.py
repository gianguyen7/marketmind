"""Run the full data pipeline: registry -> price history -> enrich -> dataset -> duckdb."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetch_markets import run as fetch_markets
from src.data.resolve_outcomes import run as enrich_snapshots
from src.data.build_dataset import run as build_dataset
from src.data.load_duckdb import run as load_duckdb


def main():
    print("=" * 60)
    print("MarketMind Data Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading registry + fetching price history...")
    markets_df, snapshots_df = fetch_markets()

    if len(snapshots_df) == 0:
        print("No snapshot data. Pipeline stopped.")
        return

    print("\n[2/4] Validating outcomes + enriching with temporal features...")
    enrich_snapshots()

    print("\n[3/4] Building modeling dataset with temporal splits...")
    build_dataset()

    print("\n[4/4] Loading into DuckDB...")
    load_duckdb()

    print("\nData pipeline complete.")


if __name__ == "__main__":
    main()
