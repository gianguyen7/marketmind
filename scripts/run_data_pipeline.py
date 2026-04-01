"""Run the full data pipeline: registry -> price history -> enrich -> dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetch_markets import run as fetch_markets
from src.data.resolve_outcomes import run as enrich_snapshots
from src.data.build_dataset import run as build_dataset


def main():
    print("=" * 60)
    print("MarketMind Data Pipeline")
    print("=" * 60)

    print("\n[1/3] Loading registry + fetching price history...")
    markets_df, snapshots_df = fetch_markets()

    if len(snapshots_df) == 0:
        print("No snapshot data. Pipeline stopped.")
        return

    print("\n[2/3] Validating outcomes + enriching with temporal features...")
    enrich_snapshots()

    print("\n[3/3] Building modeling dataset with temporal splits...")
    build_dataset()

    print("\nData pipeline complete.")


if __name__ == "__main__":
    main()
