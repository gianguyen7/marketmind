"""Run the full data pipeline: registry -> price history -> enrich -> dataset -> duckdb.

Flags:
    --gamma-fetch        Run the Gamma API bulk fetch step before the existing
                         curated-registry pipeline. Without this flag, behavior
                         is identical to the pre-B1 pipeline (safe default).
    --dry-run            Dry-run mode for --gamma-fetch: caps markets via
                         --max-markets (default 50), writes to the scratch
                         directory from configs/data.yaml:polymarket.gamma_fetch.dry_run_dir,
                         and stops after the Gamma step. Does NOT touch
                         data/raw/gamma_*.parquet or downstream pipeline stages.
    --max-markets N      Override configs/data.yaml:polymarket.gamma_fetch.max_markets
                         for this run. Useful in dry-run mode (default 50) or
                         for a quick smoke fetch. If omitted, the config value is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.data.build_dataset import run as build_dataset  # noqa: E402
from src.data.fetch_markets import run as fetch_markets  # noqa: E402
from src.data.load_duckdb import run as load_duckdb  # noqa: E402
from src.data.resolve_outcomes import run as enrich_snapshots  # noqa: E402

CONFIG_PATH = "configs/data.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MarketMind data pipeline")
    p.add_argument(
        "--gamma-fetch", action="store_true",
        help="Run Gamma API bulk fetch (Track B1) before the curated pipeline.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Dry-run mode for --gamma-fetch (caps markets, scratch dir, stops early).",
    )
    p.add_argument(
        "--max-markets", type=int, default=None,
        help="Override gamma_fetch.max_markets (default: config value, or 50 in dry-run).",
    )
    return p.parse_args()


def run_gamma_fetch(dry_run: bool, max_markets_override: int | None) -> None:
    """Invoke the Gamma fetcher with optional dry-run and max-markets overrides.

    Calls the fetch_gamma functions directly instead of fetch_gamma.run() so
    we can inject overrides without modifying fetch_gamma.py. Every decision
    is logged per the no-silent-assumptions rule.
    """
    # Import locally so `python3 scripts/run_data_pipeline.py` (no --gamma-fetch)
    # does not have to pay the fetch_gamma import cost.
    from src.data.fetch_gamma import (  # noqa: PLC0415
        fetch_all_price_histories,
        fetch_closed_markets,
        load_data_config,
    )
    from src.data.fetch_markets import _clear_checkpoint, save_raw  # noqa: PLC0415

    cfg = load_data_config(CONFIG_PATH)
    pm = cfg["polymarket"]
    gamma_cfg = pm.get("gamma_fetch", {})

    # Resolve max_markets: CLI > dry-run default > config default
    cfg_max = gamma_cfg.get("max_markets", 5000)
    if max_markets_override is not None:
        max_markets = max_markets_override
        src = "--max-markets flag"
    elif dry_run:
        max_markets = 50
        src = "dry-run default"
    else:
        max_markets = cfg_max
        src = "configs/data.yaml:polymarket.gamma_fetch.max_markets"

    # Resolve output dir
    raw_dir = cfg["data_pipeline"]["raw_dir"]
    if dry_run:
        out_dir = gamma_cfg.get("dry_run_dir", "data/raw/scratch")
        ckpt_path = str(Path(out_dir) / ".checkpoint_gamma_snapshots_dryrun.json")
    else:
        out_dir = raw_dir
        ckpt_path = gamma_cfg.get(
            "checkpoint_path", "data/raw/.checkpoint_gamma_snapshots.json",
        )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Gamma fetch  (dry_run={dry_run})")
    print("=" * 60)
    print(f"  max_markets = {max_markets}  (source: {src})")
    print(f"  min_volume_usd = {gamma_cfg.get('min_volume_usd', 1_000_000)}")
    print(f"  output dir = {out_dir}")
    print(f"  checkpoint path = {ckpt_path}")
    print(f"  binary_only = {gamma_cfg.get('binary_only', True)}")
    print(f"  closed_only = {gamma_cfg.get('closed_only', True)}")

    # Step 1: markets
    markets_df = fetch_closed_markets(
        gamma_url=pm.get("gamma_api_url", "https://gamma-api.polymarket.com"),
        min_volume=gamma_cfg.get("min_volume_usd", 1_000_000),
        max_markets=max_markets,
    )
    if len(markets_df) == 0:
        print("Gamma fetch produced 0 markets. Stopping.")
        return

    save_raw(markets_df, "gamma_markets.parquet", out_dir)

    # Step 2: price histories
    ph = pm.get("price_history", {})
    retry_cfg = pm.get("retry", {})
    rate_delay = pm.get("rate_limit", {}).get("delay_between_requests_sec", 0.4)

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
        save_raw(snapshots_df, "gamma_snapshots.parquet", out_dir)

    # Only clear the production checkpoint on success; keep dry-run checkpoint
    # around for inspection.
    if not dry_run:
        _clear_checkpoint(ckpt_path)

    # Dry-run summary: print category distribution + verification criteria
    if dry_run:
        print("\n" + "=" * 60)
        print("Dry-run summary")
        print("=" * 60)
        print(f"  Markets fetched:   {len(markets_df)}")
        print(f"  Snapshots fetched: {len(snapshots_df)}")
        if "category" in markets_df.columns:
            print("  Categories:")
            for cat, n in markets_df["category"].value_counts().items():
                print(f"    {cat}: {n}")
        res_rate = (markets_df["resolved_yes"].notna().mean()
                    if "resolved_yes" in markets_df.columns else 0.0)
        print(f"  Resolution rate: {res_rate:.1%}")
        print(f"\n  (Dry-run wrote to {out_dir} — data/raw/gamma_*.parquet untouched.)")


def run_curated_pipeline() -> None:
    """The existing 4-stage curated-registry pipeline."""
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


def main() -> None:
    args = parse_args()

    # Important: when --gamma-fetch is passed, we STOP after the Gamma step.
    # We do NOT re-run the curated pipeline, because that would:
    #   (a) make new registry API calls that the user did not ask for,
    #   (b) rewrite data/processed/{train,val,test}.parquet, which is
    #       forbidden until Track B3 per the execution plan.
    # To run the curated pipeline, invoke this script with no flags.
    if args.gamma_fetch:
        run_gamma_fetch(dry_run=args.dry_run, max_markets_override=args.max_markets)
        if args.dry_run:
            print("\nDry-run complete. Re-run with --gamma-fetch (no --dry-run) "
                  "for the full fetch.")
        else:
            print("\nGamma fetch complete. Curated pipeline NOT re-run "
                  "(data/processed/ is frozen until Track B3).")
        return

    run_curated_pipeline()


if __name__ == "__main__":
    # Touch yaml import so the linter keeps it; we use it transitively via
    # fetch_gamma's load_data_config, but importing here lets us fail fast
    # if PyYAML is missing.
    _ = yaml.__version__
    main()
