"""Load pipeline parquet outputs into a DuckDB database for EDA and querying.

Creates one table per pipeline stage, plus useful views for common queries.
Idempotent: safe to re-run after new data pulls.
"""

from pathlib import Path

import duckdb
import yaml


def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_database(config_path: str = "configs/data.yaml") -> str:
    """Load all parquet files into DuckDB. Returns the database path."""
    cfg = load_config(config_path)
    pipeline = cfg["data_pipeline"]
    db_path = pipeline["duckdb_path"]

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)

    # Map: table_name -> parquet_path
    tables = {
        "markets_registry": f"{pipeline['raw_dir']}/markets_registry.parquet",
        "snapshots_raw": f"{pipeline['raw_dir']}/market_snapshots.parquet",
        "snapshots_enriched": f"{pipeline['interim_dir']}/snapshots_enriched.parquet",
        "train": f"{pipeline['processed_dir']}/train.parquet",
        "val": f"{pipeline['processed_dir']}/val.parquet",
        "test": f"{pipeline['processed_dir']}/test.parquet",
    }

    for table_name, parquet_path in tables.items():
        p = Path(parquet_path)
        if not p.exists():
            print(f"  SKIP {table_name} — {parquet_path} not found")
            continue
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM '{parquet_path}'")
        row_count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name}: {row_count:,} rows")

    # --- Useful views for EDA ---

    con.execute("""
        CREATE OR REPLACE VIEW market_summary AS
        SELECT
            condition_id,
            any_value(question) AS question,
            any_value(theme) AS theme,
            any_value(resolved_yes) AS resolved_yes,
            any_value(volume_usd) AS volume_usd,
            count(*) AS n_snapshots,
            min(snapshot_ts) AS first_snapshot,
            max(snapshot_ts) AS last_snapshot,
            min(price_yes) AS min_price,
            max(price_yes) AS max_price,
            avg(price_yes) AS avg_price
        FROM snapshots_enriched
        GROUP BY condition_id
        ORDER BY n_snapshots DESC
    """)

    con.execute("""
        CREATE OR REPLACE VIEW theme_summary AS
        SELECT
            theme,
            count(DISTINCT condition_id) AS n_markets,
            count(*) AS n_snapshots,
            sum(DISTINCT volume_usd) AS total_volume,
            avg(resolved_yes) AS base_rate
        FROM snapshots_enriched
        GROUP BY theme
        ORDER BY n_markets DESC
    """)

    con.execute("""
        CREATE OR REPLACE VIEW split_summary AS
        SELECT 'train' AS split, count(*) AS rows, count(DISTINCT condition_id) AS markets FROM train
        UNION ALL
        SELECT 'val', count(*), count(DISTINCT condition_id) FROM val
        UNION ALL
        SELECT 'test', count(*), count(DISTINCT condition_id) FROM test
    """)

    print(f"\nViews created: market_summary, theme_summary, split_summary")

    # Print database size
    db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
    print(f"\nDatabase: {db_path} ({db_size_mb:.1f} MB)")
    con.close()
    return db_path


def run(config_path: str = "configs/data.yaml") -> str:
    print("=" * 60)
    print("Loading pipeline data into DuckDB")
    print("=" * 60)
    db_path = build_database(config_path)
    print("\nDone. Query with:")
    print(f"  python3 -c \"import duckdb; con = duckdb.connect('{db_path}'); print(con.sql('SELECT * FROM market_summary').df())\"")
    print(f"  # or: duckdb {db_path}")
    return db_path


if __name__ == "__main__":
    run()
