"""
build_dataset.py — Greyhound Prediction System v2

Parses all JSONL files from DATA_RESULTS_DIR, extracts runner-level rows,
and outputs a single flat parquet file.

Output schema contains ONLY model-relevant features:
- Race metadata: venue, date, race_number, grade, distance
- Runner metadata: box, name, trainer, last_4, best_t_d
- Target: won (binary, 1 if final_position == 1)
- Quarantined: grsa_early_odds (for market system only, NOT a model feature)

Gate check outputs:
- Row count (~124,673 expected)
- Race count (~15,294 expected)
- Date range
- Schema confirmation (no market features)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATA_RESULTS_DIR, MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Output file
OUTPUT_FILE = MODEL_DIR / "dataset.parquet"

# Schema for model features (what the model sees)
MODEL_SCHEMA = [
    # Race metadata
    "venue",
    "race_date",
    "race_number",
    "grade",
    "distance",
    # Runner metadata
    "box",
    "runner_name",
    "trainer",
    "last_4",
    "best_t_d",
    # Target
    "won",
]

# Quarantined columns (market data, NOT model features)
# These are stored separately for the market system to use later
QUARANTINED_COLUMNS = [
    "grsa_early_odds",  # GRSA scraped early-day odds
]


def parse_jsonl_file(file_path: Path) -> list[dict[str, Any]]:
    """
    Parse a single JSONL file and extract runner-level rows.

    Each race produces multiple rows (one per non-scratched runner).
    """
    rows = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                race = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line in {file_path}: {e}")
                continue

            # Extract race-level metadata
            race_date = race.get("date")
            venue = race.get("venue")
            race_number = race.get("race_number")
            grade = race.get("grade")
            distance = race.get("distance")

            # Process each runner
            runners = race.get("runners", [])
            for runner in runners:
                # Skip scratched runners
                if runner.get("is_scratched", False):
                    continue

                # Skip runners without final_position (race not run yet)
                final_position = runner.get("final_position")
                if final_position is None:
                    continue

                # Determine box (use run_box if available, else drawn_box)
                box = runner.get("run_box") or runner.get("drawn_box")

                # Target: won = 1 if final_position == 1, else 0
                won = 1 if final_position == 1 else 0

                row = {
                    # Race metadata
                    "venue": venue,
                    "race_date": race_date,
                    "race_number": race_number,
                    "grade": grade,
                    "distance": distance,
                    # Runner metadata
                    "box": box,
                    "runner_name": runner.get("name"),
                    "trainer": runner.get("trainer"),
                    "last_4": runner.get("last_4"),
                    "best_t_d": runner.get("best_t_d"),
                    # Target
                    "won": won,
                    # Quarantined market data (stored but NOT a model feature)
                    "grsa_early_odds": runner.get("odds"),
                }

                rows.append(row)

    return rows


def build_dataset() -> pd.DataFrame:
    """
    Build the full dataset by parsing all JSONL files.

    Returns:
        DataFrame with one row per runner
    """
    all_rows = []

    # Get all date directories
    date_dirs = sorted([d for d in DATA_RESULTS_DIR.iterdir() if d.is_dir()])

    logger.info(f"Found {len(date_dirs)} date directories")

    for date_dir in date_dirs:
        date_str = date_dir.name
        jsonl_files = list(date_dir.glob("*.jsonl"))

        date_rows = []
        for jsonl_file in jsonl_files:
            file_rows = parse_jsonl_file(jsonl_file)
            date_rows.extend(file_rows)

        all_rows.extend(date_rows)
        logger.info(f"  {date_str}: {len(date_rows)} runners from {len(jsonl_files)} files")

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the schema contains no forbidden market features.

    Raises:
        ValueError if forbidden columns are found
    """
    # Columns that should NEVER appear in model features
    forbidden = [
        "odds_final",
        "bsp",
        "betfair",
        "implied",
        "market",
        "price",
        "sp",
    ]

    # Check all columns except the quarantined column
    model_columns = [c for c in df.columns if c not in QUARANTINED_COLUMNS]

    for col in model_columns:
        col_lower = col.lower()
        for forbidden_word in forbidden:
            if forbidden_word in col_lower and col != "grsa_early_odds":
                raise ValueError(
                    f"Forbidden column '{col}' found in schema. "
                    f"Market data must not be in model features."
                )

    logger.info("Schema validation passed - no forbidden market features")


def count_raw_data() -> dict:
    """
    Count total raw entries in the data for comparison.
    """
    total_races = 0
    total_runners = 0

    date_dirs = sorted([d for d in DATA_RESULTS_DIR.iterdir() if d.is_dir()])

    for date_dir in date_dirs:
        for jsonl_file in date_dir.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        race = json.loads(line)
                        total_races += 1
                        total_runners += len(race.get("runners", []))
                    except json.JSONDecodeError:
                        continue

    return {"total_races": total_races, "total_runners": total_runners}


def run_gate_check(df: pd.DataFrame) -> dict:
    """
    Run gate check and return statistics.

    Returns:
        Dictionary with gate check results
    """
    # Row and race counts
    row_count = len(df)
    race_count = df.groupby(["venue", "race_date", "race_number"]).ngroups

    # Date range
    dates = pd.to_datetime(df["race_date"])
    min_date = dates.min().strftime("%Y-%m-%d")
    max_date = dates.max().strftime("%Y-%m-%d")

    # Win rate (should be ~12.5% for 8-runner races)
    win_rate = df["won"].mean()

    # Field size distribution
    field_sizes = df.groupby(["venue", "race_date", "race_number"]).size()
    avg_field_size = field_sizes.mean()

    # Schema columns
    all_columns = list(df.columns)
    model_columns = [c for c in all_columns if c not in QUARANTINED_COLUMNS]

    # Raw data counts
    raw_counts = count_raw_data()

    return {
        "row_count": row_count,
        "race_count": race_count,
        "min_date": min_date,
        "max_date": max_date,
        "win_rate": win_rate,
        "avg_field_size": avg_field_size,
        "all_columns": all_columns,
        "model_columns": model_columns,
        "quarantined_columns": QUARANTINED_COLUMNS,
        "raw_total_races": raw_counts["total_races"],
        "raw_total_runners": raw_counts["total_runners"],
    }


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("BUILD DATASET - Greyhound Prediction System v2")
    logger.info("=" * 60)

    # Build dataset
    logger.info("\n[1] Parsing JSONL files...")
    df = build_dataset()

    # Validate schema
    logger.info("\n[2] Validating schema...")
    validate_schema(df)

    # Run gate check
    logger.info("\n[3] Running gate check...")
    stats = run_gate_check(df)

    # Print gate check results
    print("\n" + "=" * 60)
    print("GATE CHECK RESULTS")
    print("=" * 60)

    print(f"\n[Raw Data (from JSONL files)]")
    print(f"  Total races:    {stats['raw_total_races']:,} (expected ~15,294)")
    print(f"  Total runners:  {stats['raw_total_runners']:,} (expected ~124,673)")

    print(f"\n[Valid Data (after filtering)]")
    print(f"  Valid races:    {stats['race_count']:,}")
    print(f"  Valid runners:  {stats['row_count']:,}")
    print(f"  Excluded:       {stats['raw_total_runners'] - stats['row_count']:,} (scratched or no result)")

    print(f"\n[Date Range]")
    print(f"  Start date:     {stats['min_date']}")
    print(f"  End date:       {stats['max_date']}")

    print(f"\n[Data Quality]")
    print(f"  Win rate:       {stats['win_rate']:.1%} (expected ~12-14%)")
    print(f"  Avg field size: {stats['avg_field_size']:.1f}")

    print(f"\n[Schema - Model Features Only]")
    for col in stats["model_columns"]:
        print(f"  - {col}")

    print(f"\n[Schema - Quarantined (Market System Use Only)]")
    for col in stats["quarantined_columns"]:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  - {col} ({non_null:,} non-null values)")

    # Save to parquet
    print(f"\n[Output]")
    print(f"  Saving to: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")

    # Final gate status
    print("\n" + "=" * 60)

    # Check: raw counts match expected
    raw_races_ok = abs(stats["raw_total_races"] - 15294) <= 100
    raw_runners_ok = abs(stats["raw_total_runners"] - 124673) <= 500

    # Check: valid data is reasonable (>90% of races have results)
    race_completion = stats["race_count"] / stats["raw_total_races"]

    # Check: no market features in model schema
    forbidden_in_schema = any(
        any(f in c.lower() for f in ["odds", "price", "market", "betfair", "sp", "bsp"])
        for c in stats["model_columns"]
    )

    if raw_races_ok and raw_runners_ok and race_completion > 0.9 and not forbidden_in_schema:
        print("✓ GATE CHECK PASSED")
        print(f"  - Raw race count: {stats['raw_total_races']:,} matches expected ~15,294")
        print(f"  - Raw runner count: {stats['raw_total_runners']:,} matches expected ~124,673")
        print(f"  - Valid races: {stats['race_count']:,} ({race_completion:.1%} of raw)")
        print(f"  - Valid runners: {stats['row_count']:,} (excludes scratched/no-result)")
        print(f"  - No market features in model schema")
    else:
        print("✗ GATE CHECK FAILED")
        if not raw_races_ok:
            print(f"  - Raw race count {stats['raw_total_races']:,} outside expected range")
        if not raw_runners_ok:
            print(f"  - Raw runner count {stats['raw_total_runners']:,} outside expected range")
        if race_completion <= 0.9:
            print(f"  - Race completion rate {race_completion:.1%} below 90%")
        if forbidden_in_schema:
            print(f"  - Market features found in model schema")

    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
