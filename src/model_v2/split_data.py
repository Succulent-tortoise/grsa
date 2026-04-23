"""
split_data.py — Greyhound Prediction System v2

Split features.parquet into three chronological partitions:
- Train: Sep 30 – Dec 15, 2025 (~70%)
- Validation: Dec 16, 2025 – Jan 15, 2026 (~15%)
- Test: Jan 16 – Feb 9, 2026 (~15%)

CRITICAL RULES:
- Test set is LOCKED until all modelling decisions are final
- Test set is written to a SEPARATE file
- Training/validation code must NOT load the test set
- Never shuffle — strict chronological order only

Gate check:
- Row counts per split
- Win rates per split
- No date overlap between splits
- Test set in separate file
"""

import json
import logging
from pathlib import Path

import pandas as pd

from config import MODEL_DIR, TRAIN_END_DATE, VAL_END_DATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Input/output files
INPUT_FILE = MODEL_DIR / "features.parquet"
TRAIN_FILE = MODEL_DIR / "train.parquet"
VAL_FILE = MODEL_DIR / "val.parquet"
TEST_FILE = MODEL_DIR / "test.parquet"
SPLIT_INFO_FILE = MODEL_DIR / "split_info.json"


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test based on date boundaries.

    Splits:
    - Train: race_date <= TRAIN_END_DATE (2025-12-15)
    - Val: TRAIN_END_DATE < race_date <= VAL_END_DATE (2026-01-15)
    - Test: race_date > VAL_END_DATE (2026-01-15)

    Args:
        df: DataFrame with 'race_date' column

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])

    train_end = pd.to_datetime(TRAIN_END_DATE)
    val_end = pd.to_datetime(VAL_END_DATE)

    # Train: Sep 30 – Dec 15, 2025
    train_mask = df["race_date_dt"] <= train_end
    train_df = df[train_mask].drop(columns=["race_date_dt"]).reset_index(drop=True)

    # Validation: Dec 16, 2025 – Jan 15, 2026
    val_mask = (df["race_date_dt"] > train_end) & (df["race_date_dt"] <= val_end)
    val_df = df[val_mask].drop(columns=["race_date_dt"]).reset_index(drop=True)

    # Test: Jan 16 – Feb 9, 2026
    test_mask = df["race_date_dt"] > val_end
    test_df = df[test_mask].drop(columns=["race_date_dt"]).reset_index(drop=True)

    return train_df, val_df, test_df


def validate_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Validate that there is no date overlap between splits.

    Returns:
        Dictionary with overlap check results
    """
    train_dates = set(pd.to_datetime(train_df["race_date"]).dt.date)
    val_dates = set(pd.to_datetime(val_df["race_date"]).dt.date)
    test_dates = set(pd.to_datetime(test_df["race_date"]).dt.date)

    train_val_overlap = train_dates & val_dates
    val_test_overlap = val_dates & test_dates
    train_test_overlap = train_dates & test_dates

    return {
        "train_val_overlap": len(train_val_overlap),
        "val_test_overlap": len(val_test_overlap),
        "train_test_overlap": len(train_test_overlap),
        "train_date_range": (min(train_dates), max(train_dates)) if train_dates else None,
        "val_date_range": (min(val_dates), max(val_dates)) if val_dates else None,
        "test_date_range": (min(test_dates), max(test_dates)) if test_dates else None,
    }


def compute_split_stats(df: pd.DataFrame, name: str) -> dict:
    """
    Compute statistics for a data split.

    Returns:
        Dictionary with split statistics
    """
    dates = pd.to_datetime(df["race_date"])

    return {
        "name": name,
        "row_count": len(df),
        "race_count": df.groupby(["venue", "race_date", "race_number"]).ngroups,
        "win_rate": df["won"].mean(),
        "min_date": dates.min().strftime("%Y-%m-%d"),
        "max_date": dates.max().strftime("%Y-%m-%d"),
        "unique_dates": dates.nunique(),
    }


def run_gate_check(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    Run comprehensive gate check on all splits.

    Returns:
        Dictionary with all gate check results
    """
    # Compute stats for each split
    train_stats = compute_split_stats(train_df, "Train")
    val_stats = compute_split_stats(val_df, "Validation")
    test_stats = compute_split_stats(test_df, "Test")

    # Validate no overlap
    overlap_results = validate_no_overlap(train_df, val_df, test_df)

    # Total counts
    total_rows = len(train_df) + len(val_df) + len(test_df)
    total_races = train_stats["race_count"] + val_stats["race_count"] + test_stats["race_count"]

    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "overlap": overlap_results,
        "total_rows": total_rows,
        "total_races": total_races,
        "train_pct": len(train_df) / total_rows * 100,
        "val_pct": len(val_df) / total_rows * 100,
        "test_pct": len(test_df) / total_rows * 100,
    }


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("DATA SPLIT - Greyhound Prediction System v2")
    logger.info("=" * 60)

    # Load features
    logger.info(f"\n[1] Loading features from {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    logger.info(f"  Loaded {len(df):,} rows")

    # Split data
    logger.info("\n[2] Splitting data chronologically...")
    logger.info(f"  Train boundary: <= {TRAIN_END_DATE}")
    logger.info(f"  Val boundary:   {TRAIN_END_DATE} < date <= {VAL_END_DATE}")
    logger.info(f"  Test boundary:  > {VAL_END_DATE}")

    train_df, val_df, test_df = split_data(df)

    # Run gate check
    logger.info("\n[3] Running gate check...")
    gate_results = run_gate_check(train_df, val_df, test_df)

    # Print gate check results
    print("\n" + "=" * 60)
    print("GATE CHECK RESULTS")
    print("=" * 60)

    print("\n[Split Statistics]")

    for split_name, stats_key, pct_key in [
        ("Train", "train", "train_pct"),
        ("Validation", "val", "val_pct"),
        ("Test", "test", "test_pct"),
    ]:
        stats = gate_results[stats_key]
        print(f"\n  {split_name}:")
        print(f"    Rows:         {stats['row_count']:,} ({gate_results[pct_key]:.1f}%)")
        print(f"    Races:        {stats['race_count']:,}")
        print(f"    Win rate:     {stats['win_rate']:.1%}")
        print(f"    Date range:   {stats['min_date']} to {stats['max_date']}")
        print(f"    Unique dates: {stats['unique_dates']}")

    print(f"\n[Total]")
    print(f"  Total rows:  {gate_results['total_rows']:,}")
    print(f"  Total races: {gate_results['total_races']:,}")

    print(f"\n[Date Overlap Check]")
    overlap = gate_results["overlap"]
    print(f"  Train-Val overlap: {overlap['train_val_overlap']} dates")
    print(f"  Val-Test overlap:  {overlap['val_test_overlap']} dates")
    print(f"  Train-Test overlap: {overlap['train_test_overlap']} dates")

    print(f"\n[Date Ranges]")
    print(f"  Train: {overlap['train_date_range'][0]} to {overlap['train_date_range'][1]}")
    print(f"  Val:   {overlap['val_date_range'][0]} to {overlap['val_date_range'][1]}")
    print(f"  Test:  {overlap['test_date_range'][0]} to {overlap['test_date_range'][1]}")

    # Save splits
    print(f"\n[4] Saving splits...")

    print(f"  Train: {TRAIN_FILE}")
    train_df.to_parquet(TRAIN_FILE, index=False)
    print(f"    {len(train_df):,} rows, {TRAIN_FILE.stat().st_size / 1024:.1f} KB")

    print(f"  Val:   {VAL_FILE}")
    val_df.to_parquet(VAL_FILE, index=False)
    print(f"    {len(val_df):,} rows, {VAL_FILE.stat().st_size / 1024:.1f} KB")

    print(f"  Test:  {TEST_FILE}")
    test_df.to_parquet(TEST_FILE, index=False)
    print(f"    {len(test_df):,} rows, {TEST_FILE.stat().st_size / 1024:.1f} KB")

    # Save split info
    split_info = {
        "train_end_date": TRAIN_END_DATE,
        "val_end_date": VAL_END_DATE,
        "splits": {
            "train": {
                "file": str(TRAIN_FILE.name),
                "row_count": gate_results["train"]["row_count"],
                "race_count": gate_results["train"]["race_count"],
                "win_rate": gate_results["train"]["win_rate"],
                "min_date": gate_results["train"]["min_date"],
                "max_date": gate_results["train"]["max_date"],
            },
            "val": {
                "file": str(VAL_FILE.name),
                "row_count": gate_results["val"]["row_count"],
                "race_count": gate_results["val"]["race_count"],
                "win_rate": gate_results["val"]["win_rate"],
                "min_date": gate_results["val"]["min_date"],
                "max_date": gate_results["val"]["max_date"],
            },
            "test": {
                "file": str(TEST_FILE.name),
                "row_count": gate_results["test"]["row_count"],
                "race_count": gate_results["test"]["race_count"],
                "win_rate": gate_results["test"]["win_rate"],
                "min_date": gate_results["test"]["min_date"],
                "max_date": gate_results["test"]["max_date"],
            },
        },
        "overlap_check": {
            "train_val": overlap["train_val_overlap"],
            "val_test": overlap["val_test_overlap"],
            "train_test": overlap["train_test_overlap"],
        },
        "test_locked": True,
        "test_locked_reason": "Test set must not be inspected until all modelling decisions are final",
    }

    with open(SPLIT_INFO_FILE, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Split info: {SPLIT_INFO_FILE}")

    # Final gate status
    print("\n" + "=" * 60)

    # Check conditions
    no_overlap = (
        overlap["train_val_overlap"] == 0
        and overlap["val_test_overlap"] == 0
        and overlap["train_test_overlap"] == 0
    )

    # Verify date boundaries match PLAN.md
    correct_boundaries = (
        overlap["train_date_range"][1].strftime("%Y-%m-%d") == TRAIN_END_DATE
        and overlap["val_date_range"][0].strftime("%Y-%m-%d") == "2025-12-16"
        and overlap["val_date_range"][1].strftime("%Y-%m-%d") == VAL_END_DATE
        and overlap["test_date_range"][0].strftime("%Y-%m-%d") == "2026-01-16"
    )

    win_rates_consistent = (
        0.10 < gate_results["train"]["win_rate"] < 0.20
        and 0.10 < gate_results["val"]["win_rate"] < 0.20
        and 0.10 < gate_results["test"]["win_rate"] < 0.20
    )

    if no_overlap and correct_boundaries and win_rates_consistent:
        print("✓ GATE CHECK PASSED")
        print(f"  - No date overlap between splits")
        print(f"  - Date boundaries match PLAN.md specification")
        print(f"  - Train: {gate_results['train_pct']:.1f}% ({gate_results['train']['row_count']:,} rows)")
        print(f"  - Val:   {gate_results['val_pct']:.1f}% ({gate_results['val']['row_count']:,} rows)")
        print(f"  - Test:  {gate_results['test_pct']:.1f}% ({gate_results['test']['row_count']:,} rows)")
        print(f"  - Win rates consistent across splits (~14%)")
        print(f"  - Test set in SEPARATE file: {TEST_FILE.name}")
        print(f"  - Test set is LOCKED until modelling decisions final")
    else:
        print("✗ GATE CHECK FAILED")
        if not no_overlap:
            print(f"  - Date overlap detected between splits")
        if not correct_boundaries:
            print(f"  - Date boundaries do not match PLAN.md specification")
        if not win_rates_consistent:
            print(f"  - Win rates inconsistent across splits")

    print("=" * 60)

    # Warning about test set
    print("\n⚠️  TEST SET LOCKED")
    print(f"   File: {TEST_FILE}")
    print("   DO NOT load this file in training or validation code.")
    print("   DO NOT inspect test outcomes until modelling is complete.")
    print("   Violation requires creating a new hold-out period.")
    print()

    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
