"""
features.py — Greyhound Prediction System v2

Feature engineering from raw race/runner data.
All features are computed purely from race/runner/form data available at race time.
NO MARKET INPUTS.

CRITICAL: Runner history features must be computed with strict chronological ordering.
For each runner, only races PRIOR to the current race date may be used.
Process runners sorted by race_date ascending.

Gate check:
- Output full feature schema
- Confirm zero forbidden substrings in column names
- Show null rates per feature
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    DATA_RESULTS_DIR,
    FORBIDDEN_FEATURE_SUBSTRINGS,
    GRADE_ENCODING,
    MODEL_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Input/output files
INPUT_FILE = MODEL_DIR / "dataset.parquet"
OUTPUT_FILE = MODEL_DIR / "features.parquet"

# EWMA decay factor (more weight on recent races)
EWMA_ALPHA = 0.4


def parse_last_4(last_4_str: str | None) -> list[int]:
    """
    Parse the TSS_DOGS last_4 string into a list of finishing positions.

    The string contains positions from most recent to oldest.
    Examples:
        "5228" -> [5, 2, 2, 8] (most recent first)
        "42" -> [4, 2]
        "-" -> [] (no form)
        "4F" -> [4, None] (F = fell/failed to finish)
        "5228x" -> [5, 2, 2, 8] (x = scratched, ignore)

    Returns:
        List of integers (positions), most recent first. Empty if no valid form.
    """
    if not last_4_str or last_4_str == "-":
        return []

    positions = []
    for char in last_4_str:
        if char.isdigit():
            positions.append(int(char))
        # F, X, etc. are non-finishes - we skip them for position calculations
        # but they still count as a race having occurred

    return positions


def count_races_in_form(last_4_str: str | None) -> int:
    """
    Count the number of races represented in the last_4 string.
    Includes non-finishes (F, X, etc.) as they indicate a race occurred.
    """
    if not last_4_str or last_4_str == "-":
        return 0

    # Count all non-dash characters as races
    return len([c for c in last_4_str if c != "-"])


def compute_form_features(positions: list[int]) -> dict[str, float]:
    """
    Compute form features from a list of finishing positions.

    Args:
        positions: List of positions, most recent first

    Returns:
        Dictionary with ewma_position, trend_slope, form_volatility
    """
    if not positions:
        return {
            "ewma_position": np.nan,
            "trend_slope": np.nan,
            "form_volatility": np.nan,
        }

    n = len(positions)

    # EWMA: more weight on recent (lower index) positions
    # Reverse to have oldest first for EWMA calculation
    positions_reversed = positions[::-1]
    weights = [(1 - EWMA_ALPHA) ** (n - 1 - i) for i in range(n)]
    weights = np.array(weights) / sum(weights)
    ewma = float(np.average(positions_reversed, weights=weights))

    # Trend slope: linear regression on positions over time
    # Negative slope = improving (lower positions)
    # x = [0, 1, 2, ...] representing race sequence (oldest to newest)
    if n >= 2:
        x = np.arange(n)
        slope, _ = np.polyfit(x, positions_reversed, 1)
        trend_slope = float(slope)
    else:
        trend_slope = np.nan

    # Volatility: standard deviation of positions
    if n >= 2:
        volatility = float(np.std(positions))
    else:
        volatility = np.nan

    return {
        "ewma_position": ewma,
        "trend_slope": trend_slope,
        "form_volatility": volatility,
    }


def encode_grade(grade: str | None) -> float:
    """
    Encode grade string to numeric value.
    Higher values = better grades.
    """
    if not grade:
        return np.nan

    grade = grade.strip()

    # Direct lookup
    if grade in GRADE_ENCODING:
        return float(GRADE_ENCODING[grade])

    # Try partial matches for variations
    grade_upper = grade.upper()
    for key, value in GRADE_ENCODING.items():
        if key.upper() in grade_upper or grade_upper in key.upper():
            return float(value)

    # Unknown grade
    return np.nan


def compute_features_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features with strict chronological ordering.

    CRITICAL: For each runner, only races PRIOR to the current race
    may be used when computing historical features.

    Process:
    1. Sort by race_date ascending
    2. Maintain runner_history dict tracking (date, position) for each runner
    3. For each row, compute days_since_last_run from history BEFORE updating
    4. After processing row, add it to history

    Args:
        df: DataFrame with raw runner data

    Returns:
        DataFrame with all features computed
    """
    logger.info("Computing features with chronological ordering...")

    # Sort by race_date to ensure chronological processing
    df = df.copy()
    df["race_date_dt"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date_dt", "race_number"]).reset_index(drop=True)

    # Runner history: runner_name -> list of (date, position)
    # This tracks COMPLETED races only (used for days_since_last_run)
    runner_history: dict[str, list[tuple[datetime, int]]] = {}

    # Pre-compute field sizes for each race
    field_size_map = (
        df.groupby(["venue", "race_date", "race_number"])
        .size()
        .to_dict()
    )

    # Pre-compute best_t_d ranks for each race
    # Group by race and rank by best_t_d (lower is better)
    df["best_t_d_rank"] = np.nan

    # First pass: compute within-race features (best_t_d_rank, field_size)
    race_groups = df.groupby(["venue", "race_date", "race_number"])

    for (venue, race_date, race_num), race_df in race_groups:
        indices = race_df.index.tolist()

        # Field size (same for all runners in race)
        field_size = len(race_df)
        for idx in indices:
            df.loc[idx, "field_size"] = field_size

        # Best time at distance rank (1 = fastest, NaN if no time)
        best_t_d_values = race_df["best_t_d"].values
        valid_mask = ~np.isnan(best_t_d_values)

        if valid_mask.any():
            # Rank: 1 = best (lowest time), ascending=True
            ranks = np.full(len(best_t_d_values), np.nan)
            ranks[valid_mask] = stats.rankdata(best_t_d_values[valid_mask], method="average")
            for i, idx in enumerate(indices):
                df.loc[idx, "best_t_d_rank"] = ranks[i]

    # Second pass: compute chronological features (days_since_last_run from actual history)
    days_since_last_run = []

    for idx, row in df.iterrows():
        runner_name = row["runner_name"]
        race_date = row["race_date_dt"]
        won = row["won"]

        # Compute days_since_last_run from history (races BEFORE this one)
        if runner_name in runner_history and runner_history[runner_name]:
            # Get the most recent race before this one
            history = runner_history[runner_name]
            # Filter to races strictly before current date
            prior_races = [(d, p) for d, p in history if d < race_date]

            if prior_races:
                last_race_date = max(d for d, p in prior_races)
                days_diff = (race_date - last_race_date).days
                days_since_last_run.append(days_diff)
            else:
                days_since_last_run.append(np.nan)
        else:
            days_since_last_run.append(np.nan)

        # AFTER computing features, add this race to history
        if runner_name not in runner_history:
            runner_history[runner_name] = []
        runner_history[runner_name].append((race_date, won))

    df["days_since_last_run"] = days_since_last_run

    # Compute form features from last_4 string
    logger.info("Computing form features from last_4 string...")

    form_features = df["last_4"].apply(lambda x: compute_form_features(parse_last_4(x)))
    df["ewma_position"] = form_features.apply(lambda x: x["ewma_position"])
    df["trend_slope"] = form_features.apply(lambda x: x["trend_slope"])
    df["form_volatility"] = form_features.apply(lambda x: x["form_volatility"])
    df["num_races"] = df["last_4"].apply(count_races_in_form)

    # Encode grade
    df["grade_encoded"] = df["grade"].apply(encode_grade)

    # Box is already present
    # Distance is already present

    return df


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns that should be in the final feature set.

    Model features:
    - box, distance, grade_encoded, field_size (race context)
    - ewma_position, trend_slope, form_volatility, days_since_last_run, num_races (form)
    - best_t_d_rank (within-race relative)

    Target:
    - won

    Metadata (kept for traceability):
    - venue, race_date, race_number, runner_name

    Quarantined (market system only):
    - grsa_early_odds
    """
    feature_columns = [
        # Race context
        "box",
        "distance",
        "grade_encoded",
        "field_size",
        # Form features
        "ewma_position",
        "trend_slope",
        "form_volatility",
        "days_since_last_run",
        "num_races",
        # Within-race relative
        "best_t_d_rank",
        # Target
        "won",
    ]

    metadata_columns = [
        "venue",
        "race_date",
        "race_number",
        "runner_name",
        "trainer",
    ]

    quarantined_columns = [
        "grsa_early_odds",
    ]

    # Keep all relevant columns
    all_columns = feature_columns + metadata_columns + quarantined_columns
    available_columns = [c for c in all_columns if c in df.columns]

    return df[available_columns], feature_columns


def validate_no_forbidden_features(columns: list[str]) -> None:
    """
    Validate that no forbidden substrings appear in column names.

    Raises:
        ValueError if forbidden substrings found
    """
    forbidden_found = []

    for col in columns:
        col_lower = col.lower()
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
            if forbidden in col_lower:
                forbidden_found.append((col, forbidden))

    if forbidden_found:
        raise ValueError(
            f"Forbidden substrings found in feature columns:\n"
            + "\n".join(f"  '{col}' contains '{f}'" for col, f in forbidden_found)
        )

    logger.info("Schema validation passed - no forbidden market features")


def run_gate_check(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """
    Run gate check and return statistics.

    Returns:
        Dictionary with gate check results
    """
    # Null rates per feature
    null_rates = {}
    for col in feature_columns:
        if col in df.columns:
            null_rate = df[col].isna().mean()
            null_rates[col] = null_rate

    # Data types
    dtypes = {col: str(df[col].dtype) for col in feature_columns if col in df.columns}

    # Basic stats
    stats_summary = {}
    for col in feature_columns:
        if col in df.columns and df[col].notna().any():
            if df[col].dtype in ["float64", "int64"]:
                stats_summary[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                }

    return {
        "row_count": len(df),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "null_rates": null_rates,
        "dtypes": dtypes,
        "stats": stats_summary,
    }


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING - Greyhound Prediction System v2")
    logger.info("=" * 60)

    # Load dataset
    logger.info(f"\n[1] Loading dataset from {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    logger.info(f"  Loaded {len(df):,} rows")

    # Compute features
    logger.info("\n[2] Computing features...")
    df = compute_features_chronologically(df)

    # Select feature columns
    logger.info("\n[3] Selecting feature columns...")
    df, feature_columns = select_feature_columns(df)

    # Validate no forbidden features
    logger.info("\n[4] Validating schema...")
    validate_no_forbidden_features(feature_columns)

    # Run gate check
    logger.info("\n[5] Running gate check...")
    gate_results = run_gate_check(df, feature_columns)

    # Print gate check results
    print("\n" + "=" * 60)
    print("GATE CHECK RESULTS")
    print("=" * 60)

    print(f"\n[Feature Schema - {gate_results['feature_count']} features]")
    for col in gate_results["feature_columns"]:
        null_pct = gate_results["null_rates"].get(col, 0) * 100
        dtype = gate_results["dtypes"].get(col, "N/A")
        print(f"  {col:25s} | {dtype:10s} | {null_pct:5.1f}% null")

    print(f"\n[Null Rate Summary]")
    for col, rate in sorted(gate_results["null_rates"].items(), key=lambda x: -x[1]):
        print(f"  {col:25s}: {rate * 100:5.1f}%")

    print(f"\n[Feature Statistics]")
    for col, stats_dict in gate_results["stats"].items():
        print(f"  {col:25s}: min={stats_dict['min']:.2f}, max={stats_dict['max']:.2f}, mean={stats_dict['mean']:.2f}")

    print(f"\n[Forbidden Substring Check]")
    print("  Checked for: " + ", ".join(FORBIDDEN_FEATURE_SUBSTRINGS))
    print("  Result: None found in feature columns")

    # Save to parquet
    print(f"\n[Output]")
    print(f"  Saving to: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")

    # Save feature columns list for model training
    feature_cols_file = MODEL_DIR / "feature_columns.json"
    import json
    with open(feature_cols_file, "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"  Feature columns saved to: {feature_cols_file}")

    # Final gate status
    print("\n" + "=" * 60)

    # Check: no high null rates (except expected ones)
    high_null_features = [
        col for col, rate in gate_results["null_rates"].items()
        if rate > 0.5 and col not in ["best_t_d_rank", "days_since_last_run"]
    ]

    # Check: all features are numeric (except target)
    non_numeric = [
        col for col in feature_columns
        if col in df.columns and col != "won" and df[col].dtype == "object"
    ]

    if not high_null_features and not non_numeric:
        print("✓ GATE CHECK PASSED")
        print(f"  - {gate_results['feature_count']} features computed")
        print(f"  - Zero forbidden substrings in feature names")
        print(f"  - All features are numeric")
        print(f"  - High-null features (>50%) are expected (best_t_d_rank, days_since_last_run)")
    else:
        print("✗ GATE CHECK FAILED")
        if high_null_features:
            print(f"  - Unexpected high-null features: {high_null_features}")
        if non_numeric:
            print(f"  - Non-numeric features: {non_numeric}")

    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
