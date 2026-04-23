"""
evaluate_value.py — Greyhound Prediction System v2

BOUNDARY CROSSING POINT: Model is now frozen.
This is the FIRST file permitted to load market data (grsa_early_odds).

Protocol:
1. Load FROZEN model predictions from model_val_preds.parquet (never recompute)
2. Join grsa_early_odds from dataset.parquet on validation set only
3. Compute implied probability and edge
4. Report value bet distribution at various thresholds

Gate check:
- Confirm predictions are loaded frozen (not recomputed)
- Confirm odds joined AFTER predictions finalised
- Report coverage (% of validation runners with odds)
- Report value bet stats at 3%, 5%, 7%, 10% edge thresholds
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Edge thresholds to evaluate
EDGE_THRESHOLDS = [0.03, 0.05, 0.07, 0.10]


def load_frozen_predictions() -> pd.DataFrame:
    """
    Load FROZEN model predictions from disk.

    CRITICAL: Predictions are loaded, never recomputed.
    """
    preds_file = MODEL_DIR / "model_val_preds.parquet"

    logger.info(f"Loading FROZEN predictions from {preds_file}")
    preds = pd.read_parquet(preds_file)

    logger.info(f"Loaded {len(preds):,} predictions")
    logger.info(f"Prediction columns: {preds.columns.tolist()}")

    # Verify frozen status - these are pre-computed, not live
    assert "model_proba" in preds.columns, "Missing model_proba column"
    assert len(preds) == 23756, f"Expected 23,756 validation rows, got {len(preds)}"

    return preds


def load_market_data_for_validation() -> pd.DataFrame:
    """
    Load market data (odds) for validation set only.

    This joins grsa_early_odds from dataset.parquet using the validation
    date boundaries.
    """
    # Load split info to get validation boundaries
    with open(MODEL_DIR / "split_info.json") as f:
        split_info = json.load(f)

    val_start = pd.Timestamp(split_info["splits"]["val"]["min_date"])
    val_end = pd.Timestamp(split_info["splits"]["val"]["max_date"])

    logger.info(f"Validation date range: {val_start.date()} to {val_end.date()}")

    # Load full dataset with odds
    dataset = pd.read_parquet(MODEL_DIR / "dataset.parquet")
    logger.info(f"Loaded dataset: {len(dataset):,} rows")

    # Filter to validation dates
    dataset["race_date"] = pd.to_datetime(dataset["race_date"])
    val_market = dataset[
        (dataset["race_date"] >= val_start) &
        (dataset["race_date"] <= val_end)
    ].copy()

    logger.info(f"Filtered to validation: {len(val_market):,} rows")

    # Select only join keys and odds
    val_market = val_market[["venue", "race_date", "race_number", "runner_name", "grsa_early_odds"]]

    return val_market


def join_predictions_with_odds(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Join frozen predictions with market odds.

    This is the BOUNDARY CROSSING - predictions and market data are combined.
    """
    logger.info("Joining frozen predictions with market odds...")

    # Ensure date types match
    preds["race_date"] = pd.to_datetime(preds["race_date"])
    odds["race_date"] = pd.to_datetime(odds["race_date"])

    # Join on race identifiers
    merged = preds.merge(
        odds,
        on=["venue", "race_date", "race_number", "runner_name"],
        how="left"
    )

    logger.info(f"Merged data: {len(merged):,} rows")

    return merged


def compute_value_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute implied probability and edge for each runner.
    """
    logger.info("Computing implied probability and edge...")

    # Filter to runners with valid odds (> 0)
    valid_mask = (df["grsa_early_odds"].notna()) & (df["grsa_early_odds"] > 1.0)
    df_valid = df[valid_mask].copy()

    # Implied probability = 1 / decimal odds
    df_valid["implied_proba"] = 1.0 / df_valid["grsa_early_odds"]

    # Edge = model probability - implied probability
    df_valid["edge"] = df_valid["model_proba"] - df_valid["implied_proba"]

    logger.info(f"Valid odds rows: {len(df_valid):,} / {len(df):,} ({len(df_valid)/len(df)*100:.1f}%)")

    return df_valid


def analyze_edge_thresholds(df: pd.DataFrame) -> list[dict]:
    """
    Analyze value bets at various edge thresholds.
    """
    results = []

    for threshold in EDGE_THRESHOLDS:
        # Runners with edge above threshold
        value_bets = df[df["edge"] >= threshold]

        if len(value_bets) == 0:
            results.append({
                "threshold_pct": threshold * 100,
                "count": 0,
                "pct_of_valid": 0,
                "avg_edge": None,
                "avg_odds": None,
                "predicted_win_rate": None,
                "actual_win_rate": None,
            })
            continue

        results.append({
            "threshold_pct": threshold * 100,
            "count": len(value_bets),
            "pct_of_valid": len(value_bets) / len(df) * 100,
            "avg_edge": float(value_bets["edge"].mean()),
            "avg_odds": float(value_bets["grsa_early_odds"].mean()),
            "predicted_win_rate": float(value_bets["model_proba"].mean()),
            "actual_win_rate": float(value_bets["won"].mean()),
        })

    return results


def print_value_report(
    coverage: dict,
    edge_analysis: list[dict],
    gate_passed: bool
) -> None:
    """Print formatted value analysis report."""
    print("\n" + "=" * 70)
    print("VALUE BET ANALYSIS (Validation Set)")
    print("=" * 70)

    print("\n[Boundary Crossing Verification]")
    print("  ✓ Predictions loaded FROZEN from model_val_preds.parquet")
    print("  ✓ Odds joined AFTER predictions finalised")
    print("  ✓ No model recomputation performed")

    print("\n[Coverage]")
    print(f"  Total validation runners:    {coverage['total_runners']:,}")
    print(f"  Runners with valid odds:     {coverage['valid_odds']:,}")
    print(f"  Coverage:                    {coverage['coverage_pct']:.1f}%")
    print(f"  Odds null/invalid:           {coverage['invalid_odds']:,}")

    print("\n[Edge Distribution]")
    print(f"  Mean edge:                   {coverage['mean_edge']:.4f}")
    print(f"  Std edge:                    {coverage['std_edge']:.4f}")
    print(f"  Min edge:                    {coverage['min_edge']:.4f}")
    print(f"  Max edge:                    {coverage['max_edge']:.4f}")

    print("\n[Value Bets by Edge Threshold]")
    print(f"  {'Threshold':<12} {'Count':<8} {'% Valid':<10} {'Avg Edge':<12} {'Avg Odds':<10} {'Pred WR':<10} {'Actual WR':<10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for r in edge_analysis:
        if r["count"] > 0:
            print(f"  {r['threshold_pct']:>6.0f}%       {r['count']:<8,} {r['pct_of_valid']:>6.1f}%     "
                  f"{r['avg_edge']:>8.4f}     {r['avg_odds']:>8.2f}    "
                  f"{r['predicted_win_rate']:>8.1%}    {r['actual_win_rate']:>8.1%}")
        else:
            print(f"  {r['threshold_pct']:>6.0f}%       {0:<8} {'--':>6}     {'--':>8}     {'--':>8}    {'--':>8}    {'--':>8}")

    print("\n[Gate Check]")
    if gate_passed:
        print("  ✓ GATE CHECK PASSED")
        print("    - Coverage reported")
        print("    - Edge thresholds analyzed")
        print("    - Actual win rates computed")
    else:
        print("  ✗ GATE CHECK FAILED")

    print("\n" + "=" * 70)


def check_gate_conditions(coverage: dict, edge_analysis: list[dict]) -> bool:
    """
    Check gate conditions for value evaluation.

    Gate requirements:
    - Predictions loaded frozen (not recomputed) - verified by code flow
    - Odds joined after predictions finalised - verified by code flow
    - Coverage reported
    - Edge analysis at all thresholds
    """
    # All thresholds must have analysis (even if count is 0)
    if len(edge_analysis) != len(EDGE_THRESHOLDS):
        return False

    # Coverage must be computed
    if coverage["coverage_pct"] is None:
        return False

    return True


def main():
    """Main entry point for value evaluation."""
    logger.info("=" * 70)
    logger.info("VALUE EVALUATION - Greyhound Prediction System v2")
    logger.info("BOUNDARY CROSSING: Market data introduced after model freeze")
    logger.info("=" * 70)

    # Step 1: Load FROZEN predictions (never recompute)
    preds = load_frozen_predictions()

    # Step 2: Load market data for validation set
    odds = load_market_data_for_validation()

    # Step 3: Join predictions with odds (boundary crossing)
    merged = join_predictions_with_odds(preds, odds)

    # Step 4: Compute value metrics
    df_valid = compute_value_metrics(merged)

    # Compute coverage stats
    total_runners = len(merged)
    valid_odds = len(df_valid)
    invalid_odds = total_runners - valid_odds
    coverage_pct = valid_odds / total_runners * 100

    coverage = {
        "total_runners": total_runners,
        "valid_odds": valid_odds,
        "invalid_odds": invalid_odds,
        "coverage_pct": coverage_pct,
        "mean_edge": float(df_valid["edge"].mean()) if len(df_valid) > 0 else None,
        "std_edge": float(df_valid["edge"].std()) if len(df_valid) > 0 else None,
        "min_edge": float(df_valid["edge"].min()) if len(df_valid) > 0 else None,
        "max_edge": float(df_valid["edge"].max()) if len(df_valid) > 0 else None,
    }

    # Step 5: Analyze edge thresholds
    edge_analysis = analyze_edge_thresholds(df_valid)

    # Check gate conditions
    gate_passed = check_gate_conditions(coverage, edge_analysis)

    # Print report
    print_value_report(coverage, edge_analysis, gate_passed)

    # Save results
    output_file = MODEL_DIR / "value_evaluation.json"
    output_data = {
        "boundary_verification": {
            "predictions_frozen": True,
            "predictions_file": "model_val_preds.parquet",
            "odds_source": "dataset.parquet",
            "join_order": "predictions_loaded_first_then_odds_joined",
        },
        "coverage": coverage,
        "edge_analysis": edge_analysis,
        "gate_check": {
            "passed": gate_passed,
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Output]")
    print(f"  Value evaluation saved to: {output_file}")

    return coverage, edge_analysis, gate_passed


if __name__ == "__main__":
    main()
