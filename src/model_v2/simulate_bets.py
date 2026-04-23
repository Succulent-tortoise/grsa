"""
simulate_bets.py — Greyhound Prediction System v2

Simulates flat-stake betting on the validation set using frozen model predictions
and early odds.

Tests all combinations of:
- Edge thresholds: 3%, 5%, 7%, 10%
- Odds bands: $2-$5, $5-$10, $10-$20, $2-$15, $2-$20, uncapped

Also runs a favourite-only baseline for comparison.

Gate check: Report whether ANY combination achieves positive ROI.

Output: models/v2/simulation_results.json
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
EDGE_THRESHOLDS = [0.03, 0.05, 0.07, 0.10]
ODDS_BANDS = [
    (2.0, 5.0, "$2-$5"),
    (5.0, 10.0, "$5-$10"),
    (10.0, 20.0, "$10-$20"),
    (2.0, 15.0, "$2-$15"),
    (2.0, 20.0, "$2-$20"),
    (2.0, float("inf"), "uncapped"),
]
STAKE = 1.0  # Flat $1 stake


def load_validation_data_with_odds() -> pd.DataFrame:
    """Load validation predictions with odds."""
    # Load frozen predictions
    preds = pd.read_parquet(MODEL_DIR / "model_val_preds.parquet")
    preds["race_date"] = pd.to_datetime(preds["race_date"])

    # Load dataset with odds
    dataset = pd.read_parquet(MODEL_DIR / "dataset.parquet")
    dataset["race_date"] = pd.to_datetime(dataset["race_date"])

    # Join
    merged = preds.merge(
        dataset[["venue", "race_date", "race_number", "runner_name", "grsa_early_odds"]],
        on=["venue", "race_date", "race_number", "runner_name"],
        how="left",
    )

    # Filter to valid odds
    valid = merged[(merged["grsa_early_odds"].notna()) & (merged["grsa_early_odds"] > 1.0)].copy()

    # Compute implied probability and edge
    valid["implied_proba"] = 1.0 / valid["grsa_early_odds"]
    valid["edge"] = valid["model_proba"] - valid["implied_proba"]

    logger.info(f"Loaded {len(valid):,} runners with valid odds")

    return valid


def simulate_combination(
    df: pd.DataFrame,
    edge_threshold: float,
    odds_low: float,
    odds_high: float,
) -> dict[str, Any]:
    """Simulate betting for a single edge/odds combination."""

    # Filter by edge and odds
    mask = (
        (df["edge"] >= edge_threshold) &
        (df["grsa_early_odds"] >= odds_low) &
        (df["grsa_early_odds"] < odds_high)
    )
    bets = df[mask].copy()

    if len(bets) == 0:
        return {
            "edge_threshold": edge_threshold,
            "odds_band": f"${odds_low:.0f}-${odds_high:.0f}" if odds_high != float("inf") else f"${odds_low:.0f}+",
            "bet_count": 0,
            "win_count": 0,
            "total_staked": 0.0,
            "total_returned": 0.0,
            "roi_pct": None,
            "predicted_win_rate": None,
            "actual_win_rate": None,
        }

    # Simulate flat $1 stakes
    bets["stake"] = STAKE
    bets["return"] = np.where(
        bets["won"] == 1,
        bets["stake"] * bets["grsa_early_odds"],  # Win: get odds × stake
        0.0  # Loss: get nothing
    )

    total_staked = bets["stake"].sum()
    total_returned = bets["return"].sum()
    win_count = bets["won"].sum()
    roi = (total_returned - total_staked) / total_staked * 100

    return {
        "edge_threshold": edge_threshold,
        "odds_band": f"${odds_low:.0f}-${odds_high:.0f}" if odds_high != float("inf") else f"${odds_low:.0f}+",
        "bet_count": int(len(bets)),
        "win_count": int(win_count),
        "total_staked": float(total_staked),
        "total_returned": float(total_returned),
        "roi_pct": float(roi),
        "predicted_win_rate": float(bets["model_proba"].mean()),
        "actual_win_rate": float(bets["won"].mean()),
        "avg_odds": float(bets["grsa_early_odds"].mean()),
        "avg_edge": float(bets["edge"].mean()),
    }


def simulate_favourite_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """
    Simulate betting on the favourite (shortest odds) in each race.

    This is the market baseline - betting on who the market thinks will win.
    """
    # Group by race and find favourite (lowest odds)
    df_with_race_id = df.copy()
    df_with_race_id["race_id"] = (
        df_with_race_id["venue"].astype(str) + "_" +
        df_with_race_id["race_date"].astype(str) + "_" +
        df_with_race_id["race_number"].astype(str)
    )

    # Find favourite in each race
    fav_idx = df_with_race_id.groupby("race_id")["grsa_early_odds"].idxmin()
    favourites = df_with_race_id.loc[fav_idx].copy()

    # Simulate betting
    favourites["stake"] = STAKE
    favourites["return"] = np.where(
        favourites["won"] == 1,
        favourites["stake"] * favourites["grsa_early_odds"],
        0.0
    )

    total_staked = favourites["stake"].sum()
    total_returned = favourites["return"].sum()
    win_count = favourites["won"].sum()
    roi = (total_returned - total_staked) / total_staked * 100

    return {
        "strategy": "favourite_baseline",
        "description": "Bet $1 on the favourite (shortest odds) in each race",
        "bet_count": int(len(favourites)),
        "win_count": int(win_count),
        "total_staked": float(total_staked),
        "total_returned": float(total_returned),
        "roi_pct": float(roi),
        "actual_win_rate": float(favourites["won"].mean()),
        "avg_odds": float(favourites["grsa_early_odds"].mean()),
    }


def print_simulation_report(
    results: list[dict],
    favourite_baseline: dict,
    any_positive: bool,
) -> None:
    """Print formatted simulation results."""

    print("\n" + "=" * 90)
    print("BETTING SIMULATION RESULTS (Validation Set)")
    print("=" * 90)

    print(f"\nFlat stake: ${STAKE:.2f} per bet")
    print(f"Total runners with valid odds: {results[0]['total_runners'] if results else 'N/A'}")

    # Group by edge threshold
    for threshold in EDGE_THRESHOLDS:
        threshold_results = [r for r in results if r["edge_threshold"] == threshold]

        print(f"\n[Edge Threshold: {threshold*100:.0f}%]")
        print("-" * 90)
        print(f"{'Odds Band':<12} {'Bets':<8} {'Wins':<8} {'Staked':<10} {'Returned':<10} {'ROI':<10} {'Pred WR':<10} {'Actual WR':<10}")
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for r in threshold_results:
            if r["bet_count"] > 0:
                roi_str = f"{r['roi_pct']:+.1f}%"
                print(f"{r['odds_band']:<12} {r['bet_count']:<8,} {r['win_count']:<8,} "
                      f"${r['total_staked']:<9,.0f} ${r['total_returned']:<9,.0f} "
                      f"{roi_str:<10} {r['predicted_win_rate']*100:>6.1f}%    {r['actual_win_rate']*100:>6.1f}%")
            else:
                print(f"{r['odds_band']:<12} {0:<8} {0:<8} $0         $0         --         --         --")

    # Favourite baseline
    print(f"\n[FAVOURITE BASELINE]")
    print("-" * 90)
    fav = favourite_baseline
    print(f"Strategy: Bet $1 on the favourite (shortest odds) in each race")
    print(f"  Bets:          {fav['bet_count']:,}")
    print(f"  Wins:          {fav['win_count']:,} ({fav['actual_win_rate']*100:.1f}%)")
    print(f"  Total Staked:  ${fav['total_staked']:,.0f}")
    print(f"  Total Returned: ${fav['total_returned']:,.0f}")
    print(f"  ROI:           {fav['roi_pct']:+.1f}%")
    print(f"  Avg Odds:      ${fav['avg_odds']:.2f}")

    # Gate check summary
    print("\n" + "=" * 90)
    print("[GATE CHECK SUMMARY]")

    # Find best performing combination
    valid_results = [r for r in results if r["bet_count"] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x["roi_pct"])
        print(f"Best ROI: {best['roi_pct']:+.1f}% at {best['edge_threshold']*100:.0f}% edge, {best['odds_band']} odds")

        # Count positive ROI combinations
        positive_count = sum(1 for r in valid_results if r["roi_pct"] > 0)
        total_combinations = len(valid_results)
        print(f"Positive ROI combinations: {positive_count}/{total_combinations}")

    if any_positive:
        print("\n✓ AT LEAST ONE COMBINATION HAS POSITIVE ROI")
    else:
        print("\n✗ NO COMBINATION HAS POSITIVE ROI")

    print(f"\nFavourite baseline ROI: {fav['roi_pct']:+.1f}%")
    print("=" * 90)


def main():
    """Main entry point for betting simulation."""
    logger.info("=" * 90)
    logger.info("BETTING SIMULATION - Greyhound Prediction System v2")
    logger.info("=" * 90)

    # Load data
    df = load_validation_data_with_odds()
    total_runners = len(df)

    # Run all combinations
    results = []
    for threshold in EDGE_THRESHOLDS:
        for odds_low, odds_high, odds_label in ODDS_BANDS:
            result = simulate_combination(df, threshold, odds_low, odds_high)
            result["total_runners"] = total_runners
            results.append(result)
            logger.info(f"Edge {threshold*100:.0f}%, {odds_label}: {result['bet_count']} bets, ROI {result['roi_pct']}%")

    # Run favourite baseline
    favourite_baseline = simulate_favourite_baseline(df)
    logger.info(f"Favourite baseline: {favourite_baseline['bet_count']} bets, ROI {favourite_baseline['roi_pct']:+.1f}%")

    # Check if any combination has positive ROI
    valid_results = [r for r in results if r["bet_count"] > 0]
    any_positive = any(r["roi_pct"] > 0 for r in valid_results) if valid_results else False

    # Print report
    print_simulation_report(results, favourite_baseline, any_positive)

    # Save results
    output_file = MODEL_DIR / "simulation_results.json"
    output_data = {
        "config": {
            "edge_thresholds": [t * 100 for t in EDGE_THRESHOLDS],
            "odds_bands": [b[2] for b in ODDS_BANDS],
            "stake": STAKE,
        },
        "total_runners_with_odds": total_runners,
        "results": results,
        "favourite_baseline": favourite_baseline,
        "gate_check": {
            "any_positive_roi": any_positive,
            "positive_combinations": sum(1 for r in valid_results if r["roi_pct"] > 0),
            "total_combinations": len(valid_results),
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Output]")
    print(f"  Simulation results saved to: {output_file}")

    return results, favourite_baseline, any_positive


if __name__ == "__main__":
    main()
