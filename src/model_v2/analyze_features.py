"""
analyze_features.py — Greyhound Prediction System v2

Computes SHAP values for the trained XGBoost model and reports feature importance.

Gate check:
- No single feature > 30% of total SHAP importance
- Flag features < 1% as removal candidates

Output:
- models/v2/feature_importance.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from config import MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_data():
    """Load trained model and validation data."""
    # Load model
    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / "model_v2.json"))
    logger.info(f"Loaded model from {MODEL_DIR / 'model_v2.json'}")

    # Load feature columns (excluding target)
    with open(MODEL_DIR / "feature_columns.json") as f:
        all_columns = json.load(f)
    feature_cols = [c for c in all_columns if c != "won"]
    logger.info(f"Feature columns: {feature_cols}")

    # Load validation data
    val_df = pd.read_parquet(MODEL_DIR / "val.parquet")
    X_val = val_df[feature_cols]
    logger.info(f"Validation data shape: {X_val.shape}")

    return model, X_val, feature_cols


def compute_shap_values(model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for the model."""
    logger.info("Computing SHAP values...")

    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    logger.info(f"SHAP values shape: {shap_values.shape}")
    return shap_values


def compute_feature_importance(
    shap_values: np.ndarray,
    feature_cols: list[str],
) -> dict:
    """
    Compute feature importance from SHAP values.

    Uses mean absolute SHAP value as importance metric.
    """
    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Total importance (for percentage calculation)
    total_importance = mean_abs_shap.sum()

    # Build importance ranking
    importance_data = []
    for i, col in enumerate(feature_cols):
        importance = mean_abs_shap[i]
        percentage = (importance / total_importance) * 100

        importance_data.append({
            "feature": col,
            "mean_abs_shap": float(importance),
            "importance_pct": float(percentage),
            "rank": 0,  # Will be set after sorting
        })

    # Sort by importance descending
    importance_data.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    # Assign ranks
    for i, item in enumerate(importance_data):
        item["rank"] = i + 1

    return {
        "features": importance_data,
        "total_importance": float(total_importance),
        "n_features": len(feature_cols),
    }


def check_gate_conditions(importance: dict) -> dict:
    """
    Check gate conditions for feature importance.

    Returns:
        Dictionary with gate check results
    """
    features = importance["features"]

    # Check for dominant feature (> 30%)
    dominant_features = [f for f in features if f["importance_pct"] > 30]

    # Check for low importance features (< 1%)
    low_importance_features = [f for f in features if f["importance_pct"] < 1]

    gate_passed = len(dominant_features) == 0

    return {
        "gate_passed": gate_passed,
        "dominant_features": dominant_features,
        "low_importance_features": low_importance_features,
        "warnings": {
            "dominant": [f"{f['feature']} ({f['importance_pct']:.1f}%)" for f in dominant_features],
            "low_importance": [f"{f['feature']} ({f['importance_pct']:.1f}%)" for f in low_importance_features],
        },
    }


def print_importance_report(importance: dict, gate: dict) -> None:
    """Print formatted feature importance report."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS (SHAP)")
    print("=" * 60)

    print(f"\nTotal features: {importance['n_features']}")
    print(f"Total importance: {importance['total_importance']:.4f}")

    print("\n[Importance Ranking]")
    print(f"  {'Rank':<6} {'Feature':<22} {'Mean|SHAP|':<12} {'Importance':<12}")
    print(f"  {'-'*6} {'-'*22} {'-'*12} {'-'*12}")

    for f in importance["features"]:
        flag = ""
        if f["importance_pct"] > 30:
            flag = " ⚠️ DOMINANT"
        elif f["importance_pct"] < 1:
            flag = " ⚪ LOW"
        print(f"  {f['rank']:<6} {f['feature']:<22} {f['mean_abs_shap']:<12.4f} {f['importance_pct']:>6.1f}%{flag}")

    print("\n[Gate Check]")
    if gate["gate_passed"]:
        print("  ✓ No feature exceeds 30% importance")
    else:
        print("  ✗ DOMINANT FEATURES DETECTED:")
        for f in gate["dominant_features"]:
            print(f"    - {f['feature']}: {f['importance_pct']:.1f}%")

    if gate["low_importance_features"]:
        print("\n[Low Importance Candidates for Removal (< 1%)]")
        for f in gate["low_importance_features"]:
            print(f"    - {f['feature']}: {f['importance_pct']:.1f}%")
    else:
        print("\n  No features below 1% importance threshold")

    print("\n" + "=" * 60)
    if gate["gate_passed"]:
        print("✓ GATE CHECK PASSED")
    else:
        print("✗ GATE CHECK FAILED — STOP AND REPORT")
    print("=" * 60)


def main():
    """Main entry point for feature importance analysis."""
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE ANALYSIS - Greyhound Prediction System v2")
    logger.info("=" * 60)

    # Load model and data
    model, X_val, feature_cols = load_model_and_data()

    # Compute SHAP values
    shap_values = compute_shap_values(model, X_val)

    # Compute feature importance
    importance = compute_feature_importance(shap_values, feature_cols)

    # Check gate conditions
    gate = check_gate_conditions(importance)

    # Print report
    print_importance_report(importance, gate)

    # Save results
    output_file = MODEL_DIR / "feature_importance.json"
    output_data = {
        "features": importance["features"],
        "total_importance": importance["total_importance"],
        "n_features": importance["n_features"],
        "gate_check": {
            "passed": gate["gate_passed"],
            "dominant_features": gate["dominant_features"],
            "low_importance_features": gate["low_importance_features"],
            "warnings": gate["warnings"],
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Output]")
    print(f"  Feature importance saved to: {output_file}")

    return importance, gate


if __name__ == "__main__":
    main()
