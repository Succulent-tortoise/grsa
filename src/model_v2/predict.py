"""
predict.py — Greyhound Prediction System v2

Production prediction script using the frozen model.

Loads the frozen XGBoost model and produces win probability predictions
for runner inputs. Validates that no forbidden market features are present.

Usage:
    python predict.py --input <json_file>
    python predict.py --test  # Run gate check against validation sample

Gate check:
- Predictions match model_val_preds.parquet for sample runners
- Forbidden feature validation raises error on market data
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    FORBIDDEN_FEATURE_SUBSTRINGS,
    MODEL_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when prediction fails due to invalid input."""
    pass


class ForbiddenFeatureError(PredictionError):
    """Raised when input contains forbidden market features."""
    pass


def load_frozen_model() -> xgb.Booster:
    """Load the frozen XGBoost model from disk."""
    model_path = MODEL_DIR / "model_v2.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = xgb.Booster()
    model.load_model(str(model_path))

    logger.info(f"Loaded frozen model from {model_path}")
    return model


def load_feature_columns() -> list[str]:
    """Load the expected feature columns from disk."""
    feature_path = MODEL_DIR / "feature_columns.json"

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {feature_path}")

    with open(feature_path) as f:
        # Exclude target column if present
        columns = [c for c in json.load(f) if c != "won"]

    logger.info(f"Loaded {len(columns)} feature columns")
    return columns


def validate_input(df: pd.DataFrame, expected_columns: list[str]) -> None:
    """
    Validate input dataframe for prediction.

    Checks:
    1. All expected columns are present
    2. No forbidden market features are present
    3. All values are numeric
    """
    input_columns = set(df.columns.str.lower())

    # Check for forbidden features
    for col in df.columns:
        col_lower = col.lower()
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
            if forbidden in col_lower:
                raise ForbiddenFeatureError(
                    f"Forbidden feature '{col}' detected in input. "
                    f"Market data must not be used for predictions."
                )

    # Check for expected columns
    expected_set = set(expected_columns)
    missing = expected_set - set(df.columns)

    if missing:
        raise PredictionError(
            f"Missing required features: {missing}"
        )

    # Check for numeric values
    for col in expected_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise PredictionError(
                    f"Feature '{col}' must be numeric, got {df[col].dtype}"
                )


def predict(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """
    Generate win probability predictions for runners.

    Args:
        model: Loaded XGBoost model
        df: DataFrame with runner features
        feature_columns: Expected feature column names

    Returns:
        Array of win probabilities (same order as input)
    """
    # Validate input
    validate_input(df, feature_columns)

    # Extract features in correct order
    X = df[feature_columns].values

    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(X, feature_names=feature_columns)

    # Generate predictions
    probabilities = model.predict(dmatrix)

    logger.info(f"Generated {len(probabilities)} predictions")
    logger.info(f"Probability range: {probabilities.min():.4f} - {probabilities.max():.4f}")

    return probabilities


def predict_from_dict(model: xgb.Booster, runners: list[dict], feature_columns: list[str]) -> list[dict]:
    """
    Generate predictions from a list of runner dictionaries.

    Args:
        model: Loaded XGBoost model
        runners: List of dicts with runner features
        feature_columns: Expected feature column names

    Returns:
        List of dicts with runner data and predictions
    """
    df = pd.DataFrame(runners)
    probabilities = predict(model, df, feature_columns)

    results = []
    for i, runner in enumerate(runners):
        result = dict(runner)
        result["model_proba"] = float(probabilities[i])
        results.append(result)

    return results


def run_gate_check() -> dict[str, Any]:
    """
    Run gate check to verify predictions match frozen model output.

    Tests:
    1. Predictions match model_val_preds.parquet for sample runners
    2. Forbidden feature validation raises error
    """
    logger.info("=" * 60)
    logger.info("GATE CHECK - Production Predictions")
    logger.info("=" * 60)

    results = {
        "test_1_prediction_match": None,
        "test_2_forbidden_feature": None,
        "gate_passed": False,
    }

    # Load model and feature columns
    model = load_frozen_model()
    feature_columns = load_feature_columns()

    # Load frozen predictions
    frozen_preds = pd.read_parquet(MODEL_DIR / "model_val_preds.parquet")

    # Load validation data with features
    val_data = pd.read_parquet(MODEL_DIR / "val.parquet")

    # Test 1: Predictions match frozen output
    logger.info("\n[Test 1] Predictions match frozen model output")

    # Sample 10 runners from validation
    sample_idx = [100, 500, 1000, 2000, 5000, 7500, 10000, 15000, 20000, 23000]
    sample_idx = [i for i in sample_idx if i < len(val_data)][:10]

    # If we don't have 10, just take first 10
    if len(sample_idx) < 10:
        sample_idx = list(range(min(10, len(val_data))))

    test_results = []
    all_match = True
    max_diff = 0.0

    for idx in sample_idx:
        runner = val_data.iloc[idx]

        # Get feature values
        runner_features = {col: runner[col] for col in feature_columns}

        # Get frozen prediction
        frozen_proba = frozen_preds.iloc[idx]["model_proba"]

        # Generate new prediction
        df = pd.DataFrame([runner_features])
        new_proba = predict(model, df, feature_columns)[0]

        # Compare
        diff = abs(frozen_proba - new_proba)
        max_diff = max(max_diff, diff)
        matches = diff < 1e-6

        test_results.append({
            "idx": int(idx),
            "frozen_proba": float(frozen_proba),
            "new_proba": float(new_proba),
            "diff": float(diff),
            "matches": bool(matches),
        })

        if not matches:
            all_match = False
            logger.warning(f"  Mismatch at idx {idx}: frozen={frozen_proba:.6f}, new={new_proba:.6f}, diff={diff:.6f}")

    results["test_1_prediction_match"] = {
        "passed": bool(all_match),
        "samples_tested": len(test_results),
        "max_difference": float(max_diff),
        "tolerance": 1e-6,
        "details": test_results,
    }

    if all_match:
        logger.info(f"  ✓ All {len(test_results)} predictions match (max diff: {max_diff:.2e})")
    else:
        logger.error("  ✗ Some predictions do not match")

    # Test 2: Forbidden feature validation
    logger.info("\n[Test 2] Forbidden feature validation")

    forbidden_test_passed = False
    error_message = None

    try:
        # Try to predict with a forbidden feature
        runner_with_odds = {col: 0.0 for col in feature_columns}
        runner_with_odds["betfair_odds"] = 5.0  # Forbidden!

        df = pd.DataFrame([runner_with_odds])
        predict(model, df, feature_columns)

        # Should not reach here
        logger.error("  ✗ ForbiddenFeatureError was not raised!")
    except ForbiddenFeatureError as e:
        forbidden_test_passed = True
        error_message = str(e)
        logger.info(f"  ✓ ForbiddenFeatureError raised correctly: {e}")
    except Exception as e:
        logger.error(f"  ✗ Wrong exception type: {type(e).__name__}: {e}")

    results["test_2_forbidden_feature"] = {
        "passed": bool(forbidden_test_passed),
        "error_type": "ForbiddenFeatureError",
        "error_message": error_message,
    }

    # Gate check summary
    results["gate_passed"] = all_match and forbidden_test_passed

    logger.info("\n" + "=" * 60)
    if results["gate_passed"]:
        logger.info("✓ GATE CHECK PASSED")
        logger.info("  - Predictions match frozen model output")
        logger.info("  - Forbidden feature validation works correctly")
    else:
        logger.error("✗ GATE CHECK FAILED")
    logger.info("=" * 60)

    return results


def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(description="Generate win probability predictions")
    parser.add_argument("--input", type=str, help="JSON file with runner features")
    parser.add_argument("--output", type=str, help="Output file for predictions")
    parser.add_argument("--test", action="store_true", help="Run gate check")
    args = parser.parse_args()

    if args.test:
        results = run_gate_check()

        # Save test results
        output_path = MODEL_DIR / "predict_test_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[Output] Test results saved to: {output_path}")

        sys.exit(0 if results["gate_passed"] else 1)

    elif args.input:
        # Load model
        model = load_frozen_model()
        feature_columns = load_feature_columns()

        # Load input
        with open(args.input) as f:
            runners = json.load(f)

        logger.info(f"Loaded {len(runners)} runners from {args.input}")

        # Generate predictions
        results = predict_from_dict(model, runners, feature_columns)

        # Output
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Predictions saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
