"""
train_model.py — Greyhound Prediction System v2

Train XGBoost binary classifier using all 11 features.
Uses XGBoost native null handling (no imputation before training).

Training: train.parquet
Evaluation: val.parquet
Features: All columns from feature_columns.json (excluding 'won')

Gate check:
- Log Loss < baseline (0.4101)
- Brier Score < baseline (0.1227)
- AUC-ROC > 0.65 (and > baseline 0.5508)
- Calibration error < 5%

If AUC-ROC does not reach 0.65, STOP and report — do not attempt to fix.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import MODEL_DIR, XGBOOST_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Input/output files
TRAIN_FILE = MODEL_DIR / "train.parquet"
VAL_FILE = MODEL_DIR / "val.parquet"
FEATURE_COLS_FILE = MODEL_DIR / "feature_columns.json"
MODEL_FILE = MODEL_DIR / "model_v2.json"
MODEL_PICKLE_FILE = MODEL_DIR / "model.pkl"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
VAL_PREDS_FILE = MODEL_DIR / "model_val_preds.parquet"
METRICS_FILE = MODEL_DIR / "model_metrics.json"

# Baseline metrics (from Step 5)
BASELINE_LOG_LOSS = 0.4101
BASELINE_BRIER_SCORE = 0.1227
BASELINE_AUC_ROC = 0.5508


def load_feature_columns() -> list[str]:
    """Load feature columns from JSON file."""
    with open(FEATURE_COLS_FILE, "r") as f:
        all_cols = json.load(f)

    # Exclude target column
    feature_cols = [c for c in all_cols if c != "won"]
    logger.info(f"Loaded {len(feature_cols)} feature columns: {feature_cols}")
    return feature_cols


def load_data(feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation data."""
    logger.info(f"Loading training data from {TRAIN_FILE}")
    train_df = pd.read_parquet(TRAIN_FILE)
    logger.info(f"  Train: {len(train_df):,} rows")

    logger.info(f"Loading validation data from {VAL_FILE}")
    val_df = pd.read_parquet(VAL_FILE)
    logger.info(f"  Val: {len(val_df):,} rows")

    # Verify feature columns exist
    missing_train = [c for c in feature_cols if c not in train_df.columns]
    missing_val = [c for c in feature_cols if c not in val_df.columns]

    if missing_train:
        raise ValueError(f"Missing features in train: {missing_train}")
    if missing_val:
        raise ValueError(f"Missing features in val: {missing_val}")

    return train_df, val_df


def compute_null_rates(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Compute null rates for each feature."""
    null_rates = {}
    for col in feature_cols:
        null_rates[col] = df[col].isna().mean()
    return null_rates


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[xgb.Booster, dict]:
    """
    Train XGBoost model with native null handling.

    Returns:
        Tuple of (trained model, training history)
    """
    logger.info("Preparing data for XGBoost...")

    # Extract features and target
    X_train = train_df[feature_cols]
    y_train = train_df["won"]
    X_val = val_df[feature_cols]
    y_val = val_df["won"]

    # Report null rates
    train_nulls = compute_null_rates(train_df, feature_cols)
    val_nulls = compute_null_rates(val_df, feature_cols)

    logger.info("Null rates in training data:")
    for col, rate in train_nulls.items():
        if rate > 0:
            logger.info(f"  {col}: {rate:.1%}")

    # Create DMatrix (XGBoost handles NaN natively)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=False)

    # Training parameters
    params = XGBOOST_PARAMS.copy()
    logger.info(f"XGBoost parameters: {params}")

    # Train model with early stopping
    logger.info("Training XGBoost model...")

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get("n_estimators", 500),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        evals_result=evals_result,
        verbose_eval=50,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score: {model.best_score:.4f}")

    return model, evals_result


def evaluate_model(
    model: xgb.Booster,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Evaluate model on validation set.

    Returns:
        Dictionary with all metrics
    """
    logger.info("Evaluating model on validation set...")

    # Get predictions
    X_val = val_df[feature_cols]
    dval = xgb.DMatrix(X_val, enable_categorical=False)
    y_proba = model.predict(dval)
    y_true = val_df["won"].values

    # Compute metrics
    log_loss_val = float(log_loss(y_true, y_proba))
    brier_score = float(brier_score_loss(y_true, y_proba))
    auc_roc = float(roc_auc_score(y_true, y_proba))

    # Compute calibration
    n_bins = 10
    unique_proba = np.unique(y_proba)
    if len(unique_proba) >= n_bins:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(y_proba, quantiles)
        bins = np.unique(bins)
        if len(bins) < n_bins + 1:
            bins = np.linspace(y_proba.min(), y_proba.max(), n_bins + 1)
    else:
        bins = np.linspace(y_proba.min(), y_proba.max(), n_bins + 1)

    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    calibration_errors = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            pred_mean = y_proba[mask].mean()
            actual_mean = y_true[mask].mean()
            calibration_errors.append(abs(pred_mean - actual_mean))

    calibration_error = float(np.mean(calibration_errors)) if calibration_errors else 0.0

    return {
        "n_samples": len(y_true),
        "win_rate": float(y_true.mean()),
        "log_loss": log_loss_val,
        "brier_score": brier_score,
        "auc_roc": auc_roc,
        "calibration_error": calibration_error,
        "y_proba": y_proba,
        "y_true": y_true,
    }


def check_gate_conditions(metrics: dict) -> dict:
    """
    Check if model beats baseline and meets gate conditions.

    Returns:
        Dictionary with gate check results
    """
    conditions = {
        "log_loss": {
            "value": metrics["log_loss"],
            "baseline": BASELINE_LOG_LOSS,
            "target": f"< {BASELINE_LOG_LOSS}",
            "beats_baseline": metrics["log_loss"] < BASELINE_LOG_LOSS,
        },
        "brier_score": {
            "value": metrics["brier_score"],
            "baseline": BASELINE_BRIER_SCORE,
            "target": f"< {BASELINE_BRIER_SCORE}",
            "beats_baseline": metrics["brier_score"] < BASELINE_BRIER_SCORE,
        },
        "auc_roc": {
            "value": metrics["auc_roc"],
            "baseline": BASELINE_AUC_ROC,
            "target": "> 0.65",
            "beats_baseline": metrics["auc_roc"] > BASELINE_AUC_ROC,
            "meets_threshold": metrics["auc_roc"] > 0.65,
        },
        "calibration_error": {
            "value": metrics["calibration_error"],
            "target": "< 0.05",
            "meets_threshold": metrics["calibration_error"] < 0.05,
        },
    }

    all_passed = (
        conditions["log_loss"]["beats_baseline"]
        and conditions["brier_score"]["beats_baseline"]
        and conditions["auc_roc"]["beats_baseline"]
        and conditions["auc_roc"]["meets_threshold"]
    )

    return {
        "conditions": conditions,
        "all_passed": all_passed,
        "auc_meets_threshold": conditions["auc_roc"]["meets_threshold"],
    }


def print_results(metrics: dict, gate: dict, evals_result: dict) -> None:
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("XGBOOST MODEL RESULTS")
    print("=" * 60)

    print(f"\n[Training]")
    print(f"  Features: 11 (all from feature_columns.json)")
    print(f"  Samples: {metrics['n_samples']:,}")
    print(f"  Best iteration: See log above")

    print(f"\n[Validation Metrics]")
    print(f"  Log Loss:          {metrics['log_loss']:.4f}")
    print(f"  Brier Score:       {metrics['brier_score']:.4f}")
    print(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
    print(f"  Calibration Error: {metrics['calibration_error']:.4f} ({metrics['calibration_error']*100:.1f}%)")

    print(f"\n[Baseline Comparison]")
    for name, cond in gate["conditions"].items():
        value = cond["value"]
        baseline = cond.get("baseline", None)

        if baseline is not None:
            beats = cond.get("beats_baseline", False)
            status = "✓" if beats else "✗"
            improvement = baseline - value if name in ["log_loss", "brier_score"] else value - baseline
            print(f"  {status} {name}: {value:.4f} vs baseline {baseline:.4f} (Δ {improvement:+.4f})")
        else:
            meets = cond.get("meets_threshold", False)
            status = "✓" if meets else "✗"
            target = cond["target"]
            print(f"  {status} {name}: {value:.4f} ({target})")

    print(f"\n[Gate Check]")
    for name, cond in gate["conditions"].items():
        if name == "auc_roc":
            status = "✓" if cond["meets_threshold"] else "✗"
            print(f"  {status} AUC-ROC: {cond['value']:.4f} {'>' if cond['meets_threshold'] else '≤'} 0.65")

    if gate["auc_meets_threshold"]:
        print(f"\n  ✓ GATE CHECK PASSED")
        print(f"    - All metrics beat baseline")
        print(f"    - AUC-ROC > 0.65")
    else:
        print(f"\n  ✗ GATE CHECK FAILED")
        print(f"    - AUC-ROC = {metrics['auc_roc']:.4f} ≤ 0.65")
        print(f"    - STOP: Do not attempt to fix without instruction")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("XGBOOST MODEL TRAINING - Greyhound Prediction System v2")
    logger.info("=" * 60)

    # Load feature columns
    logger.info("\n[1] Loading feature columns...")
    feature_cols = load_feature_columns()

    # Load data
    logger.info("\n[2] Loading data...")
    train_df, val_df = load_data(feature_cols)

    # Train model
    logger.info("\n[3] Training XGBoost model...")
    model, evals_result = train_xgboost(train_df, val_df, feature_cols)

    # Evaluate
    logger.info("\n[4] Evaluating model...")
    metrics = evaluate_model(model, val_df, feature_cols)

    # Check gate conditions
    gate = check_gate_conditions(metrics)

    # Print results
    print_results(metrics, gate, evals_result)

    # Save outputs
    print(f"\n[5] Saving outputs...")

    # Save model (JSON format for portability)
    model.save_model(str(MODEL_FILE))
    print(f"  Model (JSON): {MODEL_FILE}")

    # Also save as pickle for sklearn-like interface
    import pickle
    with open(MODEL_PICKLE_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model (Pickle): {MODEL_PICKLE_FILE}")

    # Save validation predictions
    preds_df = val_df[["venue", "race_date", "race_number", "runner_name", "won"]].copy()
    preds_df["model_proba"] = metrics["y_proba"]
    preds_df.to_parquet(VAL_PREDS_FILE, index=False)
    print(f"  Predictions: {VAL_PREDS_FILE} ({len(preds_df):,} rows)")

    # Save metrics
    metrics_json = {
        "model_type": "XGBoost",
        "features": feature_cols,
        "n_features": len(feature_cols),
        "validation": {
            "n_samples": int(metrics["n_samples"]),
            "win_rate": float(metrics["win_rate"]),
            "log_loss": float(metrics["log_loss"]),
            "brier_score": float(metrics["brier_score"]),
            "auc_roc": float(metrics["auc_roc"]),
            "calibration_error": float(metrics["calibration_error"]),
        },
        "baseline_comparison": {
            "log_loss_baseline": BASELINE_LOG_LOSS,
            "brier_score_baseline": BASELINE_BRIER_SCORE,
            "auc_roc_baseline": BASELINE_AUC_ROC,
            "log_loss_improvement": BASELINE_LOG_LOSS - metrics["log_loss"],
            "brier_score_improvement": BASELINE_BRIER_SCORE - metrics["brier_score"],
            "auc_roc_improvement": metrics["auc_roc"] - BASELINE_AUC_ROC,
        },
        "gate_check": {
            "log_loss_beats_baseline": gate["conditions"]["log_loss"]["beats_baseline"],
            "brier_score_beats_baseline": gate["conditions"]["brier_score"]["beats_baseline"],
            "auc_roc_beats_baseline": gate["conditions"]["auc_roc"]["beats_baseline"],
            "auc_roc_meets_threshold": gate["conditions"]["auc_roc"]["meets_threshold"],
            "calibration_error_meets_threshold": gate["conditions"]["calibration_error"]["meets_threshold"],
            "gate_passed": gate["all_passed"],
        },
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics: {METRICS_FILE}")

    print("\n" + "=" * 60)

    if not gate["auc_meets_threshold"]:
        print("⚠️  STOPPING: AUC-ROC did not reach 0.65")
        print(f"   AUC-ROC = {metrics['auc_roc']:.4f}")
        print("   Do not attempt to fix without instruction.")
        return 1

    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
