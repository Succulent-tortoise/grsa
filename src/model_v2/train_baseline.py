"""
train_baseline.py — Greyhound Prediction System v2

Builds a frequency-based baseline model using box position and grade only.
This establishes the floor that the full model must beat.

Model: Logistic Regression with StandardScaler
Features: box, grade_encoded only (no form features)

Training: train.parquet only
Evaluation: val.parquet

Null handling: Median imputation (no forward-fill)

Gate check:
- Log Loss and AUC-ROC on validation set
- Confirm baseline beats random (Log Loss ~0.39, AUC = 0.50)
- Output baseline predictions
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Input/output files
TRAIN_FILE = MODEL_DIR / "train.parquet"
VAL_FILE = MODEL_DIR / "val.parquet"
BASELINE_MODEL_FILE = MODEL_DIR / "baseline_model.pkl"
BASELINE_SCALER_FILE = MODEL_DIR / "baseline_scaler.pkl"
BASELINE_PREDS_FILE = MODEL_DIR / "baseline_val_preds.parquet"
BASELINE_METRICS_FILE = MODEL_DIR / "baseline_metrics.json"

# Baseline features only (no form features)
BASELINE_FEATURES = ["box", "grade_encoded"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation splits."""
    logger.info(f"Loading training data from {TRAIN_FILE}")
    train_df = pd.read_parquet(TRAIN_FILE)
    logger.info(f"  Train: {len(train_df):,} rows")

    logger.info(f"Loading validation data from {VAL_FILE}")
    val_df = pd.read_parquet(VAL_FILE)
    logger.info(f"  Val: {len(val_df):,} rows")

    return train_df, val_df


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, SimpleImputer]:
    """
    Prepare features for training.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, scaler, imputer)
    """
    # Extract features and target
    X_train = train_df[features].values
    y_train = train_df["won"].values
    X_val = val_df[features].values
    y_val = val_df["won"].values

    # Impute missing values with median (no forward-fill)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    logger.info(f"Features: {features}")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")

    # Log null handling
    for i, feat in enumerate(features):
        null_count = train_df[feat].isna().sum()
        if null_count > 0:
            median_val = imputer.statistics_[i]
            logger.info(f"  {feat}: {null_count:,} nulls imputed with median = {median_val:.2f}")

    return X_train, y_train, X_val, y_val, scaler, imputer


def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train a logistic regression baseline model.

    No class weights - we want well-calibrated probabilities for log loss.
    The baseline should predict the base rate with small adjustments from features.
    """
    logger.info("Training logistic regression baseline...")

    # No class weights - this gives better log loss (calibration)
    # The model will naturally learn to predict near the base rate
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )

    model.fit(X_train, y_train)

    logger.info(f"  Coefficients: {model.coef_}")
    logger.info(f"  Intercept: {model.intercept_}")

    return model


def evaluate_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
) -> dict:
    """
    Evaluate model performance.

    Returns:
        Dictionary with metrics
    """
    # Get predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Compute metrics
    ll = log_loss(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    # Random baseline metrics for comparison
    # Log loss for random guessing at 14.3% win rate
    win_rate = y.mean()
    random_log_loss = -(
        win_rate * np.log(win_rate) + (1 - win_rate) * np.log(1 - win_rate)
    )
    random_auc = 0.50

    return {
        "name": name,
        "log_loss": ll,
        "auc_roc": auc,
        "win_rate": win_rate,
        "random_log_loss": random_log_loss,
        "random_auc": random_auc,
        "beats_random_log_loss": ll < random_log_loss,
        "beats_random_auc": auc > random_auc,
        "y_proba": y_proba,
        "y_true": y,
    }


def run_gate_check(val_metrics: dict) -> dict:
    """
    Run gate check on validation metrics.

    Returns:
        Dictionary with gate check results
    """
    return {
        "val_log_loss": val_metrics["log_loss"],
        "val_auc_roc": val_metrics["auc_roc"],
        "random_log_loss": val_metrics["random_log_loss"],
        "random_auc": val_metrics["random_auc"],
        "beats_random_log_loss": val_metrics["beats_random_log_loss"],
        "beats_random_auc": val_metrics["beats_random_auc"],
        "gate_passed": val_metrics["beats_random_log_loss"] and val_metrics["beats_random_auc"],
    }


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("BASELINE MODEL TRAINING - Greyhound Prediction System v2")
    logger.info("=" * 60)

    # Load data
    logger.info("\n[1] Loading data...")
    train_df, val_df = load_data()

    # Prepare features
    logger.info("\n[2] Preparing features...")
    logger.info(f"  Baseline features: {BASELINE_FEATURES}")
    X_train, y_train, X_val, y_val, scaler, imputer = prepare_features(
        train_df, val_df, BASELINE_FEATURES
    )

    # Train model
    logger.info("\n[3] Training baseline model...")
    model = train_baseline(X_train, y_train)

    # Evaluate
    logger.info("\n[4] Evaluating model...")

    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")

    # Print results
    print("\n" + "=" * 60)
    print("BASELINE MODEL RESULTS")
    print("=" * 60)

    print(f"\n[Model]")
    print(f"  Type: Logistic Regression")
    print(f"  Features: {BASELINE_FEATURES}")
    print(f"  Class weights: None (calibrated probabilities)")

    print(f"\n[Feature Coefficients]")
    for feat, coef in zip(BASELINE_FEATURES, model.coef_[0]):
        print(f"  {feat:20s}: {coef:+.4f}")
    print(f"  {'Intercept':20s}: {model.intercept_[0]:+.4f}")

    print(f"\n[Training Set]")
    print(f"  Log Loss:  {train_metrics['log_loss']:.4f}")
    print(f"  AUC-ROC:   {train_metrics['auc_roc']:.4f}")

    print(f"\n[Validation Set]")
    print(f"  Log Loss:  {val_metrics['log_loss']:.4f}")
    print(f"  AUC-ROC:   {val_metrics['auc_roc']:.4f}")
    print(f"  Win rate:  {val_metrics['win_rate']:.1%}")

    print(f"\n[Random Baseline Comparison]")
    print(f"  Random Log Loss:    {val_metrics['random_log_loss']:.4f}")
    print(f"  Baseline Log Loss:  {val_metrics['log_loss']:.4f}")
    print(f"  Improvement:        {val_metrics['random_log_loss'] - val_metrics['log_loss']:.4f}")
    print(f"  Random AUC-ROC:     {val_metrics['random_auc']:.4f}")
    print(f"  Baseline AUC-ROC:   {val_metrics['auc_roc']:.4f}")
    print(f"  Improvement:        {val_metrics['auc_roc'] - val_metrics['random_auc']:.4f}")

    # Run gate check
    gate_results = run_gate_check(val_metrics)

    print(f"\n" + "=" * 60)
    print("GATE CHECK")
    print("=" * 60)

    print(f"\n[Criteria]")
    print(f"  Baseline must beat random on validation set")
    print(f"  - Log Loss: {val_metrics['log_loss']:.4f} < {val_metrics['random_log_loss']:.4f} (random)")
    print(f"  - AUC-ROC:  {val_metrics['auc_roc']:.4f} > {val_metrics['random_auc']:.4f} (random)")

    if gate_results["gate_passed"]:
        print(f"\n✓ GATE CHECK PASSED")
        print(f"  - Baseline beats random on Log Loss")
        print(f"  - Baseline beats random on AUC-ROC")
    else:
        print(f"\n✗ GATE CHECK FAILED")
        if not gate_results["beats_random_log_loss"]:
            print(f"  - Baseline does not beat random on Log Loss")
        if not gate_results["beats_random_auc"]:
            print(f"  - Baseline does not beat random on AUC-ROC")

    print("=" * 60)

    # Save model and predictions
    print(f"\n[5] Saving outputs...")

    # Save model
    import pickle

    with open(BASELINE_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model: {BASELINE_MODEL_FILE}")

    with open(BASELINE_SCALER_FILE, "wb") as f:
        pickle.dump({"scaler": scaler, "imputer": imputer}, f)
    print(f"  Scaler: {BASELINE_SCALER_FILE}")

    # Save validation predictions
    preds_df = val_df[["venue", "race_date", "race_number", "runner_name", "won"]].copy()
    preds_df["baseline_proba"] = val_metrics["y_proba"]
    preds_df.to_parquet(BASELINE_PREDS_FILE, index=False)
    print(f"  Predictions: {BASELINE_PREDS_FILE} ({len(preds_df):,} rows)")

    # Save metrics
    metrics_json = {
        "model_type": "LogisticRegression",
        "features": BASELINE_FEATURES,
        "train": {
            "log_loss": float(train_metrics["log_loss"]),
            "auc_roc": float(train_metrics["auc_roc"]),
        },
        "validation": {
            "log_loss": float(val_metrics["log_loss"]),
            "auc_roc": float(val_metrics["auc_roc"]),
            "win_rate": float(val_metrics["win_rate"]),
        },
        "random_baseline": {
            "log_loss": float(val_metrics["random_log_loss"]),
            "auc_roc": float(val_metrics["random_auc"]),
        },
        "gate_check": {
            "val_log_loss": float(gate_results["val_log_loss"]),
            "val_auc_roc": float(gate_results["val_auc_roc"]),
            "random_log_loss": float(gate_results["random_log_loss"]),
            "random_auc": float(gate_results["random_auc"]),
            "beats_random_log_loss": bool(gate_results["beats_random_log_loss"]),
            "beats_random_auc": bool(gate_results["beats_random_auc"]),
            "gate_passed": bool(gate_results["gate_passed"]),
        },
    }

    with open(BASELINE_METRICS_FILE, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics: {BASELINE_METRICS_FILE}")

    print()

    return model, val_metrics


if __name__ == "__main__":
    main()
