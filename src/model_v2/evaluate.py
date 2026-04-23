"""
evaluate.py — Greyhound Prediction System v2

Reusable evaluation module for model predictions.
Computes: Log Loss, Brier Score, AUC-ROC, and Calibration Error.

This module is called by:
- train_model.py (during/after training)
- Baseline evaluation
- Final test set evaluation

Gate check (for any model):
- Log Loss < 0.65
- Brier Score < 0.20
- AUC-ROC > 0.65
- Calibration error < 5%
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from config import MODEL_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Log Loss (cross-entropy loss).

    Lower is better. Random baseline for 14% win rate ≈ 0.41.
    Target: < 0.65
    """
    return float(log_loss(y_true, y_proba))


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Brier Score (mean squared error of probabilities).

    Lower is better. Perfect = 0.
    Target: < 0.20
    """
    return float(brier_score_loss(y_true, y_proba))


def compute_auc_roc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute AUC-ROC (Area Under ROC Curve).

    Higher is better. Random = 0.50. Perfect = 1.00.
    Target: > 0.65
    """
    return float(roc_auc_score(y_true, y_proba))


def compute_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute calibration metrics by decile.

    Uses quantile-based binning for better visualization when
    predictions are clustered in a narrow range.

    Returns:
        Dictionary with:
        - bins: list of bin boundaries
        - predicted_probs: mean predicted probability per bin
        - actual_rates: actual win rate per bin
        - counts: number of samples per bin
        - calibration_error: mean absolute calibration error
        - max_calibration_error: max absolute calibration error
    """
    # Use quantile-based binning for better spread
    # Falls back to equal-width if not enough unique values
    unique_proba = np.unique(y_proba)
    if len(unique_proba) >= n_bins:
        # Quantile-based bins
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(y_proba, quantiles)
        # Ensure bins are strictly increasing
        bins = np.unique(bins)
        if len(bins) < n_bins + 1:
            bins = np.linspace(y_proba.min(), y_proba.max(), n_bins + 1)
    else:
        # Equal-width bins
        bins = np.linspace(y_proba.min(), y_proba.max(), n_bins + 1)

    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    predicted_probs = []
    actual_rates = []
    counts = []
    calibration_errors = []

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())  # Convert to int for JSON

        if count > 0:
            pred_mean = float(y_proba[mask].mean())
            actual_mean = float(y_true[mask].mean())
            error = abs(pred_mean - actual_mean)

            predicted_probs.append(pred_mean)
            actual_rates.append(actual_mean)
            counts.append(count)
            calibration_errors.append(error)
        else:
            predicted_probs.append(None)  # Use None instead of NaN for JSON
            actual_rates.append(None)
            counts.append(0)
            calibration_errors.append(None)

    # Compute mean calibration error (excluding empty bins)
    valid_errors = [e for e in calibration_errors if e is not None]
    mean_calibration_error = float(np.mean(valid_errors)) if valid_errors else None
    max_calibration_error = float(np.max(valid_errors)) if valid_errors else None

    return {
        "bins": [float(b) for b in bins],
        "predicted_probs": predicted_probs,
        "actual_rates": actual_rates,
        "counts": counts,
        "calibration_error": mean_calibration_error,
        "max_calibration_error": max_calibration_error,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
) -> dict[str, Any]:
    """
    Full evaluation of predictions.

    Returns:
        Dictionary with all metrics
    """
    log_loss_val = compute_log_loss(y_true, y_proba)
    brier_score = compute_brier_score(y_true, y_proba)
    auc_roc = compute_auc_roc(y_true, y_proba)
    calibration = compute_calibration(y_true, y_proba)

    return {
        "model_name": model_name,
        "n_samples": len(y_true),
        "win_rate": float(y_true.mean()),
        "log_loss": log_loss_val,
        "brier_score": brier_score,
        "auc_roc": auc_roc,
        "calibration_error": calibration["calibration_error"],
        "max_calibration_error": calibration["max_calibration_error"],
        "calibration_details": {
            "bins": calibration["bins"],
            "predicted_probs": calibration["predicted_probs"],
            "actual_rates": calibration["actual_rates"],
            "counts": calibration["counts"],
        },
    }


def check_gate_conditions(metrics: dict[str, Any], phase: str = "validation") -> dict[str, Any]:
    """
    Check if metrics meet gate conditions.

    Returns:
        Dictionary with gate check results
    """
    conditions = {
        "log_loss": {
            "target": 0.65,
            "comparison": "<",
            "value": metrics["log_loss"],
            "passed": metrics["log_loss"] < 0.65,
        },
        "brier_score": {
            "target": 0.20,
            "comparison": "<",
            "value": metrics["brier_score"],
            "passed": metrics["brier_score"] < 0.20,
        },
        "auc_roc": {
            "target": 0.65,
            "comparison": ">",
            "value": metrics["auc_roc"],
            "passed": metrics["auc_roc"] > 0.65,
        },
        "calibration_error": {
            "target": 0.05,
            "comparison": "<",
            "value": metrics["calibration_error"],
            "passed": metrics["calibration_error"] < 0.05,
        },
    }

    # For baseline, only Log Loss < 0.65 is required
    # Full model requires all conditions
    if phase == "baseline":
        gate_passed = conditions["log_loss"]["passed"]
    else:
        gate_passed = all(c["passed"] for c in conditions.values())

    return {
        "phase": phase,
        "conditions": conditions,
        "gate_passed": gate_passed,
    }


def print_evaluation_report(metrics: dict[str, Any], gate: dict[str, Any] | None = None) -> None:
    """Print formatted evaluation report."""
    print(f"\n[Evaluation Results - {metrics['model_name']}]")
    print(f"  Samples:    {metrics['n_samples']:,}")
    print(f"  Win rate:   {metrics['win_rate']:.1%}")

    print(f"\n[Metrics]")
    print(f"  Log Loss:           {metrics['log_loss']:.4f}")
    print(f"  Brier Score:        {metrics['brier_score']:.4f}")
    print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
    cal_err = metrics['calibration_error']
    max_cal_err = metrics['max_calibration_error']
    if cal_err is not None:
        print(f"  Calibration Error:  {cal_err:.4f} ({cal_err*100:.1f}%)")
        print(f"  Max Calib Error:    {max_cal_err:.4f} ({max_cal_err*100:.1f}%)")
    else:
        print(f"  Calibration Error:  N/A")
        print(f"  Max Calib Error:    N/A")

    print(f"\n[Calibration by Decile]")
    print(f"  {'Decile':<8} {'Pred Range':<24} {'Pred':<10} {'Actual':<10} {'Count':<10} {'Error':<10}")
    print(f"  {'-'*8} {'-'*24} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    cal = metrics["calibration_details"]
    bins = cal["bins"]
    for i in range(min(10, len(bins) - 1)):
        pred = cal["predicted_probs"][i] if i < len(cal["predicted_probs"]) else None
        actual = cal["actual_rates"][i] if i < len(cal["actual_rates"]) else None
        count = cal["counts"][i] if i < len(cal["counts"]) else 0

        bin_low = bins[i]
        bin_high = bins[i + 1] if i + 1 < len(bins) else bins[-1]
        bin_range = f"{bin_low:.1%} - {bin_high:.1%}"

        if count > 0 and pred is not None and actual is not None:
            error = abs(pred - actual)
            print(f"  {i+1:<8} {bin_range:<24} {pred:>8.1%}   {actual:>8.1%}   {count:>8,}   {error:>8.1%}")
        else:
            print(f"  {i+1:<8} {bin_range:<24} {'--':>8}   {'--':>8}   {0:>8}   {'--':>8}")

    if gate:
        print(f"\n[Gate Check - {gate['phase'].title()}]")
        for name, cond in gate["conditions"].items():
            status = "✓" if cond["passed"] else "✗"
            comp = cond["comparison"]
            target = cond["target"]
            value = cond["value"]
            print(f"  {status} {name}: {value:.4f} {comp} {target}")

        if gate["gate_passed"]:
            print(f"\n  ✓ GATE CHECK PASSED")
        else:
            print(f"\n  ✗ GATE CHECK FAILED")


def evaluate_parquet_file(
    preds_file: Path,
    proba_col: str = "baseline_proba",
    target_col: str = "won",
    model_name: str = "Model",
    phase: str = "validation",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluate predictions from a parquet file.

    Args:
        preds_file: Path to parquet file with predictions
        proba_col: Column name for predicted probabilities
        target_col: Column name for true labels
        model_name: Name for reporting
        phase: Phase for gate check (baseline/validation/test)

    Returns:
        Tuple of (metrics, gate_results)
    """
    logger.info(f"Loading predictions from {preds_file}")
    df = pd.read_parquet(preds_file)

    y_true = df[target_col].values
    y_proba = df[proba_col].values

    metrics = evaluate_predictions(y_true, y_proba, model_name)
    gate = check_gate_conditions(metrics, phase)

    return metrics, gate


def main():
    """Main entry point for baseline evaluation."""
    logger.info("=" * 60)
    logger.info("EVALUATION MODULE - Greyhound Prediction System v2")
    logger.info("=" * 60)

    # Evaluate baseline predictions
    baseline_preds_file = MODEL_DIR / "baseline_val_preds.parquet"

    print("\n" + "=" * 60)
    print("BASELINE MODEL EVALUATION (Validation Set)")
    print("=" * 60)

    metrics, gate = evaluate_parquet_file(
        baseline_preds_file,
        proba_col="baseline_proba",
        target_col="won",
        model_name="Baseline (Logistic Regression)",
        phase="baseline",
    )

    print_evaluation_report(metrics, gate)

    # Save evaluation results
    output_file = MODEL_DIR / "baseline_evaluation.json"
    output_data = {
        "metrics": {
            "model_name": metrics["model_name"],
            "n_samples": int(metrics["n_samples"]),
            "win_rate": float(metrics["win_rate"]),
            "log_loss": float(metrics["log_loss"]),
            "brier_score": float(metrics["brier_score"]),
            "auc_roc": float(metrics["auc_roc"]),
            "calibration_error": float(metrics["calibration_error"]) if metrics["calibration_error"] is not None else None,
            "max_calibration_error": float(metrics["max_calibration_error"]) if metrics["max_calibration_error"] is not None else None,
        },
        "calibration_table": metrics["calibration_details"],
        "gate_check": {
            "phase": gate["phase"],
            "conditions": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv for kk, vv in v.items()} for k, v in gate["conditions"].items()},
            "gate_passed": bool(gate["gate_passed"]),
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Output]")
    print(f"  Evaluation saved to: {output_file}")

    print("\n" + "=" * 60)

    if gate["gate_passed"]:
        print("✓ BASELINE GATE CHECK PASSED")
        print(f"  - Log Loss: {metrics['log_loss']:.4f} < 0.65")
        print()
        print("Note: Baseline only requires Log Loss < 0.65")
        print("Full model must also meet Brier, AUC, and calibration targets.")
    else:
        print("✗ BASELINE GATE CHECK FAILED")
        print(f"  - Log Loss: {metrics['log_loss']:.4f} >= 0.65")

    print("=" * 60)

    return metrics, gate


if __name__ == "__main__":
    main()
