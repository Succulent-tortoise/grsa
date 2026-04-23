
import sys
sys.path.insert(0, 'src/model_v2')
from evaluate import evaluate_parquet_file
from predict import load_frozen_model, load_feature_columns
import pandas as pd
import xgboost as xgb

# Load test set
test = pd.read_parquet('models/v2/test.parquet')
print(f"Test set: {len(test)} runners")

# Load model and features
model = load_frozen_model()
features = load_feature_columns()

# Generate predictions
dtest = xgb.DMatrix(test[features])
test['prediction'] = model.predict(dtest)

# Save
test[['prediction', 'won']].to_parquet('models/v2/model_test_preds.parquet')

# Evaluate
metrics, gate_check = evaluate_parquet_file('models/v2/model_test_preds.parquet', proba_col='prediction', target_col='won')

# Print results
print("\n" + "="*60)
print("TEST SET EVALUATION — FINAL HONEST ASSESSMENT")
print("="*60)
print(f"Samples:        {metrics['n_samples']}")
print(f"Win Rate:       {metrics['win_rate']*100:.1f}%")
print(f"Log Loss:       {metrics['log_loss']:.4f}")
print(f"Brier Score:    {metrics['brier_score']:.4f}")
print(f"AUC-ROC:        {metrics['auc_roc']:.4f}")
print(f"Cal Error:      {metrics['calibration_error']*100:.2f}%")

print("\n" + "="*60)
print("COMPARISON: Validation vs Test")
print("="*60)
print(f"Log Loss:  Val=0.3852  Test={metrics['log_loss']:.4f}  Δ={metrics['log_loss']-0.3852:+.4f}")
print(f"Brier:     Val=0.1161  Test={metrics['brier_score']:.4f}  Δ={metrics['brier_score']-0.1161:+.4f}")
print(f"AUC-ROC:   Val=0.6827  Test={metrics['auc_roc']:.4f}  Δ={metrics['auc_roc']-0.6827:+.4f}")
print(f"Cal Error: Val=0.35%   Test={metrics['calibration_error']*100:.2f}%")

print("\n" + "="*60)
print("GATE CHECK (from config.py targets)")
print("="*60)
for name, check in gate_check['conditions'].items():
    status = "✓ PASS" if check['passed'] else "✗ FAIL"
    print(f"{name:20s} {check['comparison']} {check['target']:.2f}:  {status}  (actual: {check['value']:.4f})")

print("\n" + "="*60)
if gate_check['gate_passed']:
    print("✓ ALL GATES PASSED — Model generalizes to test set")
else:
    print("✗ GATE FAILED — Investigate overfitting")
print("="*60)
