
import sys
sys.path.insert(0, 'src/model_v2')
from evaluate import evaluate_parquet_file
from predict import load_frozen_model, load_feature_columns
import pandas as pd
import json

# Load test set
test = pd.read_parquet('models/v2/test.parquet')
print(f"Test set: {len(test)} runners")

# Load model and features
model = load_frozen_model()
features = load_feature_columns()
print(f"Model loaded with {len(features)} features")

# Generate predictions
test['prediction'] = model.predict(test[features])

# Save
test[['prediction', 'won']].to_parquet('models/v2/model_test_preds.parquet')

# Evaluate
results = evaluate_parquet_file('models/v2/model_test_preds.parquet', proba_col='prediction', target_col='won')

# Print
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)
print(json.dumps(results, indent=2))

print("\nVALIDATION vs TEST")
print(f"Log Loss:  Val=0.3852  Test={results['log_loss']:.4f}")
print(f"AUC-ROC:   Val=0.6827  Test={results['auc_roc']:.4f}")

print("\nGATE CHECK")
print(f"Log Loss < 0.65:  {'PASS' if results['log_loss'] < 0.65 else 'FAIL'}")
print(f"AUC > 0.65:       {'PASS' if results['auc_roc'] > 0.65 else 'FAIL'}")


