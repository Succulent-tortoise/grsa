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

# Evaluate - just print what it returns
results = evaluate_parquet_file('models/v2/model_test_preds.parquet', proba_col='prediction', target_col='won')
print("\nRaw results:")
print(type(results))
print(results)
