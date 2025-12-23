"""
Debug: Why are all test predictions 0?
"""
import pandas as pd
import numpy as np
import joblib
from model import Model

print("="*80)
print("DEBUG: TEST PREDICTIONS = 0 INVESTIGATION")
print("="*80)

# Load model components
model = joblib.load('lgb_model.pkl')
selected_features = joblib.load('selected_features.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load test
test = pd.read_csv('test.csv')
print(f"\n1. Test data shape: {test.shape}")
print(f"   Test columns: {test.columns.tolist()[:10]}...")

# Transform one row
test_row = test.iloc[[0]]
print(f"\n2. Transforming first test row...")
X_transformed = preprocessor.transform(test_row, is_training=False)
print(f"   Transformed shape: {X_transformed.shape}")
print(f"   Transformed columns: {len(X_transformed.columns)}")

# Check for missing imputation indicators
print(f"\n3. Checking feature alignment...")
missing_features = [f for f in selected_features if f not in X_transformed.columns]
print(f"   Missing features in test: {len(missing_features)}")
if missing_features:
    print(f"   Examples: {missing_features[:10]}")
    
    # Check if they're imputation indicators
    missing_indicators = [f for f in missing_features if '_missing' in f]
    print(f"   Missing imputation indicators: {len(missing_indicators)}")
    print(f"   Examples: {missing_indicators[:5]}")

# Prepare features (like model.predict does)
for feature in selected_features:
    if feature not in X_transformed.columns:
        X_transformed[feature] = 0  # ← BUG: Setting all to 0!

X_final = X_transformed[selected_features]

print(f"\n4. Final feature matrix:")
print(f"   Shape: {X_final.shape}")
print(f"   Sample of imputation indicators (should reflect actual missing data):")

# Check actual missing data in test vs what we're telling the model
test_row_clean = test_row.drop(columns=['forward_returns', 'risk_free_rate'], errors='ignore')
for col in ['E1', 'E20', 'V9', 'M2', 'M6']:  # Features we know have missing data
    if col in test_row_clean.columns:
        is_missing = test_row_clean[col].isnull().values[0]
        indicator_name = f'{col}_missing'
        if indicator_name in X_final.columns:
            told_model = X_final[indicator_name].values[0]
            match = (is_missing == told_model)
            symbol = "✓" if match else "✗"
            print(f"   {col}: actually_missing={is_missing}, told_model={told_model} {symbol}")

# Make prediction
y_pred = model.predict(X_final)[0]
print(f"\n5. Raw prediction: {y_pred:.8f}")
print(f"   Sign: {'positive' if y_pred > 0 else 'negative'}")
print(f"   Position (sign strategy): {2.0 if y_pred > 0 else 0.0}")

# Compare to training predictions
print(f"\n6. Comparing to training predictions...")
train = pd.read_csv('train.csv')
X_train, _ = preprocessor.fit_transform(train)
nan_mask = X_train.isnull().any(axis=1)
X_train = X_train[~nan_mask]
X_train_final = X_train[selected_features]

train_preds = model.predict(X_train_final)
print(f"   Train predictions:")
print(f"     Mean: {train_preds.mean():.8f}")
print(f"     Std: {train_preds.std():.8f}")
print(f"     % positive: {(train_preds > 0).sum() / len(train_preds) * 100:.2f}%")
print(f"     % negative: {(train_preds < 0).sum() / len(train_preds) * 100:.2f}%")

print(f"\n7. ROOT CAUSE:")
print(f"   If test predictions are all negative but train had ~40% positive,")
print(f"   likely causes:")
print(f"   1. Imputation indicators incorrectly set to 0 (should be 1 if missing)")
print(f"   2. Test data has more missing values than train")
print(f"   3. Feature distribution shift between train/test")
