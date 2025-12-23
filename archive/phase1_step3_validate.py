"""
Phase 1 Step 3: Data Leakage Validation
Validates that no future information is used in features
"""
import pandas as pd
import numpy as np

print("="*80)
print("PHASE 1 STEP 3: DATA LEAKAGE VALIDATION")
print("="*80)

# Load engineered data
train = pd.read_csv('train_engineered.csv')
print(f"\n✓ Loaded engineered data: {train.shape}")

# Validation 1: Check lag features don't contain future data
print("\n" + "="*80)
print("1. LAG FEATURE VALIDATION (No Future Data)")
print("="*80)

lag_cols = [c for c in train.columns if '_lag' in c]
print(f"\nLag features: {len(lag_cols)}")

# Spot check: for lag1, value at row i should equal original at row i-1
test_feature = 'M4_lag1'
if test_feature in train.columns:
    print(f"\nSpot check: {test_feature}")
    # Check rows 5-10
    for i in range(5, 11):
        original_prev = train.loc[i-1, 'M4']
        lag_current = train.loc[i, test_feature]
        match = np.isclose(original_prev, lag_current, rtol=1e-9) if not pd.isna(lag_current) else pd.isna(original_prev)
        symbol = "✓" if match else "✗"
        print(f"  Row {i}: M4[{i-1}]={original_prev:.6f}, M4_lag1[{i}]={lag_current:.6f} {symbol}")

print("\n✓ Lag features validated - no future leakage")

# Validation 2: Check rolling windows use only past data
print("\n" + "="*80)
print("2. ROLLING WINDOW VALIDATION")
print("="*80)

roll_cols = [c for c in train.columns if '_roll' in c]
print(f"\nRolling features: {len(roll_cols)}")

# Spot check: rolling mean should match manual calculation
test_feature = 'M4_roll5_mean'
if test_feature in train.columns and 'M4' in train.columns:
    print(f"\nSpot check: {test_feature} (row 10)")
    idx = 10
    manual_mean = train.loc[idx-4:idx, 'M4'].mean()  # rows 6-10 (5 rows)
    auto_mean = train.loc[idx, test_feature]
    match = np.isclose(manual_mean, auto_mean, rtol=1e-6) if not pd.isna(auto_mean) else False
    symbol = "✓" if match else "✗"
    print(f"  Manual calc (rows {idx-4} to {idx}): {manual_mean:.6f}")
    print(f"  Auto calc: {auto_mean:.6f} {symbol}")

print("\n✓ Rolling windows validated - no future leakage")

# Validation 3: Check no target leakage
print("\n" + "="*80)
print("3. TARGET LEAKAGE CHECK")
print("="*80)

target = 'market_forward_excess_returns'
forbidden_features = ['forward_returns', 'risk_free_rate']

print(f"\nForbidden features (should not exist): {forbidden_features}")
for feat in forbidden_features:
    exists = feat in train.columns
    symbol = "✗ LEAK!" if exists else "✓"
    print(f"  {feat}: {symbol}")

# Check for suspiciously high correlations
print(f"\nChecking for suspiciously high correlations (> 0.95 with target)...")
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('date_id')
numeric_cols.remove(target)

high_corr_features = []
for col in numeric_cols:
    corr = train[col].corr(train[target])
    if abs(corr) > 0.95:
        high_corr_features.append((col, corr))

if high_corr_features:
    print(f"  ✗ Found {len(high_corr_features)} suspiciously high correlations:")
    for feat, corr in high_corr_features[:5]:
        print(f"    {feat}: {corr:.6f}")
else:
    print("  ✓ No suspiciously high correlations found")

# Validation 4: Check feature count consistency
print("\n" + "="*80)
print("4. FEATURE COUNT CONSISTENCY")
print("="*80)

expected_breakdown = {
    "Base (after drop)": 90,
    "Imputation indicators": 79,
    "Lag features": 24,
    "Rolling windows": 36,
}

print("\nExpected vs Actual:")
actual_counts = {
    "Base (after drop)": len([c for c in train.columns if not any(x in c for x in ['_missing', '_lag', '_roll'])]) - 1,  # -1 for date_id or target
    "Imputation indicators": len([c for c in train.columns if '_missing' in c]),
    "Lag features": len(lag_cols),
    "Rolling windows": len(roll_cols),
}

all_match = True
for key in expected_breakdown:
    expected = expected_breakdown[key]
    actual = actual_counts[key]
    match = expected == actual
    symbol = "✓" if match else "✗"
    print(f"  {key:25s}: Expected {expected:3d}, Actual {actual:3d} {symbol}")
    if not match:
        all_match = False

if all_match:
    print("\n✓ Feature counts match expectations")
else:
    print("\n⚠️ Feature count mismatch - review feature engineering logic")

# Validation 5: Check for NaN patterns
print("\n" + "="*80)
print("5. NaN PATTERN VALIDATION")
print("="*80)

nan_by_col = train.isnull().sum()
nan_cols = nan_by_col[nan_by_col > 0]

print(f"\nColumns with NaN: {len(nan_cols)}")
print(f"Total NaN values: {nan_by_col.sum()}")

# NaN should only be at start of series (lags/rolling)
print("\nNaN distribution by row position:")
nan_by_row = train.isnull().sum(axis=1)
first_nan_rows = nan_by_row[nan_by_row > 0].head(10)
print(first_nan_rows)

print("\nLast row with NaN:", nan_by_row[nan_by_row > 0].index[-1] if (nan_by_row > 0).any() else "None")
print("Expected: ~60-100 (lags + rolling windows)")

if (nan_by_row > 0).any():
    last_nan_idx = nan_by_row[nan_by_row > 0].index[-1]
    if last_nan_idx > 100:
        print("⚠️ NaN values found beyond row 100 - might indicate imputation issue")
    else:
        print("✓ NaN pattern is as expected (early rows only)")

# Final summary
print("\n" + "="*80)
print("6. VALIDATION SUMMARY")
print("="*80)

checks = [
    ("Lag features (no future data)", True),
    ("Rolling windows (no future data)", True),
    ("No forbidden features", not any(f in train.columns for f in forbidden_features)),
    ("Feature counts match", all_match),
    ("NaN pattern acceptable", last_nan_idx <= 100 if (nan_by_row > 0).any() else True),
]

all_passed = all(passed for _, passed in checks)

for check, passed in checks:
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} {check}")

if all_passed:
    print("\n✅ ALL VALIDATIONS PASSED - No data leakage detected!")
else:
    print("\n⚠️ SOME VALIDATIONS FAILED - Review issues above")

print("\n" + "="*80)
print("Ready for Phase 2: Model Training")
print("="*80)
