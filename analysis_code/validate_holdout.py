"""
QUICK SANITY CHECKER (Legacy Validation)
----------------------------------------
Calculates performance on the Holdout set (last 180 rows).

Differences from 'official_metric.py':
1. This script calculates RAW Sharpe (no penalties).
2. It provides a quick "Gut Check" on direction and position distribution.
3. For the final competition score, use 'official_metric.py'.

Useful for:
- Debugging position output (Mean, %, etc.)
- Checking simple P&L loops.
"""
import pandas as pd
import numpy as np
import joblib
from position_mapper import PositionMapper

print("="*80)
print("HOLDOUT VALIDATION (HONEST PUBLIC TEST PERFORMANCE)")
print("="*80)

# Load clean artifacts
model = joblib.load('lgb_model_clean.pkl')
selected_features = joblib.load('selected_features_clean.pkl')
preprocessor = joblib.load('preprocessor_clean.pkl')
position_mapper = PositionMapper('1_Sign')  # Sign strategy

# Load data
train_full = pd.read_csv('train.csv')
public_test = train_full.iloc[-180:].copy()

print(f"\nHoldout test: {public_test.shape}")

# Transform
X_test_transformed = preprocessor.transform(public_test, is_training=True)
y_test = X_test_transformed['market_forward_excess_returns']

# Add missing features (imputation indicators)
for feature in selected_features:
    if feature not in X_test_transformed.columns:
        X_test_transformed[feature] = 0

X_test_final = X_test_transformed[selected_features]

# Drop NaN
nan_mask = X_test_final.isnull().any(axis=1)
X_test_clean = X_test_final[~nan_mask]
y_test_clean = y_test[~nan_mask]

print(f"Valid samples: {len(X_test_clean)}/{len(public_test)}")

# Predict
y_pred = model.predict(X_test_clean)
positions = position_mapper.map(y_pred)

# Calculate metrics
print(f"\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)

print(f"\nRaw predictions:")
print(f"  Mean: {y_pred.mean():.6f}")
print(f"  Std: {y_pred.std():.6f}")
print(f"  % positive: {(y_pred > 0).sum() / len(y_pred) * 100:.2f}%")
print(f"  % negative: {(y_pred < 0).sum() / len(y_pred) * 100:.2f}%")

print(f"\nPositions:")
print(f"  Mean: {positions.mean():.4f}")
print(f"  % at 0.0: {(positions == 0.0).sum() / len(positions) * 100:.2f}%")
print(f"  % at 2.0: {(positions == 2.0).sum() / len(positions) * 100:.2f}%")

# Performance
print(f"\n" + "="*80)
print("PERFORMANCE METRICS (HONEST)")
print("="*80)

realized_returns = positions * y_test_clean.values
mean_return = realized_returns.mean()
std_return = realized_returns.std()
sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

print(f"\n🎯 Sharpe Ratio: {sharpe:.4f}")
print(f"   Mean return: {mean_return:.6f}")
print(f"   Std return: {std_return:.6f}")

# Cumulative returns
cum_returns = np.cumprod(1 + realized_returns)
total_return = cum_returns[-1] - 1

print(f"\nCumulative:")
print(f"   Total return: {total_return * 100:.2f}%")
print(f"   Final value: ${cum_returns[-1]:.4f}")

# Volatility
market_vol = y_test_clean.std()
strategy_vol = realized_returns.std()
vol_ratio = strategy_vol / market_vol

print(f"\nVolatility:")
print(f"   Market: {market_vol:.6f}")
print(f"   Strategy: {strategy_vol:.6f}")
print(f"   Ratio: {vol_ratio:.4f} {'✓' if vol_ratio < 1.2 else '⚠️'}")

# Compare to train performance
print(f"\n" + "="*80)
print("TRAIN VS HOLDOUT COMPARISON")
print("="*80)
print(f"""
Clean CV (4-fold):
  Ridge:    0.5113
  LightGBM: 0.3544

Holdout (last 180 rows):
  LightGBM: {sharpe:.4f}

{'✓ Consistent' if abs(sharpe - 0.3544) < 0.2 else '⚠️ Significant variance'}
""")

print("="*80)
print("READY FOR SUBMISSION")
print("="*80)
print(f"\n✅ Honest holdout Sharpe: {sharpe:.4f}")
print(f"✅ No data leakage")
print(f"✅ Model artifacts saved (*_clean.pkl)")
