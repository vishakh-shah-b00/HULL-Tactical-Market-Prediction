"""
Debug: Investigate extreme Sharpe values
"""
import pandas as pd
import numpy as np
from preprocessor import MarketPreprocessor
from purged_kfold import PurgedKFold
import joblib

print("="*80)
print("DEBUG: INVESTIGATING EXTREME SHARPE VALUES")
print("="*80)

# Load data
train = pd.read_csv('train.csv')
preprocessor = MarketPreprocessor.load('preprocessor.pkl')
X, y = preprocessor.fit_transform(train)

# Drop NaN
nan_mask = X.isnull().any(axis=1)
X = X[~nan_mask].reset_index(drop=True)
y = y[~nan_mask].reset_index(drop=True)

# Load selected features
selected_features = joblib.load('selected_features.pkl')
X_final = X[selected_features]

print(f"\nData shape: X={X_final.shape}, y={y.shape}")

# Get first fold
cv = PurgedKFold(n_splits=5, embargo_days=20)
train_idx, test_idx = next(cv.split(X_final, y))

X_test = X_final.iloc[test_idx]
y_test = y.iloc[test_idx]

print(f"\nTest set: {len(y_test)} samples")
print(f"y_test stats:")
print(f"  mean: {y_test.mean():.8f}")
print(f"  std: {y_test.std():.8f}")
print(f"  min: {y_test.min():.6f}")
print(f"  max: {y_test.max():.6f}")

# Load model and predict
model = joblib.load('lgb_model.pkl')
y_pred = model.predict(X_test)

print(f"\ny_pred stats:")
print(f"  mean: {y_pred.mean():.8f}")
print(f"  std: {y_pred.std():.8f}")
print(f"  min: {y_pred.min():.6f}")
print(f"  max: {y_pred.max():.6f}")

# ISSUE: Sharpe calculation error
print("\n" + "="*80)
print("SHARPE CALCULATION DEBUG")
print("="*80)

# Current (wrong) calculation
excess_returns = y_pred  # Using predictions as excess returns
mean_return = excess_returns.mean()
std_return = excess_returns.std()
sharpe_wrong = (mean_return / std_return) * np.sqrt(252)

print(f"\nWRONG calculation (using y_pred):")
print(f"  mean: {mean_return:.8f}")
print(f"  std: {std_return:.8f}")
print(f"  Sharpe: {sharpe_wrong:.4f}")

# Correct calculation
actual_returns = y_test.values
predicted_returns = y_pred
strategy_returns = predicted_returns  # In reality, would be position * actual_returns
mean_strategy = strategy_returns.mean()
std_strategy = strategy_returns.std()
sharpe_correct = (mean_strategy / std_strategy) * np.sqrt(252) if std_strategy > 0 else 0

print(f"\nCORRECT calculation (should use strategy returns):")
print(f"  mean: {mean_strategy:.8f}")
print(f"  std: {std_strategy:.8f}")
print(f"  Sharpe: {sharpe_correct:.4f}")

# Even more correct: use actual test returns weighted by prediction signal
# Simplified: assume we invest proportional to predicted return sign/magnitude
print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)

print("\n❌ Current bug: Sharpe calculated on PREDICTIONS, not RETURNS")
print("✓ Should calculate: Sharpe of ACTUAL returns when following strategy")
print("\nThe correct approach:")
print("1. Predict expected return: y_pred")
print("2. Create position: pos = f(y_pred)  # e.g., sign or scaled")
print("3. Realize return: strategy_return = pos * y_actual")
print("4. Calculate Sharpe: mean(strategy_return) / std(strategy_return) * sqrt(252)")

print("\n📊 Quick fix test:")
# Assume position = sign of prediction
positions = np.sign(y_pred)
realized_returns = positions * y_test.values
mean_realized = realized_returns.mean()
std_realized = realized_returns.std()
sharpe_realized = (mean_realized / std_realized) * np.sqrt(252) if std_realized > 0 else 0

print(f"  Position strategy: sign(y_pred) * y_actual")
print(f"  Mean return: {mean_realized:.8f}")
print(f"  Std return: {std_realized:.8f}")
print(f"  Sharpe: {sharpe_realized:.4f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
FIX REQUIRED:
1. Update calculate_sharpe() to use ACTUAL returns, not predictions
2. Implement proper position mapping (sign or scaled)
3. Recalculate all metrics

Current Sharpe values are MEANINGLESS and should be disregarded.
""")
