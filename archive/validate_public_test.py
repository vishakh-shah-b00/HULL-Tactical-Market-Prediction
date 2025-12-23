"""
Validate on ACTUAL public leaderboard test set (last 180 train rows)
"""
import pandas as pd
import numpy as np
import joblib
from preprocessor import MarketPreprocessor
from position_mapper import PositionMapper

print("="*80)
print("VALIDATION ON PUBLIC LEADERBOARD TEST SET")
print("="*80)

# Load train
train = pd.read_csv('train.csv')
print(f"\nFull train: {train.shape}")

# Split: last 180 = public test, rest = our training
public_test_size = 180
our_train = train.iloc[:-public_test_size].copy()
public_test = train.iloc[-public_test_size:].copy()

print(f"Our train: {our_train.shape}")
print(f"Public test (last 180): {public_test.shape}")

# Load model
model = joblib.load('lgb_model.pkl')
selected_features = joblib.load('selected_features.pkl')
preprocessor = joblib.load('preprocessor.pkl')
position_mapper = joblib.load('position_mapper.pkl')

# Process public test
print("\n" + "="*80)
print("PROCESSING PUBLIC TEST SET")
print("="*80)

X_test_transformed = preprocessor.transform(public_test, is_training=True)
y_test = X_test_transformed['market_forward_excess_returns']

# Handle missing features
for feature in selected_features:
    if feature not in X_test_transformed.columns:
        X_test_transformed[feature] = 0

X_test_final = X_test_transformed[selected_features]

# Drop NaN rows (from lags/rolling at start)
nan_mask = X_test_final.isnull().any(axis=1)
valid_idx = ~nan_mask

X_test_clean = X_test_final[valid_idx]
y_test_clean = y_test[valid_idx]

print(f"\nAfter dropping NaN:")
print(f"  Valid samples: {len(X_test_clean)}/{len(public_test)}")

# Predict
y_pred = model.predict(X_test_clean)
positions = position_mapper.map(y_pred)

print(f"\n" + "="*80)
print("PUBLIC TEST PREDICTIONS")
print("="*80)

print(f"\nRaw predictions:")
print(f"  Mean: {y_pred.mean():.6f}")
print(f"  Std: {y_pred.std():.6f}")
print(f"  Min: {y_pred.min():.6f}")
print(f"  Max: {y_pred.max():.6f}")
print(f"  % positive: {(y_pred > 0).sum() / len(y_pred) * 100:.2f}%")
print(f"  % negative: {(y_pred < 0).sum() / len(y_pred) * 100:.2f}%")

print(f"\nPositions (sign strategy):")
print(f"  Mean: {positions.mean():.4f}")
print(f"  % at 0.0: {(positions == 0.0).sum() / len(positions) * 100:.2f}%")
print(f"  % at 2.0: {(positions == 2.0).sum() / len(positions) * 100:.2f}%")

# Calculate performance metrics
print(f"\n" + "="*80)
print("PERFORMANCE METRICS")
print("="*80)

# Realized returns
realized_returns = positions * y_test_clean.values

# Sharpe
mean_return = realized_returns.mean()
std_return = realized_returns.std()
sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

print(f"\nSharpe Ratio: {sharpe:.4f}")
print(f"  Mean return: {mean_return:.6f}")
print(f"  Std return: {std_return:.6f}")

# Cumulative returns
cum_returns = (1 + realized_returns).cumprod()
total_return = cum_returns.iloc[-1] - 1

print(f"\nCumulative Performance:")
print(f"  Total return: {total_return * 100:.2f}%")
print(f"  Final value ($1 start): ${cum_returns.iloc[-1]:.4f}")

# Volatility check
market_vol = y_test_clean.std()
strategy_vol = realized_returns.std()
vol_ratio = strategy_vol / market_vol

print(f"\nVolatility:")
print(f"  Market vol: {market_vol:.6f}")
print(f"  Strategy vol: {strategy_vol:.6f}")
print(f"  Ratio: {vol_ratio:.4f} {'✓' if vol_ratio < 1.2 else '⚠️ >1.2'}")

# Save public test predictions for submission
print(f"\n" + "="*80)
print("SAVING PUBLIC TEST PREDICTIONS")
print("="*80)

# Need to handle the NaN rows - use position=1.0 (market neutral) for them
all_positions = np.ones(len(public_test))  # Default to 1.0
all_positions[valid_idx.values] = positions  # Fill in predicted positions

submission_public = pd.DataFrame({
    'id': range(len(all_positions)),
    'position': all_positions
})

submission_public.to_csv('submission_public_test.csv', index=False)
print(f"✓ Saved submission_public_test.csv")
print(f"  Total rows: {len(submission_public)}")
print(f"  Mean position: {all_positions.mean():.4f}")

print(f"\n✅ PUBLIC LEADERBOARD VALIDATION COMPLETE")
print(f"\n🎯 Sharpe: {sharpe:.4f} on last 180 train rows (public test)")
