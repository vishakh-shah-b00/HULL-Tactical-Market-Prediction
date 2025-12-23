"""
STABILITY & CONSISTENCY CHECKER
-------------------------------
Analyzes the strategy's performance "through time" rather than just a single number.

Metrics:
1. Cumulative Returns: Did we ever lose money? (Drawdown check)
2. Rolling Sharpe (30-day): How consistent is the edge?
   - Formula: (Mean / Std) * sqrt(252) over a sliding 30-day window.
   - Goal: High % of positive windows (Consistency).

Output:
- Text stats (Min Equity, % Positive Windows)
- 'rolling_metrics.png' (Visual proof)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from position_mapper import PositionMapper

print("="*80)
print("ROLLING METRICS & DRAWDOWN ANALYSIS")
print("="*80)

# Load artifacts
model = joblib.load('lgb_model_clean.pkl')
selected_features = joblib.load('selected_features_clean.pkl')
preprocessor = joblib.load('preprocessor_clean.pkl')
position_mapper = PositionMapper('1_Sign')

# Load data
train_full = pd.read_csv('train.csv')
public_test = train_full.iloc[-180:].copy()

# Transform
X_test = preprocessor.transform(public_test, is_training=True)
for feature in selected_features:
    if feature not in X_test.columns:
        X_test[feature] = 0
X_test_final = X_test[selected_features]

# Drop NaN
nan_mask = X_test_final.isnull().any(axis=1)
X_test_clean = X_test_final[~nan_mask]
y_test_clean = X_test['market_forward_excess_returns'][~nan_mask]
dates = public_test['date_id'][~nan_mask].values

# Predict
y_pred = model.predict(X_test_clean)
positions = position_mapper.map(y_pred)
realized_returns = positions * y_test_clean.values

# 1. Cumulative Returns Analysis
cum_returns = np.cumprod(1 + realized_returns)
min_cum_ret = cum_returns.min()
final_cum_ret = cum_returns[-1]

print(f"\nCumulative Return Stats:")
print(f"  Start: 1.0000")
print(f"  Min:   {min_cum_ret:.4f} (Did it go below 1? {'Yes' if min_cum_ret < 1 else 'No'})")
print(f"  Final: {final_cum_ret:.4f}")

if min_cum_ret < 1:
    print("  -> The strategy experienced a drawdown below initial capital.")
    # Find when it happened
    below_1_indices = np.where(cum_returns < 1)[0]
    print(f"  -> Days below 1.0: {len(below_1_indices)} days")
    print(f"  -> First dip below 1: Day {below_1_indices[0]} ({dates[below_1_indices[0]]})")

# 2. Rolling Sharpe Ratio
window = 30 # 30-day rolling window
rolling_sharpe = []

print(f"\nRolling Sharpe ({window}-day window):")
for i in range(len(realized_returns)):
    if i < window:
        rolling_sharpe.append(np.nan)
    else:
        window_ret = realized_returns[i-window:i]
        if np.std(window_ret) > 0:
            s = (np.mean(window_ret) / np.std(window_ret)) * np.sqrt(252)
            rolling_sharpe.append(s)
        else:
            rolling_sharpe.append(0)

rolling_sharpe = np.array(rolling_sharpe)
valid_rolling = rolling_sharpe[~np.isnan(rolling_sharpe)]

print(f"  Mean Rolling Sharpe: {np.mean(valid_rolling):.4f}")
print(f"  Min Rolling Sharpe:  {np.min(valid_rolling):.4f}")
print(f"  Max Rolling Sharpe:  {np.max(valid_rolling):.4f}")
print(f"  % Positive Windows:  {np.mean(valid_rolling > 0)*100:.1f}%")

# Plotting
plt.figure(figsize=(12, 8))

# Plot 1: Cumulative Returns
plt.subplot(2, 1, 1)
plt.plot(cum_returns, label='Strategy Equity')
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break Even')
plt.title('Holdout Cumulative Returns')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Rolling Sharpe
plt.subplot(2, 1, 2)
plt.plot(rolling_sharpe, label=f'{window}-Day Rolling Sharpe', color='orange')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=np.mean(valid_rolling), color='orange', linestyle='--', alpha=0.5, label='Mean')
plt.title(f'{window}-Day Rolling Sharpe Ratio')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rolling_metrics.png')
print("\nSaved plot to rolling_metrics.png")
