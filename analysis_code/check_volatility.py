"""
MARKET REGIME DIAGNOSTIC
------------------------
Checks if the Holdout period (Test Set) has significantly different volatility than the Training History.

Why?
- If Volatility Ratio > 1.2, the competition metric penalizes us.
- We need to know if we are testing in a "Quiet Bull Market" or a "Crash".

Output:
- Annualized Volatility (Train vs Holdout).
- Regime Conclusion (Normal vs High Vol).
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('train.csv')

# Split into Train and Holdout (Last 180 days)
holdout_size = 180
train_df = df.iloc[:-holdout_size]
holdout_df = df.iloc[-holdout_size:]

# Calculate Annualized Volatility (Std Dev of Returns * sqrt(252))
# We use 'forward_returns' as the proxy for market returns
train_vol = train_df['forward_returns'].std() * np.sqrt(252)
holdout_vol = holdout_df['forward_returns'].std() * np.sqrt(252)

print(f"Training Volatility (Annualized): {train_vol:.2%}")
print(f"Holdout Volatility (Annualized):  {holdout_vol:.2%}")

ratio = holdout_vol / train_vol
print(f"Ratio (Holdout / Train): {ratio:.2f}")

if ratio > 1.2:
    print("CONCLUSION: YES, High Volatility Regime.")
elif ratio < 0.8:
    print("CONCLUSION: NO, Low Volatility Regime.")
else:
    print("CONCLUSION: NORMAL Volatility Regime.")
