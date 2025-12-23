"""
REGIME COMPARISON PLOTTER
-------------------------
Generates the visual evidence for Slide 13 ("Why did we score 2.82?").

Goal: Compare the statistical properties of the "Training History" (35 years) 
vs the "Holdout" (Last 6 months).

Key Findings Visualized:
1.  **Returns**: The Holdout period had much higher mean returns (Super Bull Market).
2.  **Volatility**: The Holdout volatility was normal (no crash).
3.  **Conclusion**: The strategy surfed a very profitable wave.

Output: 'exploration_plots/12_regime_comparison.png'
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Generating Regime Comparison Plot...")

# Load data
df = pd.read_csv('train.csv')

# Split into Train and Holdout (Last 180 days)
holdout_size = 180
train_df = df.iloc[:-holdout_size]
holdout_df = df.iloc[-holdout_size:]

# Setup Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.suptitle('Regime Comparison: Training History (10+ Years) vs Holdout (Last 6 Months)', fontsize=16)

# 1. Market Trend (Cumulative Returns)
# We normalize both to start at 1.0 for comparison of "Shape"
train_cum = (1 + train_df['market_forward_excess_returns']).cumprod()
holdout_cum = (1 + holdout_df['market_forward_excess_returns']).cumprod()

# Re-index for plotting (0 to N)
axes[0].plot(np.arange(len(train_cum)), train_cum.values, label='Training History', color='gray', alpha=0.7)
axes[0].set_title('Market Trend (Cumulative Excess Return)')
axes[0].set_xlabel('Trading Days')
axes[0].set_ylabel('Cumulative Return')
# Inset for Holdout? Or just separate?
# Let's plot Holdout on a secondary axis or just note the difference in slope.
# Actually, let's plot the "Average Daily Return" distribution instead.

# Alternative Plot 1: Target Distribution (The "Opportunity")
sns.kdeplot(train_df['market_forward_excess_returns'], ax=axes[0], label='Training (Past)', color='gray', fill=True, alpha=0.3)
sns.kdeplot(holdout_df['market_forward_excess_returns'], ax=axes[0], label='Holdout (Recent)', color='blue', fill=True, alpha=0.3)
axes[0].set_title('Target Distribution (Excess Returns)')
axes[0].set_xlabel('Daily Excess Return')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_xlim(-0.05, 0.05) # Zoom in

# 2. Volatility Regime (The "Risk")
# Assuming 'VIX_Close' exists or we calculate rolling std
if 'VIX_Close' in df.columns:
    vol_col = 'VIX_Close'
    label = 'VIX Level'
else:
    # Calculate realized vol proxy
    vol_col = 'realized_vol'
    df['realized_vol'] = df['market_forward_excess_returns'].rolling(20).std() * np.sqrt(252)
    train_df = df.iloc[:-holdout_size]
    holdout_df = df.iloc[-holdout_size:]
    label = 'Realized Volatility (20D)'

sns.kdeplot(train_df[vol_col].dropna(), ax=axes[1], label='Training (Past)', color='gray', fill=True, alpha=0.3)
sns.kdeplot(holdout_df[vol_col].dropna(), ax=axes[1], label='Holdout (Recent)', color='red', fill=True, alpha=0.3)
axes[1].set_title(f'Volatility Regime ({label})')
axes[1].set_xlabel(label)
axes[1].legend()

# 3. Autocorrelation (The "Memory")
# Calculate Lag-1 Autocorrelation for rolling windows?
# Or just a simple bar chart of Mean Volatility and Mean Return
metrics = pd.DataFrame({
    'Metric': ['Mean Daily Return', 'Daily Volatility', 'Skewness'],
    'Training': [
        train_df['market_forward_excess_returns'].mean(),
        train_df['market_forward_excess_returns'].std(),
        train_df['market_forward_excess_returns'].skew()
    ],
    'Holdout': [
        holdout_df['market_forward_excess_returns'].mean(),
        holdout_df['market_forward_excess_returns'].std(),
        holdout_df['market_forward_excess_returns'].skew()
    ]
})

# Normalize for bar chart? No, raw numbers are fine but small.
# Let's just print text on the plot.
axes[2].axis('off')
axes[2].set_title('Regime Statistics Comparison')
text_str = "Key Differences:\n\n"
text_str += f"MEAN RETURN:\n  Train:   {metrics.loc[0, 'Training']:.6f}\n  Holdout: {metrics.loc[0, 'Holdout']:.6f}\n  -> Holdout was {metrics.loc[0, 'Holdout']/metrics.loc[0, 'Training']:.1f}x more profitable\n\n"
text_str += f"VOLATILITY:\n  Train:   {metrics.loc[1, 'Training']:.6f}\n  Holdout: {metrics.loc[1, 'Holdout']:.6f}\n  -> Holdout was similar vol\n\n"
text_str += f"SKEWNESS:\n  Train:   {metrics.loc[2, 'Training']:.4f}\n  Holdout: {metrics.loc[2, 'Holdout']:.4f}\n  -> Holdout was more positive/negative?"

axes[2].text(0.1, 0.3, text_str, fontsize=12, family='monospace')

plt.tight_layout()
plt.savefig('exploration_plots/12_regime_comparison.png')
print("Saved exploration_plots/12_regime_comparison.png")
