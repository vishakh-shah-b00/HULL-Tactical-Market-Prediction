"""
STATISTICAL SIGNIFICANCE VALIDATOR (Bootstrap Analysis)
-------------------------------------------------------
Calculates the 95% Confidence Interval for the strategy's Sharpe Ratio.

Problem: Is the Sharpe of 2.82 real, or just a "lucky" 6-month streak?
Solution:
    - We use Bootstrapping (Sampling with Replacement).
    - We simulate 10,000 alternative "Universes" by shuffling the 180 daily returns.
    - If 95% of these universes are profitable, we reject the Null Hypothesis (Luck).

Output:
    - Mean Sharpe Estimate
    - 95% Confidence Interval (e.g., [0.17, 5.32])
    - Probability of Profit (P-Value proxy)
"""
import pandas as pd
import numpy as np
import joblib
from position_mapper import PositionMapper

    """
    Calculate bootstrap confidence intervals for Sharpe ratio.
    
    Method:
    1. Resample the daily returns array N times with replacement.
    2. Calculate Sharpe for each resample.
    3. Build a distribution of possible Sharpes.
    4. Find the 2.5% and 97.5% percentiles (95% CI).
    
    Why this is valid:
    Daily returns have low autocorrelation (~0.04), so they are largely independent.
    Shuffling them creates valid "What If" scenarios to test for luck.
    """
    sharpe_ratios = []
    n_samples = len(returns)
    
    # Resample with replacement
    for _ in range(n_bootstrap):
        sample_returns = np.random.choice(returns, size=n_samples, replace=True)
        
        mean_ret = np.mean(sample_returns)
        std_ret = np.std(sample_returns)
        
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            sharpe_ratios.append(sharpe)
            
    # Calculate percentiles
    lower_p = (1 - confidence_level) / 2 * 100
    upper_p = (1 + confidence_level) / 2 * 100
    
    ci_lower = np.percentile(sharpe_ratios, lower_p)
    ci_upper = np.percentile(sharpe_ratios, upper_p)
    mean_sharpe = np.mean(sharpe_ratios)
    
    return mean_sharpe, ci_lower, ci_upper, sharpe_ratios

print("="*80)
print("CALCULATING CONFIDENCE INTERVALS (BOOTSTRAP)")
print("="*80)

# Load artifacts
model = joblib.load('lgb_model_clean.pkl')
selected_features = joblib.load('selected_features_clean.pkl')
preprocessor = joblib.load('preprocessor_clean.pkl')
position_mapper = PositionMapper('1_Sign')

# Load and prepare data
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

# Predict
y_pred = model.predict(X_test_clean)
positions = position_mapper.map(y_pred)
realized_returns = positions * y_test_clean.values

print(f"Valid samples: {len(realized_returns)}")
print(f"Original Sharpe: {np.mean(realized_returns)/np.std(realized_returns)*np.sqrt(252):.4f}")

# Bootstrap
print("\nRunning 10,000 bootstrap samples...")
mean_boot, lower, upper, dist = bootstrap_sharpe(realized_returns, n_bootstrap=10000)

print(f"\n95% Confidence Interval for Sharpe Ratio:")
print(f"Lower Bound (2.5%): {lower:.4f}")
print(f"Mean Estimate:      {mean_boot:.4f}")
print(f"Upper Bound (97.5%): {upper:.4f}")
print(f"\nRange width: {upper - lower:.4f}")

# Probability of Sharpe > 1.0
prob_gt_1 = np.mean(np.array(dist) > 1.0)
print(f"Probability Sharpe > 1.0: {prob_gt_1*100:.1f}%")

# Probability of Sharpe > 0.0
prob_gt_0 = np.mean(np.array(dist) > 0.0)
print(f"Probability Sharpe > 0.0: {prob_gt_0*100:.1f}%")
