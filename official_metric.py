import numpy as np
import pandas as pd
import pandas.api.types
import joblib
from position_mapper import PositionMapper

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2

class ParticipantVisibleError(Exception):
    pass

"""
OFFICIAL VALIDATION SCRIPT
--------------------------
Replicates the exact competition scoring logic to validate performance on the Holdout set.

Metric: Volatility-Adjusted Sharpe Ratio
Formula: Adjusted Sharpe = Raw Sharpe / (Vol Penalty * Return Penalty)

1. Vol Penalty: Linear penalty if Strategy Vol > 1.2x Market Vol.
2. Return Penalty: Quadratic penalty if Strategy Return < Market Return.
"""
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates the Official Volatility-Adjusted Sharpe Ratio.
    
    Args:
        solution: DataFrame with 'forward_returns' and 'risk_free_rate' (Truth).
        submission: DataFrame with 'prediction' (Strategy Position).
        
    Returns:
        float: The adjusted Sharpe score.
    """
    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution = solution.copy()
    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    print(f"DEBUG INFO:")
    print(f"  Raw Sharpe: {sharpe:.4f}")
    print(f"  Strategy Vol: {strategy_volatility:.2f}%")
    print(f"  Market Vol: {market_volatility:.2f}%")
    print(f"  Vol Ratio: {strategy_volatility/market_volatility:.4f}")
    print(f"  Vol Penalty: {vol_penalty:.4f}")
    print(f"  Strategy Mean Return: {strategy_mean_excess_return*252:.4f}")
    print(f"  Market Mean Return: {market_mean_excess_return*252:.4f}")
    print(f"  Return Penalty: {return_penalty:.4f}")

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)

print("="*80)
print("OFFICIAL METRIC VALIDATION")
print("="*80)

# Load artifacts
model = joblib.load('lgb_model_clean.pkl')
selected_features = joblib.load('selected_features_clean.pkl')
preprocessor = joblib.load('preprocessor_clean.pkl')
position_mapper = PositionMapper('1_Sign')

# Load data
train_full = pd.read_csv('train.csv')
public_test = train_full.iloc[-180:].copy()

# Need 'risk_free_rate' and 'forward_returns' (which is market_forward_excess_returns + risk_free_rate?)
# Wait, the target 'market_forward_excess_returns' IS (Market - RiskFree).
# And 'forward_returns' usually means Market Return.
# So forward_returns = market_forward_excess_returns + risk_free_rate.
# Let's check columns.
# We don't have 'risk_free_rate' column in train.csv?
# Let's assume risk_free_rate is 0 for now or check if it exists.
# Actually, the problem statement says target is excess returns.
# If we assume risk_free_rate = 0, then:
# forward_returns = market_forward_excess_returns
# strategy_returns = 0*(1-pos) + pos*forward_returns = pos*market_forward_excess_returns
# This matches our 'realized_returns' calculation.

if 'risk_free_rate' not in public_test.columns:
    print("Warning: 'risk_free_rate' not found. Assuming 0.0 for calculation.")
    public_test['risk_free_rate'] = 0.0

# Construct 'forward_returns' from target if needed
# If target is excess, and we assume rf=0, then forward_returns = target.
public_test['forward_returns'] = public_test['market_forward_excess_returns'] + public_test['risk_free_rate']

# Transform
X_test = preprocessor.transform(public_test, is_training=True)
for feature in selected_features:
    if feature not in X_test.columns:
        X_test[feature] = 0
X_test_final = X_test[selected_features]

# Drop NaN
nan_mask = X_test_final.isnull().any(axis=1)
X_test_clean = X_test_final[~nan_mask]
solution_df = public_test[~nan_mask].copy()

# Predict
y_pred = model.predict(X_test_clean)
positions = position_mapper.map(y_pred)

submission_df = pd.DataFrame({'prediction': positions}, index=solution_df.index)

# Run Score
try:
    final_score = score(solution_df, submission_df, 'id')
    print(f"\n✅ OFFICIAL ADJUSTED SHARPE: {final_score:.4f}")
except Exception as e:
    print(f"\n❌ Error calculating score: {e}")
