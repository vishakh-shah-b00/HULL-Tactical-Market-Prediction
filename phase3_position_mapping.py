"""
PHASE 3: BETTING STRATEGY RESEARCH (The "Casino" Phase)
-------------------------------------------------------
Goal: Convert raw model scores (regression outputs) into portfolio allocations [0.0, 2.0].

Tested Strategies:
1. Sign-Based (Winner): If Positive -> 2.0 (Max Long), If Negative -> 0.0 (Cash).
2. Scaled: Linear mapping.
3. Sigmoid/Tanh: Non-linear mapping (Sniper).
4. Tercile: Buckets.

Outcome:
'Sign-Based' won because the model's *direction* is accurate, but its *magnitude* calibration is noisy.
This script saves the 'position_mapper.pkl' artifact used by the Model class.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from preprocessor import MarketPreprocessor

print("="*80)
print("PHASE 3: POSITION MAPPING STRATEGY")
print("="*80)

# Load model and data
print("\n[1/5] Loading model and data...")
model = joblib.load('lgb_model.pkl')
selected_features = joblib.load('selected_features.pkl')

train = pd.read_csv('train.csv')
preprocessor = MarketPreprocessor.load('preprocessor.pkl')
X, y = preprocessor.fit_transform(train)

# Drop NaN
nan_mask = X.isnull().any(axis=1)
X = X[~nan_mask].reset_index(drop=True)
y = y[~nan_mask].reset_index(drop=True)

X_final = X[selected_features]
print(f"  Data: {len(X_final)} samples")

# Get predictions
y_pred = model.predict(X_final)
print(f"  Predictions: mean={y_pred.mean():.6f}, std={y_pred.std():.6f}")

# Analyze prediction distribution
print("\n[2/5] Analyzing prediction distribution...")
print(f"  Min: {y_pred.min():.6f}")
print(f"  25%: {np.percentile(y_pred, 25):.6f}")
print(f"  50%: {np.percentile(y_pred, 50):.6f}")
print(f"  75%: {np.percentile(y_pred, 75):.6f}")
print(f"  Max: {y_pred.max():.6f}")

# Position mapping strategies
print("\n[3/5] Testing position mapping strategies...")

def position_strategy_1_sign(predictions):
    """Simple: 2 if positive, 0 if negative"""
    return np.where(predictions > 0, 2.0, 0.0)

def position_strategy_2_scaled(predictions):
    """Scaled: map predictions to [0, 2] linearly"""
    pred_min = predictions.min()
    pred_max = predictions.max()
    
    if pred_max == pred_min:
        return np.ones_like(predictions)
    
    # Linear scaling
    positions = 2.0 * (predictions - pred_min) / (pred_max - pred_min)
    return np.clip(positions, 0.0, 2.0)

def position_strategy_3_sigmoid(predictions, center=0, scale=1):
    """Sigmoid: smooth mapping around center"""
    # Sigmoid function: 2 / (1 + exp(-x/scale))
    positions = 2.0 / (1.0 + np.exp(-(predictions - center) / scale))
    return np.clip(positions, 0.0, 2.0)

def position_strategy_4_tercile(predictions):
    """Tercile: 0, 1, or 2 based on terciles"""
    tercile_33 = np.percentile(predictions, 33.33)
    tercile_67 = np.percentile(predictions, 66.67)
    
    positions = np.zeros_like(predictions)
    positions[predictions >= tercile_67] = 2.0
    positions[(predictions >= tercile_33) & (predictions < tercile_67)] = 1.0
    positions[predictions < tercile_33] = 0.0
    
    return positions

def position_strategy_5_tanh(predictions):
    """Tanh: centered at 1, range [0, 2]"""
    # tanh maps [-inf, inf] to [-1, 1]
    # We shift to [0, 2]
    normalized = predictions / (np.std(predictions) + 1e-8)
    positions = 1.0 + np.tanh(normalized)
    return np.clip(positions, 0.0, 2.0)

# Calculate Sharpe for each strategy
strategies = {
    '1_Sign': position_strategy_1_sign,
    '2_Scaled': position_strategy_2_scaled,
    '3_Sigmoid': lambda p: position_strategy_3_sigmoid(p, center=0, scale=y_pred.std()),
    '4_Tercile': position_strategy_4_tercile,
    '5_Tanh': position_strategy_5_tanh
}

results = []
for name, strategy_func in strategies.items():
    positions = strategy_func(y_pred)
    
    # Calculate realized returns
    # Position interpretation: fraction of capital invested
    # realized_return = position * market_return
    realized_returns = positions * y.values
    
    # Calculate metrics
    mean_return = realized_returns.mean()
    std_return = realized_returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # Position statistics
    pos_mean = positions.mean()
    pos_std = positions.std()
    pos_zero = (positions == 0).sum() / len(positions) * 100
    pos_two = (positions == 2).sum() / len(positions) * 100
    
    results.append({
        'Strategy': name,
        'Sharpe': sharpe,
        'Mean Return': mean_return,
        'Std Return': std_return,
        'Pos Mean': pos_mean,
        'Pos Std': pos_std,
        '% at 0': pos_zero,
        '% at 2': pos_two
    })

results_df = pd.DataFrame(results).sort_values('Sharpe', ascending=False)
print("\nStrategy Comparison:")
print(results_df.to_string(index=False))

# Select best strategy
best_strategy_name = results_df.iloc[0]['Strategy']
best_sharpe = results_df.iloc[0]['Sharpe']
print(f"\n✓ Best strategy: {best_strategy_name} with Sharpe {best_sharpe:.4f}")

# Implement best strategy
print("\n[4/5] Implementing best strategy...")
best_func = strategies[best_strategy_name]
final_positions = best_func(y_pred)

print(f"  Position distribution:")
print(f"    Mean: {final_positions.mean():.4f}")
print(f"    Std: {final_positions.std():.4f}")
print(f"    Min: {final_positions.min():.4f}")
print(f"    Max: {final_positions.max():.4f}")
print(f"    % at 0: {(final_positions == 0).sum() / len(final_positions) * 100:.2f}%")
print(f"    % at 2: {(final_positions == 2).sum() / len(final_positions) * 100:.2f}%")

# Volatility analysis
print("\n[5/5] Volatility analysis...")
realized_returns = final_positions * y.values

# Rolling volatility (20-day)
rolling_vol = pd.Series(realized_returns).rolling(20).std()
market_vol = pd.Series(y.values).rolling(20).std()

volatility_ratio = rolling_vol / market_vol
valid_ratio = volatility_ratio.dropna()

print(f"  Strategy volatility / Market volatility:")
print(f"    Mean: {valid_ratio.mean():.4f}")
print(f"    Median: {valid_ratio.median():.4f}")
print(f"    Max: {valid_ratio.max():.4f}")
print(f"    % > 1.2: {(valid_ratio > 1.2).sum() / len(valid_ratio) * 100:.2f}%")

if valid_ratio.mean() > 1.2:
    print(f"  ⚠️ Average volatility ratio {valid_ratio.mean():.4f} > 1.2 penalty threshold")
    print(f"  Consider volatility targeting or position scaling")
else:
    print(f"  ✓ Volatility ratio {valid_ratio.mean():.4f} within acceptable range")

# Save position mapping function
print("\n" + "="*80)
print("SAVING POSITION MAPPER")
print("="*80)

class PositionMapper:
    """Converts predictions to [0, 2] positions"""
    
    def __init__(self, strategy_name, strategy_params=None):
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        
    def map(self, predictions):
        """Map predictions to positions [0, 2]"""
        if self.strategy_name == '1_Sign':
            return np.where(predictions > 0, 2.0, 0.0)
        
        elif self.strategy_name == '2_Scaled':
            pred_min = predictions.min()
            pred_max = predictions.max()
            if pred_max == pred_min:
                return np.ones_like(predictions)
            positions = 2.0 * (predictions - pred_min) / (pred_max - pred_min)
            return np.clip(positions, 0.0, 2.0)
        
        elif self.strategy_name == '3_Sigmoid':
            center = self.strategy_params.get('center', 0)
            scale = self.strategy_params.get('scale', 1)
            positions = 2.0 / (1.0 + np.exp(-(predictions - center) / scale))
            return np.clip(positions, 0.0, 2.0)
        
        elif self.strategy_name == '4_Tercile':
            tercile_33 = np.percentile(predictions, 33.33)
            tercile_67 = np.percentile(predictions, 66.67)
            positions = np.zeros_like(predictions)
            positions[predictions >= tercile_67] = 2.0
            positions[(predictions >= tercile_33) & (predictions < tercile_67)] = 1.0
            return positions
        
        elif self.strategy_name == '5_Tanh':
            normalized = predictions / (np.std(predictions) + 1e-8)
            positions = 1.0 + np.tanh(normalized)
            return np.clip(positions, 0.0, 2.0)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")

# Create and save mapper
mapper = PositionMapper(
    strategy_name=best_strategy_name,
    strategy_params={'scale': y_pred.std()} if '3' in best_strategy_name else {}
)
joblib.dump(mapper, 'position_mapper.pkl')
print(f"✓ Saved position mapper: {best_strategy_name}")

print("\n✅ PHASE 3 STEP 1 COMPLETE - Position mapping strategy selected")
print(f"\nNext: Backtest with official competition scorer")
