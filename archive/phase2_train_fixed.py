"""
Phase 2: Fixed Model Training Pipeline
FIXED: Sharpe now calculated on realized returns, not predictions
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from purged_kfold import PurgedKFold
from preprocessor import MarketPreprocessor

print("="*80)
print("PHASE 2: FIXED MODEL TRAINING PIPELINE")
print("="*80)

# Step 1: Load and preprocess data
print("\n[1/6] Loading and preprocessing data...")
train = pd.read_csv('train.csv')
print(f"  Loaded train: {train.shape}")

preprocessor = MarketPreprocessor()
X, y = preprocessor.fit_transform(train)
preprocessor.save('preprocessor.pkl')

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Drop rows with NaN
nan_mask = X.isnull().any(axis=1)
n_nan = nan_mask.sum()
if n_nan > 0:
    print(f"  Dropping {n_nan} rows with NaN...")
    X = X[~nan_mask].reset_index(drop=True)
    y = y[~nan_mask].reset_index(drop=True)
    print(f"  Final: X={X.shape}, y={y.shape}")

# Step 2: Feature Selection
print("\n[2/6] Feature selection...")
print(f"  Starting features: {X.shape[1]}")

# Variance filter
variance_selector = VarianceThreshold(threshold=0.01)
X_var = variance_selector.fit_transform(X)
selected_features_var = X.columns[variance_selector.get_support()].tolist()
print(f"  After variance (>0.01): {len(selected_features_var)}")

# Correlation filter
X_var_df = pd.DataFrame(X_var, columns=selected_features_var)
corr_matrix = X_var_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_uncorr = X_var_df.drop(columns=to_drop)
print(f"  After correlation (<0.95): {X_uncorr.shape[1]}")

# Mutual information
n_features_to_keep = min(100, X_uncorr.shape[1])
mi_selector = SelectKBest(score_func=mutual_info_regression, k=n_features_to_keep)
mi_selector.fit(X_uncorr, y)
selected_features = X_uncorr.columns[mi_selector.get_support()].tolist()
print(f"  Final (mutual info top {n_features_to_keep}): {len(selected_features)}")

joblib.dump(selected_features, 'selected_features.pkl')
X_final = X[selected_features].copy()

# Step 3: Setup CV
print("\n[3/6] Setting up PurgedKFold CV...")
cv = PurgedKFold(n_splits=5, embargo_days=20)

# Step 4: FIXED Sharpe calculation
def calculate_sharpe(y_true, y_pred, position_scaling='sign'):
    """
    Calculate Sharpe ratio on REALIZED returns
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        position_scaling: 'sign' or 'scaled'
    
    Returns:
        Sharpe ratio (annualized)
    """
    # Create positions from predictions
    if position_scaling == 'sign':
        positions = np.sign(y_pred)
    elif position_scaling == 'scaled':
        # Scale predictions to [-1, 1] range
        pred_std = np.std(y_pred)
        if pred_std > 0:
            positions = np.clip(y_pred / (2 * pred_std), -1, 1)
        else:
            positions = np.sign(y_pred)
    else:
        positions = y_pred  # Use raw predictions
    
    # Calculate REALIZED returns
    realized_returns = positions * y_true
    
    # Calculate Sharpe
    mean_return = realized_returns.mean()
    std_return = realized_returns.std()
    
    if std_return == 0 or np.isnan(std_return):
        return 0
    
    sharpe = (mean_return / std_return) * np.sqrt(252)
    return sharpe

# Step 5: Train Ridge Baseline
print("\n[4/6] Training Ridge baseline...")
ridge_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_final, y)):
    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sharpe_sign = calculate_sharpe(y_test.values, y_pred, 'sign')
    sharpe_scaled = calculate_sharpe(y_test.values, y_pred, 'scaled')
    
    ridge_scores.append({
        'fold': fold_idx+1,
        'mse': mse,
        'r2': r2,
        'sharpe_sign': sharpe_sign,
        'sharpe_scaled': sharpe_scaled
    })
    
    print(f"  Fold {fold_idx+1}: MSE={mse:.6f}, R2={r2:.4f}, "
          f"Sharpe(sign)={sharpe_sign:.4f}, Sharpe(scaled)={sharpe_scaled:.4f}")

ridge_mean_sharpe = np.mean([s['sharpe_sign'] for s in ridge_scores])
print(f"\n  Ridge Mean Sharpe (sign): {ridge_mean_sharpe:.4f}")

# Step 6: Train LightGBM
print("\n[5/6] Training LightGBM...")
lgb_scores = []

lgb_params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'max_depth': 4,
    'num_leaves': 15,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 300,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'verbose': -1,
    'random_state': 42
}

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_final, y)):
    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sharpe_sign = calculate_sharpe(y_test.values, y_pred, 'sign')
    sharpe_scaled = calculate_sharpe(y_test.values, y_pred, 'scaled')
    
    lgb_scores.append({
        'fold': fold_idx+1,
        'mse': mse,
        'r2': r2,
        'sharpe_sign': sharpe_sign,
        'sharpe_scaled': sharpe_scaled,
        'best_iter': model.best_iteration
    })
    
    print(f"  Fold {fold_idx+1}: MSE={mse:.6f}, R2={r2:.4f}, "
          f"Sharpe(sign)={sharpe_sign:.4f}, Sharpe(scaled)={sharpe_scaled:.4f}, "
          f"Iters={model.best_iteration}")

lgb_mean_sharpe_sign = np.mean([s['sharpe_sign'] for s in lgb_scores])
lgb_mean_sharpe_scaled = np.mean([s['sharpe_scaled'] for s in lgb_scores])
print(f"\n  LightGBM Mean Sharpe (sign): {lgb_mean_sharpe_sign:.4f}")
print(f"  LightGBM Mean Sharpe (scaled): {lgb_mean_sharpe_scaled:.4f}")

# Train final model
print("\n[6/6] Training final LightGBM...")
train_data = lgb.Dataset(X_final, label=y)
final_lgb = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=int(np.mean([s['best_iter'] for s in lgb_scores]))
)

joblib.dump(final_lgb, 'lgb_model.pkl')
print(f"  Saved LightGBM ({final_lgb.num_trees()} trees)")

# Feature importance
importance = final_lgb.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': importance
}).sort_values('importance', ascending=False)
feature_importance.to_csv('feature_importance.csv', index=False)

# Summary
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"""
Dataset: {len(X_final)} samples, {len(selected_features)} features

Ridge Baseline:
  Mean Sharpe (sign):   {ridge_mean_sharpe:.4f}
  Mean Sharpe (scaled): {np.mean([s['sharpe_scaled'] for s in ridge_scores]):.4f}

LightGBM:
  Mean Sharpe (sign):   {lgb_mean_sharpe_sign:.4f}
  Mean Sharpe (scaled): {lgb_mean_sharpe_scaled:.4f}
  Improvement (sign):   {(lgb_mean_sharpe_sign - ridge_mean_sharpe):.4f}

Top 10 Features:
""")
print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)
print("\nRidge:")
print(pd.DataFrame(ridge_scores).to_string(index=False))
print("\nLightGBM:")
print(pd.DataFrame(lgb_scores).to_string(index=False))

print("\n✅ PHASE 2 COMPLETE - Models trained with CORRECT metrics!")
