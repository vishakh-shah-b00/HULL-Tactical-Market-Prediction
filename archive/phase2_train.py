"""
Phase 2: Complete Model Training Pipeline
Features: PurgedKFold CV, Feature Selection, Baseline Models, LightGBM
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from purged_kfold import PurgedKFold
from preprocessor import MarketPreprocessor

print("="*80)
print("PHASE 2: MODEL TRAINING PIPELINE")
print("="*80)

# Step 1: Load and preprocess data
print("\n[1/7] Loading and preprocessing data...")
train = pd.read_csv('train.csv')
print(f"  Loaded train: {train.shape}")

preprocessor = MarketPreprocessor()
X, y = preprocessor.fit_transform(train)
preprocessor.save('preprocessor.pkl')

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Features: {X.shape[1]}")

# Drop rows with NaN (from lag/rolling features at start)
nan_mask = X.isnull().any(axis=1)
n_nan = nan_mask.sum()
if n_nan > 0:
    print(f"  Dropping {n_nan} rows with NaN values...")
    X = X[~nan_mask].reset_index(drop=True)
    y = y[~nan_mask].reset_index(drop=True)
    print(f"  Final shape: X={X.shape}, y={y.shape}")

# Step 2: Feature Selection
print("\n[2/7] Feature selection...")
print(f"  Starting features: {X.shape[1]}")

# 2a: Remove low variance features
variance_selector = VarianceThreshold(threshold=0.01)
X_var = variance_selector.fit_transform(X)
selected_features_var = X.columns[variance_selector.get_support()].tolist()
print(f"  After variance filter (var > 0.01): {len(selected_features_var)}")

# 2b: Remove highly correlated features
X_var_df = pd.DataFrame(X_var, columns=selected_features_var)
corr_matrix = X_var_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_uncorr = X_var_df.drop(columns=to_drop)
print(f"  After correlation filter (corr < 0.95): {X_uncorr.shape[1]}")
print(f"  Dropped {len(to_drop)} highly correlated features")

# 2c: Select top features by mutual information
n_features_to_keep = min(100, X_uncorr.shape[1])
mi_selector = SelectKBest(score_func=mutual_info_regression, k=n_features_to_keep)
X_selected = mi_selector.fit_transform(X_uncorr, y)
selected_features = X_uncorr.columns[mi_selector.get_support()].tolist()

print(f"  Final features (mutual info top {n_features_to_keep}): {len(selected_features)}")

# Save selected features
joblib.dump(selected_features, 'selected_features.pkl')
print(f"  Saved selected features to 'selected_features.pkl'")

# Create final X with selected features
X_final = X[selected_features].copy()

# Step 3: Setup Cross-Validation
print("\n[3/7] Setting up PurgedKFold CV...")
cv = PurgedKFold(n_splits=5, embargo_days=20)
print(f"  CV: {cv.n_splits} folds, {cv.embargo_days} embargo days")

# Step 4: Define Sharpe calculation
def calculate_sharpe(y_true, y_pred, annualize=True):
    """Calculate Sharpe ratio"""
    excess_returns = y_pred  # Already excess returns
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    
    if std_return == 0:
        return 0
    
    sharpe = mean_return / std_return
    
    if annualize:
        sharpe = sharpe * np.sqrt(252)
    
    return sharpe

# Step 5: Train Ridge Baseline
print("\n[4/7] Training Ridge baseline...")
ridge_scores = []
ridge_sharpes = []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_final, y)):
    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    
    # Predict
    y_pred = ridge.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sharpe = calculate_sharpe(y_test, y_pred)
    
    ridge_scores.append({'fold': fold_idx+1, 'mse': mse, 'r2': r2, 'sharpe': sharpe})
    ridge_sharpes.append(sharpe)
    
    print(f"  Fold {fold_idx+1}: MSE={mse:.6f}, R2={r2:.4f}, Sharpe={sharpe:.4f}")

ridge_mean_sharpe = np.mean(ridge_sharpes)
print(f"\n  Ridge Mean Sharpe: {ridge_mean_sharpe:.4f}")

# Decision point
if ridge_mean_sharpe < 0.2:
    print(f"  ⚠️ Ridge Sharpe < 0.2 - Consider feature re-engineering")
else:
    print(f"  ✓ Ridge Sharpe >= 0.2 - Proceed to LightGBM")

# Step 6: Train LightGBM
print("\n[5/7] Training LightGBM...")
lgb_scores = []
lgb_sharpes = []
lgb_models = []

# LightGBM parameters (heavy regularization)
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
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train
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
    
    # Predict
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sharpe = calculate_sharpe(y_test, y_pred)
    
    lgb_scores.append({'fold': fold_idx+1, 'mse': mse, 'r2': r2, 'sharpe': sharpe, 
                       'best_iter': model.best_iteration})
    lgb_sharpes.append(sharpe)
    lgb_models.append(model)
    
    print(f"  Fold {fold_idx+1}: MSE={mse:.6f}, R2={r2:.4f}, Sharpe={sharpe:.4f}, Iters={model.best_iteration}")

lgb_mean_sharpe = np.mean(lgb_sharpes)
print(f"\n  LightGBM Mean Sharpe: {lgb_mean_sharpe:.4f}")

# Step 7: Train final LightGBM on full data
print("\n[6/7] Training final LightGBM on full data...")
train_data = lgb.Dataset(X_final, label=y)
final_lgb = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=int(np.mean([s['best_iter'] for s in lgb_scores]))
)
print(f"  Trained for {final_lgb.num_trees()} iterations")

# Save models
joblib.dump(final_lgb, 'lgb_model.pkl')
print(f"  Saved LightGBM model")

# Step 8: Feature Importance
print("\n[7/7] Feature importance analysis...")
importance = final_lgb.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
print(feature_importance.head(20).to_string(index=False))

feature_importance.to_csv('feature_importance.csv', index=False)
print(f"\nSaved feature importance to 'feature_importance.csv'")

# Final Summary
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"""
Dataset:
  Samples: {len(X_final)}
  Features (original): {X.shape[1]}
  Features (selected): {len(selected_features)}

Ridge Baseline:
  Mean Sharpe: {ridge_mean_sharpe:.4f}
  Decision: {'✓ Proceed' if ridge_mean_sharpe >= 0.2 else '⚠️ Review features'}

LightGBM:
  Mean Sharpe: {lgb_mean_sharpe:.4f}
  Improvement: {(lgb_mean_sharpe - ridge_mean_sharpe):.4f}
  Best iteration: {int(np.mean([s['best_iter'] for s in lgb_scores]))}

Files saved:
  - preprocessor.pkl
  - selected_features.pkl
  - lgb_model.pkl
  - feature_importance.csv

Next steps:
  1. Train volatility model (separate target)
  2. Implement position mapping (Phase 3)
  3. Backtest with official scorer (Phase 4)
""")

# Detailed fold results
print("\n" + "="*80)
print("DETAILED FOLD RESULTS")
print("="*80)
print("\nRidge:")
print(pd.DataFrame(ridge_scores).to_string(index=False))
print("\nLightGBM:")
print(pd.DataFrame(lgb_scores).to_string(index=False))

print("\n✅ PHASE 2 MODEL TRAINING COMPLETE")
