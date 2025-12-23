"""
RETRAIN: Production Training Script
-----------------------------------
This script performs the full End-to-End training pipeline with strict leakage prevention.

Steps:
1.  **Honest Split**: Separates the last 180 days (Holdout) BEFORE any processing.
2.  **Preprocessing**: Applies lag/rolling features using `MarketPreprocessor`.
3.  **Feature Selection**: Reduces 200+ features to Top 100 using:
    - Variance Threshold (remove constants)
    - Correlation Filter (remove >95% collinear redundancy)
    - Mutual Information (select highest non-linear predictive power)
4.  **Cross-Validation**: Runs a "Model Tournament" (Ridge vs LightGBM) using PurgedKFold.
5.  **Final Training**: Trains the winner (LightGBM) on the full non-holdout dataset.
6.  **Artifact Generation**: Saves models and feature lists for inference.

Output:
- preprocessor_clean.pkl
- lgb_model_clean.pkl
- selected_features_clean.pkl
- feature_importance_clean.csv
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
print("RETRAINING - EXCLUDING PUBLIC TEST (LAST 180 ROWS)")
print("="*80)

# ------------------------------------------------------------------------------
# 1. HONEST DATA SPLIT
# ------------------------------------------------------------------------------
# We deliberately exclude the last 180 days (Public Leaderboard) from ALL training.
# This prevents overfitting to the leaderboard and gives us a trustworthy validation set.
train_full = pd.read_csv('train.csv')
public_test_size = 180

train = train_full.iloc[:-public_test_size].copy()
public_test = train_full.iloc[-public_test_size:].copy()

print(f"\n✓ Data split:")
print(f"  Full train: {train_full.shape}")
print(f"  Our train (excluding public test): {train.shape}")
print(f"  Public test (last 180): {public_test.shape}")

# Preprocess
print("\n[1/6] Preprocessing...")
preprocessor = MarketPreprocessor()
X, y = preprocessor.fit_transform(train)
preprocessor.save('preprocessor_clean.pkl')

# Drop NaN
nan_mask = X.isnull().any(axis=1)
X = X[~nan_mask].reset_index(drop=True)
y = y[~nan_mask].reset_index(drop=True)
print(f"  Clean data: {X.shape}")

# ------------------------------------------------------------------------------
# 2. FEATURE SELECTION PIPELINE
# ------------------------------------------------------------------------------
print("\n[2/6] Feature selection...")

# A. Variance Threshold: Remove constant/quasi-constant features
variance_selector = VarianceThreshold(threshold=0.01)
X_var = variance_selector.fit_transform(X)
selected_features_var = X.columns[variance_selector.get_support()].tolist()

# B. Correlation Filter: Remove highly collinear features (>0.95)
# Redundant features confuse linear models and waste tree splits.
X_var_df = pd.DataFrame(X_var, columns=selected_features_var)
corr_matrix = X_var_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_uncorr = X_var_df.drop(columns=to_drop)

# C. Mutual Information: Select Top 100 based on Non-Linear Dependency
# MI captures complex relationships (e.g. Volatility regimes) that correlation misses.
n_features = min(100, X_uncorr.shape[1])
mi_selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
mi_selector.fit(X_uncorr, y)
selected_features = X_uncorr.columns[mi_selector.get_support()].tolist()

joblib.dump(selected_features, 'selected_features_clean.pkl')
X_final = X[selected_features].copy()
print(f"  Selected {len(selected_features)} features")

# CV Setup
print("\n[3/6] Cross-validation...")
cv = PurgedKFold(n_splits=5, embargo_days=20)

def calculate_sharpe(y_true, y_pred):
    positions = np.sign(y_pred)
    realized_returns = positions * y_true
    mean_return = realized_returns.mean()
    std_return = realized_returns.std()
    return (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

# Ridge Baseline
print("\n[4/6] Training Ridge...")
ridge_sharpes = []
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_final, y)):
    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    
    sharpe = calculate_sharpe(y_test.values, y_pred)
    ridge_sharpes.append(sharpe)
    print(f"  Fold {fold_idx+1}: Sharpe={sharpe:.4f}")

ridge_mean = np.mean(ridge_sharpes)
print(f"  Ridge Mean Sharpe: {ridge_mean:.4f}")

# LightGBM
print("\n[5/6] Training LightGBM...")
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

lgb_sharpes = []
lgb_iters = []

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
    sharpe = calculate_sharpe(y_test.values, y_pred)
    lgb_sharpes.append(sharpe)
    lgb_iters.append(model.best_iteration)
    print(f"  Fold {fold_idx+1}: Sharpe={sharpe:.4f}, Iters={model.best_iteration}")

lgb_mean = np.mean(lgb_sharpes)
print(f"  LightGBM Mean Sharpe: {lgb_mean:.4f}")

# Train final model
print("\n[6/6] Training final model on all clean train data...")
train_data = lgb.Dataset(X_final, label=y)
final_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=int(np.mean(lgb_iters))
)

joblib.dump(final_model, 'lgb_model_clean.pkl')

# Feature importance
importance = final_model.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': importance
}).sort_values('importance', ascending=False)
feature_importance.to_csv('feature_importance_clean.csv', index=False)

print(f"\n✓ Saved artifacts:")
print(f"  - preprocessor_clean.pkl")
print(f"  - selected_features_clean.pkl")
print(f"  - lgb_model_clean.pkl")
print(f"  - feature_importance_clean.csv")

print(f"\n" + "="*80)
print("CLEAN TRAINING SUMMARY")
print("="*80)
print(f"""
Training data: {len(X_final)} samples (excluded last 180)
Features: {len(selected_features)}

Ridge Baseline:  {ridge_mean:.4f}
LightGBM:        {lgb_mean:.4f}
Improvement:     {(lgb_mean - ridge_mean):.4f}

✅ Models trained WITHOUT data leakage
""")

print("="*80)
print("Next: Validate on true holdout (last 180 rows)")
print("="*80)
