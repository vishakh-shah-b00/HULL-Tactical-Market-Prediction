"""
Phase 1 Step 2: D6 Analysis and Feature Engineering Pipeline
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 1 STEP 2: D6 ANALYSIS & FEATURE ENGINEERING")
print("="*80)

# Load data
print("\n1. Loading data...")
train = pd.read_csv('train.csv')
print(f"✓ Train shape: {train.shape}")

# D6 Analysis
print("\n" + "="*80)
print("2. D6 ENCODING ANALYSIS")
print("="*80)

print("\nValue distribution:")
print(train['D6'].value_counts().sort_index())

print("\nPercentages:")
print(train['D6'].value_counts(normalize=True).sort_index() * 100)

print("\nCorrelation with target by D6 value:")
target = 'market_forward_excess_returns'
d6_target_corr = train.groupby('D6')[target].agg(['mean', 'std', 'count'])
print(d6_target_corr)

print("\nAbsolute mean return by D6:")
abs_means = train.groupby('D6')[target].mean().abs()
print(abs_means)

# Determine encoding
if abs_means[0] > abs_means[-1]:
    print("\n✓ Decision: 0 has higher absolute predictive power")
    print("  Encoding: -1 → 0, 0 → 1")
    encoding_map = {-1: 0, 0: 1}
else:
    print("\n✓ Decision: -1 has higher absolute predictive power")
    print("  Encoding: -1 → 1, 0 → 0")
    encoding_map = {-1: 1, 0: 0}

print(f"\nEncoding map: {encoding_map}")

# Apply transformation to test
print("\n" + "="*80)
print("3. BUILDING FEATURE ENGINEERING PIPELINE")
print("="*80)

# Features to drop (>60% missing + leakage)
DROP_FEATURES = ['E7', 'V10', 'S3', 'M1', 'M13', 'M14', 'forward_returns', 'risk_free_rate']
print(f"\n✓ Features to drop ({len(DROP_FEATURES)}): {DROP_FEATURES}")

# Top features for lag creation
TOP_FEATURES_FOR_LAGS = ['M4', 'V13', 'S5', 'S2', 'M2', 'D1', 'D2', 'M17', 'M12', 'E19']
print(f"\n✓ Top features for lags ({len(TOP_FEATURES_FOR_LAGS)}): {TOP_FEATURES_FOR_LAGS}")

# Features for rolling windows
FEATURES_FOR_ROLLING = ['M4', 'V13', 'S5', 'S2', 'M2', target]
print(f"\n✓ Features for rolling windows ({len(FEATURES_FOR_ROLLING)-1}): {FEATURES_FOR_ROLLING[:-1]}")

print("\n" + "="*80)
print("4. APPLYING TRANSFORMATIONS")
print("="*80)

# Step 1: Drop features
print("\n[1/6] Dropping features...")
train_clean = train.drop(columns=DROP_FEATURES, errors='ignore')
print(f"  Shape after drop: {train_clean.shape}")

# Step 2: Transform D6
print("\n[2/6] Transforming D6 encoding...")
train_clean['D6'] = train_clean['D6'].map(encoding_map)
print(f"  D6 unique values after transform: {sorted(train_clean['D6'].unique())}")

# Step 3: Create imputation indicators
print("\n[3/6] Creating imputation indicators...")
imputation_indicators = {}
for col in train_clean.columns:
    if train_clean[col].isnull().any() and col not in ['date_id', target]:
        indicator_name = f'{col}_missing'
        imputation_indicators[indicator_name] = train_clean[col].isnull().astype(int)
        
print(f"  Created {len(imputation_indicators)} imputation indicators")

# Step 4: Forward-fill missing values
print("\n[4/6] Forward-filling missing values...")
# Get columns to fill (exclude date_id and target)
cols_to_fill = [c for c in train_clean.columns if c not in ['date_id', target]]
train_clean[cols_to_fill] = train_clean[cols_to_fill].fillna(method='ffill')

# Check remaining NaN (only at the start of series)
remaining_nan = train_clean[cols_to_fill].isnull().sum()
print(f"  Remaining NaN values: {remaining_nan.sum()} total")
if remaining_nan.sum() > 0:
    print(f"  Features with NaN: {remaining_nan[remaining_nan > 0].to_dict()}")
    # Fill remaining with median (only at series start)
    train_clean[cols_to_fill] = train_clean[cols_to_fill].fillna(train_clean[cols_to_fill].median())
    print("  ✓ Filled remaining NaN with median")

# Step 5: Create lag features
print("\n[5/6] Creating lag features...")
lag_features = {}
lag_windows = {
    'M': [1, 2, 5],       # Market
    'V': [1, 5, 10],      # Volatility
    'E': [5, 10, 20],     # Macro (slow)
    'S': [1, 5],          # Sentiment
    'I': [5, 10],         # Interest
    'P': [5, 10],         # Price
    'D': [1]              # Dummy
}

lag_count = 0
for feature in TOP_FEATURES_FOR_LAGS:
    if feature not in train_clean.columns:
        continue
    
    # Determine lag windows based on prefix
    prefix = feature[0]
    lags = lag_windows.get(prefix, [1, 5])
    
    for lag in lags:
        lag_name = f'{feature}_lag{lag}'
        lag_features[lag_name] = train_clean[feature].shift(lag)
        lag_count += 1

print(f"  Created {lag_count} lag features")

# Step 6: Create rolling window features
print("\n[6/6] Creating rolling window features...")
rolling_features = {}
windows = [5, 20, 60]

for feature in FEATURES_FOR_ROLLING:
    if feature not in train_clean.columns:
        continue
    
    for window in windows:
        # Rolling mean
        roll_mean_name = f'{feature}_roll{window}_mean'
        rolling_features[roll_mean_name] = train_clean[feature].rolling(window=window).mean()
        
        # Rolling std (volatility proxy)
        roll_std_name = f'{feature}_roll{window}_std'
        rolling_features[roll_std_name] = train_clean[feature].rolling(window=window).std()

print(f"  Created {len(rolling_features)} rolling features")

# Combine all features
print("\n" + "="*80)
print("5. COMBINING ALL FEATURES")
print("="*80)

# Add imputation indicators
for name, series in imputation_indicators.items():
    train_clean[name] = series

# Add lag features
for name, series in lag_features.items():
    train_clean[name] = series

# Add rolling features
for name, series in rolling_features.items():
    train_clean[name] = series

print(f"\nFinal feature count: {train_clean.shape[1]}")
print(f"  Original features: {train.shape[1] - len(DROP_FEATURES)}")
print(f"  Imputation indicators: {len(imputation_indicators)}")
print(f"  Lag features: {len(lag_features)}")
print(f"  Rolling features: {len(rolling_features)}")

# Check for NaN
nan_count = train_clean.isnull().sum().sum()
print(f"\nTotal NaN values: {nan_count}")
if nan_count > 0:
    print("  Note: NaN values in lag/rolling features are expected at start of series")

# Save engineered features
print("\n" + "="*80)
print("6. SAVING ENGINEERED FEATURES")
print("="*80)

output_file = 'train_engineered.csv'
train_clean.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Summary statistics
print("\n" + "="*80)
print("7. FEATURE ENGINEERING SUMMARY")
print("="*80)

print(f"""
Starting features:     {train.shape[1]}
Dropped features:      {len(DROP_FEATURES)}
Base features:         {train.shape[1] - len(DROP_FEATURES)}
Added features:        {len(imputation_indicators) + len(lag_features) + len(rolling_features)}
Final features:        {train_clean.shape[1]}

Feature breakdown:
  - Base (after drop):     {train.shape[1] - len(DROP_FEATURES)}
  - Imputation indicators: {len(imputation_indicators)}
  - Lag features:          {len(lag_features)}
  - Rolling windows:       {len(rolling_features)}
  - Total:                 {train_clean.shape[1]}

D6 encoding:             {encoding_map}
""")

print("\n✅ PHASE 1 COMPLETE - Feature engineering pipeline built successfully!")
print("\nNext steps:")
print("  1. Validate no data leakage")
print("  2. Check feature correlations")
print("  3. Proceed to Phase 2: Model training")
