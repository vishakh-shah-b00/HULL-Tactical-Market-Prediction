"""
Phase 1: Data Inspection & Validation
Step 1: Inspect datatypes and validate hypotheses before feature engineering
"""
import pandas as pd
import numpy as np

def inspect_data_structure():
    """Comprehensive data structure inspection"""
    print("=" * 80)
    print("PHASE 1: DATA INSPECTION & VALIDATION")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print(f"✓ Train shape: {train.shape}")
    print(f"✓ Test shape: {test.shape}")
    
    # Datatype inspection
    print("\n" + "=" * 80)
    print("2. DATATYPE INSPECTION")
    print("=" * 80)
    
    print("\nDatatype breakdown:")
    dtype_counts = train.dtypes.value_counts()
    print(dtype_counts)
    
    print("\nColumn datatypes by group:")
    for prefix in ['date_id', 'D', 'M', 'E', 'I', 'P', 'V', 'S', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']:
        if prefix in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']:
            cols = [prefix] if prefix in train.columns else []
        else:
            cols = [c for c in train.columns if c.startswith(prefix)]
        
        if cols:
            dtypes = train[cols].dtypes.value_counts()
            print(f"\n{prefix}*: {dtypes.to_dict()}")
    
    # Check for unexpected types
    print("\n" + "=" * 80)
    print("3. UNEXPECTED DATATYPE CHECK")
    print("=" * 80)
    
    # Expected: int64 for date_id and D*, float64 for all others
    int_cols = train.select_dtypes(include=['int64', 'int32']).columns.tolist()
    float_cols = train.select_dtypes(include=['float64', 'float32']).columns.tolist()
    other_cols = [c for c in train.columns if c not in int_cols and c not in float_cols]
    
    print(f"\nInteger columns ({len(int_cols)}): {int_cols}")
    print(f"Float columns ({len(float_cols)}): {len(float_cols)} total")
    print(f"Other types ({len(other_cols)}): {other_cols}")
    
    # Validate date_id
    print("\n" + "=" * 80)
    print("4. DATE_ID VALIDATION")
    print("=" * 80)
    
    print(f"date_id dtype: {train['date_id'].dtype}")
    print(f"date_id range: [{train['date_id'].min()}, {train['date_id'].max()}]")
    print(f"date_id unique: {train['date_id'].nunique()} / {len(train)}")
    print(f"date_id sorted: {train['date_id'].is_monotonic_increasing}")
    
    # Check for gaps
    gaps = train['date_id'].diff().value_counts().sort_index()
    print(f"\ndate_id gaps:")
    print(gaps.head(10))
    
    # Missing value validation
    print("\n" + "=" * 80)
    print("5. MISSING VALUE VALIDATION (from Phase 0)")
    print("=" * 80)
    
    missing_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
    missing_features = missing_pct[missing_pct > 0]
    
    print(f"\nTotal features with missing data: {len(missing_features)}")
    print(f"\nFeatures with >60% missing (hypothesis: should drop):")
    drop_candidates = missing_features[missing_features > 60]
    for feat, pct in drop_candidates.items():
        print(f"  {feat}: {pct:.2f}%")
    
    print(f"\nFeatures with 50-60% missing (borderline):")
    borderline = missing_features[(missing_features >= 50) & (missing_features <= 60)]
    for feat, pct in borderline.items():
        print(f"  {feat}: {pct:.2f}%")
    
    # Target variable validation
    print("\n" + "=" * 80)
    print("6. TARGET VARIABLE VALIDATION")
    print("=" * 80)
    
    target = 'market_forward_excess_returns'
    print(f"\nTarget: {target}")
    print(f"  Dtype: {train[target].dtype}")
    print(f"  Missing: {train[target].isnull().sum()}")
    print(f"  Mean: {train[target].mean():.8f}")
    print(f"  Std: {train[target].std():.8f}")
    print(f"  Min: {train[target].min():.6f}")
    print(f"  Max: {train[target].max():.6f}")
    
    # Validate relationship with forward_returns
    if 'forward_returns' in train.columns:
        print(f"\nforward_returns validation:")
        print(f"  Dtype: {train['forward_returns'].dtype}")
        print(f"  Missing: {train['forward_returns'].isnull().sum()}")
        print(f"  Correlation with target: {train['forward_returns'].corr(train[target]):.6f}")
    
    # Feature group validation
    print("\n" + "=" * 80)
    print("7. FEATURE GROUP VALIDATION")
    print("=" * 80)
    
    groups = {
        'D (Dummy)': [c for c in train.columns if c.startswith('D')],
        'M (Market)': [c for c in train.columns if c.startswith('M')],
        'E (Macro)': [c for c in train.columns if c.startswith('E')],
        'I (Interest)': [c for c in train.columns if c.startswith('I')],
        'P (Price)': [c for c in train.columns if c.startswith('P')],
        'V (Volatility)': [c for c in train.columns if c.startswith('V')],
        'S (Sentiment)': [c for c in train.columns if c.startswith('S')],
    }
    
    for name, cols in groups.items():
        if cols:
            missing_avg = train[cols].isnull().mean().mean() * 100
            print(f"\n{name}:")
            print(f"  Count: {len(cols)}")
            print(f"  Avg missing: {missing_avg:.2f}%")
            print(f"  Dtypes: {train[cols].dtypes.value_counts().to_dict()}")
            print(f"  Features: {cols}")
    
    # Value range validation (check for already-scaled features)
    print("\n" + "=" * 80)
    print("8. VALUE RANGE VALIDATION (Check if pre-scaled)")
    print("=" * 80)
    
    print("\nSample feature ranges (first 10 numeric):")
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('date_id')
    
    for col in numeric_cols[:10]:
        min_val = train[col].min()
        max_val = train[col].max()
        mean_val = train[col].mean()
        print(f"  {col:30s}: [{min_val:10.6f}, {max_val:10.6f}], mean={mean_val:10.6f}")
    
    # Check for binary features (D*)
    print("\n" + "=" * 80)
    print("9. BINARY FEATURE VALIDATION (D* features)")
    print("=" * 80)
    
    d_cols = [c for c in train.columns if c.startswith('D')]
    print(f"\nD* features: {len(d_cols)}")
    for col in d_cols:
        unique_vals = train[col].unique()
        print(f"  {col}: unique values = {sorted(unique_vals)}, count = {len(unique_vals)}")
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("10. SUMMARY & HYPOTHESES VALIDATION")
    print("=" * 80)
    
    print("\n✓ Hypothesis 1: date_id is sequential integer")
    print(f"  Status: {'CONFIRMED' if train['date_id'].dtype == 'int64' and train['date_id'].is_monotonic_increasing else 'FAILED'}")
    
    print("\n✓ Hypothesis 2: D* features are binary (0/1)")
    d_is_binary = all(set(train[col].dropna().unique()).issubset({0, 1}) for col in d_cols)
    print(f"  Status: {'CONFIRMED' if d_is_binary else 'FAILED'}")
    
    print("\n✓ Hypothesis 3: 7 features have >60% missing")
    print(f"  Status: {'CONFIRMED' if len(drop_candidates) >= 7 else 'FAILED'}")
    print(f"  Actual count: {len(drop_candidates)}")
    
    print("\n✓ Hypothesis 4: Target has no missing values")
    print(f"  Status: {'CONFIRMED' if train[target].isnull().sum() == 0 else 'FAILED'}")
    
    print("\n✓ Hypothesis 5: Most features are float64")
    print(f"  Status: {'CONFIRMED' if len(float_cols) > len(int_cols) else 'FAILED'}")
    print(f"  Float: {len(float_cols)}, Int: {len(int_cols)}")
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE - Ready for feature engineering decisions")
    print("=" * 80)
    
    return train, test, drop_candidates.index.tolist()

if __name__ == "__main__":
    train, test, drop_list = inspect_data_structure()
    
    print(f"\n\n📋 RECOMMENDED ACTIONS:")
    print(f"1. DROP {len(drop_list)} features: {drop_list}")
    print(f"2. Forward-fill remaining missing values (justified by high autocorr)")
    print(f"3. Keep D* features as-is (binary, no missing)")
    print(f"4. Validate test.csv has same structure")
