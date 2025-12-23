"""
Analysis: Should we drop features with >30% missing values?
"""
import pandas as pd
import joblib

print("="*80)
print("MISSING VALUE THRESHOLD ANALYSIS")
print("="*80)

# Load original data to check missing percentages
train = pd.read_csv('train.csv')

# Calculate missing percentages
missing_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)

# Load feature importance from trained model
try:
    feature_importance = pd.read_csv('feature_importance.csv')
    has_importance = True
except:
    has_importance = False
    print("Note: feature_importance.csv not found, analysis will be limited")

# Categorize features by missing percentage
print("\n" + "="*80)
print("FEATURES BY MISSING PERCENTAGE")
print("="*80)

categories = {
    '>60% (already dropped)': missing_pct[missing_pct > 60],
    '30-60% (under consideration)': missing_pct[(missing_pct >= 30) & (missing_pct <= 60)],
    '10-30% (currently kept)': missing_pct[(missing_pct >= 10) & (missing_pct < 30)],
    '<10% (minimal missing)': missing_pct[(missing_pct > 0) & (missing_pct < 10)]
}

for category, features in categories.items():
    print(f"\n{category}: {len(features)} features")
    if len(features) > 0 and len(features) <= 15:
        for feat, pct in features.items():
            print(f"  {feat}: {pct:.2f}%")
    elif len(features) > 15:
        print(f"  (showing top 10)")
        for feat, pct in features.head(10).items():
            print(f"  {feat}: {pct:.2f}%")

# Analyze 30-60% missing features
print("\n" + "="*80)
print("DETAILED ANALYSIS: 30-60% Missing Features")
print("="*80)

target_features = missing_pct[(missing_pct >= 30) & (missing_pct <= 60)]
print(f"\nFeatures to analyze: {len(target_features)}")

if has_importance and len(target_features) > 0:
    print("\nChecking if these features made it to final model...")
    
    results = []
    for feat in target_features.index:
        # Check if in selected features
        in_model = feat in feature_importance['feature'].values
        
        if in_model:
            imp_row = feature_importance[feature_importance['feature'] == feat]
            importance = imp_row['importance'].values[0]
            rank = feature_importance[feature_importance['feature'] == feat].index[0] + 1
            
            results.append({
                'Feature': feat,
                'Missing %': target_features[feat],
                'In Model': 'Yes',
                'Importance': importance,
                'Rank': f"{rank}/{len(feature_importance)}"
            })
        else:
            results.append({
                'Feature': feat,
                'Missing %': target_features[feat],
                'In Model': 'No',
                'Importance': 0,
                'Rank': '-'
            })
    
    results_df = pd.DataFrame(results).sort_values('Importance', ascending=False)
    print("\n" + results_df.to_string(index=False))
    
    # Summary
    in_model_count = results_df['In Model'].value_counts().get('Yes', 0)
    print(f"\nSummary:")
    print(f"  Features with 30-60% missing: {len(target_features)}")
    print(f"  Made it to final model: {in_model_count}/{len(target_features)}")
    print(f"  Filtered out: {len(target_features) - in_model_count}")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if has_importance and len(target_features) > 0:
    important_ones = results_df[results_df['In Model'] == 'Yes']
    
    if len(important_ones) > 0:
        top_important = important_ones.head(3)
        print(f"\n✓ {len(important_ones)} features with 30-60% missing ARE in the final model")
        print(f"\nTop performers:")
        for _, row in top_important.iterrows():
            print(f"  • {row['Feature']}: {row['Missing %']:.1f}% missing, Rank {row['Rank']}")
        
        print(f"\n📊 Analysis:")
        print(f"  - Feature selection already filtered out {len(target_features) - in_model_count} of them")
        print(f"  - The {len(important_ones)} that remain have proven predictive value")
        print(f"  - Dropping all >30% would lose these contributors")
        
        print(f"\n✅ RECOMMENDATION: Keep current 60% threshold")
        print(f"  Reason: Feature selection is already doing its job")
        print(f"  • Mutual information filter removed low-value high-missing features")
        print(f"  • Only HIGH-VALUE features with 30-60% missing remain")
        print(f"  • Imputation (forward-fill) handles the missing data well")
    else:
        print(f"\n✓ None of the 30-60% missing features made it to final model")
        print(f"\n💡 RECOMMENDATION: Could drop >30% missing")
        print(f"  Reason: Feature selection already excluded them anyway")
        print(f"  Benefit: Slightly faster preprocessing, no performance loss")
else:
    print("\nWithout feature importance data, general recommendation:")
    print("  • >60% missing: Always drop (too sparse)")
    print("  • 30-60% missing: KEEP if feature selection includes them")
    print("  • <30% missing: Keep (manageable with imputation)")

print("\n" + "="*80)
print("TRADE-OFFS")
print("="*80)
print("""
Drop >30% missing:
  ✓ Pros: Cleaner dataset, faster preprocessing
  ✗ Cons: May lose valuable features, need to retrain

Keep current (>60%):
  ✓ Pros: Let feature selection decide, maximum flexibility
  ✗ Cons: More features to process initially

BEST PRACTICE:
  → Use feature selection (mutual info, variance) instead of hard threshold
  → Already implemented in current pipeline
  → Let the MODEL decide what's valuable, not arbitrary % cutoff
""")
