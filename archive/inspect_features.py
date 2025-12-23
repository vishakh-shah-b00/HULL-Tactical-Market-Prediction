import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def inspect_features(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    target_col = 'market_forward_excess_returns'
    
    # Define feature groups
    feature_groups = {
        'Market (M)': [c for c in df.columns if c.startswith('M')],
        'Macro (E)': [c for c in df.columns if c.startswith('E')],
        'Interest Rate (I)': [c for c in df.columns if c.startswith('I')],
        'Price (P)': [c for c in df.columns if c.startswith('P')],
        'Volatility (V)': [c for c in df.columns if c.startswith('V')],
        'Sentiment (S)': [c for c in df.columns if c.startswith('S')],
        'Dummy (D)': [c for c in df.columns if c.startswith('D')]
    }
    
    print("\nFeature Inspection Summary:")
    
    for group_name, features in feature_groups.items():
        print(f"\n--- {group_name} ---")
        if not features:
            print("No features found.")
            continue
            
        group_df = df[features + [target_col]]
        
        # Missing Values
        missing = group_df[features].isnull().mean() * 100
        print(f"Average Missing Values: {missing.mean():.2f}%")
        print(f"Features with > 50% missing: {list(missing[missing > 50].index)}")
        
        # Correlations
        corr = group_df.corr()[target_col].drop(target_col)
        print(f"Top 3 Positive Correlations:\n{corr.nlargest(3)}")
        print(f"Top 3 Negative Correlations:\n{corr.nsmallest(3)}")
        
        # Visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(group_df.corr(), annot=False, cmap='coolwarm', center=0)
        plt.title(f'Correlation Heatmap - {group_name}')
        plt.tight_layout()
        safe_name = group_name.split(' ')[0].lower()
        plt.savefig(f'corr_{safe_name}.png')
        print(f"Saved correlation heatmap to 'corr_{safe_name}.png'")

if __name__ == "__main__":
    inspect_features('train.csv')
