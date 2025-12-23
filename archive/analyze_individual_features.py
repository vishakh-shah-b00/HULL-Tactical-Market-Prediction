import pandas as pd
import numpy as np

def analyze_individual_features(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    target_col = 'market_forward_excess_returns'
    exclude_cols = ['date_id', target_col]
    
    results = []
    
    print("Analyzing features...")
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        series = df[col]
        
        # Basic Stats
        missing_pct = series.isnull().mean() * 100
        
        # Drop NaNs for stats
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            stats = {
                'Feature': col,
                'Missing %': missing_pct,
                'Mean': np.nan, 'Std': np.nan, 'Min': np.nan, 'Max': np.nan, 'Median': np.nan,
                'Skew': np.nan, 'Kurtosis': np.nan,
                'Corr with Target': np.nan,
                'Autocorr (Lag 1)': np.nan
            }
        else:
            stats = {
                'Feature': col,
                'Missing %': missing_pct,
                'Mean': clean_series.mean(),
                'Std': clean_series.std(),
                'Min': clean_series.min(),
                'Max': clean_series.max(),
                'Median': clean_series.median(),
                'Skew': clean_series.skew(),
                'Kurtosis': clean_series.kurtosis(),
                'Corr with Target': series.corr(df[target_col]),
                'Autocorr (Lag 1)': series.autocorr(lag=1)
            }
        
        results.append(stats)
    
    results_df = pd.DataFrame(results)
    
    # Sort by absolute correlation
    results_df['Abs Corr'] = results_df['Corr with Target'].abs()
    results_df = results_df.sort_values('Abs Corr', ascending=False).drop(columns=['Abs Corr'])
    
    output_file = 'feature_summary.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved feature summary to '{output_file}'")
    
    print("\nTop 10 Features by Correlation with Target:")
    print(results_df[['Feature', 'Corr with Target', 'Missing %']].head(10))

if __name__ == "__main__":
    analyze_individual_features('train.csv')
