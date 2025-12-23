"""
Phase 0: Comprehensive Data Exploration
Generates visualizations and statistical analysis for the competition dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

def load_data():
    """Load train and test data"""
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test

def plot_target_analysis(df, save_dir='exploration_plots'):
    """Analyze target variable distribution and properties"""
    print("\n=== TARGET VARIABLE ANALYSIS ===")
    
    target = 'market_forward_excess_returns'
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribution
    axes[0, 0].hist(df[target], bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df[target].mean(), color='red', linestyle='--', label=f'Mean: {df[target].mean():.6f}')
    axes[0, 0].axvline(df[target].median(), color='green', linestyle='--', label=f'Median: {df[target].median():.6f}')
    axes[0, 0].set_xlabel('Market Forward Excess Returns')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Target Distribution')
    axes[0, 0].legend()
    
    # 2. Q-Q plot (normality check)
    stats.probplot(df[target], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality Check)')
    
    # 3. Time series
    axes[1, 0].plot(df['date_id'], df[target], alpha=0.6, linewidth=0.5)
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Date ID')
    axes[1, 0].set_ylabel('Returns')
    axes[1, 0].set_title('Target Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot (outlier detection)
    axes[1, 1].boxplot(df[target], vert=True)
    axes[1, 1].set_ylabel('Returns')
    axes[1, 1].set_title('Target Box Plot (Outliers)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_target_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/01_target_analysis.png")
    
    # Statistics
    print(f"\nTarget Statistics:")
    print(f"  Mean: {df[target].mean():.6f}")
    print(f"  Std: {df[target].std():.6f}")
    print(f"  Skew: {df[target].skew():.4f}")
    print(f"  Kurtosis: {df[target].kurtosis():.4f}")
    print(f"  Min: {df[target].min():.6f}")
    print(f"  Max: {df[target].max():.6f}")

def plot_autocorrelation(df, save_dir='exploration_plots'):
    """Plot autocorrelation and partial autocorrelation"""
    print("\n=== AUTOCORRELATION ANALYSIS ===")
    
    target = 'market_forward_excess_returns'
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # ACF
    plot_acf(df[target].dropna(), lags=50, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].set_xlabel('Lags')
    
    # PACF
    plot_pacf(df[target].dropna(), lags=50, ax=axes[1], alpha=0.05)
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lags')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_autocorrelation.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/02_autocorrelation.png")

def plot_volatility_clustering(df, save_dir='exploration_plots'):
    """Visualize volatility clustering and regime changes"""
    print("\n=== VOLATILITY CLUSTERING ===")
    
    target = 'market_forward_excess_returns'
    
    # Calculate rolling volatility
    df['rolling_vol_20'] = df[target].rolling(window=20).std()
    df['rolling_vol_60'] = df[target].rolling(window=60).std()
    df['abs_returns'] = df[target].abs()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Returns with rolling volatility
    axes[0].plot(df['date_id'], df[target], alpha=0.4, linewidth=0.5, label='Returns')
    axes[0].fill_between(df['date_id'], -df['rolling_vol_20'], df['rolling_vol_20'], 
                         alpha=0.2, color='red', label='±1 Std (20-day)')
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Returns')
    axes[0].set_title('Returns with Rolling Volatility Bands')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Rolling volatility time series
    axes[1].plot(df['date_id'], df['rolling_vol_20'], label='20-day Vol', alpha=0.7)
    axes[1].plot(df['date_id'], df['rolling_vol_60'], label='60-day Vol', alpha=0.7)
    axes[1].set_ylabel('Volatility (Std)')
    axes[1].set_title('Rolling Volatility Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Absolute returns (volatility proxy)
    axes[2].bar(df['date_id'], df['abs_returns'], alpha=0.5, width=1)
    axes[2].plot(df['date_id'], df['rolling_vol_20'], color='red', linewidth=2, label='20-day MA')
    axes[2].set_xlabel('Date ID')
    axes[2].set_ylabel('Absolute Returns')
    axes[2].set_title('Absolute Returns (Volatility Clustering)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_volatility_clustering.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/03_volatility_clustering.png")

def plot_missing_data(df, save_dir='exploration_plots'):
    """Visualize missing data patterns"""
    print("\n=== MISSING DATA PATTERNS ===")
    
    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Bar chart of missing percentages
    axes[0].barh(range(len(missing_pct)), missing_pct.values, alpha=0.7)
    axes[0].set_yticks(range(len(missing_pct)))
    axes[0].set_yticklabels(missing_pct.index, fontsize=8)
    axes[0].axvline(50, color='red', linestyle='--', label='50% threshold')
    axes[0].set_xlabel('Missing %')
    axes[0].set_title('Missing Data by Feature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 2. Heatmap of missing data over time (sample)
    sample_cols = missing_pct.head(20).index.tolist()
    if sample_cols:
        missing_matrix = df[sample_cols].isnull().astype(int)
        # Sample every 50th row for visibility
        sample_matrix = missing_matrix.iloc[::50, :]
        sns.heatmap(sample_matrix.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing'}, 
                    ax=axes[1], xticklabels=False)
        axes[1].set_ylabel('Features')
        axes[1].set_xlabel('Time (sampled)')
        axes[1].set_title('Missing Data Pattern Over Time (Top 20 features)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/04_missing_data.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/04_missing_data.png")
    
    print(f"\nTop 10 features with most missing data:")
    print(missing_pct.head(10))

def plot_feature_correlations(df, save_dir='exploration_plots'):
    """Plot correlation heatmaps by feature group"""
    print("\n=== FEATURE CORRELATIONS ===")
    
    target = 'market_forward_excess_returns'
    
    # Define feature groups
    groups = {
        'Market (M)': [c for c in df.columns if c.startswith('M')],
        'Volatility (V)': [c for c in df.columns if c.startswith('V')],
        'Sentiment (S)': [c for c in df.columns if c.startswith('S')],
        'Macro (E)': [c for c in df.columns if c.startswith('E')],
    }
    
    for group_name, features in groups.items():
        if not features or len(features) < 2:
            continue
        
        # Get correlations with target
        corrs = df[features + [target]].corr()[target].drop(target).sort_values(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Correlation with target
        axes[0].barh(range(len(corrs)), corrs.values, alpha=0.7)
        axes[0].set_yticks(range(len(corrs)))
        axes[0].set_yticklabels(corrs.index, fontsize=8)
        axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Correlation with Target')
        axes[0].set_title(f'{group_name} - Correlation with Target')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # 2. Heatmap (top 15 features)
        top_features = corrs.abs().nlargest(15).index.tolist()
        if len(top_features) > 1:
            corr_matrix = df[top_features + [target]].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                       ax=axes[1], vmin=-0.5, vmax=0.5)
            axes[1].set_title(f'{group_name} - Correlation Matrix (Top 15)')
        
        plt.tight_layout()
        safe_name = group_name.split(' ')[0].lower()
        plt.savefig(f'{save_dir}/05_corr_{safe_name}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_dir}/05_corr_{safe_name}.png")

def plot_top_features(df, save_dir='exploration_plots'):
    """Visualize top predictive features"""
    print("\n=== TOP FEATURES ANALYSIS ===")
    
    target = 'market_forward_excess_returns'
    
    # Get top correlated features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target)
    numeric_cols.remove('date_id')
    
    corrs = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False).head(8)
    top_features = corrs.index.tolist()
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        # Scatter plot with trend line
        valid_data = df[[feature, target]].dropna()
        axes[idx].scatter(valid_data[feature], valid_data[target], alpha=0.1, s=10)
        
        # Add trend line
        z = np.polyfit(valid_data[feature], valid_data[target], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data[feature].min(), valid_data[feature].max(), 100)
        axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel(target)
        axes[idx].set_title(f'{feature} vs Target (corr={corrs[feature]:.4f})')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_top_features_scatter.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/06_top_features_scatter.png")

def plot_feature_distributions(df, save_dir='exploration_plots'):
    """Plot distributions of top features"""
    print("\n=== FEATURE DISTRIBUTIONS ===")
    
    target = 'market_forward_excess_returns'
    
    # Get top features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target)
    numeric_cols.remove('date_id')
    
    corrs = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False).head(6)
    top_features = corrs.index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        data = df[feature].dropna()
        axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
        axes[idx].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.3f}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{feature} Distribution')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/07_feature_distributions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/07_feature_distributions.png")

def main():
    """Run all exploration analyses"""
    import os
    
    # Create output directory
    save_dir = 'exploration_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    train, test = load_data()
    
    # Run all analyses
    plot_target_analysis(train, save_dir)
    plot_autocorrelation(train, save_dir)
    plot_volatility_clustering(train, save_dir)
    plot_missing_data(train, save_dir)
    plot_feature_correlations(train, save_dir)
    plot_top_features(train, save_dir)
    plot_feature_distributions(train, save_dir)
    
    print(f"\n✅ All exploration plots saved to '{save_dir}/' directory")
    print("\nKey Findings Summary:")
    print("1. Check target distribution for normality and outliers")
    print("2. Review autocorrelation plots for lag selection")
    print("3. Identify volatility regimes and clustering patterns")
    print("4. Note missing data patterns (temporal vs feature-specific)")
    print("5. Examine top features and their relationships with target")

if __name__ == "__main__":
    main()
