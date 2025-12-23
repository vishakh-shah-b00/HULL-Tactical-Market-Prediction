import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_time_series(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("\nData Structure:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Target variable analysis
    target_col = 'market_forward_excess_returns'
    time_col = 'date_id'
    
    if target_col in df.columns and time_col in df.columns:
        print(f"\nAnalyzing {target_col} over {time_col}...")
        
        # Basic Statistics
        print("\nTarget Variable Statistics:")
        print(df[target_col].describe())
        
        # Plotting
        plt.figure(figsize=(15, 6))
        plt.plot(df[time_col], df[target_col], label=target_col, alpha=0.7)
        plt.title(f'{target_col} over Time')
        plt.xlabel('Date ID')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        plt.savefig('time_series_plot.png')
        print("\nSaved time series plot to 'time_series_plot.png'")
        
        # Rolling Statistics
        window = 30
        rolling_mean = df[target_col].rolling(window=window).mean()
        rolling_std = df[target_col].rolling(window=window).std()
        
        plt.figure(figsize=(15, 6))
        plt.plot(df[time_col], df[target_col], label='Original', alpha=0.3)
        plt.plot(df[time_col], rolling_mean, label=f'Rolling Mean (w={window})', color='red')
        plt.plot(df[time_col], rolling_std, label=f'Rolling Std (w={window})', color='black')
        plt.title(f'Rolling Mean & Standard Deviation (Window={window})')
        plt.legend()
        plt.grid(True)
        plt.savefig('rolling_stats.png')
        print("Saved rolling stats plot to 'rolling_stats.png'")

        # Stationarity Test (ADF)
        from statsmodels.tsa.stattools import adfuller
        print("\nPerforming Augmented Dickey-Fuller Test...")
        result = adfuller(df[target_col].dropna())
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        
        if result[1] < 0.05:
            print("Result: The series is likely STATIONARY (p-value < 0.05).")
        else:
            print("Result: The series is likely NON-STATIONARY (p-value >= 0.05).")

    else:
        print(f"Columns {target_col} or {time_col} not found.")

if __name__ == "__main__":
    analyze_time_series('train.csv')
