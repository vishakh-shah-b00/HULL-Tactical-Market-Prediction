"""
Preprocessing Pipeline Class for Hull Tactical Market Prediction
Reusable, production-ready feature engineering pipeline
"""
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Tuple, Optional

class MarketPreprocessor:
    """
    Feature engineering pipeline for market prediction

    Usage:
        # Training
        preprocessor = MarketPreprocessor()
        X_train, y_train = preprocessor.fit_transform(train_df)
        preprocessor.save('preprocessor.pkl')
        
        # Inference
        preprocessor = MarketPreprocessor.load('preprocessor.pkl')
        X_test = preprocessor.transform(test_df)
    """
    
    def __init__(self):
        # Features to drop
        self.drop_features = ['E7', 'V10', 'S3', 'M1', 'M13', 'M14', 
                             'forward_returns', 'risk_free_rate']
        
        # D6 encoding (determined from analysis)
        self.d6_encoding = {-1: 1, 0: 0}
        
        # Top features for lags
        self.top_features_for_lags = ['M4', 'V13', 'S5', 'S2', 'M2', 
                                      'D1', 'D2', 'M17', 'M12', 'E19']
        
        # Features for rolling windows
        self.features_for_rolling = ['M4', 'V13', 'S5', 'S2', 'M2']
        
        # Lag windows by feature type
        self.lag_windows = {
            'M': [1, 2, 5],
            'V': [1, 5, 10],
            'E': [5, 10, 20],
            'S': [1, 5],
            'I': [5, 10],
            'P': [5, 10],
            'D': [1]
        }
        
        # Rolling windows
        self.rolling_windows = [5, 20, 60]
        
        # Fitted parameters (learned during fit)
        self.median_values_ = None
        self.feature_names_ = None
        self.target_col = 'market_forward_excess_returns'
        
    def fit(self, df: pd.DataFrame) -> 'MarketPreprocessor':
        """
        Fit the preprocessor on training data
        
        Args:
            df: Training dataframe with all features and target
            
        Returns:
            self
        """
        print("[Preprocessor] Fitting on training data...")
        
        # Store median values for imputation (only on base features)
        df_clean = df.drop(columns=self.drop_features, errors='ignore')
        cols_to_impute = [c for c in df_clean.columns if c not in ['date_id', self.target_col]]
        self.median_values_ = df_clean[cols_to_impute].median()
        
        print(f"[Preprocessor] Fitted. Stored {len(self.median_values_)} median values")
        return self
    
    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Transform dataframe using fitted parameters
        
        Args:
            df: Input dataframe
            is_training: If True, includes target column
            
        Returns:
            Transformed dataframe
        """
        print(f"[Preprocessor] Transforming {len(df)} rows...")
        df = df.copy()
        
        # Step 1: Drop features
        df = df.drop(columns=self.drop_features, errors='ignore')
        
        # Step 2: Transform D6
        if 'D6' in df.columns:
            df['D6'] = df['D6'].map(self.d6_encoding)
        
        # Step 3: Create imputation indicators
        imputation_indicators = {}
        for col in df.columns:
            if df[col].isnull().any() and col not in ['date_id', self.target_col]:
                indicator_name = f'{col}_missing'
                imputation_indicators[indicator_name] = df[col].isnull().astype(int)
        
        # Step 4: Forward-fill and median imputation
        cols_to_fill = [c for c in df.columns if c not in ['date_id', self.target_col]]
        df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill')
        
        # Fill remaining NaN with fitted medians
        if self.median_values_ is not None:
            for col in cols_to_fill:
                if col in self.median_values_ and df[col].isnull().any():
                    df[col] = df[col].fillna(self.median_values_[col])
        
        # Step 5: Create lag features
        lag_features = {}
        for feature in self.top_features_for_lags:
            if feature not in df.columns:
                continue
            
            prefix = feature[0]
            lags = self.lag_windows.get(prefix, [1, 5])
            
            for lag in lags:
                lag_name = f'{feature}_lag{lag}'
                lag_features[lag_name] = df[feature].shift(lag)
        
        # Step 6: Create rolling window features
        rolling_features = {}
        for feature in self.features_for_rolling:
            if feature not in df.columns:
                continue
            
            for window in self.rolling_windows:
                roll_mean_name = f'{feature}_roll{window}_mean'
                rolling_features[roll_mean_name] = df[feature].rolling(window=window).mean()
                
                roll_std_name = f'{feature}_roll{window}_std'
                rolling_features[roll_std_name] = df[feature].rolling(window=window).std()
        
        # Step 7: Add target rolling features if in training mode
        if is_training and self.target_col in df.columns:
            for window in self.rolling_windows:
                roll_mean_name = f'{self.target_col}_roll{window}_mean'
                rolling_features[roll_mean_name] = df[self.target_col].rolling(window=window).mean()
                
                roll_std_name = f'{self.target_col}_roll{window}_std'
                rolling_features[roll_std_name] = df[self.target_col].rolling(window=window).std()
        
        # Combine all features
        for name, series in imputation_indicators.items():
            df[name] = series
        
        for name, series in lag_features.items():
            df[name] = series
        
        for name, series in rolling_features.items():
            df[name] = series
        
        # Store feature names (excluding date_id and target)
        if self.feature_names_ is None:
            self.feature_names_ = [c for c in df.columns if c not in ['date_id', self.target_col]]
        
        print(f"[Preprocessor] Transformed to {df.shape[1]} features")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform in one step, return X and y
        
        Args:
            df: Training dataframe
            
        Returns:
            X: Feature matrix
            y: Target variable
        """
        self.fit(df)
        df_transformed = self.transform(df, is_training=True)
        
        X = df_transformed[self.feature_names_]
        y = df_transformed[self.target_col]
        
        return X, y
    
    def save(self, filepath: str):
        """Save fitted preprocessor to disk"""
        joblib.dump(self, filepath)
        print(f"[Preprocessor] Saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'MarketPreprocessor':
        """Load fitted preprocessor from disk"""
        preprocessor = joblib.load(filepath)
        print(f"[Preprocessor] Loaded from {filepath}")
        return preprocessor


if __name__ == "__main__":
    print("="*80)
    print("TESTING PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load data
    train = pd.read_csv('train.csv')
    print(f"\n✓ Loaded train data: {train.shape}")
    
    # Initialize and fit
    preprocessor = MarketPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train)
    
    print(f"\n✓ Fit transform complete:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Features: {len(preprocessor.feature_names_)}")
    
    # Save pipeline
    preprocessor.save('preprocessor.pkl')
    
    # Test loading
    loaded_preprocessor = MarketPreprocessor.load('preprocessor.pkl')
    print(f"\n✓ Loaded preprocessor has {len(loaded_preprocessor.feature_names_)} features")
    
    print("\n✅ PREPROCESSING PIPELINE CLASS READY FOR USE")
