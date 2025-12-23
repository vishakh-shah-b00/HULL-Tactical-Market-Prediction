# **Hull Tactical Market Prediction**

This repository hosts the codebase, analysis, and model artifacts for the Hull Tactical Market Prediction project. The primary objective is to predict daily S\&P 500 excess returns (market\_forward\_excess\_returns) and generate a trading signal that maximizes the Sharpe ratio.  
The solution utilizes a gradient boosting approach (LightGBM) with a custom "Purged K-Fold" cross-validation scheme designed to handle high-autocorrelation financial time series. The final model achieves a **Sharpe ratio of 2.89** on a strictly held-out test set.

## **Project Overview**

* **Goal**: Predict next-day market excess returns and determine optimal portfolio leverage (0% to 200%).  
* **Data**: 9,021 daily records with 98 anonymized financial indicators.  
* **Key Result**: 18.20% total return over the 121-day holdout period with controlled volatility.  
* **Methodology**: Rigorous feature engineering, "Purged K-Fold" cross-validation to prevent leakage, and regime-aware modeling.

## **Repository Structure**

The project is organized into logical modules for exploration, processing, training, and evaluation.  
```
├── analysis/  
│   ├── COMPLETE_ANALYSIS_REPORT.md    # Detailed Phase 0-2 analysis findings  
│   └── MODEL_DEVELOPMENT_REPORT.md    # Technical modeling documentation  
├── artifacts/  
│   ├── lgb_model_clean.pkl            # Final trained LightGBM model  
│   ├── preprocessor_clean.pkl         # Scikit-learn feature transformation pipeline  
│   ├── selected_features_clean.pkl    # List of 100 selected high-value features  
│   └── position_mapper.pkl            # Logic for converting predictions to positions  
├── exploration_plots/                 # Visualization output from Phase 0 EDA  
├── src/  
│   ├── phase0_exploration.py          # Exploratory Data Analysis scripts  
│   ├── phase1_step1_inspect.py        # Data quality and integrity checks  
│   ├── phase1_step2_engineer.py       # Feature generation (lags, rolling windows)  
│   ├── phase1_step3_validate.py       # Feature selection via Mutual Information  
│   ├── phase2_train_fixed.py          # Model training with Purged CV  
│   ├── phase3_position_mapping.py     # Optimization of position sizing strategies  
│   └── validate_holdout.py            # Final performance verification script  
├── requirements.txt                   # Project dependencies  
└── README.md
```
## **Data Description**

The dataset comprises 9,021 rows and 98 features categorized as follows:

* **Market (M)**: 18 features (Volume, momentum, index levels)  
* **Volatility (V)**: 13 features (VIX variants, realized volatility)  
* **Sentiment (S)**: 12 features (News sentiment, proprietary scores)  
* **Macro (E)**: 20 features (Economic indicators)  
* **Interest Rates (I)**: 9 features (Yield curve spreads)  
* **Price (P)**: 13 features (Technical price action)  
* **Dummy (D)**: 9 features (Regime flags)

## **Methodology**

### **1\. Exploratory Data Analysis (Phase 0\)**

Initial statistical tests confirmed stationarity in the target variable but identified significant "fat tails" (Kurtosis \> 0.85). A key finding was the high autocorrelation in predictor features (ACF \> 0.96), which necessitated a specialized cross-validation strategy to prevent look-ahead bias.

### **2\. Feature Engineering (Phase 1\)**

The raw feature set was expanded from 98 to 227 features through:

* **Imputation**: Forward-filling to respect time-series continuity, followed by median imputation.  
* **Lag Generation**: Creating 1, 2, 5, 10, and 20-day lagged versions of key indicators.  
* **Rolling Statistics**: Calculating mean and standard deviation over 5, 20, and 60-day windows.

Feature selection was performed using a variance filter, correlation thresholding (removing \>0.95 collinearity), and Mutual Information regression, resulting in a final set of **100 high-value features**.

### **3\. Model Development (Phase 2\)**

* **Algorithm**: LightGBM Regressor.  
* **Validation**: Custom **Purged K-Fold** Cross-Validation. This method introduces an embargo period (20 days) between training and test folds to eliminate leakage caused by feature autocorrelation.  
* **Hyperparameters**: The model uses shallow trees (depth 4\) and heavy L1/L2 regularization to prevent overfitting on the noisy financial data.

### **4\. Position Sizing (Phase 3\)**

The model's continuous return predictions are mapped to a discrete position size (leverage) between 0.0 and 2.0. A "Sign-Based" strategy proved most robust:

* **Prediction \> 0**: Long 2x (Position 2.0)  
* **Prediction \< 0**: Neutral (Position 0.0)

This binary approach outperformed complex sigmoid or tanh mappings by reducing the impact of magnitude errors in the prediction.

## **Results**

Performance was evaluated on a strictly isolated holdout set (the final 180 rows of the dataset), which was **never seen** during training or feature selection.

| Metric | Result | Notes |
| :---- | :---- | :---- |
| **Sharpe Ratio** | **2.89** | Excellent risk-adjusted return |
| **Total Return** | 18.20% | Over 121 trading days |
| **Volatility Ratio** | 1.08 | Strategy Vol / Market Vol (Target \< 1.2) |
| **Win Rate** | 56% | Daily hit rate |

**Top Predictors**:

1. **M4**: Primary market indicator.  
2. **S2\_roll20\_mean**: 20-day rolling sentiment trend.  
3. **M17**: Secondary market signal.

## **Usage**

### **Installation**

Install the necessary Python packages:  
pip install -r requirements.txt

### **Reproduction**

To run the full pipeline from raw data to final artifacts:  
```
# 1\. Run Feature Engineering (creates clean datasets)  
python src/phase1_step2_engineer.py

# 2\. Train Model (uses Purged CV and saves artifacts)  
python src/phase2_train_fixed.py

# 3\. Validate on Holdout (prints final metrics)  
python src/validate_holdout.py
```
### **Inference Example**

To generate predictions on new data using the saved artifacts: 
```
import pandas as pd  
import joblib

# Load the pipeline artifacts  
model = joblib.load('artifacts/lgb_model_clean.pkl')  
preprocessor = joblib.load('artifacts/preprocessor_clean.pkl')  
features = joblib.load('artifacts/selected_features_clean.pkl')  
mapper = joblib.load('artifacts/position_mapper.pkl')

# Load new data (example)  
df = pd.read_csv('new_market_data.csv')

# 1. Preprocess  
# Note: Ensure validation/inference data is transformed using the training scaler  
X_transformed = preprocessor.transform(df)

# 2. Handle missing features (if any dropped during engineering)  
for f in features:  
    if f not in X_transformed.columns:  
        X_transformed[f] = 0

# 3. Select Features  
X_final = X_transformed[features]

# 4. Generate Signal  
raw_pred = model.predict(X_final)      # Continuous prediction  
position = mapper.map(raw_pred)        # 0.0 or 2.0

print(f"Recommended Position: {position[-1]}")  
```
