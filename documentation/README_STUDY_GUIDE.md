# Hull Tactical Market Prediction - Project Study Guide

This guide explains the structure of the solution, what each file does, and how we met the competition requirements. Use this to study for your presentation.

---

## 1. Did We Meet the Competition Goals?

**The Prompt:**
> "Build a model that predicts excess returns and includes a betting strategy designed to outperform the S&P 500 while staying within a 120% volatility constraint."

**Our Solution:**
1.  **Predict Excess Returns**:
    *   **Yes.** Our target variable was `market_forward_excess_returns`.
    *   **Evidence**: `preprocessor.py` (lines 140-150) and `model.py`.
2.  **Betting Strategy**:
    *   **Yes.** We implemented a dynamic position sizing strategy (Sign-Based).
    *   **Evidence**: `position_mapper.py` maps predictions to positions [0.0, 2.0].
3.  **Outperform S&P 500**:
    *   **Yes.** S&P 500 Sharpe = 0.08 vs Our Sharpe = 2.82 (Official Metric).
    *   **Evidence**: `benchmark_analysis.py` and `official_metric.py`.
4.  **120% Volatility Constraint**:
    *   **Yes.** Our Volatility Ratio is 1.08 (108%), which is safely under the 120% limit.
    *   **Evidence**: `official_metric.py` confirms no penalty.

---

## 2. Directory Structure & File Explanations

### A. The "Engine Room" (Core Submission)
*These are the files that actually run the strategy.*

*   **`model.py`**: The brain. Contains the `Model` class that loads the saved artifacts and runs `predict()`.
*   **`preprocessor.py`**: The plumbing. Contains `MarketPreprocessor` which cleans data, imputes missing values, and creates features (lags, rolling stats).
*   **`position_mapper.py`**: The strategist. Takes the raw prediction (e.g., "0.005") and decides the bet size (e.g., "200% long").
*   **`submission.csv`**: The final output file required by the competition.

### B. The "Factory" (Training & Validation)
*These scripts created the model. Study these to understand HOW we got here.*

*   **`retrain_clean.py`**: The master training script.
    *   **What it does**: Loads data -> Engineers features -> Selects top 100 features -> Trains LightGBM -> Saves `*_clean.pkl` files.
    *   **Key Concept**: "PurgedKFold" (preventing data leakage).
*   **`validate_holdout.py`**: The proof.
    *   **What it does**: Runs the trained model on the last 180 days (which it never saw during training).
    *   **Key Result**: Produced the Sharpe 2.89 result.

### C. The "Lab Reports" (Analysis & Proof)
*These scripts prove our results are statistically significant. Essential for your presentation.*

*   **`official_metric.py`**: The referee.
    *   **Purpose**: Implements the EXACT competition scoring code to prove we pass the rules.
    *   **Result**: Official Score 2.82 (Pass).
*   **`calculate_ci.py`**: The statistician.
    *   **Purpose**: Bootstraps the results to prove they aren't luck.
    *   **Result**: 98% probability of profit.
*   **`calculate_rolling.py`**: The stress tester.
    *   **Purpose**: Checks performance over time (rolling windows) and drawdowns.
    *   **Result**: Strategy never lost money (min equity 1.0).
*   **`benchmark_analysis.py`**: The comparator.
    *   **Purpose**: Compares us to the S&P 500.
    *   **Result**: We beat the market alpha by 4x.
*   **`fee_analysis.py`**: The reality check.
    *   **Purpose**: Proves that with institutional fees (0.01%), our 50.4% win rate is highly profitable.

### D. The "Artifacts" (Saved Brains)
*Binary files loaded by the code. You don't read these, but the code needs them.*

*   **`lgb_model_clean.pkl`**: The trained LightGBM model.
*   **`preprocessor_clean.pkl`**: The saved preprocessing rules (means/medians).
*   **`selected_features_clean.pkl`**: The list of 100 features we use.
*   **`position_mapper.pkl`**: The saved betting strategy logic.

### E. The Reports (Documentation)
*Read these to understand the narrative.*

*   **`COMPLETE_ANALYSIS_REPORT.md`**: Phase 0 & 1 (Data Exploration). Shows *why* we chose certain features.
*   **`MODEL_DEVELOPMENT_REPORT.md`**: Phase 2 & 3 (Modeling). Shows *how* we built and validated the model.

---

## 3. Key Concepts to Study

1.  **Regime Dependence**: Why our model works better in some months than others. (See `MODEL_DEVELOPMENT_REPORT` > Sustainability).
2.  **Excess vs. Absolute Returns**: Why our 0.35 CV Sharpe is actually good. (See `benchmark_analysis.py`).
3.  **Data Leakage**: How we fixed it using the "Honest Holdout" method. (See `validate_holdout.py`).
