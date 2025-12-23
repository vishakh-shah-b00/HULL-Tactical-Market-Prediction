"""
Phase 3: Final predict() function for Kaggle submission
Compatible with competition API - FIXED for test data
"""
import pandas as pd
import numpy as np
import joblib

class Model:
    """
    INFERENCE WRAPPER FOR SUBMISSION
    --------------------------------
    This class defines the API expected by the competition environment.
    It loads the trained artifacts (model, scaler, feature list) and 
    generates predictions for new data.
    """
    
    def __init__(self):
        """Load all required artifacts"""
        self.preprocessor = joblib.load('preprocessor.pkl')
        self.model = joblib.load('lgb_model.pkl')
        self.selected_features = joblib.load('selected_features.pkl')
        self.position_mapper = joblib.load('position_mapper.pkl')
        
        print("✓ Model loaded successfully")
        print(f"  Features: {len(self.selected_features)}")
        print(f"  Position strategy: {self.position_mapper.strategy_name}")
    
    def predict(self, X_test_df, current_holdings):
        """
        Generate predictions for test data
        
        Args:
            X_test_df: pd.DataFrame with same columns as training data
            current_holdings: float, current position (not used)
            
        Returns:
            float: position allocation [0.0, 2.0]
        """
        # Preprocess
        X_transformed = self.preprocessor.transform(X_test_df, is_training=False)
        
        # Handle missing features (imputation indicators may not exist in test)
        for feature in self.selected_features:
            if feature not in X_transformed.columns:
                # Add missing feature as zeros (indicates no imputation needed)
                X_transformed[feature] = 0
        
        # Select features in correct order
        X_final = X_transformed[self.selected_features]
        
        # Handle any remaining NaN
        if X_final.isnull().any().any():
            X_final = X_final.ffill().bfill().fillna(0)
        
        # Predict returns
        y_pred = self.model.predict(X_final)
        
        # Map to position (handle scalar or array)
        if isinstance(y_pred, (float, np.floating, int)):
            position = self.position_mapper.map(np.array([y_pred]))[0]
        else:
            positions = self.position_mapper.map(y_pred)
            position = positions[-1] if len(positions) > 0 else 1.0
        
        # Ensure in range [0, 2]
        position = float(np.clip(position, 0.0, 2.0))
        
        return position


def test_predict_function():
    """Test the predict function with local data"""
    print("="*80)
    print("TESTING PREDICT() FUNCTION")
    print("="*80)
    
    # Initialize model
    model = Model()
    
    # Load test data
    print("\nLoading test data...")
    test = pd.read_csv('test.csv')
    print(f"  Test data: {test.shape}")
    
    # Test single prediction
    print("\n[1/2] Testing single-row prediction...")
    test_row = test.iloc[[0]]
    
    try:
        position = model.predict(test_row, current_holdings=1.0)
        print(f"  ✓ Single prediction: {position:.4f}")
        print(f"  Type: {type(position)}")
        print(f"  Valid range: {0.0 <= position <= 2.0}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test batch prediction
    print("\n[2/2] Testing batch prediction...")
    try:
        positions = []
        for idx in range(min(100, len(test))):
            test_row = test.iloc[[idx]]
            pos = model.predict(test_row, current_holdings=1.0)
            positions.append(pos)
        
        positions = np.array(positions)
        print(f"  ✓ Generated {len(positions)} predictions")
        print(f"  Mean: {positions.mean():.4f}")
        print(f"  Std: {positions.std():.4f}")
        print(f"  Min: {positions.min():.4f}")
        print(f"  Max: {positions.max():.4f}")
        print(f"  % at 0: {(positions == 0).sum() / len(positions) * 100:.2f}%")
        print(f"  % at 2: {(positions == 2).sum() / len(positions) * 100:.2f}%")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ PREDICT() FUNCTION TESTS PASSED")
    return True


if __name__ == "__main__":
    success = test_predict_function()
    
    if success:
        print("\n" + "="*80)
        print("READY FOR SUBMISSION")
        print("="*80)
        print("To create submission file:")
        print("  python3 create_submission.py")
