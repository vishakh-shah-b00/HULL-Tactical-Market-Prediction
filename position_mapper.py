"""
Position Mapper - Converts predictions to [0, 2] positions
"""
import numpy as np

class PositionMapper:
    """Converts predictions to [0, 2] positions"""
    
    def __init__(self, strategy_name, strategy_params=None):
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        
    def map(self, predictions):
        """Map predictions to positions [0, 2]"""
        if self.strategy_name == '1_Sign':
            return np.where(predictions > 0, 2.0, 0.0)
        
        elif self.strategy_name == '2_Scaled':
            pred_min = predictions.min()
            pred_max = predictions.max()
            if pred_max == pred_min:
                return np.ones_like(predictions)
            positions = 2.0 * (predictions - pred_min) / (pred_max - pred_min)
            return np.clip(positions, 0.0, 2.0)
        
        elif self.strategy_name == '3_Sigmoid':
            center = self.strategy_params.get('center', 0)
            scale = self.strategy_params.get('scale', 1)
            positions = 2.0 / (1.0 + np.exp(-(predictions - center) / scale))
            return np.clip(positions, 0.0, 2.0)
        
        elif self.strategy_name == '4_Tercile':
            tercile_33 = np.percentile(predictions, 33.33)
            tercile_67 = np.percentile(predictions, 66.67)
            positions = np.zeros_like(predictions)
            positions[predictions >= tercile_67] = 2.0
            positions[(predictions >= tercile_33) & (predictions < tercile_67)] = 1.0
            return positions
        
        elif self.strategy_name == '5_Tanh':
            normalized = predictions / (np.std(predictions) + 1e-8)
            positions = 1.0 + np.tanh(normalized)
            return np.clip(positions, 0.0, 2.0)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
