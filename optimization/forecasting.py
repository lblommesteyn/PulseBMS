import numpy as np

class EnergyForecaster:
    """Time-series forecasting for load and generation"""
    
    def __init__(self):
        self.history = []
        
    def add_observation(self, value: float):
        self.history.append(value)
        if len(self.history) > 1000:
            self.history.pop(0)
            
    def predict_next_24h(self) -> list:
        """Simple Holt-Winters exponential smoothing simulation"""
        if not self.history:
            return [0.0] * 24
            
        last_val = self.history[-1]
        # Simulate a daily profile
        profile = np.sin(np.linspace(0, 2*np.pi, 24)) * 10
        
        return [last_val + p + np.random.normal(0, 2) for p in profile]
