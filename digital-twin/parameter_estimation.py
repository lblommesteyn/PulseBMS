import numpy as np
from sklearn.linear_model import RANSACRegressor

class ParameterEstimator:
    """Online parameter estimation for Equivalent Circuit Models (ECM)"""
    
    def __init__(self):
        self.voltage_buffer = []
        self.current_buffer = []
        
    def add_data(self, voltage, current):
        self.voltage_buffer.append(voltage)
        self.current_buffer.append(current)
        if len(self.voltage_buffer) > 1000:
            self.voltage_buffer.pop(0)
            self.current_buffer.pop(0)
            
    def estimate_r0(self):
        """Estimate internal resistance R0 using RANSAC to reject outliers"""
        if len(self.voltage_buffer) < 50:
            return 0.05 # Default
            
        dV = np.diff(self.voltage_buffer)
        dI = np.diff(self.current_buffer)
        
        # R0 approx dV/dI
        mask = np.abs(dI) > 0.5 # Filter for significant current jumps
        if np.sum(mask) < 10:
            return 0.05
            
        X = dI[mask].reshape(-1, 1)
        y = dV[mask].reshape(-1, 1)
        
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        
        r0 = np.abs(ransac.estimator_.coef_[0][0])
        return r0
