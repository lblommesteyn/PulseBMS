import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SensorReadings:
    cell_voltages: List[float]
    cell_temps: List[float]
    pack_current: float
    pack_voltage: float

class BatterySensors:
    def __init__(self, num_cells=12, noise_level=0.005):
        self.num_cells = num_cells
        self.noise_level = noise_level
        self.base_voltage = 3.7
        
    def read_all(self) -> SensorReadings:
        """Simulate ADC readings with noise"""
        # Simulate slight imbalance
        voltages = [
            self.base_voltage + random.uniform(-0.02, 0.02) + random.gauss(0, self.noise_level) 
            for _ in range(self.num_cells)
        ]
        
        # Simulate hotspot
        temps = [
            25.0 + random.uniform(0, 2.0) + (5.0 if i == 5 else 0) 
            for i in range(self.num_cells)
        ]
        
        current = random.gauss(0, 0.5) # Idle noise
        
        return SensorReadings(
            cell_voltages=voltages,
            cell_temps=temps,
            pack_current=current,
            pack_voltage=sum(voltages)
        )
