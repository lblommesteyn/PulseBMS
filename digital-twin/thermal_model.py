import numpy as np

class PackThermalModel:
    """Nodal thermal model for battery pack"""
    
    def __init__(self, num_nodes=12):
        self.num_nodes = num_nodes
        self.temps = np.ones(num_nodes) * 25.0
        self.c_p = 750.0 # Specific heat capacity J/kgK
        self.mass = 0.5 # kg per cell
        self.r_th_amb = 5.0 # Thermal resistance to ambient
        
    def step(self, power_heat_generation: list, dt: float, ambient_temp: float, coolant_flow: float):
        """Update node temperatures based on heat gen and cooling"""
        new_temps = self.temps.copy()
        
        for i in range(self.num_nodes):
            q_gen = power_heat_generation[i]
            
            # Convection to ambient
            q_amb = (self.temps[i] - ambient_temp) / self.r_th_amb
            
            # Active cooling
            h_cool = coolant_flow * 10.0 # Simplified heat transfer coeff relation
            q_cool = h_cool * (self.temps[i] - 20.0) # Assume 20C coolant
            
            # Energy balance: m*Cp*dT/dt = Qin - Qout
            dT_dt = (q_gen - q_amb - q_cool) / (self.mass * self.c_p)
            
            new_temps[i] += dT_dt * dt
            
        self.temps = new_temps
        return self.temps
