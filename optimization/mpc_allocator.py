"""
PulseBMS Enhanced - Model Predictive Control (MPC) Baseline Allocator
Optimizes power allocation across heterogeneous battery fleet while respecting safety constraints
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
import cvxpy as cp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BatteryState:
    """Current state of a battery pack"""
    device_id: str
    soc: float              # State of charge (%)
    soh: float              # State of health (%)
    voltage: float          # Current voltage (V)
    current: float          # Current current (A)
    temperature: float      # Temperature (°C)
    power_capability: float # Maximum power capability (W)
    energy_capacity: float  # Available energy capacity (Wh)
    chemistry: str          # Battery chemistry
    last_updated: datetime


@dataclass
class SafetyConstraints:
    """Safety constraints for battery operation"""
    min_soc: float = 10.0          # Minimum SoC (%)
    max_soc: float = 90.0          # Maximum SoC (%)
    min_voltage: float = 2.5       # Minimum cell voltage (V)
    max_voltage: float = 4.2       # Maximum cell voltage (V)
    max_current: float = 100.0     # Maximum current (A)
    max_temperature: float = 45.0  # Maximum temperature (°C)
    max_power_charge: float = 5000.0    # Maximum charging power (W)
    max_power_discharge: float = 5000.0 # Maximum discharge power (W)


@dataclass
class PowerDemand:
    """Power demand profile"""
    timestamp: datetime
    demand_kw: float        # Power demand in kW
    horizon_minutes: int    # Prediction horizon
    profile: List[float]    # Power profile over horizon


@dataclass
class AllocationResult:
    """Result of power allocation optimization"""
    device_id: str
    allocated_power: float  # Allocated power in W
    predicted_soc: List[float]  # SoC evolution over horizon
    predicted_voltage: List[float]  # Voltage evolution
    cost: float            # Allocation cost
    constraints_satisfied: bool
    degradation_cost: float


class BatteryModel:
    """Simple battery model for MPC predictions"""
    
    def __init__(self):
        # Model parameters for different chemistries
        self.chemistry_params = {
            "LFP": {
                "ocv_params": [3.0, 3.3, 3.35, 3.4, 3.45],  # Open circuit voltage curve
                "soc_points": [0, 25, 50, 75, 100],
                "resistance": 0.05,  # Internal resistance (Ohm)
                "efficiency": 0.95,  # Round-trip efficiency
                "degradation_factor": 0.0001  # Degradation per cycle
            },
            "NMC": {
                "ocv_params": [3.0, 3.7, 3.8, 3.9, 4.2],
                "soc_points": [0, 25, 50, 75, 100],
                "resistance": 0.08,
                "efficiency": 0.92,
                "degradation_factor": 0.0002
            },
            "LCO": {
                "ocv_params": [3.0, 3.6, 3.75, 3.85, 4.1],
                "soc_points": [0, 25, 50, 75, 100],
                "resistance": 0.07,
                "efficiency": 0.93,
                "degradation_factor": 0.00015
            }
        }
    
    def get_ocv(self, soc: float, chemistry: str) -> float:
        """Get open circuit voltage for given SoC and chemistry"""
        chemistry = chemistry.upper()
        if chemistry not in self.chemistry_params:
            chemistry = "NMC"  # Default
        
        params = self.chemistry_params[chemistry]
        ocv_curve = params["ocv_params"]
        soc_points = params["soc_points"]
        
        # Linear interpolation
        return np.interp(soc, soc_points, ocv_curve)
    
    def get_voltage(self, soc: float, current: float, chemistry: str) -> float:
        """Get terminal voltage considering internal resistance"""
        ocv = self.get_ocv(soc, chemistry)
        resistance = self.chemistry_params.get(chemistry.upper(), self.chemistry_params["NMC"])["resistance"]
        return ocv - (current * resistance)
    
    def predict_soc(self, initial_soc: float, power_profile: List[float], 
                   dt: float, capacity_ah: float, chemistry: str) -> List[float]:
        """Predict SoC evolution given power profile"""
        soc_profile = [initial_soc]
        current_soc = initial_soc
        
        efficiency = self.chemistry_params.get(chemistry.upper(), self.chemistry_params["NMC"])["efficiency"]
        
        for power in power_profile:
            # Estimate current from power (simplified)
            voltage = self.get_voltage(current_soc, 0, chemistry)  # No-load voltage
            current = power / voltage if voltage > 0 else 0
            
            # Apply efficiency
            if power < 0:  # Charging
                current *= efficiency
            else:  # Discharging
                current /= efficiency
            
            # Update SoC
            soc_change = -(current * dt) / (capacity_ah * 36)  # Convert to percentage
            current_soc = np.clip(current_soc + soc_change, 0, 100)
            soc_profile.append(current_soc)
        
        return soc_profile[:-1]  # Remove last element to match input length
    
    def calculate_degradation_cost(self, power_profile: List[float], soc_profile: List[float], 
                                  chemistry: str, soh: float) -> float:
        """Calculate degradation cost for given power profile"""
        chemistry = chemistry.upper()
        degradation_factor = self.chemistry_params.get(chemistry, self.chemistry_params["NMC"])["degradation_factor"]
        
        # Calculate stress factors
        power_stress = np.mean(np.abs(power_profile)) / 1000.0  # Normalize to kW
        soc_stress = np.mean([(abs(s - 50) / 50) ** 2 for s in soc_profile])  # Stress from deviation from 50%
        health_factor = (1 - soh / 100) + 0.1  # Degraded batteries have higher cost
        
        total_stress = power_stress * soc_stress * health_factor
        return degradation_factor * total_stress * len(power_profile)


class MPCAllocator:
    """Model Predictive Control power allocator"""
    
    def __init__(self, prediction_horizon: int = 24, control_horizon: int = 6, dt_minutes: float = 15.0):
        self.prediction_horizon = prediction_horizon  # Number of time steps to predict
        self.control_horizon = control_horizon        # Number of control moves
        self.dt_minutes = dt_minutes                  # Time step in minutes
        self.dt_hours = dt_minutes / 60.0            # Time step in hours
        
        self.battery_model = BatteryModel()
        
        # Optimization weights
        self.weights = {
            "tracking": 1000.0,      # Weight for tracking power demand
            "soc_balance": 10.0,     # Weight for SoC balancing
            "degradation": 50.0,     # Weight for degradation minimization
            "power_smoothing": 5.0,  # Weight for power smoothing
            "efficiency": 20.0       # Weight for efficiency maximization
        }
        
        # Optimization tolerances
        self.tolerance = {
            "power_balance": 0.1,    # kW tolerance for power balance
            "soc_limits": 1.0,       # % tolerance for SoC limits
            "voltage_limits": 0.1    # V tolerance for voltage limits
        }
    
    async def allocate_power(self, battery_states: List[BatteryState], 
                           power_demand: PowerDemand, 
                           safety_constraints: Dict[str, SafetyConstraints]) -> List[AllocationResult]:
        """
        Allocate power across battery fleet using MPC optimization
        """
        try:
            logger.info(f"Starting MPC allocation for {len(battery_states)} batteries")
            
            # Prepare optimization problem
            n_batteries = len(battery_states)
            n_steps = min(self.prediction_horizon, len(power_demand.profile))
            
            if n_batteries == 0:
                logger.warning("No batteries available for allocation")
                return []
            
            # Create decision variables (power allocation for each battery at each time step)
            P = cp.Variable((n_batteries, n_steps))  # Power allocation matrix [W]
            
            # Create objective function
            objective = self._create_objective(P, battery_states, power_demand, n_steps)
            
            # Create constraints
            constraints = self._create_constraints(P, battery_states, power_demand, safety_constraints, n_steps)
            
            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(objective), constraints)
            
            # Use appropriate solver
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except:
                try:
                    problem.solve(solver=cp.SCS, verbose=False)
                except:
                    problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                # Extract results
                power_allocation = P.value
                results = await self._create_allocation_results(
                    battery_states, power_allocation, power_demand, safety_constraints
                )
                
                logger.info(f"MPC allocation completed successfully. Status: {problem.status}")
                return results
            else:
                logger.error(f"MPC optimization failed: {problem.status}")
                # Return fallback allocation
                return await self._fallback_allocation(battery_states, power_demand)
                
        except Exception as e:
            logger.error(f"MPC allocation failed: {e}")
            return await self._fallback_allocation(battery_states, power_demand)
    
    def _create_objective(self, P: cp.Variable, battery_states: List[BatteryState], 
                         power_demand: PowerDemand, n_steps: int) -> cp.Expression:
        """Create MPC objective function"""
        
        n_batteries = len(battery_states)
        total_cost = 0
        
        # 1. Power tracking cost - minimize deviation from total demand
        demand_profile = np.array(power_demand.profile[:n_steps]) * 1000  # Convert kW to W
        total_allocated = cp.sum(P, axis=0)  # Sum across all batteries for each time step
        tracking_cost = cp.sum_squares(total_allocated - demand_profile)
        total_cost += self.weights["tracking"] * tracking_cost
        
        # 2. SoC balancing cost - encourage balanced SoC across batteries
        if n_batteries > 1:
            soc_targets = np.array([state.soc for state in battery_states])
            soc_mean = np.mean(soc_targets)
            
            # Predict SoC evolution for each battery
            for i, state in enumerate(battery_states):
                capacity_ah = state.energy_capacity / state.voltage  # Approximate capacity
                power_profile = P[i, :]
                
                # Simple SoC prediction
                soc_change = cp.sum(power_profile) * self.dt_hours / (capacity_ah * 36)  # Simplified
                final_soc = state.soc - soc_change  # Negative because positive power = discharge
                
                soc_deviation = (final_soc - soc_mean) ** 2
                total_cost += self.weights["soc_balance"] * soc_deviation
        
        # 3. Degradation cost - penalize high power and unbalanced operation
        for i, state in enumerate(battery_states):
            # Power stress
            power_stress = cp.sum_squares(P[i, :]) / 1e6  # Normalize
            
            # Health factor - degraded batteries should be used less
            health_factor = (1 - state.soh / 100) + 0.1
            degradation_cost = health_factor * power_stress
            
            total_cost += self.weights["degradation"] * degradation_cost
        
        # 4. Power smoothing cost - encourage smooth power profiles
        for i in range(n_batteries):
            for t in range(n_steps - 1):
                power_change = P[i, t+1] - P[i, t]
                total_cost += self.weights["power_smoothing"] * cp.square(power_change)
        
        # 5. Efficiency cost - prefer batteries with better efficiency
        for i, state in enumerate(battery_states):
            # Efficiency factor based on chemistry and health
            chemistry_efficiency = self.battery_model.chemistry_params.get(
                state.chemistry.upper(), self.battery_model.chemistry_params["NMC"]
            )["efficiency"]
            health_efficiency = state.soh / 100
            total_efficiency = chemistry_efficiency * health_efficiency
            
            # Penalize use of less efficient batteries
            efficiency_cost = (1 - total_efficiency) * cp.sum_squares(P[i, :])
            total_cost += self.weights["efficiency"] * efficiency_cost
        
        return total_cost
    
    def _create_constraints(self, P: cp.Variable, battery_states: List[BatteryState],
                           power_demand: PowerDemand, safety_constraints: Dict[str, SafetyConstraints],
                           n_steps: int) -> List[cp.Constraint]:
        """Create MPC constraints"""
        
        constraints = []
        n_batteries = len(battery_states)
        
        # 1. Power balance constraint - total allocation should meet demand (with tolerance)
        demand_profile = np.array(power_demand.profile[:n_steps]) * 1000  # Convert kW to W
        total_allocated = cp.sum(P, axis=0)
        
        for t in range(n_steps):
            tolerance_w = self.tolerance["power_balance"] * 1000  # Convert to W
            constraints.append(total_allocated[t] >= demand_profile[t] - tolerance_w)
            constraints.append(total_allocated[t] <= demand_profile[t] + tolerance_w)
        
        # 2. Individual battery power limits
        for i, state in enumerate(battery_states):
            device_constraints = safety_constraints.get(state.device_id, SafetyConstraints())
            
            # Power capability limits
            max_discharge = min(state.power_capability, device_constraints.max_power_discharge)
            max_charge = -min(state.power_capability, device_constraints.max_power_charge)
            
            for t in range(n_steps):
                constraints.append(P[i, t] <= max_discharge)  # Discharge limit
                constraints.append(P[i, t] >= max_charge)     # Charge limit
        
        # 3. SoC constraints - maintain SoC within safe limits
        for i, state in enumerate(battery_states):
            device_constraints = safety_constraints.get(state.device_id, SafetyConstraints())
            capacity_ah = state.energy_capacity / state.voltage  # Approximate capacity
            
            for t in range(n_steps):
                # Calculate cumulative SoC change up to time t
                cumulative_power = cp.sum(P[i, :t+1])
                soc_change = cumulative_power * self.dt_hours / (capacity_ah * 36)  # Simplified
                predicted_soc = state.soc - soc_change  # Negative because positive power = discharge
                
                # SoC limits with tolerance
                constraints.append(predicted_soc >= device_constraints.min_soc - self.tolerance["soc_limits"])
                constraints.append(predicted_soc <= device_constraints.max_soc + self.tolerance["soc_limits"])
        
        # 4. Voltage constraints (simplified)
        for i, state in enumerate(battery_states):
            device_constraints = safety_constraints.get(state.device_id, SafetyConstraints())
            
            # Simplified voltage constraint based on current SoC and power
            for t in range(n_steps):
                # Estimate current from power
                estimated_current = P[i, t] / state.voltage
                
                # Basic voltage limits (this is simplified - real implementation would be more complex)
                constraints.append(estimated_current <= device_constraints.max_current)
                constraints.append(estimated_current >= -device_constraints.max_current)
        
        return constraints
    
    async def _create_allocation_results(self, battery_states: List[BatteryState], 
                                       power_allocation: np.ndarray, power_demand: PowerDemand,
                                       safety_constraints: Dict[str, SafetyConstraints]) -> List[AllocationResult]:
        """Create allocation results from optimization solution"""
        
        results = []
        n_steps = power_allocation.shape[1]
        
        for i, state in enumerate(battery_states):
            # Extract power profile for this battery
            power_profile = power_allocation[i, :].tolist()
            
            # Predict SoC evolution
            capacity_ah = state.energy_capacity / state.voltage
            predicted_soc = self.battery_model.predict_soc(
                state.soc, power_profile, self.dt_hours, capacity_ah, state.chemistry
            )
            
            # Predict voltage evolution
            predicted_voltage = []
            for j, power in enumerate(power_profile):
                current = power / state.voltage if state.voltage > 0 else 0
                soc = predicted_soc[j] if j < len(predicted_soc) else state.soc
                voltage = self.battery_model.get_voltage(soc, current, state.chemistry)
                predicted_voltage.append(voltage)
            
            # Calculate costs
            degradation_cost = self.battery_model.calculate_degradation_cost(
                power_profile, predicted_soc, state.chemistry, state.soh
            )
            
            # Check constraint satisfaction
            device_constraints = safety_constraints.get(state.device_id, SafetyConstraints())
            constraints_satisfied = self._check_constraints_satisfied(
                power_profile, predicted_soc, predicted_voltage, device_constraints
            )
            
            # Calculate total cost (simplified)
            total_cost = degradation_cost + abs(power_profile[0]) * 0.001  # Simple cost model
            
            result = AllocationResult(
                device_id=state.device_id,
                allocated_power=power_profile[0],  # First time step power
                predicted_soc=predicted_soc,
                predicted_voltage=predicted_voltage,
                cost=total_cost,
                constraints_satisfied=constraints_satisfied,
                degradation_cost=degradation_cost
            )
            
            results.append(result)
        
        return results
    
    def _check_constraints_satisfied(self, power_profile: List[float], soc_profile: List[float],
                                   voltage_profile: List[float], constraints: SafetyConstraints) -> bool:
        """Check if constraints are satisfied"""
        
        # Check SoC limits
        if any(soc < constraints.min_soc or soc > constraints.max_soc for soc in soc_profile):
            return False
        
        # Check voltage limits
        if any(v < constraints.min_voltage or v > constraints.max_voltage for v in voltage_profile):
            return False
        
        # Check power limits
        if any(abs(p) > max(constraints.max_power_charge, constraints.max_power_discharge) for p in power_profile):
            return False
        
        return True
    
    async def _fallback_allocation(self, battery_states: List[BatteryState], 
                                 power_demand: PowerDemand) -> List[AllocationResult]:
        """Fallback allocation when MPC optimization fails"""
        
        logger.warning("Using fallback allocation strategy")
        
        results = []
        total_demand_w = power_demand.demand_kw * 1000  # Convert to W
        
        if not battery_states:
            return results
        
        # Simple proportional allocation based on available capacity
        total_capacity = sum(state.energy_capacity * (state.soc / 100) for state in battery_states)
        
        for state in battery_states:
            if total_capacity > 0:
                available_energy = state.energy_capacity * (state.soc / 100)
                allocation_ratio = available_energy / total_capacity
                allocated_power = total_demand_w * allocation_ratio
                
                # Clamp to power capability
                allocated_power = np.clip(allocated_power, -state.power_capability, state.power_capability)
            else:
                allocated_power = 0.0
            
            # Create simple prediction
            predicted_soc = [state.soc] * self.prediction_horizon
            predicted_voltage = [state.voltage] * self.prediction_horizon
            
            result = AllocationResult(
                device_id=state.device_id,
                allocated_power=allocated_power,
                predicted_soc=predicted_soc,
                predicted_voltage=predicted_voltage,
                cost=abs(allocated_power) * 0.001,
                constraints_satisfied=True,
                degradation_cost=0.0
            )
            
            results.append(result)
        
        return results


class MPCCoordinator:
    """Coordinator that interfaces MPC allocator with the rest of the system"""
    
    def __init__(self):
        self.mpc_allocator = MPCAllocator()
        self.last_allocation_time = None
        self.allocation_interval = 300  # 5 minutes
        
    async def coordinate_power_allocation(self, fleet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main coordination function called by the backend"""
        
        try:
            # Extract battery states from fleet data
            battery_states = self._extract_battery_states(fleet_data)
            
            # Extract power demand
            power_demand = self._extract_power_demand(fleet_data)
            
            # Extract safety constraints
            safety_constraints = self._extract_safety_constraints(fleet_data)
            
            # Check if allocation is needed
            current_time = datetime.utcnow()
            if (self.last_allocation_time and 
                (current_time - self.last_allocation_time).total_seconds() < self.allocation_interval):
                
                # Return cached allocation if available
                return {"status": "cached", "message": "Using previous allocation"}
            
            # Run MPC allocation
            allocation_results = await self.mpc_allocator.allocate_power(
                battery_states, power_demand, safety_constraints
            )
            
            self.last_allocation_time = current_time
            
            # Format results for the backend
            allocation_commands = {}
            total_allocated_power = 0
            
            for result in allocation_results:
                allocation_commands[result.device_id] = {
                    "power_setpoint": result.allocated_power,
                    "predicted_soc": result.predicted_soc,
                    "cost": result.cost,
                    "constraints_ok": result.constraints_satisfied
                }
                total_allocated_power += result.allocated_power
            
            logger.info(f"Allocated {total_allocated_power/1000:.2f} kW across {len(allocation_results)} batteries")
            
            return {
                "status": "success",
                "timestamp": current_time.isoformat(),
                "total_allocated_kw": total_allocated_power / 1000,
                "allocations": allocation_commands,
                "horizon_minutes": self.mpc_allocator.prediction_horizon * self.mpc_allocator.dt_minutes
            }
            
        except Exception as e:
            logger.error(f"Power allocation coordination failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _extract_battery_states(self, fleet_data: Dict[str, Any]) -> List[BatteryState]:
        """Extract battery states from fleet data"""
        
        battery_states = []
        devices = fleet_data.get("devices", [])
        
        for device in devices:
            try:
                state = BatteryState(
                    device_id=device["device_id"],
                    soc=device.get("measurements", {}).get("soc", 50.0),
                    soh=device.get("measurements", {}).get("soh", 80.0),
                    voltage=device.get("measurements", {}).get("voltage", 400.0),
                    current=device.get("measurements", {}).get("current", 0.0),
                    temperature=device.get("measurements", {}).get("temperature", 25.0),
                    power_capability=device.get("specifications", {}).get("max_power", 5000.0),
                    energy_capacity=device.get("specifications", {}).get("capacity_kwh", 50.0) * 1000,  # Convert to Wh
                    chemistry=device.get("specifications", {}).get("chemistry", "NMC"),
                    last_updated=datetime.utcnow()
                )
                battery_states.append(state)
            except Exception as e:
                logger.warning(f"Failed to extract state for device {device.get('device_id', 'unknown')}: {e}")
        
        return battery_states
    
    def _extract_power_demand(self, fleet_data: Dict[str, Any]) -> PowerDemand:
        """Extract power demand from fleet data"""
        
        demand_data = fleet_data.get("power_demand", {})
        
        return PowerDemand(
            timestamp=datetime.utcnow(),
            demand_kw=demand_data.get("current_demand_kw", 10.0),
            horizon_minutes=demand_data.get("horizon_minutes", 360),  # 6 hours
            profile=demand_data.get("profile", [10.0] * 24)  # Default flat profile
        )
    
    def _extract_safety_constraints(self, fleet_data: Dict[str, Any]) -> Dict[str, SafetyConstraints]:
        """Extract safety constraints from fleet data"""
        
        constraints = {}
        devices = fleet_data.get("devices", [])
        
        for device in devices:
            device_id = device["device_id"]
            safety_config = device.get("safety_constraints", {})
            
            constraints[device_id] = SafetyConstraints(
                min_soc=safety_config.get("min_soc", 10.0),
                max_soc=safety_config.get("max_soc", 90.0),
                min_voltage=safety_config.get("min_voltage", 2.5),
                max_voltage=safety_config.get("max_voltage", 4.2),
                max_current=safety_config.get("max_current", 100.0),
                max_temperature=safety_config.get("max_temperature", 45.0),
                max_power_charge=safety_config.get("max_power_charge", 5000.0),
                max_power_discharge=safety_config.get("max_power_discharge", 5000.0)
            )
        
        return constraints


# Example usage and testing
if __name__ == "__main__":
    async def test_mpc_allocator():
        """Test the MPC allocator"""
        
        # Create test battery states
        battery_states = [
            BatteryState(
                device_id="battery_001",
                soc=60.0, soh=85.0, voltage=400.0, current=0.0, temperature=25.0,
                power_capability=5000.0, energy_capacity=50000.0, chemistry="NMC",
                last_updated=datetime.utcnow()
            ),
            BatteryState(
                device_id="battery_002", 
                soc=45.0, soh=90.0, voltage=350.0, current=0.0, temperature=22.0,
                power_capability=3000.0, energy_capacity=30000.0, chemistry="LFP",
                last_updated=datetime.utcnow()
            )
        ]
        
        # Create test power demand
        power_demand = PowerDemand(
            timestamp=datetime.utcnow(),
            demand_kw=5.0,
            horizon_minutes=360,
            profile=[5.0, 4.5, 4.0, 3.5, 3.0, 4.0] + [5.0] * 18  # 24 hour profile
        )
        
        # Create safety constraints
        safety_constraints = {
            "battery_001": SafetyConstraints(),
            "battery_002": SafetyConstraints()
        }
        
        # Test allocation
        allocator = MPCAllocator()
        results = await allocator.allocate_power(battery_states, power_demand, safety_constraints)
        
        print(f"Allocation completed. Results for {len(results)} batteries:")
        for result in results:
            print(f"  {result.device_id}: {result.allocated_power:.2f}W, Cost: {result.cost:.4f}")
    
    # Run test
    asyncio.run(test_mpc_allocator())
