"""
PulseBMS Enhanced - Digital Twin Service
PyBaMM-based battery physics modeling and degradation forecasting
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

import pybamm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import motor.motor_asyncio
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BatteryModelRequest(BaseModel):
    """Request for battery model simulation"""
    device_id: str
    chemistry: str
    current_soc: float
    current_soh: float
    temperature: float
    power_profile: List[float]  # Power profile in W
    time_steps: List[float]     # Time steps in hours
    model_parameters: Dict[str, Any] = {}


class BatteryModelResponse(BaseModel):
    """Response from battery model simulation"""
    device_id: str
    voltage_profile: List[float]
    current_profile: List[float]
    temperature_profile: List[float]
    soc_profile: List[float]
    capacity_fade: float
    resistance_growth: float
    expected_eol: Optional[str]
    cycle_life_remaining: float
    calendar_life_remaining: float


@dataclass
class BatteryParameters:
    """Battery model parameters"""
    chemistry: str
    nominal_capacity: float  # Ah
    nominal_voltage: float   # V
    
    # Electrochemical parameters
    diffusion_coefficient: float = 3.9e-14
    reaction_rate: float = 2.334e-11
    conductivity: float = 1.0
    
    # Degradation parameters
    sei_layer_resistance: float = 0.001
    plating_rate: float = 1e-12
    cycling_factor: float = 1.0
    calendar_factor: float = 1.0


class PyBaMMModelManager:
    """Manages PyBaMM battery models for different chemistries"""
    
    def __init__(self):
        self.models = {}
        self.parameters = {}
        self.solvers = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize PyBaMM models for different chemistries"""
        
        # LFP (Lithium Iron Phosphate) model
        self.models["LFP"] = pybamm.lithium_ion.DFN()
        self.parameters["LFP"] = pybamm.ParameterValues("Chen2020")
        
        # NMC (Lithium Nickel Manganese Cobalt) model  
        self.models["NMC"] = pybamm.lithium_ion.DFN()
        self.parameters["NMC"] = pybamm.ParameterValues("NMC_OKane2022")
        
        # LCO (Lithium Cobalt Oxide) model
        self.models["LCO"] = pybamm.lithium_ion.DFN()
        self.parameters["LCO"] = pybamm.ParameterValues("Marquis2019")
        
        # Initialize solvers
        for chemistry in ["LFP", "NMC", "LCO"]:
            try:
                solver = pybamm.CasadiSolver(mode="fast")
                self.solvers[chemistry] = solver
                logger.info(f"Initialized PyBaMM model for {chemistry}")
            except Exception as e:
                logger.warning(f"Failed to initialize {chemistry} model: {e}")
                # Fallback to basic solver
                self.solvers[chemistry] = pybamm.ScipySolver()
    
    def get_model(self, chemistry: str) -> Tuple[pybamm.BaseModel, pybamm.ParameterValues, pybamm.BaseSolver]:
        """Get model, parameters, and solver for given chemistry"""
        chemistry = chemistry.upper()
        if chemistry not in self.models:
            chemistry = "NMC"  # Default fallback
        
        return self.models[chemistry], self.parameters[chemistry], self.solvers[chemistry]
    
    def update_parameters(self, chemistry: str, custom_params: Dict[str, Any]):
        """Update model parameters for degradation or customization"""
        chemistry = chemistry.upper()
        if chemistry in self.parameters:
            param_values = self.parameters[chemistry].copy()
            
            # Update with custom parameters
            for key, value in custom_params.items():
                if key in param_values:
                    param_values[key] = value
            
            return param_values
        
        return self.parameters.get(chemistry, self.parameters["NMC"])


class DegradationModel:
    """Battery degradation modeling for capacity fade and resistance growth"""
    
    def __init__(self):
        self.degradation_models = {
            "LFP": self._lfp_degradation,
            "NMC": self._nmc_degradation,
            "LCO": self._lco_degradation
        }
    
    def _lfp_degradation(self, cycles: float, calendar_age: float, temperature: float, 
                        soc_avg: float, depth_of_discharge: float) -> Tuple[float, float]:
        """LFP degradation model"""
        # Cycle degradation (Arrhenius-based)
        activation_energy = 24000  # J/mol
        gas_constant = 8.314  # J/(mol·K)
        ref_temp = 298.15  # K
        
        temp_factor = np.exp(activation_energy / gas_constant * (1/ref_temp - 1/(temperature + 273.15)))
        
        # Cycle capacity fade
        cycle_fade = 0.0002 * cycles * temp_factor * (depth_of_discharge ** 0.5)
        
        # Calendar aging
        calendar_fade = 0.0001 * np.sqrt(calendar_age * 365) * temp_factor * (soc_avg / 100) ** 0.5
        
        # Resistance growth
        resistance_growth = 0.0005 * cycles * temp_factor + 0.0002 * calendar_age
        
        total_capacity_fade = cycle_fade + calendar_fade
        
        return min(total_capacity_fade, 0.3), min(resistance_growth, 2.0)  # Cap at 30% and 200%
    
    def _nmc_degradation(self, cycles: float, calendar_age: float, temperature: float,
                        soc_avg: float, depth_of_discharge: float) -> Tuple[float, float]:
        """NMC degradation model"""
        activation_energy = 31000  # J/mol (higher than LFP)
        gas_constant = 8.314
        ref_temp = 298.15
        
        temp_factor = np.exp(activation_energy / gas_constant * (1/ref_temp - 1/(temperature + 273.15)))
        
        # More aggressive degradation for NMC
        cycle_fade = 0.0003 * cycles * temp_factor * (depth_of_discharge ** 0.8)
        calendar_fade = 0.0002 * np.sqrt(calendar_age * 365) * temp_factor * (soc_avg / 100) ** 0.75
        
        resistance_growth = 0.0008 * cycles * temp_factor + 0.0003 * calendar_age
        
        total_capacity_fade = cycle_fade + calendar_fade
        
        return min(total_capacity_fade, 0.4), min(resistance_growth, 3.0)
    
    def _lco_degradation(self, cycles: float, calendar_age: float, temperature: float,
                        soc_avg: float, depth_of_discharge: float) -> Tuple[float, float]:
        """LCO degradation model"""
        activation_energy = 28000  # J/mol
        gas_constant = 8.314
        ref_temp = 298.15
        
        temp_factor = np.exp(activation_energy / gas_constant * (1/ref_temp - 1/(temperature + 273.15)))
        
        # LCO is sensitive to high SoC
        soc_stress = 1.0 + 0.5 * max(0, (soc_avg - 80) / 20)  # Extra stress above 80% SoC
        
        cycle_fade = 0.00025 * cycles * temp_factor * (depth_of_discharge ** 0.6) * soc_stress
        calendar_fade = 0.00015 * np.sqrt(calendar_age * 365) * temp_factor * soc_stress
        
        resistance_growth = 0.0006 * cycles * temp_factor + 0.00025 * calendar_age
        
        total_capacity_fade = cycle_fade + calendar_fade
        
        return min(total_capacity_fade, 0.35), min(resistance_growth, 2.5)
    
    def predict_degradation(self, chemistry: str, cycles: float, calendar_age: float,
                           temperature: float, soc_avg: float, depth_of_discharge: float) -> Tuple[float, float]:
        """Predict capacity fade and resistance growth"""
        chemistry = chemistry.upper()
        
        if chemistry in self.degradation_models:
            return self.degradation_models[chemistry](
                cycles, calendar_age, temperature, soc_avg, depth_of_discharge
            )
        else:
            # Default to NMC model
            return self._nmc_degradation(cycles, calendar_age, temperature, soc_avg, depth_of_discharge)


class DigitalTwinService:
    """Main digital twin service for battery modeling"""
    
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017", 
                 redis_url: str = "redis://localhost:6379"):
        self.model_manager = PyBaMMModelManager()
        self.degradation_model = DegradationModel()
        
        # Database connections
        self.mongodb_client = None
        self.mongodb_db = None
        self.redis_client = None
        
        # Model cache
        self.model_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
            self.mongodb_db = self.mongodb_client.pulsebms
            
            self.redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            await self.redis_client.ping()
            
            logger.info("Digital Twin Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Digital Twin Service: {e}")
    
    async def simulate_battery(self, request: BatteryModelRequest) -> BatteryModelResponse:
        """Run battery simulation and return predictions"""
        try:
            # Get battery configuration
            device_config = await self._get_device_config(request.device_id)
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Returning cached result for device {request.device_id}")
                return BatteryModelResponse(**cached_result)
            
            # Run PyBaMM simulation
            simulation_results = await self._run_pybamm_simulation(request, device_config)
            
            # Calculate degradation
            degradation_results = await self._calculate_degradation(request, device_config, simulation_results)
            
            # Create response
            response = BatteryModelResponse(
                device_id=request.device_id,
                voltage_profile=simulation_results["voltage"],
                current_profile=simulation_results["current"],
                temperature_profile=simulation_results["temperature"],
                soc_profile=simulation_results["soc"],
                capacity_fade=degradation_results["capacity_fade"],
                resistance_growth=degradation_results["resistance_growth"],
                expected_eol=degradation_results["expected_eol"],
                cycle_life_remaining=degradation_results["cycle_life_remaining"],
                calendar_life_remaining=degradation_results["calendar_life_remaining"]
            )
            
            # Cache result
            await self._cache_result(cache_key, response.dict())
            
            # Store prediction in database
            await self._store_prediction(request.device_id, response)
            
            logger.info(f"Completed simulation for device {request.device_id}")
            return response
            
        except Exception as e:
            logger.error(f"Simulation failed for device {request.device_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
    
    async def _run_pybamm_simulation(self, request: BatteryModelRequest, device_config: Dict) -> Dict[str, List[float]]:
        """Run PyBaMM simulation"""
        chemistry = request.chemistry.upper()
        model, param_values, solver = self.model_manager.get_model(chemistry)
        
        # Update parameters based on device configuration
        param_values = self.model_manager.update_parameters(chemistry, {
            "Nominal cell capacity [A.h]": device_config.get("nominal_capacity", 50.0),
            "Ambient temperature [K]": request.temperature + 273.15,
            **request.model_parameters
        })
        
        try:
            # Create experiment from power profile
            experiment_steps = []
            for i, power in enumerate(request.power_profile):
                if i < len(request.time_steps):
                    duration = request.time_steps[i]
                    if power > 0:  # Discharge
                        experiment_steps.append(f"Discharge at {abs(power)}W for {duration} hours")
                    elif power < 0:  # Charge
                        experiment_steps.append(f"Charge at {abs(power)}W for {duration} hours")
                    else:  # Rest
                        experiment_steps.append(f"Rest for {duration} hours")
            
            if not experiment_steps:
                # Default rest experiment
                experiment_steps = ["Rest for 1 hour"]
            
            experiment = pybamm.Experiment(experiment_steps)
            
            # Run simulation
            sim = pybamm.Simulation(model, parameter_values=param_values, experiment=experiment)
            solution = sim.solve()
            
            # Extract results
            time_data = solution["Time [h]"].data
            voltage_data = solution["Terminal voltage [V]"].data
            current_data = solution["Current [A]"].data
            temperature_data = np.full_like(time_data, request.temperature)  # Simplified
            
            # Calculate SoC profile
            initial_soc = request.current_soc
            capacity = device_config.get("nominal_capacity", 50.0)
            soc_data = [initial_soc]
            
            for i in range(1, len(current_data)):
                dt = time_data[i] - time_data[i-1]
                charge_change = current_data[i] * dt  # Ah
                soc_change = -(charge_change / capacity) * 100  # Negative because current is positive for discharge
                new_soc = soc_data[-1] + soc_change
                soc_data.append(np.clip(new_soc, 0, 100))
            
            return {
                "voltage": voltage_data.tolist(),
                "current": current_data.tolist(),
                "temperature": temperature_data.tolist(),
                "soc": soc_data
            }
            
        except Exception as e:
            logger.error(f"PyBaMM simulation failed: {e}")
            # Return simplified fallback simulation
            return self._fallback_simulation(request)
    
    def _fallback_simulation(self, request: BatteryModelRequest) -> Dict[str, List[float]]:
        """Fallback simulation when PyBaMM fails"""
        num_points = len(request.power_profile)
        
        # Simple voltage model based on SoC
        base_voltage = 3.3 if request.chemistry == "LFP" else 3.7
        voltage_profile = [base_voltage + (request.current_soc - 50) * 0.01 for _ in range(num_points)]
        
        # Current from power and voltage
        current_profile = [p / v if v > 0 else 0 for p, v in zip(request.power_profile, voltage_profile)]
        
        # Constant temperature
        temperature_profile = [request.temperature] * num_points
        
        # SoC change based on current
        soc_profile = [request.current_soc]
        for i in range(1, num_points):
            dt = request.time_steps[i-1] if i-1 < len(request.time_steps) else 1.0
            current = current_profile[i-1]
            capacity = 50.0  # Default capacity
            soc_change = -(current * dt / capacity) * 100
            new_soc = soc_profile[-1] + soc_change
            soc_profile.append(np.clip(new_soc, 0, 100))
        
        return {
            "voltage": voltage_profile,
            "current": current_profile,
            "temperature": temperature_profile,
            "soc": soc_profile
        }
    
    async def _calculate_degradation(self, request: BatteryModelRequest, device_config: Dict, 
                                    simulation_results: Dict) -> Dict[str, Any]:
        """Calculate degradation predictions"""
        
        # Get historical data for degradation calculation
        historical_data = await self._get_historical_data(request.device_id)
        
        # Extract parameters for degradation model
        cycles = historical_data.get("total_cycles", 100.0)
        calendar_age = historical_data.get("age_years", 1.0)
        avg_temperature = np.mean(simulation_results["temperature"])
        avg_soc = np.mean(simulation_results["soc"])
        
        # Calculate depth of discharge from SoC profile
        soc_values = simulation_results["soc"]
        if len(soc_values) > 1:
            soc_range = max(soc_values) - min(soc_values)
            depth_of_discharge = soc_range
        else:
            depth_of_discharge = 20.0  # Default
        
        # Predict degradation
        capacity_fade, resistance_growth = self.degradation_model.predict_degradation(
            request.chemistry, cycles, calendar_age, avg_temperature, avg_soc, depth_of_discharge
        )
        
        # Calculate remaining life
        current_soh = request.current_soh
        remaining_capacity = (current_soh / 100) * (1 - capacity_fade)
        eol_threshold = 0.8  # 80% capacity = end of life
        
        if remaining_capacity > eol_threshold:
            # Estimate cycles until EOL
            degradation_rate = capacity_fade / cycles if cycles > 0 else 0.0001
            cycles_to_eol = (remaining_capacity - eol_threshold) / degradation_rate if degradation_rate > 0 else 10000
            cycle_life_remaining = min(cycles_to_eol / 5000.0, 1.0) * 100  # Normalize to percentage
        else:
            cycle_life_remaining = 0.0
        
        # Calendar life (simplified)
        calendar_life_remaining = max(0, (10 - calendar_age) / 10 * 100)  # Assume 10 year calendar life
        
        # Expected EOL date
        if cycle_life_remaining > 0 and cycles_to_eol < 10000:
            days_per_cycle = 1.0  # Assume 1 cycle per day
            days_to_eol = cycles_to_eol * days_per_cycle
            expected_eol = (datetime.utcnow() + timedelta(days=days_to_eol)).isoformat()
        else:
            expected_eol = None
        
        return {
            "capacity_fade": capacity_fade,
            "resistance_growth": resistance_growth,
            "expected_eol": expected_eol,
            "cycle_life_remaining": cycle_life_remaining,
            "calendar_life_remaining": calendar_life_remaining
        }
    
    async def _get_device_config(self, device_id: str) -> Dict:
        """Get device configuration from database"""
        try:
            if self.mongodb_db:
                config = await self.mongodb_db.device_configurations.find_one({"device_id": device_id})
                if config:
                    return config
        except Exception as e:
            logger.warning(f"Failed to get device config: {e}")
        
        # Return default configuration
        return {
            "device_id": device_id,
            "nominal_capacity": 50.0,
            "nominal_voltage": 400.0,
            "chemistry": "NMC"
        }
    
    async def _get_historical_data(self, device_id: str) -> Dict:
        """Get historical data for degradation calculation"""
        try:
            if self.mongodb_db:
                # Get aggregated historical data
                pipeline = [
                    {"$match": {"device_id": device_id}},
                    {"$group": {
                        "_id": None,
                        "total_records": {"$sum": 1},
                        "avg_soc": {"$avg": "$measurements.soc"},
                        "avg_temperature": {"$avg": "$measurements.temperature"}
                    }}
                ]
                
                result = await self.mongodb_db.telemetry_data.aggregate(pipeline).to_list(1)
                if result:
                    data = result[0]
                    # Estimate cycles from total records (rough approximation)
                    estimated_cycles = data["total_records"] / 3600  # Assume 1 hour per cycle
                    return {
                        "total_cycles": estimated_cycles,
                        "age_years": 1.0,  # Default
                        "avg_soc": data.get("avg_soc", 50.0),
                        "avg_temperature": data.get("avg_temperature", 25.0)
                    }
        except Exception as e:
            logger.warning(f"Failed to get historical data: {e}")
        
        return {"total_cycles": 100.0, "age_years": 1.0, "avg_soc": 50.0, "avg_temperature": 25.0}
    
    def _generate_cache_key(self, request: BatteryModelRequest) -> str:
        """Generate cache key for simulation request"""
        key_data = f"{request.device_id}_{request.chemistry}_{request.current_soc}_{request.current_soh}"
        key_data += f"_{hash(str(request.power_profile))}_{hash(str(request.time_steps))}"
        return f"simulation:{hash(key_data)}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached simulation result"""
        try:
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict):
        """Cache simulation result"""
        try:
            if self.redis_client:
                await self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(result))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _store_prediction(self, device_id: str, response: BatteryModelResponse):
        """Store prediction in database"""
        try:
            if self.mongodb_db:
                prediction_doc = {
                    "device_id": device_id,
                    "prediction_timestamp": datetime.utcnow(),
                    "horizon_hours": 24,  # Default horizon
                    "predictions": response.dict(),
                    "model_metadata": {
                        "model_type": "PyBaMM",
                        "version": "1.0.0"
                    }
                }
                
                await self.mongodb_db.digital_twin_predictions.insert_one(prediction_doc)
        except Exception as e:
            logger.warning(f"Failed to store prediction: {e}")


# FastAPI application
app = FastAPI(title="PulseBMS Digital Twin Service", version="1.0.0")
digital_twin = DigitalTwinService()


@app.on_event("startup")
async def startup_event():
    await digital_twin.initialize()


@app.post("/simulate", response_model=BatteryModelResponse)
async def simulate_battery(request: BatteryModelRequest):
    """Run battery simulation and return predictions"""
    return await digital_twin.simulate_battery(request)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "digital-twin"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
