"""
PulseBMS Enhanced - System Simulation Runner
Orchestrates the entire system loop to demonstrate functionality:
1. Initialize virtual fleet
2. Get RL actions (Charge/Discharge)
3. Simulate physics (Digital Twin)
4. Push telemetry to Backend
"""

import asyncio
import logging
import random
import time
import requests
import json
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [SIMULATION] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
BACKEND_URL = "http://localhost:8000/api/v1"
DIGITAL_TWIN_URL = "http://localhost:8001"
NUM_DEVICES = 5
SIMULATION_SPEED = 1.0  # Seconds per step

class SystemSimulator:
    def __init__(self):
        self.devices = []
        self.market_price = 0.12
        
    def initialize_fleet(self):
        """Create virtual devices in the backend"""
        logger.info("Initializing virtual fleet...")
        
        chemistries = ["LFP", "NMC", "LCO"]
        
        for i in range(NUM_DEVICES):
            device_id = f"sim_battery_{i+1:03d}"
            chemistry = chemistries[i % len(chemistries)]
            
            # 1. Register Device
            device_payload = {
                "device_id": device_id,
                "device_type": "battery_pack",
                "location": {"site": "Simulation_Lab", "rack": f"R-{i+1}"},
                "specifications": {
                    "chemistry": chemistry,
                    "capacity_kwh": 50.0 + (i * 5),
                    "max_power": 5000.0 + (i * 1000),
                    "voltage_nominal": 400.0
                },
                "safety_constraints": {
                    "min_soc": 10.0,
                    "max_soc": 90.0,
                    "max_current": 100.0,
                    "max_temperature": 45.0
                }
            }
            
            try:
                # Check if exists first (optional, or just ignore error)
                requests.post(f"{BACKEND_URL}/devices/register", json=device_payload)
                
                # Initialize local state
                self.devices.append({
                    "device_id": device_id,
                    "chemistry": chemistry,
                    "soc": 50.0 + random.uniform(-10, 10),
                    "soh": 95.0 - (i * 2),
                    "voltage": 400.0,
                    "current": 0.0,
                    "temperature": 25.0,
                    "power_capability": device_payload["specifications"]["max_power"]
                })
                logger.info(f"Registered device {device_id} ({chemistry})")
                
            except Exception as e:
                logger.error(f"Failed to register {device_id}: {e}")

    def update_market(self):
        """Simulate changing electricity prices"""
        # Simple sine wave price model
        base = 0.12
        var = 0.05
        now = time.time()
        self.market_price = base + var * np.sin(now / 10.0)

    def run_step(self):
        """Execute one simulation step for all devices"""
        self.update_market()
        
        for device in self.devices:
            # 1. Get Action (Simulate RL Policy Decision)
            # In a real deployment, we'd call the RL service. Here we simulate 'intelligent' decisions.
            action = self._get_rl_action(device)
            target_power = action * device["power_capability"]
            
            # 2. Simulate Physics (Digital Twin)
            # We use the Digital Twin service to get the 'next state' based on physics
            next_state = self._simulate_physics(device, target_power)
            
            if next_state:
                # Update local state
                device.update(next_state)
            
            # 3. Push Telemetry
            self._push_telemetry(device)
            
    def _get_rl_action(self, device):
        """
        Simulate RL Policy:
        - If Price High -> Discharge (Sell)
        - If Price Low -> Charge (Buy)
        - Respect SoC Constraints
        """
        soc = device["soc"]
        price = self.market_price
        
        # Simple threshold policy acting as 'trained agent'
        if soc > 80: return 1.0 # Force Discharge
        if soc < 20: return -1.0 # Force Charge
        
        if price > 0.14:
            return 0.8 # Sell
        elif price < 0.10:
            return -0.8 # Buy
        return 0.0 # Hold

    def _simulate_physics(self, device, power):
        """Call Digital Twin Service or Fallback"""
        try:
            payload = {
                "device_id": device["device_id"],
                "chemistry": device["chemistry"],
                "current_soc": device["soc"],
                "current_soh": device["soh"],
                "temperature": device["temperature"],
                "power_profile": [power],
                "time_steps": [5.0/60.0], # 5 minutes step
                "model_parameters": {}
            }
            
            # Using timeout to avoid blocking
            resp = requests.post(f"{DIGITAL_TWIN_URL}/simulate", json=payload, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                # Return the FINAL state from the simulation profile
                return {
                    "soc": data["soc_profile"][-1],
                    "voltage": data["voltage_profile"][-1],
                    "current": data["current_profile"][-1],
                    "temperature": data["temperature_profile"][-1],
                    # Keep SoH slowly degrading
                    "soh": device["soh"] - (data["capacity_fade"] * 100) # Simplified
                }
        except Exception as e:
            # logger.warning(f"Digital Twin unavailable, using simple physics: {e}")
            pass
            
        # Fallback Physics
        dt = 5.0 / 60.0 # hours
        capacity = 50.0 * 1000 # Wh
        energy = power * dt
        soc_change = -(energy / capacity) * 100
        new_soc = min(100, max(0, device["soc"] + soc_change))
        
        # Temp rise
        temp_change = (abs(power) * 0.05 * dt) / 10.0
        new_temp = device["temperature"] + temp_change - ((device["temperature"] - 25) * 0.1)
        
        return {
            "soc": new_soc,
            "voltage": 400.0 + (new_soc - 50)*0.5,
            "current": power / 400.0,
            "temperature": new_temp,
            "soh": device["soh"]
        }

    def _push_telemetry(self, device):
        """Send data to backend"""
        telemetry = {
            "device_id": device["device_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "measurements": {
                "soc": device["soc"],
                "soh": device["soh"],
                "voltage": device["voltage"],
                "current": device["current"],
                "temperature": device["temperature"],
                "power": device["voltage"] * device["current"]
            },
            "alarms": [],
            "warnings": []
        }
        
        # Add simulated alarms
        if device["temperature"] > 40:
            telemetry["warnings"].append("High Temperature")
        
        try:
            requests.post(f"{BACKEND_URL}/telemetry/store", json=telemetry, timeout=1)
        except Exception:
            pass

    def run(self):
        self.initialize_fleet()
        logger.info("Simulation loop started. Press Ctrl+C to stop.")
        
        step = 0
        while True:
            try:
                start = time.time()
                self.run_step()
                
                # Log occasional status
                if step % 5 == 0:
                    logger.info(f"Step {step}: Market Price=${self.market_price:.3f}/kWh")
                    for d in self.devices:
                        logger.info(f"  [{d['device_id']}] SoC:{d['soc']:.1f}% Temp:{d['temperature']:.1f}C Power:{d['current']*d['voltage']:.0f}W")
                
                step += 1
                time.sleep(SIMULATION_SPEED)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Simulation Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    # Wait for services to potentially come up
    print("Waiting 5s for services...")
    time.sleep(5)
    
    sim = SystemSimulator()
    sim.run()
