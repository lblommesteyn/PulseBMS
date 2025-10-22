"""
PulseBMS Enhanced - Reinforcement Learning (RL) Optimization Policy
Safe RL policy for battery power allocation with safety constraints and degradation awareness
"""

import asyncio
import logging
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BatteryState:
    """Battery state for RL environment"""
    device_id: str
    soc: float              # State of charge (%)
    soh: float              # State of health (%)
    voltage: float          # Terminal voltage (V)
    current: float          # Current (A)
    temperature: float      # Temperature (°C)
    power_capability: float # Max power capability (W)
    energy_capacity: float  # Energy capacity (Wh)
    chemistry: str          # Battery chemistry
    cycles: float           # Total cycles
    calendar_age: float     # Calendar age (years)


@dataclass
class MarketSignal:
    """Market signals for price-aware optimization"""
    timestamp: datetime
    electricity_price: float    # $/kWh
    demand_charge: float       # $/kW
    frequency_regulation: float # $/kW
    reserve_capacity: float    # $/kW
    carbon_intensity: float    # kg CO2/kWh


@dataclass
class SafetyViolation:
    """Safety violation tracking"""
    device_id: str
    violation_type: str
    severity: float
    timestamp: datetime
    value: float
    limit: float


class SafetyLayer:
    """Safety layer that enforces hard constraints on RL actions"""
    
    def __init__(self):
        self.violation_history = []
        self.emergency_stop_threshold = 3
        
    def check_action_safety(self, action: np.ndarray, battery_states: List[BatteryState],
                           safety_constraints: Dict[str, Any]) -> Tuple[np.ndarray, List[SafetyViolation]]:
        """Check and modify actions to ensure safety"""
        
        safe_action = action.copy()
        violations = []
        
        for i, (power_allocation, state) in enumerate(zip(action, battery_states)):
            device_constraints = safety_constraints.get(state.device_id, {})
            
            # Check SoC limits with power projection
            projected_soc = self._project_soc(state, power_allocation, horizon_minutes=15)
            
            if projected_soc < device_constraints.get("min_soc", 10.0):
                violation = SafetyViolation(
                    device_id=state.device_id,
                    violation_type="soc_low",
                    severity=1.0,
                    timestamp=datetime.utcnow(),
                    value=projected_soc,
                    limit=device_constraints.get("min_soc", 10.0)
                )
                violations.append(violation)
                safe_action[i] = min(safe_action[i], 0)  # No discharge
                
            elif projected_soc > device_constraints.get("max_soc", 90.0):
                safe_action[i] = max(safe_action[i], 0)  # No charge
            
            # Check power limits
            max_power = min(state.power_capability, device_constraints.get("max_power_discharge", 5000.0))
            min_power = -min(state.power_capability, device_constraints.get("max_power_charge", 5000.0))
            
            if power_allocation > max_power:
                safe_action[i] = max_power
            elif power_allocation < min_power:
                safe_action[i] = min_power
        
        self.violation_history.extend(violations)
        return safe_action, violations
    
    def _project_soc(self, state: BatteryState, power: float, horizon_minutes: float) -> float:
        """Project SoC given power allocation over time horizon"""
        capacity_ah = state.energy_capacity / state.voltage
        energy_change_wh = power * (horizon_minutes / 60)
        soc_change = -(energy_change_wh / state.energy_capacity) * 100
        return state.soc + soc_change
    
    def get_violation_penalty(self) -> float:
        """Calculate penalty based on recent violations"""
        recent_violations = [v for v in self.violation_history 
                           if (datetime.utcnow() - v.timestamp).total_seconds() < 3600]
        
        if not recent_violations:
            return 0.0
        
        total_penalty = sum(v.severity for v in recent_violations)
        return min(total_penalty, 10.0)


class BatteryFleetEnvironment(gym.Env):
    """Gymnasium environment for battery fleet optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.max_batteries = config.get("max_batteries", 10)
        self.episode_length = config.get("episode_length", 288)
        self.dt_minutes = config.get("dt_minutes", 5.0)
        
        # Action space: power allocation for each battery (-1 to 1, normalized)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.max_batteries,), 
            dtype=np.float32
        )
        
        # Observation space
        obs_dim = self.max_batteries * 8 + 5 + 3  # Battery states + market + time
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize environment state
        self.battery_states = []
        self.market_signals = []
        self.safety_layer = SafetyLayer()
        self.current_step = 0
        self.episode_return = 0.0
        
        # Performance tracking
        self.total_energy_delivered = 0.0
        self.total_degradation_cost = 0.0
        self.total_safety_violations = 0
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_return = 0.0
        self.total_energy_delivered = 0.0
        self.total_degradation_cost = 0.0
        self.total_safety_violations = 0
        
        # Initialize battery states
        self.battery_states = self._generate_initial_battery_states()
        
        # Generate market signals for episode
        self.market_signals = self._generate_market_signals()
        
        # Clear safety history
        self.safety_layer.violation_history = []
        
        observation = self._get_observation()
        info = {"episode_start": True}
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """Execute one step in the environment"""
        
        # Apply safety layer
        safe_action, violations = self.safety_layer.check_action_safety(
            action, self.battery_states, self._get_safety_constraints()
        )
        
        # Convert normalized actions to actual power allocations
        power_allocations = self._denormalize_actions(safe_action)
        
        # Simulate battery dynamics
        self._update_battery_states(power_allocations)
        
        # Calculate reward
        reward = self._calculate_reward(power_allocations, violations)
        
        # Update environment state
        self.current_step += 1
        self.episode_return += reward
        self.total_safety_violations += len(violations)
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Get next observation
        observation = self._get_observation()
        
        # Prepare info
        info = {
            "power_allocations": power_allocations.tolist(),
            "safety_violations": len(violations),
            "total_energy": self.total_energy_delivered,
            "degradation_cost": self.total_degradation_cost,
            "episode_return": self.episode_return
        }
        
        if done:
            info["episode_metrics"] = self._get_episode_metrics()
        
        return observation, reward, done, False, info
    
    def _generate_initial_battery_states(self) -> List[BatteryState]:
        """Generate initial battery states with random variation"""
        
        states = []
        chemistries = ["LFP", "NMC", "LCO"]
        
        for i in range(self.max_batteries):
            chemistry = np.random.choice(chemistries)
            
            state = BatteryState(
                device_id=f"battery_{i:03d}",
                soc=np.random.uniform(20, 80),
                soh=np.random.uniform(70, 95),
                voltage=np.random.uniform(350, 420),
                current=0.0,
                temperature=np.random.uniform(20, 30),
                power_capability=np.random.uniform(3000, 8000),
                energy_capacity=np.random.uniform(30000, 80000),
                chemistry=chemistry,
                cycles=np.random.uniform(50, 500),
                calendar_age=np.random.uniform(0.5, 3.0)
            )
            states.append(state)
        
        return states
    
    def _generate_market_signals(self) -> List[MarketSignal]:
        """Generate market signals for the episode"""
        
        signals = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        for step in range(self.episode_length):
            timestamp = base_time + timedelta(minutes=step * self.dt_minutes)
            hour = timestamp.hour
            
            # Realistic electricity price profile
            if 6 <= hour <= 9 or 17 <= hour <= 21:  # Peak hours
                base_price = 0.15
            elif 22 <= hour <= 5:  # Off-peak
                base_price = 0.08
            else:  # Mid-peak
                base_price = 0.12
            
            price_noise = np.random.normal(0, 0.02)
            electricity_price = max(0.05, base_price + price_noise)
            
            signal = MarketSignal(
                timestamp=timestamp,
                electricity_price=electricity_price,
                demand_charge=np.random.uniform(10, 20),
                frequency_regulation=np.random.uniform(15, 25),
                reserve_capacity=np.random.uniform(5, 15),
                carbon_intensity=np.random.uniform(0.4, 0.8)
            )
            signals.append(signal)
        
        return signals
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        
        obs = []
        
        # Battery states (normalized)
        for i in range(self.max_batteries):
            if i < len(self.battery_states):
                state = self.battery_states[i]
                obs.extend([
                    state.soc / 100.0,
                    state.soh / 100.0,
                    (state.voltage - 300) / 200.0,
                    np.tanh(state.current / 100.0),
                    (state.temperature - 25) / 25.0,
                    state.power_capability / 10000.0,
                    state.energy_capacity / 100000.0,
                    state.cycles / 1000.0
                ])
            else:
                obs.extend([0.0] * 8)
        
        # Market signals
        if self.current_step < len(self.market_signals):
            signal = self.market_signals[self.current_step]
            obs.extend([
                signal.electricity_price / 0.3,
                signal.demand_charge / 30.0,
                signal.frequency_regulation / 30.0,
                signal.reserve_capacity / 20.0,
                signal.carbon_intensity / 1.0
            ])
        else:
            obs.extend([0.0] * 5)
        
        # Time features
        if self.current_step < len(self.market_signals):
            timestamp = self.market_signals[self.current_step].timestamp
            hour = timestamp.hour / 24.0
            day_of_week = timestamp.weekday() / 7.0
            month = timestamp.month / 12.0
            obs.extend([hour, day_of_week, month])
        else:
            obs.extend([0.0, 0.0, 0.0])
        
        return np.array(obs, dtype=np.float32)
    
    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Convert normalized actions to actual power allocations"""
        
        power_allocations = np.zeros(len(self.battery_states))
        
        for i, (action, state) in enumerate(zip(actions, self.battery_states)):
            if i < len(self.battery_states):
                power_allocations[i] = action * state.power_capability
        
        return power_allocations
    
    def _update_battery_states(self, power_allocations: np.ndarray):
        """Update battery states based on power allocations"""
        
        for i, (state, power) in enumerate(zip(self.battery_states, power_allocations)):
            if i >= len(self.battery_states):
                break
                
            # Update SoC based on power
            dt_hours = self.dt_minutes / 60.0
            capacity_ah = state.energy_capacity / state.voltage
            current = power / state.voltage if state.voltage > 0 else 0
            
            efficiency = 0.95 if power < 0 else 0.92
            effective_current = current * efficiency if power < 0 else current / efficiency
            
            soc_change = -(effective_current * dt_hours) / capacity_ah * 100
            state.soc = np.clip(state.soc + soc_change, 0, 100)
            
            # Update other states
            state.current = current
            
            # Simple voltage model
            soc_factor = (state.soc - 50) / 50
            chemistry_base = {"LFP": 3.3, "NMC": 3.7, "LCO": 3.6}.get(state.chemistry, 3.7)
            state.voltage = chemistry_base + soc_factor * 0.3 - abs(current) * 0.01
            
            # Update temperature
            power_loss = abs(power) * 0.05
            temp_rise = power_loss / 1000.0
            state.temperature = 25.0 + temp_rise
            
            # Update cycles
            if abs(current) > 1.0:
                state.cycles += dt_hours / 2.0
        
        # Track total energy delivered
        total_power = sum(power_allocations)
        energy_delivered = total_power * (self.dt_minutes / 60.0) / 1000.0
        self.total_energy_delivered += abs(energy_delivered)
    
    def _calculate_reward(self, power_allocations: np.ndarray, violations: List[SafetyViolation]) -> float:
        """Calculate reward for current step"""
        
        reward = 0.0
        
        # Economic reward
        if self.current_step < len(self.market_signals):
            signal = self.market_signals[self.current_step]
            total_power_kw = sum(power_allocations) / 1000.0
            
            dt_hours = self.dt_minutes / 60.0
            energy_kwh = total_power_kw * dt_hours
            revenue = energy_kwh * signal.electricity_price
            
            if total_power_kw > 0:  # Discharging
                grid_service_bonus = total_power_kw * signal.frequency_regulation * dt_hours
                revenue += grid_service_bonus
            
            reward += revenue * 100
        
        # SoC balancing reward
        if len(self.battery_states) > 1:
            soc_values = [state.soc for state in self.battery_states]
            soc_std = np.std(soc_values)
            reward -= soc_std * 0.1
        
        # Degradation penalty
        degradation_cost = 0.0
        for state, power in zip(self.battery_states, power_allocations):
            optimal_soc = 60.0
            soc_stress = abs(state.soc - optimal_soc) / 40.0
            power_stress = abs(power) / state.power_capability
            health_factor = (1 - state.soh / 100) + 0.1
            
            step_degradation = soc_stress * power_stress * health_factor * 0.01
            degradation_cost += step_degradation
        
        self.total_degradation_cost += degradation_cost
        reward -= degradation_cost * 50
        
        # Safety penalty
        safety_penalty = self.safety_layer.get_violation_penalty()
        reward -= safety_penalty * 20
        
        return reward
    
    def _get_safety_constraints(self) -> Dict[str, Any]:
        """Get safety constraints for all batteries"""
        
        constraints = {}
        for state in self.battery_states:
            constraints[state.device_id] = {
                "min_soc": 15.0,
                "max_soc": 85.0,
                "max_power_discharge": state.power_capability,
                "max_power_charge": state.power_capability,
                "max_temperature": 45.0
            }
        
        return constraints
    
    def _get_episode_metrics(self) -> Dict[str, float]:
        """Get episode performance metrics"""
        
        return {
            "total_energy_delivered_kwh": self.total_energy_delivered,
            "total_degradation_cost": self.total_degradation_cost,
            "total_safety_violations": self.total_safety_violations,
            "episode_return": self.episode_return,
            "avg_soc": np.mean([state.soc for state in self.battery_states]),
            "soc_std": np.std([state.soc for state in self.battery_states]),
            "avg_soh": np.mean([state.soh for state in self.battery_states])
        }


class RLPolicyTrainer:
    """RL policy trainer with safety constraints"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.env = None
        
    def create_environment(self) -> BatteryFleetEnvironment:
        """Create training environment"""
        
        env_config = {
            "max_batteries": self.config.get("max_batteries", 5),
            "episode_length": self.config.get("episode_length", 288),
            "dt_minutes": self.config.get("dt_minutes", 5.0)
        }
        
        return BatteryFleetEnvironment(env_config)
    
    def train_policy(self, total_timesteps: int = 100000, model_type: str = "PPO") -> None:
        """Train RL policy"""
        
        logger.info(f"Starting RL policy training with {model_type}")
        
        # Create environment
        self.env = self.create_environment()
        
        # Create model
        if model_type == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1
            )
        elif model_type == "SAC":
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                verbose=1
            )
        
        # Train model
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        logger.info("RL policy training completed")
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str, model_type: str = "PPO"):
        """Load trained model"""
        if model_type == "PPO":
            self.model = PPO.load(path)
        elif model_type == "SAC":
            self.model = SAC.load(path)
        
        logger.info(f"Model loaded from {path}")


class RLPolicyDeployment:
    """Deployment wrapper for trained RL policy"""
    
    def __init__(self, model_path: str, model_type: str = "PPO"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.safety_layer = SafetyLayer()
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        try:
            if self.model_type == "PPO":
                self.model = PPO.load(self.model_path)
            elif self.model_type == "SAC":
                self.model = SAC.load(self.model_path)
            
            logger.info(f"RL model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            self.model = None
    
    async def get_power_allocation(self, battery_states: List[Dict], market_data: Dict) -> Dict[str, float]:
        """Get power allocation from RL policy"""
        
        if not self.model:
            logger.warning("RL model not loaded, using fallback allocation")
            return self._fallback_allocation(battery_states)
        
        try:
            # Convert battery states to observation
            observation = self._create_observation(battery_states, market_data)
            
            # Get action from RL policy
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Apply safety layer
            safe_action, violations = self.safety_layer.check_action_safety(
                action, self._convert_to_battery_states(battery_states), {}
            )
            
            # Convert to power allocations
            allocations = {}
            for i, state in enumerate(battery_states[:len(safe_action)]):
                power_capability = state.get("power_capability", 5000.0)
                allocations[state["device_id"]] = float(safe_action[i] * power_capability)
            
            return allocations
            
        except Exception as e:
            logger.error(f"RL policy prediction failed: {e}")
            return self._fallback_allocation(battery_states)
    
    def _create_observation(self, battery_states: List[Dict], market_data: Dict) -> np.ndarray:
        """Create observation for RL model"""
        
        obs = []
        max_batteries = 10  # Match training config
        
        # Battery states
        for i in range(max_batteries):
            if i < len(battery_states):
                state = battery_states[i]
                obs.extend([
                    state.get("soc", 50.0) / 100.0,
                    state.get("soh", 80.0) / 100.0,
                    (state.get("voltage", 400.0) - 300) / 200.0,
                    np.tanh(state.get("current", 0.0) / 100.0),
                    (state.get("temperature", 25.0) - 25) / 25.0,
                    state.get("power_capability", 5000.0) / 10000.0,
                    state.get("energy_capacity", 50000.0) / 100000.0,
                    state.get("cycles", 100.0) / 1000.0
                ])
            else:
                obs.extend([0.0] * 8)
        
        # Market signals
        obs.extend([
            market_data.get("electricity_price", 0.12) / 0.3,
            market_data.get("demand_charge", 15.0) / 30.0,
            market_data.get("frequency_regulation", 20.0) / 30.0,
            market_data.get("reserve_capacity", 10.0) / 20.0,
            market_data.get("carbon_intensity", 0.6) / 1.0
        ])
        
        # Time features
        now = datetime.utcnow()
        obs.extend([
            now.hour / 24.0,
            now.weekday() / 7.0,
            now.month / 12.0
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _convert_to_battery_states(self, battery_states: List[Dict]) -> List[BatteryState]:
        """Convert dict battery states to BatteryState objects"""
        
        states = []
        for state_dict in battery_states:
            state = BatteryState(
                device_id=state_dict["device_id"],
                soc=state_dict.get("soc", 50.0),
                soh=state_dict.get("soh", 80.0),
                voltage=state_dict.get("voltage", 400.0),
                current=state_dict.get("current", 0.0),
                temperature=state_dict.get("temperature", 25.0),
                power_capability=state_dict.get("power_capability", 5000.0),
                energy_capacity=state_dict.get("energy_capacity", 50000.0),
                chemistry=state_dict.get("chemistry", "NMC"),
                cycles=state_dict.get("cycles", 100.0),
                calendar_age=state_dict.get("calendar_age", 1.0)
            )
            states.append(state)
        
        return states
    
    def _fallback_allocation(self, battery_states: List[Dict]) -> Dict[str, float]:
        """Fallback allocation when RL fails"""
        
        allocations = {}
        for state in battery_states:
            # Simple allocation based on SoC
            soc = state.get("soc", 50.0)
            power_capability = state.get("power_capability", 5000.0)
            
            if soc > 60:  # High SoC, prefer discharge
                allocations[state["device_id"]] = power_capability * 0.3
            elif soc < 40:  # Low SoC, prefer charge
                allocations[state["device_id"]] = -power_capability * 0.3
            else:  # Balanced SoC
                allocations[state["device_id"]] = 0.0
        
        return allocations


# Example usage
if __name__ == "__main__":
    # Training example
    config = {
        "max_batteries": 5,
        "episode_length": 288,
        "dt_minutes": 5.0,
        "log_path": "./logs/rl_training/"
    }
    
    trainer = RLPolicyTrainer(config)
    trainer.train_policy(total_timesteps=50000, model_type="PPO")
    trainer.save_model("./models/rl_policy_ppo.zip")
