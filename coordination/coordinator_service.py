"""
PulseBMS Enhanced - Coordinator Service
Integrates MPC, RL, and Digital Twin services for optimal power allocation
"""

import asyncio
import logging
import json
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import motor.motor_asyncio
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    MPC_ONLY = "mpc_only"
    RL_ONLY = "rl_only"
    HYBRID_MPC_RL = "hybrid_mpc_rl"
    ADAPTIVE = "adaptive"


@dataclass
class AllocationDecision:
    """Power allocation decision with metadata"""
    device_id: str
    allocated_power_w: float
    strategy_used: str
    confidence_score: float
    safety_score: float
    expected_soc: float
    expected_degradation: float
    timestamp: datetime


@dataclass
class CoordinationMetrics:
    """Performance metrics for coordination"""
    total_energy_delivered_kwh: float
    total_revenue_dollars: float
    total_degradation_cost: float
    safety_violations_count: int
    strategy_performance: Dict[str, float]
    fleet_efficiency: float
    soc_balance_score: float


class DigitalTwinClient:
    """Client for digital twin service"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def get_battery_forecast(self, device_id: str, current_state: Dict, 
                                 power_profile: List[float], time_steps: List[float]) -> Optional[Dict]:
        """Get battery forecast from digital twin"""
        
        if not self.session:
            await self.initialize()
        
        try:
            request_data = {
                "device_id": device_id,
                "chemistry": current_state.get("chemistry", "NMC"),
                "current_soc": current_state.get("soc", 50.0),
                "current_soh": current_state.get("soh", 80.0),
                "temperature": current_state.get("temperature", 25.0),
                "power_profile": power_profile,
                "time_steps": time_steps,
                "model_parameters": {}
            }
            
            async with self.session.post(
                f"{self.base_url}/simulate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Digital twin forecast received for {device_id}")
                    return result
                else:
                    logger.warning(f"Digital twin request failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Digital twin client error: {e}")
            return None


class SafetyMonitor:
    """Safety monitoring and emergency stop functionality"""
    
    def __init__(self, emergency_stop_threshold: int = 5):
        self.emergency_stop_threshold = emergency_stop_threshold
        self.violation_history = []
        self.emergency_stop_active = False
        self.last_safety_check = datetime.utcnow()
    
    def check_safety_constraints(self, allocations: List[AllocationDecision], 
                               battery_states: List[Dict]) -> tuple[bool, List[str]]:
        """Check if allocations satisfy safety constraints"""
        
        violations = []
        
        for allocation, state in zip(allocations, battery_states):
            device_id = allocation.device_id
            
            # Check SoC projection
            if allocation.expected_soc < 10.0:
                violations.append(f"{device_id}: SoC too low ({allocation.expected_soc:.1f}%)")
            elif allocation.expected_soc > 95.0:
                violations.append(f"{device_id}: SoC too high ({allocation.expected_soc:.1f}%)")
            
            # Check power limits
            max_power = state.get("power_capability", 5000.0) * 0.9  # 10% safety margin
            if abs(allocation.allocated_power_w) > max_power:
                violations.append(f"{device_id}: Power limit exceeded ({allocation.allocated_power_w:.0f}W > {max_power:.0f}W)")
            
            # Check temperature
            current_temp = state.get("temperature", 25.0)
            if current_temp > 40.0:
                violations.append(f"{device_id}: Temperature too high ({current_temp:.1f}°C)")
        
        # Track violations
        self.violation_history.extend(violations)
        
        # Check for emergency stop condition
        recent_violations = [v for v in self.violation_history 
                           if (datetime.utcnow() - self.last_safety_check).total_seconds() < 3600]
        
        if len(recent_violations) >= self.emergency_stop_threshold:
            self.emergency_stop_active = True
            logger.critical(f"EMERGENCY STOP ACTIVATED: {len(recent_violations)} safety violations in last hour")
        
        self.last_safety_check = datetime.utcnow()
        
        return len(violations) == 0, violations
    
    def reset_emergency_stop(self):
        """Reset emergency stop (manual intervention required)"""
        self.emergency_stop_active = False
        self.violation_history = []
        logger.info("Emergency stop reset by operator")


class PerformanceTracker:
    """Track and analyze performance of different optimization strategies"""
    
    def __init__(self):
        self.strategy_metrics = {}
        self.recent_decisions = []
        self.performance_window = 50
    
    def record_decision(self, decisions: List[AllocationDecision], actual_results: Dict[str, Any]):
        """Record allocation decision and its results"""
        
        for decision in decisions:
            strategy = decision.strategy_used
            
            if strategy not in self.strategy_metrics:
                self.strategy_metrics[strategy] = {
                    "total_decisions": 0,
                    "successful_decisions": 0,
                    "total_energy": 0.0,
                    "total_revenue": 0.0,
                    "safety_violations": 0
                }
            
            metrics = self.strategy_metrics[strategy]
            metrics["total_decisions"] += 1
            
            # Determine if decision was successful
            actual_power = actual_results.get("power_delivered", {}).get(decision.device_id, 0.0)
            power_error = abs(actual_power - decision.allocated_power_w) / max(abs(decision.allocated_power_w), 1.0)
            
            if power_error < 0.1:  # Within 10% of target
                metrics["successful_decisions"] += 1
            
            # Update metrics
            metrics["total_energy"] += abs(actual_power) * (5.0 / 60.0) / 1000.0  # kWh
            metrics["total_revenue"] += actual_results.get("revenue", 0.0)
            
            if actual_results.get("safety_violations", 0) > 0:
                metrics["safety_violations"] += 1
        
        # Store recent decisions
        self.recent_decisions.extend(decisions)
        if len(self.recent_decisions) > self.performance_window:
            self.recent_decisions = self.recent_decisions[-self.performance_window:]
    
    def get_strategy_performance(self, strategy: str) -> float:
        """Get performance score for a strategy (0-1)"""
        
        if strategy not in self.strategy_metrics:
            return 0.5  # Default neutral score
        
        metrics = self.strategy_metrics[strategy]
        
        if metrics["total_decisions"] == 0:
            return 0.5
        
        # Calculate success rate
        success_rate = metrics["successful_decisions"] / metrics["total_decisions"]
        
        # Calculate safety score
        safety_score = 1.0 - (metrics["safety_violations"] / metrics["total_decisions"])
        
        # Weighted combination
        performance_score = 0.7 * success_rate + 0.3 * safety_score
        
        return np.clip(performance_score, 0.0, 1.0)
    
    def get_best_strategy(self) -> str:
        """Get the best performing strategy"""
        
        if not self.strategy_metrics:
            return OptimizationStrategy.MPC_ONLY.value
        
        best_strategy = max(self.strategy_metrics.keys(), 
                          key=lambda s: self.get_strategy_performance(s))
        
        return best_strategy


class CoordinatorService:
    """Main coordinator service that orchestrates all optimization components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Service configuration
        self.digital_twin_url = self.config.get("digital_twin_url", "http://localhost:8001")
        self.backend_api_url = self.config.get("backend_api_url", "http://localhost:8000")
        self.optimization_strategy = OptimizationStrategy(
            self.config.get("optimization_strategy", "mpc_only")
        )
        
        # Initialize components
        self.digital_twin_client = DigitalTwinClient(self.digital_twin_url)
        self.safety_monitor = SafetyMonitor()
        self.performance_tracker = PerformanceTracker()
        
        # Database connections
        self.mongodb_client = None
        self.mongodb_db = None
        self.redis_client = None
        
        # State tracking
        self.last_allocation_time = None
        self.current_allocations = {}
        self.fleet_state = {}
        
        # Performance metrics
        self.metrics = CoordinationMetrics(
            total_energy_delivered_kwh=0.0,
            total_revenue_dollars=0.0,
            total_degradation_cost=0.0,
            safety_violations_count=0,
            strategy_performance={},
            fleet_efficiency=0.0,
            soc_balance_score=0.0
        )
    
    async def initialize(self):
        """Initialize coordinator service"""
        
        logger.info("Initializing Coordinator Service...")
        
        try:
            # Initialize database connections
            mongodb_url = self.config.get("mongodb_url", "mongodb://localhost:27017")
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
            self.mongodb_db = self.mongodb_client.pulsebms
            
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Initialize digital twin client
            await self.digital_twin_client.initialize()
            
            logger.info("Coordinator Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Coordinator Service: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown coordinator service"""
        
        logger.info("Shutting down Coordinator Service...")
        
        if self.digital_twin_client:
            await self.digital_twin_client.close()
        
        if self.mongodb_client:
            self.mongodb_client.close()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def coordinate_power_allocation(self, fleet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main coordination function - orchestrates all optimization strategies"""
        
        try:
            current_time = datetime.utcnow()
            
            # Check if allocation is needed
            allocation_interval = self.config.get("allocation_interval_seconds", 300)  # 5 minutes
            if (self.last_allocation_time and 
                (current_time - self.last_allocation_time).total_seconds() < allocation_interval):
                return {"status": "cached", "message": "Using previous allocation"}
            
            # Check emergency stop
            if self.safety_monitor.emergency_stop_active:
                return await self._handle_emergency_stop()
            
            # Extract fleet information
            battery_states = self._extract_battery_states(fleet_data)
            power_demand = self._extract_power_demand(fleet_data)
            market_data = self._extract_market_data(fleet_data)
            
            if not battery_states:
                logger.warning("No battery states available for coordination")
                return {"status": "error", "message": "No batteries available"}
            
            # Get forecasts from digital twin (if needed)
            forecasts = await self._get_digital_twin_forecasts(battery_states, power_demand)
            
            # Determine optimization strategy
            strategy = self._select_optimization_strategy()
            
            # Generate allocations based on strategy
            allocations = await self._generate_allocations(
                strategy, battery_states, power_demand, market_data, forecasts
            )
            
            # Apply safety checks
            safe_allocations, safety_violations = await self._apply_safety_checks(allocations, battery_states)
            
            # Record decision for performance tracking
            self._record_allocation_decision(safe_allocations, strategy)
            
            # Update state
            self.last_allocation_time = current_time
            self.current_allocations = {alloc.device_id: alloc.allocated_power_w for alloc in safe_allocations}
            
            # Prepare response
            response = {
                "status": "success",
                "timestamp": current_time.isoformat(),
                "strategy_used": strategy.value,
                "total_allocated_kw": sum(alloc.allocated_power_w for alloc in safe_allocations) / 1000.0,
                "allocations": {alloc.device_id: {
                    "power_setpoint_w": alloc.allocated_power_w,
                    "expected_soc": alloc.expected_soc,
                    "confidence": alloc.confidence_score,
                    "safety_score": alloc.safety_score
                } for alloc in safe_allocations},
                "safety_violations": safety_violations,
                "forecast_horizon_hours": 6,
                "performance_metrics": self._get_current_metrics()
            }
            
            # Store coordination result
            await self._store_coordination_result(response)
            
            logger.info(f"Power allocation completed using {strategy.value} strategy: "
                       f"{len(safe_allocations)} batteries, "
                       f"{sum(alloc.allocated_power_w for alloc in safe_allocations)/1000:.2f} kW total")
            
            return response
            
        except Exception as e:
            logger.error(f"Coordination failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _handle_emergency_stop(self) -> Dict[str, Any]:
        """Handle emergency stop condition"""
        
        logger.critical("Emergency stop active - all allocations set to zero")
        
        return {
            "status": "emergency_stop",
            "message": "Emergency stop active due to safety violations",
            "timestamp": datetime.utcnow().isoformat(),
            "allocations": {},
            "action_required": "Manual intervention required to reset emergency stop"
        }
    
    def _select_optimization_strategy(self) -> OptimizationStrategy:
        """Select the best optimization strategy based on current conditions"""
        
        if self.optimization_strategy == OptimizationStrategy.ADAPTIVE:
            # Use performance-based adaptive strategy selection
            best_strategy_name = self.performance_tracker.get_best_strategy()
            
            # Map string back to enum
            strategy_map = {
                "mpc_only": OptimizationStrategy.MPC_ONLY,
                "rl_only": OptimizationStrategy.RL_ONLY,
                "hybrid_mpc_rl": OptimizationStrategy.HYBRID_MPC_RL
            }
            
            return strategy_map.get(best_strategy_name, OptimizationStrategy.MPC_ONLY)
        else:
            return self.optimization_strategy
    
    async def _generate_allocations(self, strategy: OptimizationStrategy, 
                                  battery_states: List[Dict], power_demand: Dict,
                                  market_data: Dict, forecasts: Dict) -> List[AllocationDecision]:
        """Generate power allocations using selected strategy"""
        
        allocations = []
        
        if strategy == OptimizationStrategy.MPC_ONLY:
            allocations = await self._generate_mpc_allocations(battery_states, power_demand, forecasts)
            
        elif strategy == OptimizationStrategy.RL_ONLY:
            allocations = await self._generate_rl_allocations(battery_states, market_data)
            
        elif strategy == OptimizationStrategy.HYBRID_MPC_RL:
            mpc_allocations = await self._generate_mpc_allocations(battery_states, power_demand, forecasts)
            rl_allocations = await self._generate_rl_allocations(battery_states, market_data)
            
            # Weighted combination (70% MPC, 30% RL)
            allocations = self._combine_allocations(mpc_allocations, rl_allocations, 0.7, 0.3)
        
        return allocations
    
    async def _generate_mpc_allocations(self, battery_states: List[Dict], 
                                      power_demand: Dict, forecasts: Dict) -> List[AllocationDecision]:
        """Generate allocations using MPC (simplified implementation)"""
        
        allocations = []
        total_demand_w = power_demand.get("current_demand_kw", 10.0) * 1000
        
        # Simple proportional allocation based on available capacity
        total_capacity = sum(state.get("energy_capacity", 50000) * (state.get("soc", 50) / 100) 
                           for state in battery_states)
        
        for state in battery_states:
            device_id = state["device_id"]
            
            if total_capacity > 0:
                available_energy = state.get("energy_capacity", 50000) * (state.get("soc", 50) / 100)
                allocation_ratio = available_energy / total_capacity
                allocated_power = total_demand_w * allocation_ratio
                
                # Clamp to power capability
                power_capability = state.get("power_capability", 5000)
                allocated_power = np.clip(allocated_power, -power_capability, power_capability)
            else:
                allocated_power = 0.0
            
            # Simple SoC projection
            current_soc = state.get("soc", 50.0)
            capacity_wh = state.get("energy_capacity", 50000.0)
            energy_change = allocated_power * (5.0 / 60.0)  # 5 minutes
            soc_change = -(energy_change / capacity_wh) * 100
            expected_soc = np.clip(current_soc + soc_change, 0, 100)
            
            allocation = AllocationDecision(
                device_id=device_id,
                allocated_power_w=allocated_power,
                strategy_used="mpc_only",
                confidence_score=0.9,
                safety_score=0.95,
                expected_soc=expected_soc,
                expected_degradation=abs(allocated_power) * 1e-6,
                timestamp=datetime.utcnow()
            )
            allocations.append(allocation)
        
        return allocations
    
    async def _generate_rl_allocations(self, battery_states: List[Dict], 
                                     market_data: Dict) -> List[AllocationDecision]:
        """Generate allocations using RL (placeholder implementation)"""
        
        allocations = []
        
        for state in battery_states:
            device_id = state["device_id"]
            
            # Simple RL-style allocation based on SoC and market price
            soc = state.get("soc", 50.0)
            power_capability = state.get("power_capability", 5000.0)
            electricity_price = market_data.get("electricity_price", 0.12)
            
            # Simple policy: discharge when price is high and SoC is high
            if electricity_price > 0.15 and soc > 60:
                allocated_power = power_capability * 0.5
            elif electricity_price < 0.10 and soc < 40:
                allocated_power = -power_capability * 0.3
            else:
                allocated_power = 0.0
            
            # SoC projection
            current_soc = soc
            capacity_wh = state.get("energy_capacity", 50000.0)
            energy_change = allocated_power * (5.0 / 60.0)
            soc_change = -(energy_change / capacity_wh) * 100
            expected_soc = np.clip(current_soc + soc_change, 0, 100)
            
            allocation = AllocationDecision(
                device_id=device_id,
                allocated_power_w=allocated_power,
                strategy_used="rl_only",
                confidence_score=0.8,
                safety_score=0.9,
                expected_soc=expected_soc,
                expected_degradation=abs(allocated_power) * 1.2e-6,
                timestamp=datetime.utcnow()
            )
            allocations.append(allocation)
        
        return allocations
    
    def _combine_allocations(self, mpc_allocations: List[AllocationDecision], 
                           rl_allocations: List[AllocationDecision],
                           mpc_weight: float, rl_weight: float) -> List[AllocationDecision]:
        """Combine MPC and RL allocations with weighted average"""
        
        combined_allocations = []
        
        # Create device mapping
        mpc_map = {alloc.device_id: alloc for alloc in mpc_allocations}
        rl_map = {alloc.device_id: alloc for alloc in rl_allocations}
        
        all_devices = set(mpc_map.keys()) | set(rl_map.keys())
        
        for device_id in all_devices:
            mpc_alloc = mpc_map.get(device_id)
            rl_alloc = rl_map.get(device_id)
            
            if mpc_alloc and rl_alloc:
                # Weighted combination
                combined_power = (mpc_weight * mpc_alloc.allocated_power_w + 
                                rl_weight * rl_alloc.allocated_power_w)
                combined_soc = (mpc_weight * mpc_alloc.expected_soc + 
                              rl_weight * rl_alloc.expected_soc)
                combined_confidence = (mpc_weight * mpc_alloc.confidence_score + 
                                     rl_weight * rl_alloc.confidence_score)
                combined_safety = min(mpc_alloc.safety_score, rl_alloc.safety_score)
                combined_degradation = (mpc_weight * mpc_alloc.expected_degradation + 
                                      rl_weight * rl_alloc.expected_degradation)
                
                allocation = AllocationDecision(
                    device_id=device_id,
                    allocated_power_w=combined_power,
                    strategy_used="hybrid_mpc_rl",
                    confidence_score=combined_confidence,
                    safety_score=combined_safety,
                    expected_soc=combined_soc,
                    expected_degradation=combined_degradation,
                    timestamp=datetime.utcnow()
                )
                combined_allocations.append(allocation)
                
            elif mpc_alloc:
                mpc_alloc.strategy_used = "hybrid_mpc_rl"
                combined_allocations.append(mpc_alloc)
                
            elif rl_alloc:
                rl_alloc.strategy_used = "hybrid_mpc_rl"
                combined_allocations.append(rl_alloc)
        
        return combined_allocations
    
    async def _apply_safety_checks(self, allocations: List[AllocationDecision], 
                                 battery_states: List[Dict]) -> tuple[List[AllocationDecision], List[str]]:
        """Apply safety checks and modify allocations if needed"""
        
        # Check safety constraints
        safe, violations = self.safety_monitor.check_safety_constraints(allocations, battery_states)
        
        if not safe:
            logger.warning(f"Safety violations detected: {violations}")
            # Apply safety corrections
            for allocation in allocations:
                if allocation.expected_soc < 15.0:
                    allocation.allocated_power_w = min(allocation.allocated_power_w, 0)  # No discharge
                elif allocation.expected_soc > 90.0:
                    allocation.allocated_power_w = max(allocation.allocated_power_w, 0)  # No charge
                
                # Update safety score
                allocation.safety_score *= 0.8
        
        return allocations, violations
    
    async def _get_digital_twin_forecasts(self, battery_states: List[Dict], 
                                        power_demand: Dict) -> Dict[str, Any]:
        """Get forecasts from digital twin service"""
        
        forecasts = {}
        
        # For now, return empty forecasts (digital twin integration can be expanded)
        return forecasts
    
    def _extract_battery_states(self, fleet_data: Dict[str, Any]) -> List[Dict]:
        """Extract battery states from fleet data"""
        return fleet_data.get("devices", [])
    
    def _extract_power_demand(self, fleet_data: Dict[str, Any]) -> Dict:
        """Extract power demand from fleet data"""
        return fleet_data.get("power_demand", {"current_demand_kw": 10.0})
    
    def _extract_market_data(self, fleet_data: Dict[str, Any]) -> Dict:
        """Extract market data from fleet data"""
        return fleet_data.get("market_data", {"electricity_price": 0.12})
    
    def _record_allocation_decision(self, allocations: List[AllocationDecision], strategy: OptimizationStrategy):
        """Record allocation decision for performance tracking"""
        
        # Simplified - in real implementation would get actual results
        actual_results = {
            "power_delivered": {alloc.device_id: alloc.allocated_power_w for alloc in allocations},
            "revenue": sum(abs(alloc.allocated_power_w) for alloc in allocations) * 0.0001,
            "safety_violations": 0
        }
        
        self.performance_tracker.record_decision(allocations, actual_results)
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        return {
            "total_energy_delivered_kwh": self.metrics.total_energy_delivered_kwh,
            "total_revenue_dollars": self.metrics.total_revenue_dollars,
            "safety_violations_count": self.metrics.safety_violations_count,
            "strategy_performance": {
                strategy: self.performance_tracker.get_strategy_performance(strategy)
                for strategy in self.performance_tracker.strategy_metrics.keys()
            }
        }
    
    async def _store_coordination_result(self, result: Dict[str, Any]):
        """Store coordination result in database"""
        
        try:
            if self.mongodb_db:
                await self.mongodb_db.coordination_results.insert_one({
                    "timestamp": datetime.utcnow(),
                    "result": result
                })
        except Exception as e:
            logger.warning(f"Failed to store coordination result: {e}")


# FastAPI service wrapper
from fastapi import FastAPI

app = FastAPI(title="PulseBMS Coordinator Service", version="1.0.0")
coordinator = CoordinatorService()


@app.on_event("startup")
async def startup_event():
    await coordinator.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    await coordinator.shutdown()


@app.post("/coordinate")
async def coordinate_power_allocation(fleet_data: Dict[str, Any]):
    """Coordinate power allocation across battery fleet"""
    return await coordinator.coordinate_power_allocation(fleet_data)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "coordinator"
    }


@app.post("/emergency-stop/reset")
async def reset_emergency_stop():
    """Reset emergency stop (manual intervention)"""
    coordinator.safety_monitor.reset_emergency_stop()
    return {"status": "emergency_stop_reset", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
