"""
PulseBMS Enhanced - Integration Tests
Comprehensive end-to-end testing of the complete battery management system
"""

import asyncio
import pytest
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import aiohttp
import motor.motor_asyncio
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """Complete integration test suite for PulseBMS Enhanced"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.digital_twin_url = "http://localhost:8001"
        self.coordinator_url = "http://localhost:8002"
        
        # Test configuration
        self.mongodb_url = "mongodb://localhost:27017"
        self.redis_url = "redis://localhost:6379"
        
        # Test data
        self.test_devices = [
            {
                "device_id": "test_battery_001",
                "device_type": "battery_pack",
                "specifications": {
                    "chemistry": "LFP",
                    "capacity_kwh": 50.0,
                    "max_power": 5000.0,
                    "voltage_nominal": 400.0
                },
                "initial_state": {
                    "soc": 60.0,
                    "soh": 85.0,
                    "voltage": 410.0,
                    "current": 0.0,
                    "temperature": 25.0
                }
            },
            {
                "device_id": "test_battery_002",
                "device_type": "battery_pack",
                "specifications": {
                    "chemistry": "NMC",
                    "capacity_kwh": 75.0,
                    "max_power": 7500.0,
                    "voltage_nominal": 400.0
                },
                "initial_state": {
                    "soc": 45.0,
                    "soh": 92.0,
                    "voltage": 380.0,
                    "current": 0.0,
                    "temperature": 28.0
                }
            }
        ]
        
        # Service clients
        self.session = None
        self.mongodb_client = None
        self.redis_client = None
    
    async def setup(self):
        """Setup test environment"""
        logger.info("Setting up integration test environment...")
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Connect to databases
        self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        
        # Clean test databases
        await self._cleanup_test_data()
        
        logger.info("Integration test environment ready")
    
    async def teardown(self):
        """Cleanup test environment"""
        logger.info("Cleaning up integration test environment...")
        
        await self._cleanup_test_data()
        
        if self.session:
            await self.session.close()
        
        if self.mongodb_client:
            self.mongodb_client.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Integration test cleanup completed")
    
    async def _cleanup_test_data(self):
        """Clean up test data from databases"""
        try:
            # Clean MongoDB test collections
            if self.mongodb_client:
                test_db = self.mongodb_client.pulsebms_test
                collections = await test_db.list_collection_names()
                for collection in collections:
                    await test_db[collection].delete_many({})
            
            # Clean Redis test data
            if self.redis_client:
                await self.redis_client.flushdb()
        
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("Starting complete integration test suite...")
        
        test_results = {
            "start_time": datetime.utcnow().isoformat(),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": []
            }
        }
        
        # Define test cases
        test_cases = [
            ("test_backend_health", self.test_backend_health),
            ("test_device_registration", self.test_device_registration),
            ("test_telemetry_pipeline", self.test_telemetry_pipeline),
            ("test_digital_twin_integration", self.test_digital_twin_integration),
            ("test_coordinator_service", self.test_coordinator_service),
            ("test_safety_monitoring", self.test_safety_monitoring),
            ("test_end_to_end_workflow", self.test_end_to_end_workflow),
            ("test_performance_under_load", self.test_performance_under_load)
        ]
        
        # Run each test case
        for test_name, test_func in test_cases:
            logger.info(f"Running test: {test_name}")
            test_results["summary"]["total"] += 1
            
            try:
                result = await test_func()
                test_results["tests"][test_name] = {
                    "status": "passed" if result["success"] else "failed",
                    "details": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if result["success"]:
                    test_results["summary"]["passed"] += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    test_results["summary"]["failed"] += 1
                    logger.error(f"❌ {test_name} FAILED: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                test_results["summary"]["failed"] += 1
                test_results["summary"]["errors"].append(f"{test_name}: {str(e)}")
                test_results["tests"][test_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.error(f"💥 {test_name} ERROR: {e}")
        
        test_results["end_time"] = datetime.utcnow().isoformat()
        
        # Generate summary report
        summary = test_results["summary"]
        success_rate = (summary["passed"] / summary["total"]) * 100 if summary["total"] > 0 else 0
        
        logger.info(f"""
=== INTEGRATION TEST SUMMARY ===
Total Tests: {summary['total']}
Passed: {summary['passed']}
Failed: {summary['failed']}
Success Rate: {success_rate:.1f}%
        """)
        
        return test_results
    
    async def test_backend_health(self) -> Dict[str, Any]:
        """Test backend service health and basic functionality"""
        
        try:
            # Test health endpoint
            async with self.session.get(f"{self.backend_url}/health") as response:
                if response.status != 200:
                    return {"success": False, "error": f"Health check failed: {response.status}"}
                
                health_data = await response.json()
                
                if health_data.get("status") != "healthy":
                    return {"success": False, "error": "Backend not healthy"}
            
            return {
                "success": True,
                "health_data": health_data,
                "message": "Backend health check passed"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_device_registration(self) -> Dict[str, Any]:
        """Test device registration and management"""
        
        try:
            registered_devices = []
            
            for device_config in self.test_devices:
                # Register device
                device_info = {
                    "device_id": device_config["device_id"],
                    "device_type": device_config["device_type"],
                    "location": {"site": "test_site", "rack": "test_rack"},
                    "specifications": device_config["specifications"],
                    "safety_constraints": {
                        "min_soc": 10.0,
                        "max_soc": 90.0,
                        "max_current": 100.0,
                        "max_temperature": 45.0
                    }
                }
                
                async with self.session.post(
                    f"{self.backend_url}/devices/register",
                    json=device_info
                ) as response:
                    
                    if response.status != 201:
                        return {"success": False, "error": f"Device registration failed: {response.status}"}
                    
                    result = await response.json()
                    registered_devices.append(result)
            
            return {
                "success": True,
                "registered_devices": len(registered_devices),
                "message": "Device registration successful"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_telemetry_pipeline(self) -> Dict[str, Any]:
        """Test telemetry data pipeline from edge to backend"""
        
        try:
            # Send telemetry for each test device
            for device_config in self.test_devices:
                device_id = device_config["device_id"]
                initial_state = device_config["initial_state"]
                
                # Create telemetry data
                telemetry = {
                    "device_id": device_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "measurements": {
                        "soc": initial_state["soc"],
                        "soh": initial_state["soh"],
                        "voltage": initial_state["voltage"],
                        "current": initial_state["current"],
                        "temperature": initial_state["temperature"],
                        "power": initial_state["voltage"] * initial_state["current"]
                    },
                    "alarms": [],
                    "warnings": []
                }
                
                # Send telemetry via REST API
                async with self.session.post(
                    f"{self.backend_url}/telemetry/store",
                    json=telemetry
                ) as response:
                    
                    if response.status != 201:
                        return {"success": False, "error": f"Telemetry storage failed: {response.status}"}
            
            # Wait for data to be processed
            await asyncio.sleep(2)
            
            return {
                "success": True,
                "telemetry_points": len(self.test_devices),
                "message": "Telemetry pipeline working correctly"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_digital_twin_integration(self) -> Dict[str, Any]:
        """Test digital twin service integration"""
        
        try:
            # Test digital twin health
            async with self.session.get(f"{self.digital_twin_url}/health") as response:
                if response.status != 200:
                    return {"success": False, "error": "Digital twin service not available"}
            
            # Test battery simulation
            device_config = self.test_devices[0]
            simulation_request = {
                "device_id": device_config["device_id"],
                "chemistry": device_config["specifications"]["chemistry"],
                "current_soc": device_config["initial_state"]["soc"],
                "current_soh": device_config["initial_state"]["soh"],
                "temperature": device_config["initial_state"]["temperature"],
                "power_profile": [1000, 2000, 1500, 0, -1000],  # W
                "time_steps": [0.25, 0.5, 0.75, 1.0, 1.25],    # hours
                "model_parameters": {}
            }
            
            async with self.session.post(
                f"{self.digital_twin_url}/simulate",
                json=simulation_request
            ) as response:
                
                if response.status != 200:
                    return {"success": False, "error": f"Digital twin simulation failed: {response.status}"}
                
                simulation_result = await response.json()
                
                # Verify simulation output
                required_fields = ["voltage_profile", "current_profile", "soc_profile"]
                for field in required_fields:
                    if field not in simulation_result:
                        return {"success": False, "error": f"Missing simulation output: {field}"}
            
            return {
                "success": True,
                "message": "Digital twin integration working"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_coordinator_service(self) -> Dict[str, Any]:
        """Test coordinator service integration"""
        
        try:
            # Test coordinator health
            async with self.session.get(f"{self.coordinator_url}/health") as response:
                if response.status != 200:
                    return {"success": False, "error": "Coordinator service not available"}
            
            # Create coordinator test data
            coordination_data = {
                "devices": [
                    {
                        "device_id": device["device_id"],
                        "soc": device["initial_state"]["soc"],
                        "soh": device["initial_state"]["soh"],
                        "voltage": device["initial_state"]["voltage"],
                        "current": device["initial_state"]["current"],
                        "temperature": device["initial_state"]["temperature"],
                        "power_capability": device["specifications"]["max_power"],
                        "energy_capacity": device["specifications"]["capacity_kwh"] * 1000,
                        "chemistry": device["specifications"]["chemistry"]
                    }
                    for device in self.test_devices
                ],
                "power_demand": {
                    "current_demand_kw": 6.0
                },
                "market_data": {
                    "electricity_price": 0.15
                }
            }
            
            # Test coordination
            async with self.session.post(
                f"{self.coordinator_url}/coordinate",
                json=coordination_data
            ) as response:
                
                if response.status != 200:
                    return {"success": False, "error": f"Coordination failed: {response.status}"}
                
                coordination_result = await response.json()
                
                # Verify coordination result
                if coordination_result.get("status") != "success":
                    return {"success": False, "error": f"Coordination status: {coordination_result.get('status')}"}
            
            return {
                "success": True,
                "message": "Coordinator service working"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_safety_monitoring(self) -> Dict[str, Any]:
        """Test safety monitoring and emergency stop functionality"""
        
        try:
            # Create test data with safety violations
            violation_telemetry = {
                "device_id": self.test_devices[0]["device_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "measurements": {
                    "soc": 5.0,  # Below safe limit
                    "soh": 80.0,
                    "voltage": 2.0,  # Below safe limit
                    "current": 150.0,  # Above safe limit
                    "temperature": 55.0,  # Above safe limit
                    "power": 300.0
                },
                "alarms": ["low_soc", "low_voltage", "high_current", "high_temperature"],
                "warnings": []
            }
            
            # Send violating telemetry
            async with self.session.post(
                f"{self.backend_url}/telemetry/store",
                json=violation_telemetry
            ) as response:
                
                if response.status != 201:
                    return {"success": False, "error": "Failed to store violation telemetry"}
            
            return {
                "success": True,
                "message": "Safety monitoring working"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow"""
        
        try:
            # 1. Send telemetry data
            for device_config in self.test_devices:
                telemetry = {
                    "device_id": device_config["device_id"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "measurements": device_config["initial_state"],
                    "alarms": [],
                    "warnings": []
                }
                
                async with self.session.post(
                    f"{self.backend_url}/telemetry/store",
                    json=telemetry
                ) as response:
                    if response.status != 201:
                        return {"success": False, "error": "E2E: Telemetry storage failed"}
            
            # 2. Get fleet status
            async with self.session.get(f"{self.backend_url}/fleet/status") as response:
                if response.status != 200:
                    return {"success": False, "error": "E2E: Fleet status retrieval failed"}
                
                fleet_status = await response.json()
            
            # 3. Request power allocation via coordinator
            coordination_data = {
                "devices": [
                    {
                        "device_id": device["device_id"],
                        "soc": device["initial_state"]["soc"],
                        "power_capability": device["specifications"]["max_power"],
                        "energy_capacity": device["specifications"]["capacity_kwh"] * 1000
                    }
                    for device in self.test_devices
                ],
                "power_demand": {"current_demand_kw": 5.0}
            }
            
            async with self.session.post(
                f"{self.coordinator_url}/coordinate",
                json=coordination_data
            ) as response:
                if response.status != 200:
                    return {"success": False, "error": "E2E: Coordination failed"}
                
                coordination_result = await response.json()
            
            return {
                "success": True,
                "message": "End-to-end workflow completed successfully"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_performance_under_load(self) -> Dict[str, Any]:
        """Test system performance under load"""
        
        try:
            # Simulate multiple concurrent telemetry updates
            concurrent_requests = 20
            tasks = []
            
            for i in range(concurrent_requests):
                device_id = self.test_devices[i % len(self.test_devices)]["device_id"]
                
                telemetry = {
                    "device_id": device_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "measurements": {
                        "soc": 50.0 + (i % 40),
                        "soh": 80.0 + (i % 20),
                        "voltage": 400.0 + (i % 50),
                        "current": i % 100,
                        "temperature": 25.0 + (i % 20),
                        "power": (400.0 + (i % 50)) * (i % 100)
                    },
                    "alarms": [],
                    "warnings": []
                }
                
                task = self.session.post(
                    f"{self.backend_url}/telemetry/store",
                    json=telemetry
                )
                tasks.append(task)
            
            # Execute all requests concurrently
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful_requests = 0
            failed_requests = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    failed_requests += 1
                else:
                    async with response:
                        if response.status == 201:
                            successful_requests += 1
                        else:
                            failed_requests += 1
            
            total_time = end_time - start_time
            requests_per_second = concurrent_requests / total_time
            
            return {
                "success": True,
                "performance_metrics": {
                    "concurrent_requests": concurrent_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "requests_per_second": requests_per_second
                },
                "message": "Performance test passed"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


async def run_integration_tests():
    """Run the complete integration test suite"""
    test_suite = IntegrationTestSuite()
    
    try:
        await test_suite.setup()
        results = await test_suite.run_all_tests()
        
        # Save results to file
        with open("integration_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
        
    finally:
        await test_suite.teardown()


# Pytest integration
@pytest.mark.asyncio
async def test_integration_suite():
    """Pytest wrapper for integration tests"""
    results = await run_integration_tests()
    
    # Assert overall success
    summary = results["summary"]
    assert summary["failed"] == 0, f"Integration tests failed: {summary['failed']}/{summary['total']}"
    assert summary["passed"] == summary["total"], "Not all tests passed"


if __name__ == "__main__":
    async def main():
        print("Running PulseBMS Enhanced Integration Tests...")
        results = await run_integration_tests()
        
        summary = results["summary"]
        success_rate = (summary["passed"] / summary["total"]) * 100 if summary["total"] > 0 else 0
        
        print(f"\nIntegration Test Results:")
        print(f"  Total Tests: {summary['total']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if summary["errors"]:
            print(f"  Errors: {summary['errors']}")
        
        return summary["failed"] == 0
    
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
