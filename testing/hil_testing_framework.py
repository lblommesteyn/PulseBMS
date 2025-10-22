"""
PulseBMS Enhanced - Hardware-in-the-Loop (HIL) Testing Framework
Framework for testing battery management system with real hardware devices
"""

import asyncio
import logging
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import serial
import struct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HILTestStatus(Enum):
    """HIL test execution status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EMERGENCY_STOP = "emergency_stop"


class HardwareInterface(Enum):
    """Supported hardware interface types"""
    SERIAL_RS485 = "serial_rs485"
    MODBUS_TCP = "modbus_tcp"
    MOCK_DEVICE = "mock_device"


@dataclass
class HILTestConfig:
    """Configuration for HIL testing"""
    test_name: str
    description: str
    duration_minutes: float
    
    # Hardware configuration
    hardware_devices: List[Dict[str, Any]] = field(default_factory=list)
    
    # Test parameters
    power_profile: List[float] = field(default_factory=list)  # Power profile in W
    time_intervals: List[float] = field(default_factory=list)  # Time intervals in seconds
    
    # Safety limits
    max_voltage: float = 4.2  # V per cell
    min_voltage: float = 2.5  # V per cell
    max_current: float = 100.0  # A
    max_temperature: float = 45.0  # °C
    
    # Data collection
    sampling_rate_hz: float = 1.0
    log_telemetry: bool = True


@dataclass
class HILTestResult:
    """Results from HIL test execution"""
    test_config: HILTestConfig
    status: HILTestStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    
    # Collected data
    telemetry_data: List[Dict[str, Any]] = field(default_factory=list)
    command_data: List[Dict[str, Any]] = field(default_factory=list)
    error_log: List[str] = field(default_factory=list)
    
    # Test metrics
    success_rate: float = 0.0
    safety_violations: int = 0
    avg_voltage: float = 0.0
    avg_current: float = 0.0
    avg_temperature: float = 0.0
    energy_delivered_wh: float = 0.0


class HardwareDevice:
    """Base class for hardware device interfaces"""
    
    def __init__(self, device_id: str, interface_type: HardwareInterface, config: Dict[str, Any]):
        self.device_id = device_id
        self.interface_type = interface_type
        self.config = config
        self.connected = False
        self.last_telemetry = {}
        
    async def connect(self) -> bool:
        """Connect to hardware device"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from hardware device"""
        raise NotImplementedError
    
    async def send_command(self, command: Dict[str, Any]) -> bool:
        """Send command to hardware device"""
        raise NotImplementedError
    
    async def read_telemetry(self) -> Dict[str, Any]:
        """Read telemetry from hardware device"""
        raise NotImplementedError
    
    async def emergency_stop(self) -> bool:
        """Execute emergency stop"""
        raise NotImplementedError


class MockBatteryDevice(HardwareDevice):
    """Mock battery device for testing without real hardware"""
    
    def __init__(self, device_id: str, config: Dict[str, Any]):
        super().__init__(device_id, HardwareInterface.MOCK_DEVICE, config)
        
        # Battery state simulation
        self.soc = config.get("initial_soc", 50.0)  # %
        self.soh = config.get("initial_soh", 85.0)  # %
        self.voltage = config.get("initial_voltage", 400.0)  # V
        self.current = 0.0  # A
        self.temperature = config.get("initial_temperature", 25.0)  # °C
        self.capacity_ah = config.get("capacity_ah", 100.0)  # Ah
        
        # Simulation parameters
        self.internal_resistance = config.get("internal_resistance", 0.05)  # Ohm
        self.efficiency = config.get("efficiency", 0.95)
        
        # Command tracking
        self.target_power = 0.0
    
    async def connect(self) -> bool:
        """Connect to mock device"""
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connected = True
        logger.info(f"Mock device {self.device_id} connected")
        return True
    
    async def disconnect(self):
        """Disconnect from mock device"""
        self.connected = False
        logger.info(f"Mock device {self.device_id} disconnected")
    
    async def send_command(self, command: Dict[str, Any]) -> bool:
        """Send command to mock device"""
        if not self.connected:
            return False
        
        try:
            command_type = command.get("type", "power_setpoint")
            
            if command_type == "power_setpoint":
                self.target_power = command.get("power_w", 0.0)
                logger.debug(f"Device {self.device_id} received power setpoint: {self.target_power}W")
                
            elif command_type == "emergency_stop":
                self.target_power = 0.0
                logger.warning(f"Device {self.device_id} emergency stop activated")
                
            return True
            
        except Exception as e:
            logger.error(f"Command failed for device {self.device_id}: {e}")
            return False
    
    async def read_telemetry(self) -> Dict[str, Any]:
        """Read telemetry from mock device"""
        if not self.connected:
            return {}
        
        # Update battery state based on current power
        dt = 1.0 / 60.0  # 1 minute time step
        
        # Update current based on power and voltage
        if self.voltage > 0:
            self.current = self.target_power / self.voltage
        
        # Update SoC based on current
        if self.capacity_ah > 0:
            soc_change = -(self.current * dt / self.capacity_ah) * 100
            self.soc = np.clip(self.soc + soc_change, 0, 100)
        
        # Update voltage based on SoC (simplified OCV curve)
        if self.soc > 80:
            ocv = 4.1
        elif self.soc > 20:
            ocv = 3.3 + (self.soc - 20) / 60 * 0.8
        else:
            ocv = 3.0 + self.soc / 20 * 0.3
        
        # Terminal voltage with resistance drop
        self.voltage = ocv - (self.current * self.internal_resistance)
        
        # Update temperature based on power losses
        power_loss = abs(self.current) ** 2 * self.internal_resistance
        self.temperature = 25.0 + power_loss * 0.1  # Simplified thermal model
        
        # Create telemetry data
        telemetry = {
            "device_id": self.device_id,
            "timestamp": datetime.utcnow().isoformat(),
            "measurements": {
                "soc": float(self.soc),
                "soh": float(self.soh),
                "voltage": float(self.voltage),
                "current": float(self.current),
                "temperature": float(self.temperature),
                "power": float(self.current * self.voltage)
            },
            "status": {
                "connected": self.connected,
                "fault_status": 0,
                "warning_status": 0
            }
        }
        
        self.last_telemetry = telemetry
        return telemetry
    
    async def emergency_stop(self) -> bool:
        """Execute emergency stop"""
        self.target_power = 0.0
        self.current = 0.0
        logger.critical(f"Emergency stop executed for device {self.device_id}")
        return True


class SerialBatteryDevice(HardwareDevice):
    """Serial RS485 battery device interface"""
    
    def __init__(self, device_id: str, config: Dict[str, Any]):
        super().__init__(device_id, HardwareInterface.SERIAL_RS485, config)
        
        self.port = config.get("port", "COM1")
        self.baudrate = config.get("baudrate", 9600)
        self.timeout = config.get("timeout", 1.0)
        self.device_address = config.get("device_address", 1)
        
        self.serial_connection = None
    
    async def connect(self) -> bool:
        """Connect to serial device"""
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            self.connected = True
            logger.info(f"Serial device {self.device_id} connected on {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to serial device {self.device_id}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from serial device"""
        if self.serial_connection:
            self.serial_connection.close()
        self.connected = False
    
    async def send_command(self, command: Dict[str, Any]) -> bool:
        """Send command to serial device using Modbus RTU protocol"""
        if not self.connected or not self.serial_connection:
            return False
        
        try:
            command_type = command.get("type", "power_setpoint")
            
            if command_type == "power_setpoint":
                power_w = int(command.get("power_w", 0))
                
                # Create Modbus RTU frame
                function_code = 0x06  # Write Single Register
                register_address = 0x1000  # Power setpoint register
                
                frame = struct.pack('>BBHH', 
                                  self.device_address, 
                                  function_code,
                                  register_address, 
                                  power_w & 0xFFFF)
                
                # Calculate CRC16
                crc = self._calculate_crc16(frame)
                frame += struct.pack('<H', crc)
                
                # Send command
                self.serial_connection.write(frame)
                response = self.serial_connection.read(8)
                
                return len(response) >= 8
                
        except Exception as e:
            logger.error(f"Command failed for device {self.device_id}: {e}")
            return False
    
    async def read_telemetry(self) -> Dict[str, Any]:
        """Read telemetry from serial device"""
        if not self.connected or not self.serial_connection:
            return {}
        
        try:
            # Read multiple registers using Modbus RTU
            function_code = 0x03
            start_address = 0x2000
            num_registers = 10
            
            frame = struct.pack('>BBHH', 
                              self.device_address, 
                              function_code,
                              start_address, 
                              num_registers)
            
            crc = self._calculate_crc16(frame)
            frame += struct.pack('<H', crc)
            
            self.serial_connection.write(frame)
            response = self.serial_connection.read(25)
            
            if len(response) >= 25:
                # Parse response
                byte_count = response[2]
                data = response[3:3+byte_count]
                values = struct.unpack(f'>{byte_count//2}H', data)
                
                telemetry = {
                    "device_id": self.device_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "measurements": {
                        "soc": values[0] / 100.0,
                        "soh": values[1] / 100.0,
                        "voltage": values[2] / 100.0,
                        "current": (values[3] - 32768) / 100.0,
                        "temperature": (values[4] - 2731) / 10.0,
                        "power": (values[5] - 32768) / 10.0
                    },
                    "status": {
                        "connected": True,
                        "fault_status": values[6],
                        "warning_status": values[7]
                    }
                }
                
                return telemetry
            
            return {}
                
        except Exception as e:
            logger.error(f"Telemetry read failed for device {self.device_id}: {e}")
            return {}
    
    async def emergency_stop(self) -> bool:
        """Execute emergency stop"""
        return await self.send_command({"type": "power_setpoint", "power_w": 0})
    
    def _calculate_crc16(self, data: bytes) -> int:
        """Calculate CRC16 for Modbus RTU"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc


class HILTestExecutor:
    """Hardware-in-the-loop test executor"""
    
    def __init__(self):
        self.devices: Dict[str, HardwareDevice] = {}
        self.current_test: Optional[HILTestConfig] = None
        self.test_status = HILTestStatus.IDLE
        self.test_start_time: Optional[datetime] = None
        
        # Data collection
        self.telemetry_data: List[Dict[str, Any]] = []
        self.command_data: List[Dict[str, Any]] = []
        self.error_log: List[str] = []
        
        # Safety monitoring
        self.safety_violations = 0
        self.emergency_stop_triggered = False
    
    async def initialize_devices(self, device_configs: List[Dict[str, Any]]) -> bool:
        """Initialize and connect to hardware devices"""
        
        logger.info(f"Initializing {len(device_configs)} hardware devices...")
        
        for device_config in device_configs:
            device_id = device_config["device_id"]
            interface_type = HardwareInterface(device_config.get("interface_type", "mock_device"))
            
            try:
                # Create device based on interface type
                if interface_type == HardwareInterface.MOCK_DEVICE:
                    device = MockBatteryDevice(device_id, device_config)
                elif interface_type == HardwareInterface.SERIAL_RS485:
                    device = SerialBatteryDevice(device_id, device_config)
                else:
                    logger.error(f"Unsupported interface type: {interface_type}")
                    continue
                
                # Connect to device
                if await device.connect():
                    self.devices[device_id] = device
                    logger.info(f"Device {device_id} initialized successfully")
                else:
                    logger.error(f"Failed to connect to device {device_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to initialize device {device_id}: {e}")
                return False
        
        logger.info(f"All {len(self.devices)} devices initialized successfully")
        return True
    
    async def run_test(self, test_config: HILTestConfig) -> HILTestResult:
        """Execute HIL test"""
        
        logger.info(f"Starting HIL test: {test_config.test_name}")
        
        self.current_test = test_config
        self.test_status = HILTestStatus.INITIALIZING
        self.test_start_time = datetime.utcnow()
        
        # Clear previous data
        self.telemetry_data.clear()
        self.command_data.clear()
        self.error_log.clear()
        self.safety_violations = 0
        self.emergency_stop_triggered = False
        
        try:
            # Initialize devices if not already done
            if not self.devices:
                if not await self.initialize_devices(test_config.hardware_devices):
                    raise Exception("Failed to initialize hardware devices")
            
            self.test_status = HILTestStatus.RUNNING
            
            # Execute test with monitoring
            await self._execute_test_with_monitoring()
            
            # Generate results
            results = self._generate_test_results()
            
            if self.emergency_stop_triggered:
                self.test_status = HILTestStatus.EMERGENCY_STOP
            elif self.safety_violations > 0:
                self.test_status = HILTestStatus.FAILED
            else:
                self.test_status = HILTestStatus.COMPLETED
            
            logger.info(f"HIL test completed: {self.test_status.value}")
            return results
            
        except Exception as e:
            self.test_status = HILTestStatus.FAILED
            self.error_log.append(f"Test execution failed: {str(e)}")
            logger.error(f"HIL test failed: {e}")
            return self._generate_test_results()
    
    async def _execute_test_with_monitoring(self):
        """Execute test sequence with concurrent monitoring"""
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start test execution task
        execution_task = asyncio.create_task(self._execute_test_sequence())
        
        try:
            # Wait for test execution to complete
            await execution_task
        finally:
            # Cancel monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _execute_test_sequence(self):
        """Execute the test power profile sequence"""
        
        if not self.current_test:
            return
        
        power_profile = self.current_test.power_profile
        time_intervals = self.current_test.time_intervals
        
        if len(power_profile) != len(time_intervals):
            raise ValueError("Power profile and time intervals must have same length")
        
        logger.info(f"Executing test sequence with {len(power_profile)} steps")
        
        for i, (power_w, interval_s) in enumerate(zip(power_profile, time_intervals)):
            if self.emergency_stop_triggered:
                logger.warning("Test sequence stopped due to emergency stop")
                break
            
            # Send power commands to all devices
            for device_id, device in self.devices.items():
                # Distribute power proportionally (simplified)
                device_power = power_w / len(self.devices)
                
                command = {
                    "type": "power_setpoint",
                    "power_w": device_power,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                success = await device.send_command(command)
                
                # Log command
                self.command_data.append({
                    "device_id": device_id,
                    "command": command,
                    "success": success,
                    "timestamp": datetime.utcnow()
                })
                
                if not success:
                    self.error_log.append(f"Command failed for device {device_id} at step {i}")
            
            logger.debug(f"Test step {i+1}/{len(power_profile)}: {power_w}W for {interval_s}s")
            
            # Wait for the specified interval
            await asyncio.sleep(interval_s)
        
        # Set all devices to zero power at end of test
        for device_id, device in self.devices.items():
            await device.send_command({"type": "power_setpoint", "power_w": 0})
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop for data collection and safety checks"""
        
        if not self.current_test:
            return
        
        sampling_interval = 1.0 / self.current_test.sampling_rate_hz
        
        while self.test_status == HILTestStatus.RUNNING:
            try:
                # Collect telemetry from all devices
                for device_id, device in self.devices.items():
                    telemetry = await device.read_telemetry()
                    
                    if telemetry:
                        self.telemetry_data.append(telemetry)
                        
                        # Safety checks
                        measurements = telemetry.get("measurements", {})
                        await self._check_safety_limits(device_id, measurements)
                
                # Wait for next sampling interval
                await asyncio.sleep(sampling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_log.append(f"Monitoring error: {str(e)}")
    
    async def _check_safety_limits(self, device_id: str, measurements: Dict[str, Any]):
        """Check safety limits and trigger emergency stop if needed"""
        
        if not self.current_test:
            return
        
        violations = []
        
        # Check voltage limits
        voltage = measurements.get("voltage", 0)
        if voltage > self.current_test.max_voltage:
            violations.append(f"Voltage too high: {voltage}V")
        elif voltage < self.current_test.min_voltage:
            violations.append(f"Voltage too low: {voltage}V")
        
        # Check current limits
        current = abs(measurements.get("current", 0))
        if current > self.current_test.max_current:
            violations.append(f"Current too high: {current}A")
        
        # Check temperature limits
        temperature = measurements.get("temperature", 0)
        if temperature > self.current_test.max_temperature:
            violations.append(f"Temperature too high: {temperature}°C")
        
        if violations:
            self.safety_violations += len(violations)
            
            for violation in violations:
                error_msg = f"Safety violation on {device_id}: {violation}"
                self.error_log.append(error_msg)
                logger.warning(error_msg)
            
            # Trigger emergency stop for critical violations
            if (voltage > self.current_test.max_voltage * 1.1 or 
                current > self.current_test.max_current * 1.2 or
                temperature > self.current_test.max_temperature + 5):
                
                await self._trigger_emergency_stop("Critical safety violation detected")
    
    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop for all devices"""
        
        self.emergency_stop_triggered = True
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Send emergency stop to all devices
        for device_id, device in self.devices.items():
            try:
                await device.emergency_stop()
            except Exception as e:
                logger.error(f"Failed to send emergency stop to device {device_id}: {e}")
        
        self.error_log.append(f"Emergency stop triggered: {reason}")
    
    def _generate_test_results(self) -> HILTestResult:
        """Generate comprehensive test results"""
        
        end_time = datetime.utcnow()
        duration_seconds = (end_time - self.test_start_time).total_seconds() if self.test_start_time else 0
        
        # Calculate metrics
        success_rate = self._calculate_success_rate()
        avg_voltage, avg_current, avg_temperature, energy_delivered = self._calculate_summary_stats()
        
        results = HILTestResult(
            test_config=self.current_test,
            status=self.test_status,
            start_time=self.test_start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            telemetry_data=self.telemetry_data.copy(),
            command_data=self.command_data.copy(),
            error_log=self.error_log.copy(),
            success_rate=success_rate,
            safety_violations=self.safety_violations,
            avg_voltage=avg_voltage,
            avg_current=avg_current,
            avg_temperature=avg_temperature,
            energy_delivered_wh=energy_delivered
        )
        
        return results
    
    def _calculate_success_rate(self) -> float:
        """Calculate command success rate"""
        if not self.command_data:
            return 0.0
        
        successful_commands = sum(1 for cmd in self.command_data if cmd.get("success", False))
        return successful_commands / len(self.command_data) * 100.0
    
    def _calculate_summary_stats(self) -> tuple[float, float, float, float]:
        """Calculate summary statistics from telemetry data"""
        
        if not self.telemetry_data:
            return 0.0, 0.0, 0.0, 0.0
        
        voltages = []
        currents = []
        temperatures = []
        powers = []
        
        for data in self.telemetry_data:
            measurements = data.get("measurements", {})
            voltages.append(measurements.get("voltage", 0))
            currents.append(measurements.get("current", 0))
            temperatures.append(measurements.get("temperature", 0))
            powers.append(measurements.get("power", 0))
        
        avg_voltage = np.mean(voltages) if voltages else 0.0
        avg_current = np.mean(currents) if currents else 0.0
        avg_temperature = np.mean(temperatures) if temperatures else 0.0
        energy_delivered = sum(abs(p) for p in powers) * (1.0 / 3600.0)  # Wh
        
        return avg_voltage, avg_current, avg_temperature, energy_delivered
    
    async def cleanup(self):
        """Cleanup resources and disconnect devices"""
        for device in self.devices.values():
            try:
                await device.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting device {device.device_id}: {e}")
        
        self.devices.clear()


# Example usage
if __name__ == "__main__":
    async def run_example_test():
        """Run an example HIL test"""
        
        # Create test configuration
        test_config = HILTestConfig(
            test_name="Basic Power Profile Test",
            description="Test basic power delivery profile with mock devices",
            duration_minutes=5.0,
            hardware_devices=[
                {
                    "device_id": "battery_001",
                    "interface_type": "mock_device",
                    "initial_soc": 60.0,
                    "initial_soh": 85.0,
                    "capacity_ah": 100.0
                },
                {
                    "device_id": "battery_002", 
                    "interface_type": "mock_device",
                    "initial_soc": 45.0,
                    "initial_soh": 90.0,
                    "capacity_ah": 80.0
                }
            ],
            power_profile=[1000, 2000, 1500, 500, 0, -1000, 0],  # W
            time_intervals=[30, 30, 30, 30, 30, 30, 30],  # seconds
            sampling_rate_hz=0.5
        )
        
        # Create and run test
        executor = HILTestExecutor()
        
        try:
            results = await executor.run_test(test_config)
            
            print(f"Test completed with status: {results.status.value}")
            print(f"Duration: {results.duration_seconds:.1f} seconds")
            print(f"Success rate: {results.success_rate:.1f}%")
            print(f"Safety violations: {results.safety_violations}")
            print(f"Energy delivered: {results.energy_delivered_wh:.2f} Wh")
            print(f"Average voltage: {results.avg_voltage:.2f} V")
            print(f"Average current: {results.avg_current:.2f} A")
            print(f"Average temperature: {results.avg_temperature:.1f} °C")
            
            if results.error_log:
                print(f"Errors: {results.error_log}")
                
        finally:
            await executor.cleanup()
    
    asyncio.run(run_example_test())
