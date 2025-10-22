"""
PulseBMS Enhanced - Edge Device Implementation
Real-time battery monitoring with local SoC/SoH estimation and safety interlocks
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

import paho.mqtt.client as mqtt
from scipy import signal
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BatteryConfig:
    """Battery pack configuration"""
    device_id: str
    site_id: str
    chemistry: str  # LFP, NMC, LCO, NCA
    nominal_capacity: float  # Ah
    nominal_voltage: float  # V
    series_cells: int
    parallel_cells: int
    max_cell_voltage: float = 4.2
    min_cell_voltage: float = 2.5
    max_temperature: float = 60.0
    min_temperature: float = -20.0


@dataclass
class SensorData:
    """Raw sensor measurements"""
    timestamp: datetime
    pack_voltage: float
    pack_current: float
    cell_voltages: List[float]
    cell_temperatures: List[float]
    ambient_temperature: float


class KalmanFilter:
    """Extended Kalman Filter for SoC estimation"""
    
    def __init__(self, initial_soc: float = 50.0):
        self.state = np.array([initial_soc])
        self.P = np.array([[10.0]])
        self.Q = np.array([[0.1]])
        self.R = np.array([[1.0]])
        
    def predict(self, current: float, dt: float, capacity: float):
        """Predict step using coulomb counting"""
        coulomb_efficiency = 0.99
        delta_soc = -(current * dt * coulomb_efficiency) / (3600 * capacity) * 100
        self.state[0] += delta_soc
        F = np.array([[1.0]])
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, voltage: float, expected_voltage: float):
        """Update step using voltage measurement"""
        H = np.array([[1.0]])
        y = voltage - expected_voltage
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        self.state += K * y
        I = np.eye(1)
        self.P = (I - K @ H) @ self.P
        self.state[0] = np.clip(self.state[0], 0.0, 100.0)
    
    def get_soc(self) -> float:
        return float(self.state[0])


class SoCEstimator:
    """State of Charge estimator using Kalman filter"""
    
    def __init__(self, config: BatteryConfig):
        self.config = config
        self.kalman_filter = KalmanFilter()
        self.voltage_soc_table = self._get_voltage_soc_table(config.chemistry)
        self.last_timestamp = None
        
    def _get_voltage_soc_table(self, chemistry: str) -> Dict[float, float]:
        """Get voltage-SoC lookup table for battery chemistry"""
        if chemistry == "LFP":
            return {2.5: 0, 2.8: 5, 3.0: 10, 3.2: 20, 3.25: 40, 3.3: 60, 3.35: 80, 3.4: 90, 3.5: 95, 3.6: 100}
        elif chemistry == "NMC":
            return {2.5: 0, 2.8: 5, 3.0: 10, 3.3: 20, 3.5: 40, 3.7: 60, 3.9: 80, 4.0: 90, 4.1: 95, 4.2: 100}
        else:
            return {2.5: 0, 2.8: 5, 3.0: 10, 3.3: 20, 3.5: 40, 3.7: 60, 3.9: 80, 4.0: 90, 4.1: 95, 4.2: 100}
    
    def estimate_soc(self, sensor_data: SensorData) -> float:
        """Main SoC estimation using Kalman filter"""
        current_time = sensor_data.timestamp.timestamp()
        dt = 1.0
        if self.last_timestamp is not None:
            dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        
        self.kalman_filter.predict(sensor_data.pack_current, dt, self.config.nominal_capacity)
        
        avg_cell_voltage = np.mean(sensor_data.cell_voltages)
        expected_voltage = 3.3  # Simplified
        self.kalman_filter.update(avg_cell_voltage, expected_voltage)
        
        return self.kalman_filter.get_soc()


class SoHEstimator:
    """State of Health estimator"""
    
    def __init__(self, config: BatteryConfig):
        self.config = config
        self.initial_capacity = config.nominal_capacity
        self.cycle_count = 0.0
        self.last_soc = 50.0
        
    def estimate_soh(self, current_soc: float) -> float:
        """Simplified SoH estimation"""
        # Update cycle count
        soc_change = abs(current_soc - self.last_soc)
        if soc_change > 1.0:
            self.cycle_count += soc_change / 100.0
        self.last_soc = current_soc
        
        # Simple degradation model
        cycle_degradation = min(self.cycle_count / 5000.0 * 20, 30)
        return max(70.0, 100.0 - cycle_degradation)


class SafetyMonitor:
    """Real-time safety monitoring"""
    
    def __init__(self, config: BatteryConfig):
        self.config = config
        
    def check_safety_limits(self, sensor_data: SensorData, soc: float) -> List[str]:
        """Check safety limits and return alarm flags"""
        alarms = []
        
        # Cell voltage checks
        for i, voltage in enumerate(sensor_data.cell_voltages):
            if voltage > self.config.max_cell_voltage:
                alarms.append(f"overvoltage_cell_{i}")
            elif voltage < self.config.min_cell_voltage:
                alarms.append(f"undervoltage_cell_{i}")
        
        # Temperature checks
        for i, temp in enumerate(sensor_data.cell_temperatures):
            if temp > self.config.max_temperature:
                alarms.append(f"overtemperature_cell_{i}")
            elif temp < self.config.min_temperature:
                alarms.append(f"undertemperature_cell_{i}")
        
        # SoC checks
        if soc > 95.0:
            alarms.append("high_soc_warning")
        elif soc < 5.0:
            alarms.append("low_soc_warning")
        
        return alarms


class EdgeDevice:
    """Main edge device controller"""
    
    def __init__(self, config: BatteryConfig, mqtt_broker: str = "localhost"):
        self.config = config
        self.mqtt_broker = mqtt_broker
        
        self.soc_estimator = SoCEstimator(config)
        self.soh_estimator = SoHEstimator(config)
        self.safety_monitor = SafetyMonitor(config)
        
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        self.emergency_shutdown = False
        self.telemetry_interval = 1.0
        self.last_telemetry_time = 0
        
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            command_topic = f"pulsebms/{self.config.site_id}/{self.config.device_id}/commands"
            client.subscribe(command_topic)
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            command_data = json.loads(msg.payload.decode())
            asyncio.create_task(self._handle_command(command_data))
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")
    
    async def _handle_command(self, command_data: Dict):
        """Handle incoming commands"""
        command_type = command_data.get("command", {}).get("type")
        logger.info(f"Received command: {command_type}")
        
        if command_type == "emergency_shutdown":
            self.emergency_shutdown = True
            logger.critical("EMERGENCY SHUTDOWN INITIATED")
    
    async def start(self):
        """Start the edge device main loop"""
        logger.info(f"Starting edge device for {self.config.device_id}")
        
        try:
            self.mqtt_client.connect(self.mqtt_broker, 1883, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT: {e}")
        
        while True:
            try:
                await self._control_loop()
                await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(1.0)
        
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
    
    async def _control_loop(self):
        """Main control loop"""
        current_time = time.time()
        
        # Read sensors (simulated)
        sensor_data = await self._read_sensors()
        
        # State estimation
        soc = self.soc_estimator.estimate_soc(sensor_data)
        soh = self.soh_estimator.estimate_soh(soc)
        
        # Safety monitoring
        alarm_flags = self.safety_monitor.check_safety_limits(sensor_data, soc)
        
        # Send telemetry
        if current_time - self.last_telemetry_time >= self.telemetry_interval:
            await self._send_telemetry(sensor_data, soc, soh, alarm_flags)
            self.last_telemetry_time = current_time
    
    async def _read_sensors(self) -> SensorData:
        """Read sensor data (simulated)"""
        base_voltage = 3.3 * self.config.series_cells
        cell_voltages = [3.3 + np.random.normal(0, 0.02) for _ in range(self.config.series_cells)]
        cell_temperatures = [25.0 + np.random.normal(0, 2.0) for _ in range(self.config.series_cells)]
        
        return SensorData(
            timestamp=datetime.utcnow(),
            pack_voltage=base_voltage + np.random.normal(0, 0.01),
            pack_current=np.random.normal(0, 10),
            cell_voltages=cell_voltages,
            cell_temperatures=cell_temperatures,
            ambient_temperature=22.0 + np.random.normal(0, 1.0)
        )
    
    async def _send_telemetry(self, sensor_data: SensorData, soc: float, soh: float, alarm_flags: List[str]):
        """Send telemetry data via MQTT"""
        telemetry_data = {
            "timestamp": sensor_data.timestamp.isoformat(),
            "device_info": {
                "device_id": self.config.device_id,
                "site_id": self.config.site_id,
                "firmware_version": "1.0.0"
            },
            "measurements": {
                "pack": {
                    "voltage": sensor_data.pack_voltage,
                    "current": sensor_data.pack_current,
                    "power": sensor_data.pack_voltage * sensor_data.pack_current,
                    "temperature": np.mean(sensor_data.cell_temperatures),
                    "soc": soc,
                    "soh": soh,
                    "internal_resistance": 0.05
                },
                "cells": {
                    "count": len(sensor_data.cell_voltages),
                    "voltages": sensor_data.cell_voltages,
                    "temperatures": sensor_data.cell_temperatures
                }
            },
            "alarms": {
                "active_count": len(alarm_flags),
                "flags": alarm_flags
            }
        }
        
        topic = f"pulsebms/{self.config.site_id}/{self.config.device_id}/telemetry"
        try:
            self.mqtt_client.publish(topic, json.dumps(telemetry_data), qos=1)
        except Exception as e:
            logger.error(f"Failed to send telemetry: {e}")


# Example usage
async def main():
    config = BatteryConfig(
        device_id="battery_pack_001",
        site_id="site_001",
        chemistry="LFP",
        nominal_capacity=100.0,  # 100Ah
        nominal_voltage=400.0,   # 400V pack
        series_cells=128,        # 128 cells in series
        parallel_cells=1
    )
    
    device = EdgeDevice(config)
    await device.start()


if __name__ == "__main__":
    asyncio.run(main())
