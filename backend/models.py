"""
PulseBMS Enhanced - Data Models
Pydantic models for API requests/responses and data validation
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from pydantic import BaseModel, Field, validator


class BatteryChemistry(str, Enum):
    """Supported battery chemistries"""
    LFP = "LFP"  # Lithium Iron Phosphate
    NMC = "NMC"  # Lithium Nickel Manganese Cobalt
    LCO = "LCO"  # Lithium Cobalt Oxide
    NCA = "NCA"  # Lithium Nickel Cobalt Aluminum


class DeviceStatus(str, Enum):
    """Device operational status"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class CommandType(str, Enum):
    """Edge device command types"""
    START_CHARGE = "start_charge"
    STOP_CHARGE = "stop_charge"
    START_DISCHARGE = "start_discharge"
    STOP_DISCHARGE = "stop_discharge"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    UPDATE_PARAMETERS = "update_parameters"
    REQUEST_STATUS = "request_status"


class TelemetryData(BaseModel):
    """Battery pack telemetry data"""
    site_id: str
    device_id: str
    timestamp: datetime
    
    # Pack-level measurements
    voltage: float = Field(ge=0, description="Pack voltage (V)")
    current: float = Field(description="Pack current (A), positive=discharge")
    temperature: float = Field(description="Average pack temperature (°C)")
    power: Optional[float] = Field(default=None, description="Instantaneous power (W)")
    
    # State estimates
    soc: float = Field(ge=0, le=100, description="State of Charge (%)")
    soh: float = Field(ge=0, le=100, description="State of Health (%)")
    
    # Cell-level data
    cell_voltages: List[float] = Field(default_factory=list, description="Individual cell voltages (V)")
    cell_temperatures: List[float] = Field(default_factory=list, description="Individual cell temperatures (°C)")
    
    # Safety and status
    alarm_flags: List[str] = Field(default_factory=list, description="Active alarm conditions")
    status: DeviceStatus = DeviceStatus.ONLINE
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('power', always=True)
    def calculate_power(cls, v, values):
        """Calculate power if not provided"""
        if v is None and 'voltage' in values and 'current' in values:
            return values['voltage'] * values['current']
        return v


class DeviceInfo(BaseModel):
    """Battery device/pack information"""
    device_id: str
    site_id: str
    name: str
    
    # Hardware specifications
    chemistry: BatteryChemistry
    nominal_capacity: float = Field(gt=0, description="Nominal capacity (Ah)")
    nominal_voltage: float = Field(gt=0, description="Nominal voltage (V)")
    max_charge_power: float = Field(gt=0, description="Maximum charge power (W)")
    max_discharge_power: float = Field(gt=0, description="Maximum discharge power (W)")
    
    # Physical configuration
    series_cells: int = Field(gt=0, description="Number of cells in series")
    parallel_cells: int = Field(gt=0, description="Number of cells in parallel")
    
    # Lifecycle information
    manufacturing_date: Optional[datetime] = None
    installation_date: Optional[datetime] = None
    total_cycles: int = Field(ge=0, default=0)
    total_energy_throughput: float = Field(ge=0, default=0.0, description="Total energy (kWh)")
    
    # Current status
    status: DeviceStatus = DeviceStatus.OFFLINE
    last_seen: Optional[datetime] = None
    firmware_version: Optional[str] = None
    
    # Location and grouping
    location: Optional[str] = None
    rack_position: Optional[str] = None
    
    class Config:
        use_enum_values = True


class DeviceCommand(BaseModel):
    """Command to send to edge device"""
    command_id: str
    device_id: str
    site_id: str
    command_type: CommandType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeout_seconds: int = Field(default=30, gt=0)
    
    class Config:
        use_enum_values = True


class CommandResponse(BaseModel):
    """Response from edge device command"""
    command_id: str
    device_id: str
    site_id: str
    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SafetyConstraints(BaseModel):
    """Safety operating constraints for a device"""
    device_id: str
    
    # Voltage constraints
    max_cell_voltage: float = Field(gt=0, default=4.2)
    min_cell_voltage: float = Field(gt=0, default=2.5)
    max_pack_voltage: Optional[float] = None
    min_pack_voltage: Optional[float] = None
    
    # Current constraints
    max_charge_current: float = Field(gt=0, default=50.0)
    max_discharge_current: float = Field(gt=0, default=100.0)
    
    # Temperature constraints
    max_cell_temperature: float = Field(default=60.0)
    min_cell_temperature: float = Field(default=-20.0)
    max_temperature_delta: float = Field(gt=0, default=10.0)
    
    # SoC constraints
    max_soc: float = Field(ge=0, le=100, default=95.0)
    min_soc: float = Field(ge=0, le=100, default=5.0)
    
    # Power constraints
    max_charge_power: Optional[float] = None
    max_discharge_power: Optional[float] = None
    
    # Degradation constraints
    max_cycle_depth: float = Field(ge=0, le=100, default=80.0)
    max_c_rate: float = Field(gt=0, default=1.0)


class FleetStatus(BaseModel):
    """Fleet-wide status summary"""
    site_id: str
    total_devices: int
    online_devices: int
    total_capacity: float = Field(description="Total capacity (Ah)")
    available_capacity: float = Field(description="Available capacity (Ah)")
    total_power: float = Field(description="Current total power (W)")
    average_soc: float = Field(ge=0, le=100)
    average_soh: float = Field(ge=0, le=100)
    
    # Aggregated constraints
    max_charge_power: float
    max_discharge_power: float
    
    # Alarms and warnings
    active_alarms: List[str] = Field(default_factory=list)
    devices_in_alarm: int = 0
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PowerAllocation(BaseModel):
    """Power allocation command from coordinator"""
    site_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Site-level targets
    total_power_target: float = Field(description="Total site power target (W)")
    duration_minutes: int = Field(gt=0, default=60)
    
    # Device-specific allocations
    device_allocations: Dict[str, float] = Field(description="Power allocation per device (W)")
    
    # Optimization metadata
    optimization_method: str = Field(default="MPC")
    cost_function_value: Optional[float] = None
    constraints_satisfied: bool = True
    
    # Forecasting data
    demand_forecast: List[float] = Field(default_factory=list, description="Demand forecast (W)")
    degradation_forecast: Dict[str, float] = Field(default_factory=dict, description="Degradation per device")


class DigitalTwinRequest(BaseModel):
    """Request for digital twin simulation"""
    device_id: str
    chemistry: BatteryChemistry
    
    # Current state
    current_soc: float = Field(ge=0, le=100)
    current_soh: float = Field(ge=0, le=100)
    temperature: float
    
    # Simulation parameters
    power_profile: List[float] = Field(description="Power profile for simulation (W)")
    time_steps: List[float] = Field(description="Time steps for simulation (hours)")
    
    # Model parameters (optional overrides)
    model_parameters: Dict[str, Any] = Field(default_factory=dict)


class DigitalTwinResponse(BaseModel):
    """Response from digital twin simulation"""
    device_id: str
    
    # Simulation results
    voltage_profile: List[float]
    current_profile: List[float]
    temperature_profile: List[float]
    soc_profile: List[float]
    
    # Degradation predictions
    capacity_fade: float = Field(description="Predicted capacity fade (%)")
    resistance_growth: float = Field(description="Predicted resistance growth (%)")
    expected_eol: Optional[datetime] = Field(description="Predicted end-of-life")
    
    # Health metrics
    cycle_life_remaining: float = Field(description="Remaining cycle life (%)")
    calendar_life_remaining: float = Field(description="Remaining calendar life (%)")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    message: str
    data: Any = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str] = Field(default_factory=dict)
    uptime_seconds: float
