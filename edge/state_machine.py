import asyncio
import logging
from enum import Enum, auto
from .sensors import BatterySensors
from .actuators import Actuators
from .network import NetworkManager

logger = logging.getLogger("edge.fsm")

class State(Enum):
    INIT = auto()
    IDLE = auto()
    PRECHARGE = auto()
    ACTIVE = auto()
    FAULT = auto()
    SHUTDOWN = auto()

class EdgeStateMachine:
    def __init__(self, device_id: str):
        self.state = State.INIT
        self.sensors = BatterySensors()
        self.actuators = Actuators()
        self.network = NetworkManager(device_id)
        
    async def run_loop(self):
        self.network.connect()
        
        while True:
            try:
                # 1. Read Sensors
                data = self.sensors.read_all()
                
                # 2. Update State Logic
                await self._process_state(data)
                
                # 3. Publish Telemetry
                self.network.publish_telemetry({
                    "state": self.state.name,
                    "pack_voltage": data.pack_voltage,
                    "pack_current": data.pack_current
                })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"FSM Error: {e}")
                self.state = State.FAULT

    async def _process_state(self, data):
        if self.state == State.INIT:
            logger.info("System Initializing...")
            await asyncio.sleep(1)
            self.state = State.IDLE
            
        elif self.state == State.IDLE:
            if data.pack_voltage > 10.0: # Sim condition
                self.state = State.PRECHARGE
                
        elif self.state == State.PRECHARGE:
            self.actuators.close_contactors()
            self.state = State.ACTIVE
            
        elif self.state == State.ACTIVE:
            # Check limits
            if max(data.cell_temps) > 60.0:
                self.state = State.FAULT
            
            # Thermal Management
            if max(data.cell_temps) > 35.0:
                self.actuators.set_cooling(80.0)
            else:
                self.actuators.set_cooling(0.0)
                
        elif self.state == State.FAULT:
            self.actuators.open_contactors()
            logger.critical("FAULT STATE. CONTACTORS OPEN.")
