import logging

logger = logging.getLogger("edge.actuators")

class ContactorState:
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PRECHARGE = "PRECHARGE"

class Actuators:
    def __init__(self):
        self.main_contactor = ContactorState.OPEN
        self.precharge_relay = ContactorState.OPEN
        self.cooling_pump_speed = 0.0 # 0-100%

    def close_contactors(self):
        """Sequence to safely close contactors"""
        logger.info("Starting precharge sequence...")
        self.precharge_relay = ContactorState.CLOSED
        # Simulate delay
        self.main_contactor = ContactorState.CLOSED
        self.precharge_relay = ContactorState.OPEN
        logger.info("Contactors CLOSED. HV Active.")

    def open_contactors(self):
        logger.info("Opening contactors. HV Inactive.")
        self.main_contactor = ContactorState.OPEN
        self.precharge_relay = ContactorState.OPEN

    def set_cooling(self, percent: float):
        self.cooling_pump_speed = max(0.0, min(100.0, percent))
        logger.info(f"Cooling pump set to {self.cooling_pump_speed}%")
