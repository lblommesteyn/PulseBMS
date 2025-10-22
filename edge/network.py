import json
import logging
import paho.mqtt.client as mqtt

logger = logging.getLogger("edge.network")

class NetworkManager:
    def __init__(self, device_id: str, broker: str = "localhost"):
        self.device_id = device_id
        self.client = mqtt.Client(client_id=device_id)
        self.broker = broker
        self.connected = False
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT Broker")
        else:
            logger.error(f"Failed to connect: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        logger.warning("Disconnected from MQTT Broker")

    def connect(self):
        try:
            self.client.connect(self.broker, 1883, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Connection failed: {e}")

    def publish_telemetry(self, data: dict):
        if not self.connected:
            return
        
        topic = f"pulsebms/devices/{self.device_id}/telemetry"
        payload = json.dumps(data)
        self.client.publish(topic, payload, qos=1)

    def shutdown(self):
        self.client.loop_stop()
        self.client.disconnect()
