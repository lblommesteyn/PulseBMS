import json
import logging
import httpx
import paho.mqtt.client as mqtt

logger = logging.getLogger("edge.network")

class NetworkManager:
    def __init__(self, device_id: str, broker: str = "localhost"):
        self.device_id = device_id
        self.client = mqtt.Client(client_id=device_id)
        self.broker = broker
        self.connected = False
        self.http_fallback = False
        self.api_url = "http://localhost:8000/api/v1"
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            self.http_fallback = False
            logger.info("Connected to MQTT Broker")
        else:
            logger.error(f"Failed to connect to MQTT: {rc}")
            self.http_fallback = True

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        logger.warning("Disconnected from MQTT Broker")
        self.http_fallback = True

    def connect(self):
        try:
            self.client.connect(self.broker, 1883, 60)
            self.client.loop_start()
        except Exception as e:
            logger.warning(f"MQTT Connection failed: {e}. Switching to HTTP Fallback.")
            self.http_fallback = True
            self.connected = True # Virtual connection via HTTP

    def publish_telemetry(self, data: dict):
        if self.http_fallback:
            try:
                # HTTP Fallback
                url = f"{self.api_url}/telemetry/{self.device_id}"
                # Convert timestamp objects to str if needed
                # But data is likely already dict/json serializable
                httpx.post(url, json=data, timeout=2.0)
            except Exception as e:
                logger.debug(f"HTTP Telemetry failed: {e}")
            return

        if not self.connected:
            return
        
        try:
            topic = f"pulsebms/devices/{self.device_id}/telemetry"
            payload = json.dumps(data)
            self.client.publish(topic, payload, qos=1)
        except Exception as e:
            logger.error(f"MQTT Publish failed: {e}. Enabling HTTP Fallback.")
            self.http_fallback = True

    def shutdown(self):
        if not self.http_fallback:
            self.client.loop_stop()
            self.client.disconnect()
