"""
PulseBMS Enhanced - MQTT Client Implementation
Handles bidirectional communication with edge devices
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage

from .config import settings
from .models import TelemetryData, DeviceCommand

logger = logging.getLogger(__name__)


class MQTTClient:
    """Async MQTT client for PulseBMS communication"""
    
    def __init__(self, broker_host: str, broker_port: int, 
                 username: str = "", password: str = ""):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        
        self.client = mqtt.Client()
        self.connected = False
        self.websocket_manager = None
        
        # Message handlers
        self.telemetry_handlers: Dict[str, Callable] = {}
        self.command_handlers: Dict[str, Callable] = {}
        
        # Set up MQTT callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
        
        # Authentication
        if username and password:
            self.client.username_pw_set(username, password)
    
    async def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}")
            
            # Connect to broker
            result = self.client.connect(
                self.broker_host, 
                self.broker_port, 
                settings.MQTT_KEEPALIVE
            )
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                # Start the network loop
                self.client.loop_start()
                
                # Wait for connection
                await asyncio.sleep(1)
                
                if self.connected:
                    # Subscribe to telemetry topics
                    await self._subscribe_to_topics()
                    logger.info("MQTT client connected and subscribed successfully")
                    return True
                else:
                    logger.error("MQTT connection failed")
                    return False
            else:
                logger.error(f"MQTT connection failed with code: {result}")
                return False
                
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info("MQTT client disconnected")
    
    async def _subscribe_to_topics(self):
        """Subscribe to MQTT topics"""
        topics = [
            (settings.MQTT_TOPIC_TELEMETRY, settings.MQTT_QOS),
            (settings.MQTT_TOPIC_COMMANDS, settings.MQTT_QOS),
            (settings.MQTT_TOPIC_COORDINATOR, settings.MQTT_QOS),
        ]
        
        for topic, qos in topics:
            result, _ = self.client.subscribe(topic, qos)
            if result == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Subscribed to topic: {topic}")
            else:
                logger.error(f"Failed to subscribe to topic: {topic}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful MQTT connection"""
        if rc == 0:
            self.connected = True
            logger.info("MQTT client connected successfully")
        else:
            logger.error(f"MQTT connection failed with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        self.connected = False
        logger.warning(f"MQTT client disconnected with code: {rc}")
    
    def _on_message(self, client, userdata, msg: MQTTMessage):
        """Handle incoming MQTT messages"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.debug(f"Received message on topic: {topic}")
            
            # Parse topic to extract site_id and device_id
            topic_parts = topic.split('/')
            if len(topic_parts) >= 4:
                site_id = topic_parts[1]
                device_id = topic_parts[2]
                message_type = topic_parts[3]
                
                # Handle different message types
                if message_type == "telemetry":
                    asyncio.create_task(self._handle_telemetry(site_id, device_id, payload))
                elif message_type == "commands":
                    asyncio.create_task(self._handle_command_response(site_id, device_id, payload))
                elif message_type == "allocation" and topic_parts[2] == "coordinator":
                    asyncio.create_task(self._handle_coordinator_message(site_id, payload))
            else:
                logger.warning(f"Invalid topic format: {topic}")
                
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for successful message publish"""
        logger.debug(f"Message published with mid: {mid}")
    
    async def _handle_telemetry(self, site_id: str, device_id: str, payload: str):
        """Handle telemetry data from edge devices"""
        try:
            # Parse JSON payload
            data = json.loads(payload)
            
            # Create telemetry object
            telemetry = TelemetryData(
                site_id=site_id,
                device_id=device_id,
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
                voltage=data.get('voltage', 0.0),
                current=data.get('current', 0.0),
                temperature=data.get('temperature', 0.0),
                soc=data.get('soc', 0.0),
                soh=data.get('soh', 100.0),
                cell_voltages=data.get('cell_voltages', []),
                cell_temperatures=data.get('cell_temperatures', []),
                metadata=data.get('metadata', {})
            )
            
            # Store in database (implement database storage)
            await self._store_telemetry(telemetry)
            
            # Forward to WebSocket clients
            if self.websocket_manager:
                await self.websocket_manager.broadcast_telemetry(telemetry.dict())
            
            logger.debug(f"Processed telemetry from {site_id}/{device_id}")
            
        except Exception as e:
            logger.error(f"Error processing telemetry from {site_id}/{device_id}: {e}")
    
    async def _handle_command_response(self, site_id: str, device_id: str, payload: str):
        """Handle command responses from edge devices"""
        try:
            data = json.loads(payload)
            logger.info(f"Command response from {site_id}/{device_id}: {data}")
            
            # Forward to WebSocket clients
            if self.websocket_manager:
                await self.websocket_manager.broadcast_command_response(site_id, device_id, data)
                
        except Exception as e:
            logger.error(f"Error processing command response from {site_id}/{device_id}: {e}")
    
    async def _handle_coordinator_message(self, site_id: str, payload: str):
        """Handle coordinator allocation messages"""
        try:
            data = json.loads(payload)
            logger.info(f"Coordinator message for site {site_id}: {data}")
            
            # Forward to WebSocket clients
            if self.websocket_manager:
                await self.websocket_manager.broadcast_coordinator_update(site_id, data)
                
        except Exception as e:
            logger.error(f"Error processing coordinator message for site {site_id}: {e}")
    
    async def _store_telemetry(self, telemetry: TelemetryData):
        """Store telemetry data in database"""
        # TODO: Implement database storage
        # This will be implemented when we create the database module
        pass
    
    async def send_command(self, site_id: str, device_id: str, command: DeviceCommand) -> bool:
        """Send command to edge device"""
        try:
            topic = f"pulsebms/{site_id}/{device_id}/commands"
            payload = json.dumps(command.dict())
            
            result = self.client.publish(topic, payload, qos=settings.MQTT_QOS)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Command sent to {site_id}/{device_id}: {command.command_type}")
                return True
            else:
                logger.error(f"Failed to send command to {site_id}/{device_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending command to {site_id}/{device_id}: {e}")
            return False
    
    async def send_coordinator_allocation(self, site_id: str, allocation: Dict[str, Any]) -> bool:
        """Send power allocation from coordinator"""
        try:
            topic = f"pulsebms/{site_id}/coordinator/allocation"
            payload = json.dumps(allocation)
            
            result = self.client.publish(topic, payload, qos=settings.MQTT_QOS)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Allocation sent to site {site_id}")
                return True
            else:
                logger.error(f"Failed to send allocation to site {site_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending allocation to site {site_id}: {e}")
            return False
    
    def set_websocket_manager(self, websocket_manager):
        """Set WebSocket manager for real-time updates"""
        self.websocket_manager = websocket_manager
    
    def add_telemetry_handler(self, device_id: str, handler: Callable):
        """Add custom telemetry handler for specific device"""
        self.telemetry_handlers[device_id] = handler
    
    def add_command_handler(self, device_id: str, handler: Callable):
        """Add custom command handler for specific device"""
        self.command_handlers[device_id] = handler
