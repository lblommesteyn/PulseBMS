"""
PulseBMS Enhanced - WebSocket Manager
Handles real-time communication with dashboard clients
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_subscriptions: Dict[str, List[str]] = {}  # client_id -> [site_ids]
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_subscriptions[client_id] = []
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                self.disconnect(client_id)
    
    async def send_personal_json(self, data: Dict[str, Any], client_id: str):
        """Send JSON data to specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Error sending JSON to client {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_message(self, message: str):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON data to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Error broadcasting JSON to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_to_site(self, site_id: str, data: Dict[str, Any]):
        """Broadcast data to clients subscribed to specific site"""
        disconnected_clients = []
        
        for client_id, subscriptions in self.client_subscriptions.items():
            if site_id in subscriptions and client_id in self.active_connections:
                try:
                    websocket = self.active_connections[client_id]
                    await websocket.send_json(data)
                except Exception as e:
                    logger.error(f"Error sending site data to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def subscribe_to_site(self, client_id: str, site_id: str):
        """Subscribe client to site updates"""
        if client_id in self.client_subscriptions:
            if site_id not in self.client_subscriptions[client_id]:
                self.client_subscriptions[client_id].append(site_id)
                logger.info(f"Client {client_id} subscribed to site {site_id}")
    
    async def unsubscribe_from_site(self, client_id: str, site_id: str):
        """Unsubscribe client from site updates"""
        if client_id in self.client_subscriptions:
            if site_id in self.client_subscriptions[client_id]:
                self.client_subscriptions[client_id].remove(site_id)
                logger.info(f"Client {client_id} unsubscribed from site {site_id}")
    
    async def broadcast_telemetry(self, telemetry_data: Dict[str, Any]):
        """Broadcast telemetry data to relevant clients"""
        site_id = telemetry_data.get('site_id')
        
        message = {
            "type": "telemetry",
            "data": telemetry_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if site_id:
            await self.broadcast_to_site(site_id, message)
        else:
            await self.broadcast_json(message)
    
    async def broadcast_device_status(self, site_id: str, device_id: str, status_data: Dict[str, Any]):
        """Broadcast device status update"""
        message = {
            "type": "device_status",
            "site_id": site_id,
            "device_id": device_id,
            "data": status_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_site(site_id, message)
    
    async def broadcast_fleet_status(self, site_id: str, fleet_data: Dict[str, Any]):
        """Broadcast fleet-wide status update"""
        message = {
            "type": "fleet_status",
            "site_id": site_id,
            "data": fleet_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_site(site_id, message)
    
    async def broadcast_command_response(self, site_id: str, device_id: str, response_data: Dict[str, Any]):
        """Broadcast command response from device"""
        message = {
            "type": "command_response",
            "site_id": site_id,
            "device_id": device_id,
            "data": response_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_site(site_id, message)
    
    async def broadcast_coordinator_update(self, site_id: str, allocation_data: Dict[str, Any]):
        """Broadcast coordinator allocation update"""
        message = {
            "type": "coordinator_allocation",
            "site_id": site_id,
            "data": allocation_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_site(site_id, message)
    
    async def broadcast_alert(self, site_id: str, alert_data: Dict[str, Any], severity: str = "info"):
        """Broadcast alert/alarm to clients"""
        message = {
            "type": "alert",
            "site_id": site_id,
            "severity": severity,
            "data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_site(site_id, message)
    
    async def send_dashboard_config(self, client_id: str, config_data: Dict[str, Any]):
        """Send dashboard configuration to specific client"""
        message = {
            "type": "dashboard_config",
            "data": config_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_personal_json(message, client_id)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about connected clients"""
        return {
            "total_connections": len(self.active_connections),
            "clients": [
                {
                    "client_id": client_id,
                    "subscriptions": subscriptions
                }
                for client_id, subscriptions in self.client_subscriptions.items()
            ]
        }
