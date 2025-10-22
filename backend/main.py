"""
PulseBMS Enhanced - Backend API Server
Main FastAPI application with async MQTT integration
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_database
from .mqtt_client import MQTTClient
from .routers import devices, telemetry, fleet, health
from .websocket_manager import WebSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
mqtt_client: MQTTClient = None
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    logger.info("Starting PulseBMS Enhanced Backend")
    
    # Initialize database connections
    await init_database()
    
    # Initialize MQTT client
    global mqtt_client
    mqtt_client = MQTTClient(
        broker_host=settings.MQTT_BROKER_HOST,
        broker_port=settings.MQTT_BROKER_PORT,
        username=settings.MQTT_USERNAME,
        password=settings.MQTT_PASSWORD
    )
    
    # Connect to MQTT broker
    await mqtt_client.connect()
    
    # Set up MQTT message routing to WebSocket
    mqtt_client.set_websocket_manager(websocket_manager)
    
    logger.info("Backend services initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down PulseBMS Enhanced Backend")
    if mqtt_client:
        await mqtt_client.disconnect()


# Create FastAPI application
app = FastAPI(
    title="PulseBMS Enhanced API",
    description="Advanced Battery Management System for Second-Life EV Batteries",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(devices.router, prefix="/api/v1", tags=["devices"])
app.include_router(telemetry.router, prefix="/api/v1", tags=["telemetry"])
app.include_router(fleet.router, prefix="/api/v1", tags=["fleet"])

# Serve static files (dashboard build)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now (can add client commands later)
            await websocket_manager.send_personal_message(f"Echo: {data}", client_id)
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PulseBMS Enhanced API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Advanced Battery Management System for Second-Life EV Batteries"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )
