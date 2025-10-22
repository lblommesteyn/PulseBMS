"""
PulseBMS Enhanced - Configuration Management
"""

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "PulseBMS Enhanced"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Database - MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "pulsebms"
    
    # Database - PostgreSQL
    POSTGRES_URL: str = "postgresql://postgres:password@localhost:5432/pulsebms"
    
    # MQTT Broker
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: str = ""
    MQTT_KEEPALIVE: int = 60
    MQTT_QOS: int = 1
    
    # MQTT Topics
    MQTT_TOPIC_TELEMETRY: str = "pulsebms/+/+/telemetry"
    MQTT_TOPIC_COMMANDS: str = "pulsebms/+/+/commands"
    MQTT_TOPIC_COORDINATOR: str = "pulsebms/+/coordinator/allocation"
    
    # Redis (for caching and session management)
    REDIS_URL: str = "redis://localhost:6379"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # Digital Twin Service
    DIGITAL_TWIN_URL: str = "http://localhost:8001"
    PYBAMM_CACHE_SIZE: int = 100
    
    # Coordinator Service
    COORDINATOR_URL: str = "http://localhost:8002"
    MPC_HORIZON: int = 24  # hours
    RL_UPDATE_INTERVAL: int = 300  # seconds
    
    # Safety Constraints
    MAX_CELL_VOLTAGE: float = 4.2  # V
    MIN_CELL_VOLTAGE: float = 2.5  # V
    MAX_CELL_TEMP: float = 60.0    # °C
    MIN_CELL_TEMP: float = -20.0   # °C
    MAX_DISCHARGE_CURRENT: float = 100.0  # A
    MAX_CHARGE_CURRENT: float = 50.0      # A
    
    # Performance Settings
    TELEMETRY_BATCH_SIZE: int = 100
    WEBSOCKET_PING_INTERVAL: int = 20
    WEBSOCKET_PING_TIMEOUT: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
