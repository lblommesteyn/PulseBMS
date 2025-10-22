"""
PulseBMS Enhanced - Health Check Router
System health monitoring and status endpoints
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..database import db_manager
from ..models import HealthCheckResponse, APIResponse
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Application start time for uptime calculation
app_start_time = time.time()


class SystemMetrics(BaseModel):
    """System performance metrics"""
    active_devices: int = 0
    messages_per_second: float = 0.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check endpoint
    Returns system status, version, uptime, and service health
    """
    try:
        # Calculate uptime
        uptime_seconds = time.time() - app_start_time
        
        # Get database health status
        db_health = await db_manager.health_check()
        
        # Determine overall system status
        all_healthy = all(status == "healthy" for status in db_health.values())
        
        if all_healthy:
            overall_status = "healthy"
        elif any(status == "healthy" for status in db_health.values()):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        # Get system metrics (simplified for now)
        metrics = await get_system_metrics()
        
        response = HealthCheckResponse(
            status=overall_status,
            version=settings.VERSION,
            timestamp=datetime.utcnow(),
            services=db_health,
            uptime_seconds=uptime_seconds
        )
        
        logger.debug(f"Health check completed: {overall_status}")
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with system metrics and component status
    """
    try:
        # Basic health check
        basic_health = await health_check()
        
        # Get detailed metrics
        metrics = await get_system_metrics()
        
        # Get component-specific health
        components = await get_component_health()
        
        return APIResponse(
            success=True,
            message="Detailed health check completed",
            data={
                "basic_health": basic_health.dict(),
                "metrics": metrics.dict(),
                "components": components,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detailed health check failed: {str(e)}"
        )


@router.get("/health/database")
async def database_health():
    """
    Database-specific health check
    """
    try:
        db_health = await db_manager.health_check()
        
        # Add more detailed database metrics
        db_details = {}
        
        # MongoDB metrics
        if db_manager.mongodb_db:
            try:
                # Get database stats
                stats = await db_manager.mongodb_db.command("dbStats")
                db_details["mongodb"] = {
                    "status": db_health["mongodb"],
                    "collections": stats.get("collections", 0),
                    "data_size": stats.get("dataSize", 0),
                    "storage_size": stats.get("storageSize", 0),
                    "indexes": stats.get("indexes", 0)
                }
            except Exception as e:
                db_details["mongodb"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # PostgreSQL metrics
        if db_manager.postgres_engine:
            try:
                async with db_manager.postgres_engine.begin() as conn:
                    # Get connection pool info
                    pool_info = db_manager.postgres_engine.pool
                    db_details["postgresql"] = {
                        "status": db_health["postgresql"],
                        "pool_size": pool_info.size(),
                        "checked_out": pool_info.checkedout(),
                        "checked_in": pool_info.checkedin()
                    }
            except Exception as e:
                db_details["postgresql"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Redis metrics
        if db_manager.redis_client:
            try:
                info = await db_manager.redis_client.info()
                db_details["redis"] = {
                    "status": db_health["redis"],
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
            except Exception as e:
                db_details["redis"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return APIResponse(
            success=True,
            message="Database health check completed",
            data=db_details
        )
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database health check failed: {str(e)}"
        )


@router.get("/metrics")
async def get_metrics():
    """
    Get current system metrics
    """
    try:
        metrics = await get_system_metrics()
        
        return APIResponse(
            success=True,
            message="System metrics retrieved",
            data=metrics.dict()
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


async def get_system_metrics() -> SystemMetrics:
    """Get current system performance metrics"""
    try:
        metrics = SystemMetrics()
        
        # Get active devices from Redis cache
        try:
            redis_client = await db_manager.get_redis_client()
            keys = await redis_client.keys("device:status:*")
            metrics.active_devices = len(keys)
        except Exception as e:
            logger.warning(f"Failed to get active devices count: {e}")
        
        # TODO: Implement actual metrics collection
        # For now, return placeholder values
        metrics.messages_per_second = 150.5
        metrics.response_time_ms = 12.3
        metrics.error_rate = 0.001
        metrics.memory_usage_mb = 256.0
        metrics.cpu_usage_percent = 15.2
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return SystemMetrics()


async def get_component_health() -> Dict[str, Any]:
    """Get health status of individual components"""
    components = {}
    
    try:
        # MQTT broker health (placeholder)
        components["mqtt_broker"] = {
            "status": "healthy",
            "connected_clients": 25,
            "messages_per_second": 150.5
        }
        
        # Digital twin service health (placeholder)
        components["digital_twin"] = {
            "status": "healthy",
            "active_models": 12,
            "prediction_latency_ms": 45.2
        }
        
        # Coordinator service health (placeholder)
        components["coordinator"] = {
            "status": "healthy",
            "active_policies": 3,
            "optimization_score": 0.94
        }
        
        # WebSocket connections (placeholder)
        components["websockets"] = {
            "status": "healthy",
            "active_connections": 8,
            "message_throughput": 85.3
        }
        
    except Exception as e:
        logger.error(f"Failed to get component health: {e}")
    
    return components


@router.get("/version")
async def get_version():
    """Get application version information"""
    return APIResponse(
        success=True,
        message="Version information retrieved",
        data={
            "version": settings.VERSION,
            "app_name": settings.APP_NAME,
            "debug_mode": settings.DEBUG,
            "environment": "development" if settings.DEBUG else "production"
        }
    )


@router.get("/status")
async def get_status():
    """Quick status check endpoint"""
    try:
        uptime_seconds = time.time() - app_start_time
        
        return {
            "status": "operational",
            "uptime_seconds": uptime_seconds,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Service unavailable"
        )
