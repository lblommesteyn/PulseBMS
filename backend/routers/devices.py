"""
PulseBMS Enhanced - Device Management Router
Device configuration, registration, and status management
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError

from ..database import db_manager, cache_device_status, get_cached_device_status
from ..models import (
    DeviceInfo, DeviceStatus, SafetyConstraints, APIResponse,
    BatteryChemistry, TelemetryData
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/devices", response_model=APIResponse)
async def register_device(device_info: DeviceInfo):
    """
    Register a new battery device/pack in the system
    """
    try:
        # Validate device configuration
        if device_info.nominal_capacity <= 0:
            raise HTTPException(
                status_code=400,
                detail="Nominal capacity must be greater than 0"
            )
        
        if device_info.series_cells <= 0 or device_info.parallel_cells <= 0:
            raise HTTPException(
                status_code=400,
                detail="Cell configuration must have positive values"
            )
        
        # Check if device already exists
        postgres_session = await db_manager.get_postgres_session()
        
        # TODO: Implement PostgreSQL device storage
        # For now, we'll store in MongoDB as well
        mongodb_collection = await db_manager.get_mongodb_collection("device_configurations")
        
        existing_device = await mongodb_collection.find_one({
            "device_id": device_info.device_id
        })
        
        if existing_device:
            raise HTTPException(
                status_code=409,
                detail=f"Device {device_info.device_id} already exists"
            )
        
        # Store device configuration
        device_doc = device_info.dict()
        device_doc["created_at"] = datetime.utcnow()
        device_doc["updated_at"] = datetime.utcnow()
        
        result = await mongodb_collection.insert_one(device_doc)
        
        # Create default safety constraints
        safety_constraints = SafetyConstraints(device_id=device_info.device_id)
        await store_safety_constraints(safety_constraints)
        
        # Cache device status
        await cache_device_status(
            device_info.device_id,
            {
                "status": device_info.status.value,
                "last_seen": device_info.last_seen.isoformat() if device_info.last_seen else None,
                "site_id": device_info.site_id
            }
        )
        
        logger.info(f"Registered new device: {device_info.device_id}")
        
        return APIResponse(
            success=True,
            message=f"Device {device_info.device_id} registered successfully",
            data={
                "device_id": device_info.device_id,
                "mongodb_id": str(result.inserted_id)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register device {device_info.device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register device: {str(e)}"
        )


@router.get("/devices/{device_id}", response_model=APIResponse)
async def get_device(device_id: str):
    """
    Get device configuration and current status
    """
    try:
        # Get device configuration from MongoDB
        mongodb_collection = await db_manager.get_mongodb_collection("device_configurations")
        device_doc = await mongodb_collection.find_one({"device_id": device_id})
        
        if not device_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Device {device_id} not found"
            )
        
        # Get cached status
        cached_status = await get_cached_device_status(device_id)
        
        # Get safety constraints
        safety_constraints = await get_safety_constraints(device_id)
        
        # Get latest telemetry summary
        telemetry_summary = await get_latest_telemetry_summary(device_id)
        
        # Combine all information
        device_data = {
            "configuration": device_doc,
            "current_status": cached_status,
            "safety_constraints": safety_constraints,
            "latest_telemetry": telemetry_summary
        }
        
        return APIResponse(
            success=True,
            message=f"Device {device_id} information retrieved",
            data=device_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get device {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get device: {str(e)}"
        )


@router.put("/devices/{device_id}", response_model=APIResponse)
async def update_device(device_id: str, device_update: dict):
    """
    Update device configuration
    """
    try:
        # Get existing device
        mongodb_collection = await db_manager.get_mongodb_collection("device_configurations")
        existing_device = await mongodb_collection.find_one({"device_id": device_id})
        
        if not existing_device:
            raise HTTPException(
                status_code=404,
                detail=f"Device {device_id} not found"
            )
        
        # Update fields
        update_data = {
            **device_update,
            "updated_at": datetime.utcnow()
        }
        
        # Remove fields that shouldn't be updated
        protected_fields = ["device_id", "_id", "created_at"]
        for field in protected_fields:
            update_data.pop(field, None)
        
        # Update in MongoDB
        result = await mongodb_collection.update_one(
            {"device_id": device_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            logger.warning(f"No changes made to device {device_id}")
        
        # Update cache if status changed
        if "status" in device_update:
            cached_status = await get_cached_device_status(device_id) or {}
            cached_status["status"] = device_update["status"]
            await cache_device_status(device_id, cached_status)
        
        logger.info(f"Updated device: {device_id}")
        
        return APIResponse(
            success=True,
            message=f"Device {device_id} updated successfully",
            data={"modified_count": result.modified_count}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update device {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update device: {str(e)}"
        )


@router.delete("/devices/{device_id}", response_model=APIResponse)
async def delete_device(device_id: str, force: bool = Query(False)):
    """
    Delete a device from the system
    """
    try:
        # Check if device exists
        mongodb_collection = await db_manager.get_mongodb_collection("device_configurations")
        device = await mongodb_collection.find_one({"device_id": device_id})
        
        if not device:
            raise HTTPException(
                status_code=404,
                detail=f"Device {device_id} not found"
            )
        
        # Check if device is currently active (unless force delete)
        if not force:
            cached_status = await get_cached_device_status(device_id)
            if cached_status and cached_status.get("status") == "online":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot delete active device {device_id}. Use force=true to override."
                )
        
        # Delete device configuration
        result = await mongodb_collection.delete_one({"device_id": device_id})
        
        # Delete safety constraints
        safety_collection = await db_manager.get_mongodb_collection("safety_constraints")
        await safety_collection.delete_many({"device_id": device_id})
        
        # Clear cache
        redis_client = await db_manager.get_redis_client()
        await redis_client.delete(f"device:status:{device_id}")
        
        logger.info(f"Deleted device: {device_id}")
        
        return APIResponse(
            success=True,
            message=f"Device {device_id} deleted successfully",
            data={"deleted_count": result.deleted_count}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete device {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete device: {str(e)}"
        )


@router.get("/devices", response_model=APIResponse)
async def list_devices(
    site_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    chemistry: Optional[BatteryChemistry] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    List devices with optional filtering
    """
    try:
        # Build query filter
        query_filter = {}
        
        if site_id:
            query_filter["site_id"] = site_id
        
        if status:
            query_filter["status"] = status
        
        if chemistry:
            query_filter["chemistry"] = chemistry.value
        
        # Get devices from MongoDB
        mongodb_collection = await db_manager.get_mongodb_collection("device_configurations")
        
        # Get total count
        total_count = await mongodb_collection.count_documents(query_filter)
        
        # Get paginated results
        cursor = mongodb_collection.find(query_filter).skip(offset).limit(limit)
        devices = await cursor.to_list(length=limit)
        
        # Enhance with cached status for each device
        enhanced_devices = []
        for device in devices:
            cached_status = await get_cached_device_status(device["device_id"])
            device["cached_status"] = cached_status
            enhanced_devices.append(device)
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(enhanced_devices)} devices",
            data={
                "devices": enhanced_devices,
                "total_count": total_count,
                "offset": offset,
                "limit": limit
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list devices: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list devices: {str(e)}"
        )


@router.get("/devices/{device_id}/safety", response_model=APIResponse)
async def get_device_safety_constraints(device_id: str):
    """
    Get safety constraints for a device
    """
    try:
        safety_constraints = await get_safety_constraints(device_id)
        
        if not safety_constraints:
            raise HTTPException(
                status_code=404,
                detail=f"Safety constraints not found for device {device_id}"
            )
        
        return APIResponse(
            success=True,
            message=f"Safety constraints for device {device_id}",
            data=safety_constraints
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get safety constraints for {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get safety constraints: {str(e)}"
        )


@router.put("/devices/{device_id}/safety", response_model=APIResponse)
async def update_device_safety_constraints(device_id: str, constraints: SafetyConstraints):
    """
    Update safety constraints for a device
    """
    try:
        # Verify device exists
        device_collection = await db_manager.get_mongodb_collection("device_configurations")
        device = await device_collection.find_one({"device_id": device_id})
        
        if not device:
            raise HTTPException(
                status_code=404,
                detail=f"Device {device_id} not found"
            )
        
        # Ensure constraint device_id matches
        constraints.device_id = device_id
        
        # Update safety constraints
        result = await store_safety_constraints(constraints)
        
        logger.info(f"Updated safety constraints for device: {device_id}")
        
        return APIResponse(
            success=True,
            message=f"Safety constraints updated for device {device_id}",
            data={"updated": result}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update safety constraints for {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update safety constraints: {str(e)}"
        )


# Helper functions

async def store_safety_constraints(constraints: SafetyConstraints) -> bool:
    """Store or update safety constraints in database"""
    try:
        mongodb_collection = await db_manager.get_mongodb_collection("safety_constraints")
        
        constraints_doc = constraints.dict()
        constraints_doc["updated_at"] = datetime.utcnow()
        
        result = await mongodb_collection.replace_one(
            {"device_id": constraints.device_id},
            constraints_doc,
            upsert=True
        )
        
        return result.acknowledged
        
    except Exception as e:
        logger.error(f"Failed to store safety constraints: {e}")
        return False


async def get_safety_constraints(device_id: str) -> Optional[dict]:
    """Get safety constraints from database"""
    try:
        mongodb_collection = await db_manager.get_mongodb_collection("safety_constraints")
        return await mongodb_collection.find_one({"device_id": device_id})
        
    except Exception as e:
        logger.error(f"Failed to get safety constraints: {e}")
        return None


async def get_latest_telemetry_summary(device_id: str) -> Optional[dict]:
    """Get latest telemetry summary for a device"""
    try:
        mongodb_collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        # Get the most recent telemetry record
        cursor = mongodb_collection.find(
            {"device_id": device_id}
        ).sort("timestamp", -1).limit(1)
        
        latest = await cursor.to_list(length=1)
        
        if latest:
            telemetry = latest[0]
            return {
                "timestamp": telemetry.get("timestamp"),
                "voltage": telemetry.get("measurements", {}).get("voltage"),
                "current": telemetry.get("measurements", {}).get("current"),
                "soc": telemetry.get("measurements", {}).get("soc"),
                "soh": telemetry.get("measurements", {}).get("soh"),
                "temperature": telemetry.get("measurements", {}).get("temperature"),
                "alarm_flags": telemetry.get("safety", {}).get("alarm_flags", [])
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get latest telemetry for {device_id}: {e}")
        return None
