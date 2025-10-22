"""
PulseBMS Enhanced - Fleet Management Router
Fleet-wide operations, status monitoring, and coordination
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
import pymongo

from ..database import db_manager, get_cached_device_status
from ..models import FleetStatus, PowerAllocation, APIResponse, DeviceStatus
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/fleet/{site_id}/status", response_model=APIResponse)
async def get_fleet_status(site_id: str):
    """
    Get comprehensive fleet status for a site
    """
    try:
        # Get all devices for the site
        devices = await get_site_devices(site_id)
        
        if not devices:
            raise HTTPException(
                status_code=404,
                detail=f"No devices found for site {site_id}"
            )
        
        # Calculate fleet metrics
        fleet_metrics = await calculate_fleet_metrics(site_id, devices)
        
        # Get active alarms across the fleet
        fleet_alarms = await get_fleet_alarms(site_id)
        
        # Get current power allocation status
        allocation_status = await get_current_allocation_status(site_id)
        
        fleet_status = FleetStatus(
            site_id=site_id,
            total_devices=fleet_metrics["total_devices"],
            online_devices=fleet_metrics["online_devices"],
            total_capacity=fleet_metrics["total_capacity"],
            available_capacity=fleet_metrics["available_capacity"],
            total_power=fleet_metrics["total_power"],
            average_soc=fleet_metrics["average_soc"],
            average_soh=fleet_metrics["average_soh"],
            max_charge_power=fleet_metrics["max_charge_power"],
            max_discharge_power=fleet_metrics["max_discharge_power"],
            active_alarms=fleet_alarms["alarm_types"],
            devices_in_alarm=fleet_alarms["devices_in_alarm"]
        )
        
        return APIResponse(
            success=True,
            message=f"Fleet status for site {site_id}",
            data={
                "fleet_status": fleet_status.dict(),
                "device_details": fleet_metrics["device_details"],
                "allocation_status": allocation_status,
                "fleet_alarms": fleet_alarms
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fleet status for site {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fleet status: {str(e)}"
        )


@router.get("/fleet/{site_id}/devices", response_model=APIResponse)
async def get_fleet_devices(
    site_id: str,
    include_offline: bool = Query(True, description="Include offline devices"),
    include_telemetry: bool = Query(False, description="Include latest telemetry")
):
    """
    Get all devices in a fleet with optional telemetry data
    """
    try:
        devices = await get_site_devices(site_id)
        
        if not devices:
            return APIResponse(
                success=True,
                message=f"No devices found for site {site_id}",
                data={"devices": [], "count": 0}
            )
        
        # Filter devices based on parameters
        filtered_devices = []
        for device in devices:
            cached_status = await get_cached_device_status(device["device_id"])
            device_status = cached_status.get("status", "offline") if cached_status else "offline"
            
            # Filter offline devices if requested
            if not include_offline and device_status == "offline":
                continue
            
            device_info = {
                **device,
                "current_status": cached_status,
                "online": device_status == "online"
            }
            
            # Include telemetry if requested
            if include_telemetry and device_status == "online":
                latest_telemetry = await get_device_latest_telemetry(device["device_id"])
                device_info["latest_telemetry"] = latest_telemetry
            
            filtered_devices.append(device_info)
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(filtered_devices)} devices for site {site_id}",
            data={
                "site_id": site_id,
                "devices": filtered_devices,
                "count": len(filtered_devices),
                "total_devices": len(devices)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fleet devices for site {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fleet devices: {str(e)}"
        )


@router.get("/fleet/{site_id}/summary", response_model=APIResponse)
async def get_fleet_summary(
    site_id: str,
    period: str = Query("24h", description="Summary period", regex="^(1h|6h|24h|7d|30d)$")
):
    """
    Get fleet performance summary over a specified period
    """
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        period_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        start_time = end_time - period_map[period]
        
        # Get fleet devices
        devices = await get_site_devices(site_id)
        device_ids = [d["device_id"] for d in devices]
        
        # Get aggregated fleet metrics
        fleet_summary = await calculate_fleet_summary(site_id, device_ids, start_time, end_time)
        
        return APIResponse(
            success=True,
            message=f"Fleet summary for site {site_id} over {period}",
            data={
                "site_id": site_id,
                "period": period,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "summary": fleet_summary
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fleet summary for site {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fleet summary: {str(e)}"
        )


@router.post("/fleet/{site_id}/allocation", response_model=APIResponse)
async def set_power_allocation(site_id: str, allocation: PowerAllocation):
    """
    Set power allocation for fleet (typically called by coordinator service)
    """
    try:
        # Validate allocation
        if allocation.site_id != site_id:
            raise HTTPException(
                status_code=400,
                detail="Site ID in URL must match allocation site ID"
            )
        
        # Verify all devices in allocation exist
        devices = await get_site_devices(site_id)
        device_ids = {d["device_id"] for d in devices}
        
        for device_id in allocation.device_allocations:
            if device_id not in device_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Device {device_id} not found in site {site_id}"
                )
        
        # Store allocation in database
        allocation_doc = allocation.dict()
        allocation_doc["created_at"] = datetime.utcnow()
        
        mongodb_collection = await db_manager.get_mongodb_collection("power_allocations")
        result = await mongodb_collection.insert_one(allocation_doc)
        
        # Cache current allocation in Redis
        redis_client = await db_manager.get_redis_client()
        allocation_key = f"allocation:current:{site_id}"
        await redis_client.setex(
            allocation_key,
            3600,  # 1 hour TTL
            str(allocation_doc)
        )
        
        # TODO: Send allocation commands to devices via MQTT
        # This would be implemented when MQTT client is integrated
        
        logger.info(f"Set power allocation for site {site_id}")
        
        return APIResponse(
            success=True,
            message=f"Power allocation set for site {site_id}",
            data={
                "allocation_id": str(result.inserted_id),
                "total_allocated_power": sum(allocation.device_allocations.values()),
                "device_count": len(allocation.device_allocations)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set power allocation for site {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set power allocation: {str(e)}"
        )


@router.get("/fleet/{site_id}/allocation/current", response_model=APIResponse)
async def get_current_power_allocation(site_id: str):
    """
    Get current power allocation for a site
    """
    try:
        allocation_status = await get_current_allocation_status(site_id)
        
        if not allocation_status:
            return APIResponse(
                success=True,
                message=f"No current allocation found for site {site_id}",
                data={"allocation": None}
            )
        
        return APIResponse(
            success=True,
            message=f"Current power allocation for site {site_id}",
            data={"allocation": allocation_status}
        )
        
    except Exception as e:
        logger.error(f"Failed to get current allocation for site {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current allocation: {str(e)}"
        )


@router.get("/fleet/{site_id}/alarms", response_model=APIResponse)
async def get_fleet_alarms(site_id: str):
    """
    Get active alarms across the entire fleet
    """
    try:
        fleet_alarms = await get_fleet_alarms(site_id)
        
        return APIResponse(
            success=True,
            message=f"Fleet alarms for site {site_id}",
            data=fleet_alarms
        )
        
    except Exception as e:
        logger.error(f"Failed to get fleet alarms for site {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fleet alarms: {str(e)}"
        )


@router.get("/fleet/{site_id}/performance", response_model=APIResponse)
async def get_fleet_performance_metrics(
    site_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """
    Get detailed performance metrics for the fleet
    """
    try:
        # Set default time range (last 24 hours)
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=1)
        
        # Get performance metrics
        performance_data = await calculate_fleet_performance(site_id, start_time, end_time)
        
        return APIResponse(
            success=True,
            message=f"Fleet performance metrics for site {site_id}",
            data={
                "site_id": site_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "performance": performance_data
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fleet performance for site {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fleet performance: {str(e)}"
        )


# Helper functions

async def get_site_devices(site_id: str) -> List[dict]:
    """Get all devices for a specific site"""
    try:
        mongodb_collection = await db_manager.get_mongodb_collection("device_configurations")
        cursor = mongodb_collection.find({"site_id": site_id})
        return await cursor.to_list(length=1000)
        
    except Exception as e:
        logger.error(f"Failed to get site devices: {e}")
        return []


async def calculate_fleet_metrics(site_id: str, devices: List[dict]) -> Dict[str, Any]:
    """Calculate comprehensive fleet metrics"""
    try:
        total_devices = len(devices)
        online_devices = 0
        total_capacity = 0.0
        available_capacity = 0.0
        total_power = 0.0
        soc_values = []
        soh_values = []
        max_charge_power = 0.0
        max_discharge_power = 0.0
        device_details = []
        
        for device in devices:
            device_id = device["device_id"]
            
            # Get cached status
            cached_status = await get_cached_device_status(device_id)
            is_online = cached_status and cached_status.get("status") == "online"
            
            if is_online:
                online_devices += 1
            
            # Add device capacity
            capacity = device.get("nominal_capacity", 0)
            total_capacity += capacity
            
            # Calculate available capacity based on SoH and online status
            soh = device.get("soh", 100) if is_online else 0
            if is_online:
                available_capacity += capacity * (soh / 100)
                soh_values.append(soh)
                
                # Get current power and SoC from cached status
                current_power = cached_status.get("current_power", 0)
                current_soc = cached_status.get("current_soc", 0)
                
                total_power += current_power
                soc_values.append(current_soc)
            
            # Add power limits
            max_charge_power += device.get("max_charge_power", 0)
            max_discharge_power += device.get("max_discharge_power", 0)
            
            # Device detail
            device_details.append({
                "device_id": device_id,
                "name": device.get("name"),
                "online": is_online,
                "capacity": capacity,
                "soc": cached_status.get("current_soc", 0) if cached_status else 0,
                "soh": soh,
                "power": cached_status.get("current_power", 0) if cached_status else 0
            })
        
        return {
            "total_devices": total_devices,
            "online_devices": online_devices,
            "total_capacity": total_capacity,
            "available_capacity": available_capacity,
            "total_power": total_power,
            "average_soc": sum(soc_values) / len(soc_values) if soc_values else 0,
            "average_soh": sum(soh_values) / len(soh_values) if soh_values else 0,
            "max_charge_power": max_charge_power,
            "max_discharge_power": max_discharge_power,
            "device_details": device_details
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate fleet metrics: {e}")
        return {}


async def get_fleet_alarms(site_id: str) -> Dict[str, Any]:
    """Get active alarms across the fleet"""
    try:
        # Get recent telemetry with alarms (last hour)
        start_time = datetime.utcnow() - timedelta(hours=1)
        
        mongodb_collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        cursor = mongodb_collection.find({
            "site_id": site_id,
            "timestamp": {"$gte": start_time},
            "safety.alarm_flags": {"$exists": True, "$not": {"$size": 0}}
        })
        
        alarm_records = await cursor.to_list(length=1000)
        
        # Process alarms
        device_alarms = {}
        all_alarm_types = set()
        
        for record in alarm_records:
            device_id = record["device_id"]
            alarm_flags = record.get("safety", {}).get("alarm_flags", [])
            
            if device_id not in device_alarms:
                device_alarms[device_id] = set()
            
            for flag in alarm_flags:
                device_alarms[device_id].add(flag)
                all_alarm_types.add(flag)
        
        return {
            "devices_in_alarm": len(device_alarms),
            "alarm_types": list(all_alarm_types),
            "device_alarms": {k: list(v) for k, v in device_alarms.items()},
            "total_alarm_count": sum(len(v) for v in device_alarms.values())
        }
        
    except Exception as e:
        logger.error(f"Failed to get fleet alarms: {e}")
        return {"devices_in_alarm": 0, "alarm_types": [], "device_alarms": {}}


async def get_current_allocation_status(site_id: str) -> Optional[Dict[str, Any]]:
    """Get current power allocation status from Redis cache"""
    try:
        redis_client = await db_manager.get_redis_client()
        allocation_key = f"allocation:current:{site_id}"
        
        allocation_data = await redis_client.get(allocation_key)
        if allocation_data:
            return eval(allocation_data)  # In production, use proper JSON parsing
        
        # If not in cache, get latest from MongoDB
        mongodb_collection = await db_manager.get_mongodb_collection("power_allocations")
        cursor = mongodb_collection.find({
            "site_id": site_id
        }).sort("timestamp", pymongo.DESCENDING).limit(1)
        
        latest_allocation = await cursor.to_list(length=1)
        return latest_allocation[0] if latest_allocation else None
        
    except Exception as e:
        logger.error(f"Failed to get current allocation status: {e}")
        return None


async def get_device_latest_telemetry(device_id: str) -> Optional[Dict[str, Any]]:
    """Get latest telemetry for a specific device"""
    try:
        mongodb_collection = await db_manager.get_mongodb_collection("telemetry_data")
        cursor = mongodb_collection.find({
            "device_id": device_id
        }).sort("timestamp", pymongo.DESCENDING).limit(1)
        
        latest_data = await cursor.to_list(length=1)
        return latest_data[0] if latest_data else None
        
    except Exception as e:
        logger.error(f"Failed to get latest telemetry for {device_id}: {e}")
        return None


async def calculate_fleet_summary(site_id: str, device_ids: List[str], 
                                  start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Calculate fleet summary statistics over a time period"""
    try:
        mongodb_collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        # MongoDB aggregation pipeline for fleet summary
        pipeline = [
            {
                "$match": {
                    "site_id": site_id,
                    "device_id": {"$in": device_ids},
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_records": {"$sum": 1},
                    "avg_fleet_power": {"$avg": "$measurements.power"},
                    "total_energy": {"$sum": "$measurements.power"},
                    "avg_fleet_soc": {"$avg": "$measurements.soc"},
                    "avg_fleet_temperature": {"$avg": "$measurements.temperature"},
                    "unique_devices": {"$addToSet": "$device_id"},
                    "total_alarms": {
                        "$sum": {
                            "$size": {
                                "$ifNull": ["$safety.alarm_flags", []]
                            }
                        }
                    }
                }
            }
        ]
        
        cursor = mongodb_collection.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        
        if not results:
            return {"no_data": True}
        
        summary = results[0]
        
        # Calculate additional metrics
        total_hours = (end_time - start_time).total_seconds() / 3600
        energy_throughput = abs(summary.get("total_energy", 0)) * total_hours / 1000  # kWh
        
        return {
            "active_devices": len(summary.get("unique_devices", [])),
            "data_points": summary.get("total_records", 0),
            "avg_fleet_power": round(summary.get("avg_fleet_power", 0), 2),
            "energy_throughput_kwh": round(energy_throughput, 3),
            "avg_fleet_soc": round(summary.get("avg_fleet_soc", 0), 2),
            "avg_fleet_temperature": round(summary.get("avg_fleet_temperature", 0), 2),
            "total_alarms": summary.get("total_alarms", 0),
            "uptime_percent": round((summary.get("total_records", 0) / (len(device_ids) * total_hours * 3600)) * 100, 2)
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate fleet summary: {e}")
        return {"error": str(e)}


async def calculate_fleet_performance(site_id: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Calculate detailed fleet performance metrics"""
    try:
        # TODO: Implement detailed performance calculations
        # This would include efficiency metrics, degradation rates, optimization performance, etc.
        
        return {
            "efficiency_score": 0.94,
            "optimization_performance": 0.89,
            "average_degradation_rate": 0.002,
            "power_utilization": 0.87,
            "availability": 0.98
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate fleet performance: {e}")
        return {}
