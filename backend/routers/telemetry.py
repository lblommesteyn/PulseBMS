"""
PulseBMS Enhanced - Telemetry Router
Telemetry data retrieval and analysis endpoints
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
import pymongo

from ..database import db_manager, get_device_telemetry, store_telemetry_data
from ..models import TelemetryData, APIResponse
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/telemetry/{device_id}", response_model=APIResponse)
async def get_device_telemetry_data(
    device_id: str,
    start_time: Optional[datetime] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time (ISO format)"),
    resolution: str = Query("raw", description="Data resolution", regex="^(raw|minute|hour|day)$"),
    metrics: Optional[List[str]] = Query(None, description="Specific metrics to retrieve"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of data points")
):
    """
    Get telemetry data for a specific device with filtering and aggregation options
    """
    try:
        # Set default time range if not provided (last 24 hours)
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=1)
        
        # Validate time range
        if start_time >= end_time:
            raise HTTPException(
                status_code=400,
                detail="Start time must be before end time"
            )
        
        # Check if time range is too large (prevent excessive data retrieval)
        time_diff = end_time - start_time
        if time_diff.days > 90:
            raise HTTPException(
                status_code=400,
                detail="Time range cannot exceed 90 days"
            )
        
        # Get telemetry data based on resolution
        if resolution == "raw":
            telemetry_data = await get_raw_telemetry(device_id, start_time, end_time, limit)
        else:
            telemetry_data = await get_aggregated_telemetry(device_id, start_time, end_time, resolution, limit)
        
        # Filter metrics if specified
        if metrics:
            telemetry_data = filter_telemetry_metrics(telemetry_data, metrics)
        
        # Calculate statistics
        statistics = calculate_telemetry_statistics(telemetry_data)
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(telemetry_data)} telemetry data points for device {device_id}",
            data={
                "device_id": device_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "resolution": resolution,
                "data_points": len(telemetry_data),
                "data": telemetry_data,
                "statistics": statistics
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get telemetry for device {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve telemetry data: {str(e)}"
        )


@router.get("/telemetry/{device_id}/latest", response_model=APIResponse)
async def get_latest_telemetry(device_id: str):
    """
    Get the most recent telemetry data for a device
    """
    try:
        mongodb_collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        # Get latest telemetry record
        cursor = mongodb_collection.find(
            {"device_id": device_id}
        ).sort("timestamp", pymongo.DESCENDING).limit(1)
        
        latest_data = await cursor.to_list(length=1)
        
        if not latest_data:
            raise HTTPException(
                status_code=404,
                detail=f"No telemetry data found for device {device_id}"
            )
        
        # Calculate time since last update
        latest_record = latest_data[0]
        timestamp = latest_record.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        time_since_last = (datetime.utcnow() - timestamp).total_seconds()
        
        return APIResponse(
            success=True,
            message=f"Latest telemetry for device {device_id}",
            data={
                "device_id": device_id,
                "timestamp": latest_record.get("timestamp"),
                "time_since_last_seconds": time_since_last,
                "measurements": latest_record.get("measurements", {}),
                "safety": latest_record.get("safety", {}),
                "metadata": latest_record.get("metadata", {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest telemetry for {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get latest telemetry: {str(e)}"
        )


@router.get("/telemetry/{device_id}/summary", response_model=APIResponse)
async def get_telemetry_summary(
    device_id: str,
    period: str = Query("24h", description="Summary period", regex="^(1h|6h|24h|7d|30d)$")
):
    """
    Get telemetry summary statistics for a device over a specified period
    """
    try:
        # Calculate time range based on period
        end_time = datetime.utcnow()
        
        period_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        start_time = end_time - period_map[period]
        
        # Get aggregated data
        mongodb_collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        # MongoDB aggregation pipeline for statistics
        pipeline = [
            {
                "$match": {
                    "device_id": device_id,
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "avg_voltage": {"$avg": "$measurements.voltage"},
                    "min_voltage": {"$min": "$measurements.voltage"},
                    "max_voltage": {"$max": "$measurements.voltage"},
                    "avg_current": {"$avg": "$measurements.current"},
                    "min_current": {"$min": "$measurements.current"},
                    "max_current": {"$max": "$measurements.current"},
                    "avg_temperature": {"$avg": "$measurements.temperature"},
                    "min_temperature": {"$min": "$measurements.temperature"},
                    "max_temperature": {"$max": "$measurements.temperature"},
                    "avg_soc": {"$avg": "$measurements.soc"},
                    "min_soc": {"$min": "$measurements.soc"},
                    "max_soc": {"$max": "$measurements.soc"},
                    "avg_soh": {"$avg": "$measurements.soh"},
                    "min_soh": {"$min": "$measurements.soh"},
                    "max_soh": {"$max": "$measurements.soh"},
                    "total_energy": {"$sum": "$measurements.power"},
                    "alarm_count": {
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
            raise HTTPException(
                status_code=404,
                detail=f"No telemetry data found for device {device_id} in the specified period"
            )
        
        summary = results[0]
        
        # Calculate additional metrics
        total_hours = (end_time - start_time).total_seconds() / 3600
        energy_throughput = abs(summary.get("total_energy", 0)) * total_hours / 1000  # Convert to kWh
        
        return APIResponse(
            success=True,
            message=f"Telemetry summary for device {device_id} over {period}",
            data={
                "device_id": device_id,
                "period": period,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "data_points": summary.get("count", 0),
                "voltage": {
                    "avg": round(summary.get("avg_voltage", 0), 2),
                    "min": round(summary.get("min_voltage", 0), 2),
                    "max": round(summary.get("max_voltage", 0), 2)
                },
                "current": {
                    "avg": round(summary.get("avg_current", 0), 2),
                    "min": round(summary.get("min_current", 0), 2),
                    "max": round(summary.get("max_current", 0), 2)
                },
                "temperature": {
                    "avg": round(summary.get("avg_temperature", 0), 2),
                    "min": round(summary.get("min_temperature", 0), 2),
                    "max": round(summary.get("max_temperature", 0), 2)
                },
                "soc": {
                    "avg": round(summary.get("avg_soc", 0), 2),
                    "min": round(summary.get("min_soc", 0), 2),
                    "max": round(summary.get("max_soc", 0), 2)
                },
                "soh": {
                    "avg": round(summary.get("avg_soh", 0), 2),
                    "min": round(summary.get("min_soh", 0), 2),
                    "max": round(summary.get("max_soh", 0), 2)
                },
                "energy_throughput_kwh": round(energy_throughput, 3),
                "alarm_count": summary.get("alarm_count", 0),
                "uptime_percent": round((summary.get("count", 0) / (total_hours * 3600)) * 100, 2)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get telemetry summary for {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get telemetry summary: {str(e)}"
        )


@router.post("/telemetry/{device_id}", response_model=APIResponse)
async def store_telemetry(device_id: str, telemetry: TelemetryData):
    """
    Store telemetry data for a device (used for testing or manual data entry)
    """
    try:
        # Ensure device_id matches
        telemetry.device_id = device_id
        
        # Store telemetry data
        telemetry_doc = telemetry.dict()
        result = await store_telemetry_data(telemetry_doc)
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to store telemetry data"
            )
        
        logger.info(f"Stored telemetry data for device {device_id}")
        
        return APIResponse(
            success=True,
            message=f"Telemetry data stored for device {device_id}",
            data={"mongodb_id": str(result)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store telemetry for {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store telemetry data: {str(e)}"
        )


@router.get("/telemetry/{device_id}/alarms", response_model=APIResponse)
async def get_device_alarms(
    device_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    severity: Optional[str] = Query(None, regex="^(warning|alarm|critical)$")
):
    """
    Get alarm history for a device
    """
    try:
        # Set default time range (last 7 days)
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        mongodb_collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        # Build query
        query = {
            "device_id": device_id,
            "timestamp": {"$gte": start_time, "$lte": end_time},
            "safety.alarm_flags": {"$exists": True, "$not": {"$size": 0}}
        }
        
        # Get telemetry records with alarms
        cursor = mongodb_collection.find(query).sort("timestamp", pymongo.DESCENDING)
        alarm_records = await cursor.to_list(length=1000)
        
        # Process alarm data
        alarms = []
        for record in alarm_records:
            alarm_flags = record.get("safety", {}).get("alarm_flags", [])
            for flag in alarm_flags:
                # Filter by severity if specified
                if severity:
                    # Simple severity mapping (in real implementation, this would be more sophisticated)
                    flag_severity = "warning"
                    if "critical" in flag.lower():
                        flag_severity = "critical"
                    elif "alarm" in flag.lower():
                        flag_severity = "alarm"
                    
                    if flag_severity != severity:
                        continue
                
                alarms.append({
                    "timestamp": record.get("timestamp"),
                    "device_id": device_id,
                    "alarm_flag": flag,
                    "severity": flag_severity,
                    "measurements": record.get("measurements", {}),
                    "metadata": record.get("metadata", {})
                })
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(alarms)} alarms for device {device_id}",
            data={
                "device_id": device_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "alarm_count": len(alarms),
                "alarms": alarms
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alarms for device {device_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alarm history: {str(e)}"
        )


# Helper functions

async def get_raw_telemetry(device_id: str, start_time: datetime, end_time: datetime, limit: int) -> List[dict]:
    """Get raw telemetry data from MongoDB"""
    try:
        mongodb_collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        cursor = mongodb_collection.find({
            "device_id": device_id,
            "timestamp": {"$gte": start_time, "$lte": end_time}
        }).sort("timestamp", pymongo.DESCENDING).limit(limit)
        
        return await cursor.to_list(length=limit)
        
    except Exception as e:
        logger.error(f"Failed to get raw telemetry: {e}")
        return []


async def get_aggregated_telemetry(device_id: str, start_time: datetime, end_time: datetime, 
                                   resolution: str, limit: int) -> List[dict]:
    """Get aggregated telemetry data based on resolution"""
    try:
        # For now, return raw data (in production, this would use pre-aggregated collections)
        # TODO: Implement proper aggregation based on resolution
        return await get_raw_telemetry(device_id, start_time, end_time, limit)
        
    except Exception as e:
        logger.error(f"Failed to get aggregated telemetry: {e}")
        return []


def filter_telemetry_metrics(telemetry_data: List[dict], metrics: List[str]) -> List[dict]:
    """Filter telemetry data to include only specified metrics"""
    filtered_data = []
    
    for record in telemetry_data:
        filtered_record = {
            "timestamp": record.get("timestamp"),
            "device_id": record.get("device_id"),
            "measurements": {}
        }
        
        measurements = record.get("measurements", {})
        for metric in metrics:
            if metric in measurements:
                filtered_record["measurements"][metric] = measurements[metric]
        
        filtered_data.append(filtered_record)
    
    return filtered_data


def calculate_telemetry_statistics(telemetry_data: List[dict]) -> dict:
    """Calculate basic statistics for telemetry data"""
    if not telemetry_data:
        return {}
    
    # Extract numeric values for each metric
    metrics = {}
    for record in telemetry_data:
        measurements = record.get("measurements", {})
        for key, value in measurements.items():
            if isinstance(value, (int, float)):
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
    
    # Calculate statistics
    statistics = {}
    for metric, values in metrics.items():
        if values:
            statistics[metric] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[0] if values else None
            }
    
    return statistics
