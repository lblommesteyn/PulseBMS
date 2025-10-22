import logging
from typing import Dict, List, Optional
from datetime import datetime
from ..database import db_manager

logger = logging.getLogger("service.telemetry")

class TelemetryService:
    def __init__(self):
        self.collection_name = "telemetry_data"

    async def store_telemetry(self, device_id: str, data: Dict):
        """Store telemetry and trigger real-time updates"""
        collection = await db_manager.get_mongodb_collection(self.collection_name)
        
        # Add metadata
        record = {
            "device_id": device_id,
            "received_at": datetime.utcnow(),
            "data": data
        }
        
        await collection.insert_one(record)
        
        # Here we would also push to TimescaleDB or InfluxDB in production
        logger.debug(f"Stored telemetry for {device_id}")

    async def get_latest(self, device_id: str) -> Optional[Dict]:
        collection = await db_manager.get_mongodb_collection(self.collection_name)
        cursor = collection.find({"device_id": device_id}).sort("received_at", -1).limit(1)
        results = await cursor.to_list(length=1)
        return results[0] if results else None

    async def get_history(self, device_id: str, start: datetime, end: datetime) -> List[Dict]:
        collection = await db_manager.get_mongodb_collection(self.collection_name)
        query = {
            "device_id": device_id,
            "received_at": {"$gte": start, "$lte": end}
        }
        cursor = collection.find(query).sort("received_at", 1)
        return await cursor.to_list(length=1000)

telemetry_service = TelemetryService()
