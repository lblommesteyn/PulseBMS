import logging
from typing import List, Optional, Dict
from datetime import datetime
from ..database import db_manager, cache_device_status
from ..models import DeviceInfo, SafetyConstraints

logger = logging.getLogger("service.devices")

class DeviceService:
    def __init__(self):
        self.collection_name = "device_configurations"

    async def register_device(self, device_info: DeviceInfo) -> str:
        """Register a new device with validation logic"""
        collection = await db_manager.get_mongodb_collection(self.collection_name)
        
        # Check uniqueness
        if await collection.find_one({"device_id": device_info.device_id}):
            raise ValueError(f"Device {device_info.device_id} already exists")

        doc = device_info.dict()
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        
        result = await collection.insert_one(doc)
        
        # Init default safety
        await self.set_safety_constraints(SafetyConstraints(device_id=device_info.device_id))
        
        logger.info(f"Registered device {device_info.device_id}")
        return str(result.inserted_id)

    async def get_device(self, device_id: str) -> Optional[Dict]:
        collection = await db_manager.get_mongodb_collection(self.collection_name)
        return await collection.find_one({"device_id": device_id})

    async def list_devices(self, site_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        collection = await db_manager.get_mongodb_collection(self.collection_name)
        query = {}
        if site_id:
            query["site_id"] = site_id
            
        cursor = collection.find(query).limit(limit)
        return await cursor.to_list(length=limit)

    async def set_safety_constraints(self, constraints: SafetyConstraints):
        collection = await db_manager.get_mongodb_collection("safety_constraints")
        await collection.replace_one(
            {"device_id": constraints.device_id},
            constraints.dict(),
            upsert=True
        )

device_service = DeviceService()
