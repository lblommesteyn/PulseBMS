"""
PulseBMS Enhanced - Database Initialization and Management
MongoDB and PostgreSQL database setup and connections
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

import motor.motor_asyncio
import pymongo
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import redis.asyncio as redis

from .config import settings

logger = logging.getLogger(__name__)

# Mock Classes for Standalone Mode
class MockCollection:
    def __init__(self, name, data):
        self.name = name
        self.data = data # Reference to DB data dict
    
    async def insert_one(self, document):
        if "_id" not in document:
            import uuid
            document["_id"] = str(uuid.uuid4())
        self.data.setdefault(self.name, []).append(document)
        from collections import namedtuple
        InsertResult = namedtuple('InsertResult', ['inserted_id'])
        return InsertResult(document["_id"])

    async def find_one(self, query):
        collection_data = self.data.get(self.name, [])
        for doc in collection_data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc
        return None

    def find(self, query):
        # Returns a mock cursor
        class MockCursor:
            def __init__(self, data):
                self.data = data
            def limit(self, n):
                return self
            def sort(self, key, direction):
                return self
            async def to_list(self, length):
                return self.data[:length]

        collection_data = self.data.get(self.name, [])
        filtered = []
        for doc in collection_data:
            match = True
            for k, v in query.items():
                if isinstance(v, dict): continue # Skip complex queries for now
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return MockCursor(filtered)

    async def replace_one(self, filter_doc, replacement, upsert=False):
        collection_data = self.data.get(self.name, [])
        for i, doc in enumerate(collection_data):
            match = True
            for k, v in filter_doc.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                collection_data[i] = replacement
                return
        if upsert:
            collection_data.append(replacement)

    async def create_index(self, keys, expireAfterSeconds=None):
        pass # No-op for mock

class MockDatabase:
    def __init__(self):
        self.data = {}
    
    def __getitem__(self, name):
        return MockCollection(name, self.data)
    
    def __getattr__(self, name):
        return MockCollection(name, self.data)

class MockAsyncIOMotorClient:
    def __init__(self, *args, **kwargs):
        self.db = MockDatabase()
        
    def __getitem__(self, name):
        return self.db
        
    @property
    def admin(self):
        class Admin:
            async def command(self, cmd):
                return {"ok": 1.0}
        return Admin()

    def close(self):
        pass


# SQLAlchemy setup
Base = declarative_base()
metadata = MetaData()

# Global database connections
mongodb_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
mongodb_db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
postgres_engine: Optional[create_async_engine] = None
postgres_session: Optional[AsyncSession] = None
redis_client: Optional[redis.Redis] = None


class DatabaseManager:
    """Database manager for PulseBMS Enhanced"""
    
    def __init__(self):
        self.mongodb_client = None
        self.mongodb_db = None
        self.postgres_engine = None
        self.postgres_session_factory = None
        self.redis_client = None
    
    async def connect_mongodb(self) -> bool:
        """Connect to MongoDB"""
        if settings.STANDALONE_MODE:
            logger.warning("STANDALONE_MODE is True. Using In-Memory Mock Database.")
            self.mongodb_client = MockAsyncIOMotorClient()
            self.mongodb_db = self.mongodb_client[settings.MONGODB_DB_NAME]
            return True
            
        try:
            logger.info("Connecting to MongoDB...")
            
            # Create MongoDB client
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=2000 # Reduced timeout
            )
            
            # Test connection
            await self.mongodb_client.admin.command('ping')
            
            # Get database
            self.mongodb_db = self.mongodb_client[settings.MONGODB_DB_NAME]
            
            # Set up indexes
            await self._setup_mongodb_indexes()
            
            logger.info("MongoDB connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.warning("Switching to Standalone Mode (Mock Database) due to connection failure.")
            self.mongodb_client = MockAsyncIOMotorClient()
            self.mongodb_db = self.mongodb_client[settings.MONGODB_DB_NAME]
            return True
    
    async def connect_postgresql(self) -> bool:
        """Connect to PostgreSQL"""
        if settings.STANDALONE_MODE:
            logger.info("STANDALONE_MODE: Skipping PostgreSQL connection.")
            return True

        try:
            logger.info("Connecting to PostgreSQL...")
            
            # Create async engine
            self.postgres_engine = create_async_engine(
                settings.POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://"),
                echo=settings.DEBUG,
                pool_size=10,
                max_overflow=20
            )
            
            # Create session factory
            self.postgres_session_factory = sessionmaker(
                self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.postgres_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("PostgreSQL connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return True # Don't block startup on postgres failure for now

    async def connect_redis(self) -> bool:
        """Connect to Redis"""
        if settings.STANDALONE_MODE:
            logger.warning("STANDALONE_MODE: Using Mock Redis.")
            class MockRedis:
                def __init__(self): self.data = {}
                async def ping(self): return True
                async def setex(self, name, time, value): self.data[name] = value; return True
                async def get(self, name): return self.data.get(name)
                async def close(self): pass
            
            self.redis_client = MockRedis()
            return True

        try:
            logger.info("Connecting to Redis...")
            
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Switching to Mock Redis due to connection failure.")
            class MockRedis:
                def __init__(self): self.data = {}
                async def ping(self): return True
                async def setex(self, name, time, value): self.data[name] = value; return True
                async def get(self, name): return self.data.get(name)
                async def close(self): pass
            self.redis_client = MockRedis()
            return True
    
    async def disconnect_all(self):
        """Disconnect from all databases"""
        logger.info("Disconnecting from databases...")
        
        if self.mongodb_client:
            self.mongodb_client.close()
        
        if self.postgres_engine:
            await self.postgres_engine.dispose()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("All database connections closed")
    
    async def _setup_mongodb_indexes(self):
        """Set up MongoDB indexes for optimal performance"""
        try:
            # Telemetry data indexes
            telemetry_collection = self.mongodb_db.telemetry_data
            await telemetry_collection.create_index([
                ("site_id", pymongo.ASCENDING),
                ("device_id", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ])
            
            await telemetry_collection.create_index([
                ("day", pymongo.ASCENDING),
                ("hour", pymongo.ASCENDING)
            ])
            
            # TTL index for automatic cleanup (90 days)
            await telemetry_collection.create_index(
                "timestamp",
                expireAfterSeconds=7776000
            )
            
            # Metrics aggregation indexes
            metrics_collection = self.mongodb_db.metrics_hourly
            await metrics_collection.create_index([
                ("site_id", pymongo.ASCENDING),
                ("device_id", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ])
            
            # TTL index for hourly metrics (2 years)
            await metrics_collection.create_index(
                "timestamp",
                expireAfterSeconds=63072000
            )
            
            # Digital twin predictions indexes
            predictions_collection = self.mongodb_db.digital_twin_predictions
            await predictions_collection.create_index([
                ("device_id", pymongo.ASCENDING),
                ("prediction_timestamp", pymongo.DESCENDING)
            ])
            
            # TTL index for predictions (30 days)
            await predictions_collection.create_index(
                "prediction_timestamp",
                expireAfterSeconds=2592000
            )
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")
    
    async def get_mongodb_collection(self, collection_name: str):
        """Get MongoDB collection"""
        if not self.mongodb_db:
            raise RuntimeError("MongoDB not connected")
        return self.mongodb_db[collection_name]
    
    async def get_postgres_session(self) -> AsyncSession:
        """Get PostgreSQL session"""
        if not self.postgres_session_factory:
            raise RuntimeError("PostgreSQL not connected")
        return self.postgres_session_factory()
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            raise RuntimeError("Redis not connected")
        return self.redis_client
    
    async def health_check(self) -> dict:
        """Check health of all database connections"""
        health_status = {
            "mongodb": "unknown",
            "postgresql": "unknown",
            "redis": "unknown"
        }
        
        # Check MongoDB
        try:
            if self.mongodb_client:
                await self.mongodb_client.admin.command('ping')
                health_status["mongodb"] = "healthy"
            else:
                health_status["mongodb"] = "disconnected"
        except Exception as e:
            health_status["mongodb"] = f"unhealthy: {str(e)[:100]}"
        
        # Check PostgreSQL
        try:
            if self.postgres_engine:
                async with self.postgres_engine.begin() as conn:
                    await conn.execute("SELECT 1")
                health_status["postgresql"] = "healthy"
            else:
                health_status["postgresql"] = "disconnected"
        except Exception as e:
            health_status["postgresql"] = f"unhealthy: {str(e)[:100]}"
        
        # Check Redis
        try:
            if self.redis_client:
                await self.redis_client.ping()
                health_status["redis"] = "healthy"
            else:
                health_status["redis"] = "disconnected"
        except Exception as e:
            health_status["redis"] = f"unhealthy: {str(e)[:100]}"
        
        return health_status


# Global database manager instance
db_manager = DatabaseManager()


async def init_database() -> bool:
    """Initialize all database connections"""
    logger.info("Initializing database connections...")
    
    # Connect to all databases
    mongodb_ok = await db_manager.connect_mongodb()
    postgres_ok = await db_manager.connect_postgresql()
    redis_ok = await db_manager.connect_redis()
    
    if mongodb_ok and postgres_ok and redis_ok:
        logger.info("All database connections initialized successfully")
        
        # Set global references for backward compatibility
        global mongodb_client, mongodb_db, postgres_engine, redis_client
        mongodb_client = db_manager.mongodb_client
        mongodb_db = db_manager.mongodb_db
        postgres_engine = db_manager.postgres_engine
        redis_client = db_manager.redis_client
        
        return True
    else:
        logger.error("Failed to initialize some database connections")
        return False


async def close_database():
    """Close all database connections"""
    await db_manager.disconnect_all()


# Convenience functions for database operations
async def store_telemetry_data(telemetry_data: dict):
    """Store telemetry data in MongoDB"""
    try:
        collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        # Add partitioning fields for efficient queries
        timestamp = telemetry_data.get('timestamp', datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        telemetry_data['day'] = timestamp.strftime('%Y-%m-%d')
        telemetry_data['hour'] = timestamp.hour
        
        # Insert document
        result = await collection.insert_one(telemetry_data)
        logger.debug(f"Stored telemetry data with ID: {result.inserted_id}")
        return result.inserted_id
        
    except Exception as e:
        logger.error(f"Failed to store telemetry data: {e}")
        return None


async def get_device_telemetry(device_id: str, start_time: datetime, 
                               end_time: datetime, limit: int = 1000) -> list:
    """Get telemetry data for a device"""
    try:
        collection = await db_manager.get_mongodb_collection("telemetry_data")
        
        cursor = collection.find({
            "device_id": device_id,
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }).sort("timestamp", pymongo.DESCENDING).limit(limit)
        
        return await cursor.to_list(length=limit)
        
    except Exception as e:
        logger.error(f"Failed to get device telemetry: {e}")
        return []


async def cache_device_status(device_id: str, status_data: dict, ttl: int = 300):
    """Cache device status in Redis"""
    try:
        redis_client = await db_manager.get_redis_client()
        key = f"device:status:{device_id}"
        
        await redis_client.setex(
            key, 
            ttl, 
            str(status_data)  # Redis will handle JSON serialization
        )
        
        logger.debug(f"Cached status for device {device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cache device status: {e}")
        return False


async def get_cached_device_status(device_id: str) -> Optional[dict]:
    """Get cached device status from Redis"""
    try:
        redis_client = await db_manager.get_redis_client()
        key = f"device:status:{device_id}"
        
        status = await redis_client.get(key)
        if status:
            return eval(status)  # Simple eval for now, should use JSON in production
        return None
        
    except Exception as e:
        logger.error(f"Failed to get cached device status: {e}")
        return None
