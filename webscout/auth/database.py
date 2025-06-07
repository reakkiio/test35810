# webscout/auth/database.py

import json
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import threading
import logging

try:
    import motor.motor_asyncio
    HAS_MOTOR = True
except ImportError:
    HAS_MOTOR = False

from .models import User, APIKey, RateLimitEntry

logger = logging.getLogger(__name__)


class JSONDatabase:
    """JSON file-based database fallback."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.users_file = self.data_dir / "users.json"
        self.api_keys_file = self.data_dir / "api_keys.json"
        self.rate_limits_file = self.data_dir / "rate_limits.json"
        
        self._lock = threading.RLock()
        
        # Initialize files if they don't exist
        for file_path in [self.users_file, self.api_keys_file, self.rate_limits_file]:
            if not file_path.exists():
                self._write_json(file_path, [])
    
    def _read_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read JSON file safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _write_json(self, file_path: Path, data: List[Dict[str, Any]]) -> None:
        """Write JSON file safely."""
        with self._lock:
            # Write to temporary file first, then rename for atomicity
            temp_file = file_path.with_suffix('.tmp')
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                temp_file.replace(file_path)
            except Exception as e:
                if temp_file.exists():
                    temp_file.unlink()
                raise e
    
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        users = self._read_json(self.users_file)
        
        # Check if user already exists
        for existing_user in users:
            if existing_user.get("username") == user.username:
                raise ValueError(f"User with username '{user.username}' already exists")
        
        users.append(user.to_dict())
        self._write_json(self.users_file, users)
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        users = self._read_json(self.users_file)
        for user_data in users:
            if user_data.get("id") == user_id:
                return User.from_dict(user_data)
        return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        users = self._read_json(self.users_file)
        for user_data in users:
            if user_data.get("username") == username:
                return User.from_dict(user_data)
        return None

    async def get_user_by_telegram_id(self, telegram_id: str) -> Optional[User]:
        """Get user by Telegram ID."""
        users = self._read_json(self.users_file)
        for user_data in users:
            if user_data.get("telegram_id") == telegram_id:
                return User.from_dict(user_data)
        return None
    
    async def create_api_key(self, api_key: APIKey) -> APIKey:
        """Create a new API key."""
        api_keys = self._read_json(self.api_keys_file)
        
        # Check if key already exists
        for existing_key in api_keys:
            if existing_key.get("key") == api_key.key:
                raise ValueError("API key already exists")
        
        api_keys.append(api_key.to_dict())
        self._write_json(self.api_keys_file, api_keys)
        return api_key
    
    async def get_api_key(self, key: str) -> Optional[APIKey]:
        """Get API key by key value."""
        api_keys = self._read_json(self.api_keys_file)
        for key_data in api_keys:
            if key_data.get("key") == key:
                return APIKey.from_dict(key_data)
        return None
    
    async def update_api_key(self, api_key: APIKey) -> APIKey:
        """Update an existing API key."""
        api_keys = self._read_json(self.api_keys_file)
        
        for i, key_data in enumerate(api_keys):
            if key_data.get("id") == api_key.id:
                api_keys[i] = api_key.to_dict()
                self._write_json(self.api_keys_file, api_keys)
                return api_key
        
        raise ValueError(f"API key with ID '{api_key.id}' not found")
    
    async def get_api_keys_by_user(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        api_keys = self._read_json(self.api_keys_file)
        user_keys = []
        
        for key_data in api_keys:
            if key_data.get("user_id") == user_id:
                user_keys.append(APIKey.from_dict(key_data))
        
        return user_keys
    
    async def get_rate_limit_entry(self, api_key_id: str) -> Optional[RateLimitEntry]:
        """Get rate limit entry for API key."""
        rate_limits = self._read_json(self.rate_limits_file)
        
        for entry_data in rate_limits:
            if entry_data.get("api_key_id") == api_key_id:
                return RateLimitEntry.from_dict(entry_data)
        
        return None
    
    async def update_rate_limit_entry(self, entry: RateLimitEntry) -> RateLimitEntry:
        """Update rate limit entry."""
        rate_limits = self._read_json(self.rate_limits_file)
        
        for i, entry_data in enumerate(rate_limits):
            if entry_data.get("api_key_id") == entry.api_key_id:
                rate_limits[i] = entry.to_dict()
                self._write_json(self.rate_limits_file, rate_limits)
                return entry
        
        # Create new entry if not found
        rate_limits.append(entry.to_dict())
        self._write_json(self.rate_limits_file, rate_limits)
        return entry
    
    async def get_all_rate_limit_entries(self) -> list:
        """Return all rate limit entries (for maintenance/cleanup)."""
        # Only for JSONDatabase
        entries = self._read_json(self.rate_limits_file)
        return [RateLimitEntry.from_dict(e) for e in entries]


class MongoDatabase:
    """MongoDB database implementation."""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017", database_name: str = "webscout"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to MongoDB."""
        if not HAS_MOTOR:
            logger.warning("motor package not available, cannot connect to MongoDB")
            return False

        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.database_name]

            # Test connection
            await self.client.admin.command('ping')
            self._connected = True
            logger.info("Connected to MongoDB successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            return False
    
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        # Check if user exists
        existing = await self.db.users.find_one({"username": user.username})
        if existing:
            raise ValueError(f"User with username '{user.username}' already exists")
        
        await self.db.users.insert_one(user.to_dict())
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        user_data = await self.db.users.find_one({"id": user_id})
        return User.from_dict(user_data) if user_data else None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        if not self._connected:
            raise RuntimeError("Database not connected")

        user_data = await self.db.users.find_one({"username": username})
        return User.from_dict(user_data) if user_data else None

    async def get_user_by_telegram_id(self, telegram_id: str) -> Optional[User]:
        """Get user by Telegram ID."""
        if not self._connected:
            raise RuntimeError("Database not connected")

        user_data = await self.db.users.find_one({"telegram_id": telegram_id})
        return User.from_dict(user_data) if user_data else None
    
    async def create_api_key(self, api_key: APIKey) -> APIKey:
        """Create a new API key."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        # Check if key exists
        existing = await self.db.api_keys.find_one({"key": api_key.key})
        if existing:
            raise ValueError("API key already exists")
        
        await self.db.api_keys.insert_one(api_key.to_dict())
        return api_key
    
    async def get_api_key(self, key: str) -> Optional[APIKey]:
        """Get API key by key value."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        key_data = await self.db.api_keys.find_one({"key": key})
        return APIKey.from_dict(key_data) if key_data else None
    
    async def update_api_key(self, api_key: APIKey) -> APIKey:
        """Update an existing API key."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        result = await self.db.api_keys.update_one(
            {"id": api_key.id},
            {"$set": api_key.to_dict()}
        )
        
        if result.matched_count == 0:
            raise ValueError(f"API key with ID '{api_key.id}' not found")
        
        return api_key
    
    async def get_api_keys_by_user(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        cursor = self.db.api_keys.find({"user_id": user_id})
        keys = []
        async for key_data in cursor:
            keys.append(APIKey.from_dict(key_data))
        return keys
    
    async def get_rate_limit_entry(self, api_key_id: str) -> Optional[RateLimitEntry]:
        """Get rate limit entry for API key."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        entry_data = await self.db.rate_limits.find_one({"api_key_id": api_key_id})
        return RateLimitEntry.from_dict(entry_data) if entry_data else None
    
    async def update_rate_limit_entry(self, entry: RateLimitEntry) -> RateLimitEntry:
        """Update rate limit entry."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        await self.db.rate_limits.update_one(
            {"api_key_id": entry.api_key_id},
            {"$set": entry.to_dict()},
            upsert=True
        )
        return entry
    
    async def get_all_rate_limit_entries(self) -> list:
        """Return all rate limit entries (for maintenance/cleanup) from MongoDB."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        entries = []
        cursor = self.db.rate_limits.find({})
        async for entry_data in cursor:
            entries.append(RateLimitEntry.from_dict(entry_data))
        return entries


class DatabaseManager:
    """Database manager that handles MongoDB with JSON fallback."""
    
    def __init__(self, mongo_connection_string: Optional[str] = None, data_dir: str = "data"):
        self.mongo_connection_string = mongo_connection_string or os.getenv("MONGODB_URL")
        self.data_dir = data_dir
        
        self.mongo_db = None
        self.json_db = JSONDatabase(data_dir)
        self.use_mongo = False
        
        logger.info(f"Database manager initialized with data_dir: {data_dir}")
    
    async def initialize(self) -> None:
        """Initialize database connection."""
        if self.mongo_connection_string:
            try:
                self.mongo_db = MongoDatabase(self.mongo_connection_string)
                self.use_mongo = await self.mongo_db.connect()
                if self.use_mongo:
                    logger.info("Using MongoDB as primary database")
                else:
                    logger.info("MongoDB connection failed, falling back to JSON database")
            except Exception as e:
                logger.warning(f"MongoDB initialization failed: {e}, using JSON database")
                self.use_mongo = False
        else:
            logger.info("No MongoDB connection string provided, using JSON database")
    
    @property
    def db(self) -> Union[MongoDatabase, JSONDatabase]:
        """Get the active database instance."""
        return self.mongo_db if self.use_mongo else self.json_db
    
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        return await self.db.create_user(user)
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return await self.db.get_user_by_id(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return await self.db.get_user_by_username(username)

    async def get_user_by_telegram_id(self, telegram_id: str) -> Optional[User]:
        """Get user by Telegram ID."""
        return await self.db.get_user_by_telegram_id(telegram_id)
    
    async def create_api_key(self, api_key: APIKey) -> APIKey:
        """Create a new API key."""
        return await self.db.create_api_key(api_key)
    
    async def get_api_key(self, key: str) -> Optional[APIKey]:
        """Get API key by key value."""
        return await self.db.get_api_key(key)
    
    async def update_api_key(self, api_key: APIKey) -> APIKey:
        """Update an existing API key."""
        return await self.db.update_api_key(api_key)
    
    async def get_api_keys_by_user(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        return await self.db.get_api_keys_by_user(user_id)
    
    async def get_rate_limit_entry(self, api_key_id: str) -> Optional[RateLimitEntry]:
        """Get rate limit entry for API key."""
        return await self.db.get_rate_limit_entry(api_key_id)
    
    async def update_rate_limit_entry(self, entry: RateLimitEntry) -> RateLimitEntry:
        """Update rate limit entry."""
        return await self.db.update_rate_limit_entry(entry)
    
    def get_status(self) -> Dict[str, str]:
        """Get database status."""
        return {
            "type": "MongoDB" if self.use_mongo else "JSON",
            "status": "connected" if (self.use_mongo and self.mongo_db._connected) or (not self.use_mongo) else "disconnected"
        }
