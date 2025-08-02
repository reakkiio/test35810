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

try:
    from supabase import create_client, Client #type: ignore
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

from .models import User, APIKey, RateLimitEntry, RequestLog

logger = logging.getLogger(__name__)


class JSONDatabase:
    """JSON file-based database fallback."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.users_file = self.data_dir / "users.json"
        self.api_keys_file = self.data_dir / "api_keys.json"
        self.rate_limits_file = self.data_dir / "rate_limits.json"
        self.request_logs_file = self.data_dir / "request_logs.json"
        
        self._lock = threading.RLock()
        
        # Initialize files if they don't exist
        for file_path in [self.users_file, self.api_keys_file, self.rate_limits_file, self.request_logs_file]:
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
    
    async def create_request_log(self, request_log: RequestLog) -> RequestLog:
        """Create a new request log entry."""
        request_logs = self._read_json(self.request_logs_file)
        request_logs.append(request_log.to_dict())
        self._write_json(self.request_logs_file, request_logs)
        return request_log
    
    async def get_request_logs(self, limit: int = 100, offset: int = 0) -> List[RequestLog]:
        """Get request logs with pagination."""
        request_logs = self._read_json(self.request_logs_file)
        # Sort by created_at descending (newest first)
        request_logs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        # Apply pagination
        paginated_logs = request_logs[offset:offset + limit]
        return [RequestLog.from_dict(log_data) for log_data in paginated_logs]


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
    
    async def create_request_log(self, request_log: RequestLog) -> RequestLog:
        """Create a new request log entry."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        await self.db.request_logs.insert_one(request_log.to_dict())
        return request_log
    
    async def get_request_logs(self, limit: int = 100, offset: int = 0) -> List[RequestLog]:
        """Get request logs with pagination."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        cursor = self.db.request_logs.find({}).sort("created_at", -1).skip(offset).limit(limit)
        logs = []
        async for log_data in cursor:
            logs.append(RequestLog.from_dict(log_data))
        return logs


class SupabaseDatabase:
    """Supabase database implementation."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client: Optional[Client] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Supabase."""
        if not HAS_SUPABASE:
            logger.warning("supabase package not available, cannot connect to Supabase")
            return False

        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            # Test connection by trying to access a table
            self.client.table('users').select('id').limit(1).execute()
            self._connected = True
            logger.info("Connected to Supabase successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Supabase: {e}")
            self._connected = False
            return False
    
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('users').insert(user.to_dict()).execute()
            if result.data:
                return user
            else:
                raise ValueError("Failed to create user")
        except Exception as e:
            if "duplicate key" in str(e).lower() or "already exists" in str(e).lower():
                raise ValueError(f"User with username '{user.username}' already exists")
            raise e
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('users').select('*').eq('id', user_id).execute()
            if result.data:
                return User.from_dict(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        if not self._connected:
            raise RuntimeError("Database not connected")

        try:
            result = self.client.table('users').select('*').eq('username', username).execute()
            if result.data:
                return User.from_dict(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None

    async def get_user_by_telegram_id(self, telegram_id: str) -> Optional[User]:
        """Get user by Telegram ID."""
        if not self._connected:
            raise RuntimeError("Database not connected")

        try:
            result = self.client.table('users').select('*').eq('telegram_id', int(telegram_id)).execute()
            if result.data:
                return User.from_dict(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error getting user by telegram_id: {e}")
            return None
    
    async def create_api_key(self, api_key: APIKey) -> APIKey:
        """Create a new API key."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('api_keys').insert(api_key.to_dict()).execute()
            if result.data:
                return api_key
            else:
                raise ValueError("Failed to create API key")
        except Exception as e:
            if "duplicate key" in str(e).lower() or "already exists" in str(e).lower():
                raise ValueError("API key already exists")
            raise e
    
    async def get_api_key(self, key: str) -> Optional[APIKey]:
        """Get API key by key value."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('api_keys').select('*').eq('key', key).execute()
            if result.data:
                return APIKey.from_dict(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error getting API key: {e}")
            return None
    
    async def update_api_key(self, api_key: APIKey) -> APIKey:
        """Update an existing API key."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('api_keys').update(api_key.to_dict()).eq('id', api_key.id).execute()
            if result.data:
                return api_key
            else:
                raise ValueError(f"API key with ID '{api_key.id}' not found")
        except Exception as e:
            logger.error(f"Error updating API key: {e}")
            raise e
    
    async def get_api_keys_by_user(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('api_keys').select('*').eq('user_id', user_id).execute()
            return [APIKey.from_dict(key_data) for key_data in result.data]
        except Exception as e:
            logger.error(f"Error getting API keys by user: {e}")
            return []
    
    async def get_rate_limit_entry(self, api_key_id: str) -> Optional[RateLimitEntry]:
        """Get rate limit entry for API key."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('rate_limits').select('*').eq('api_key_id', api_key_id).execute()
            if result.data:
                return RateLimitEntry.from_dict(result.data[0])
            return None
        except Exception as e:
            logger.error(f"Error getting rate limit entry: {e}")
            return None
    
    async def update_rate_limit_entry(self, entry: RateLimitEntry) -> RateLimitEntry:
        """Update rate limit entry."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            # Try to update first
            result = self.client.table('rate_limits').update(entry.to_dict()).eq('api_key_id', entry.api_key_id).execute()
            if not result.data:
                # If no rows were updated, insert new entry
                result = self.client.table('rate_limits').insert(entry.to_dict()).execute()
            return entry
        except Exception as e:
            logger.error(f"Error updating rate limit entry: {e}")
            raise e
    
    async def get_all_rate_limit_entries(self) -> list:
        """Return all rate limit entries (for maintenance/cleanup) from Supabase."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('rate_limits').select('*').execute()
            return [RateLimitEntry.from_dict(entry_data) for entry_data in result.data]
        except Exception as e:
            logger.error(f"Error getting all rate limit entries: {e}")
            return []
    
    async def create_request_log(self, request_log: RequestLog) -> RequestLog:
        """Create a new request log entry."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('request_logs').insert(request_log.to_dict()).execute()
            if result.data:
                return request_log
            else:
                raise ValueError("Failed to create request log")
        except Exception as e:
            logger.error(f"Error creating request log: {e}")
            raise e
    
    async def get_request_logs(self, limit: int = 100, offset: int = 0) -> List[RequestLog]:
        """Get request logs with pagination."""
        if not self._connected:
            raise RuntimeError("Database not connected")
        
        try:
            result = self.client.table('request_logs').select('*').order('created_at', desc=True).range(offset, offset + limit - 1).execute()
            return [RequestLog.from_dict(log_data) for log_data in result.data]
        except Exception as e:
            logger.error(f"Error getting request logs: {e}")
            return []


class DatabaseManager:
    """Database manager with flexible backend support."""
    
    def __init__(self, mongo_connection_string: Optional[str] = None, data_dir: str = "data"):
        self.mongo_connection_string = mongo_connection_string
        self.data_dir = data_dir
        self.supabase_url = self._get_supabase_url()
        self.supabase_key = self._get_supabase_key()
        
        # Database instances
        self.supabase_db = None
        self.mongo_db = None
        self.json_db = None
        self.active_db = None
        
        logger.info("🔗 Database manager initialized with flexible backend support")
    
    def _get_supabase_url(self) -> Optional[str]:
        """Get Supabase URL from environment variables or GitHub secrets."""
        # Try environment variable first
        url = os.getenv("SUPABASE_URL")
        if url:
            return url
        
        # Try to get from GitHub secrets (if running in GitHub Actions)
        github_url = os.getenv("GITHUB_SUPABASE_URL")  # GitHub Actions secret
        if github_url:
            return github_url
        
        # Don't log error during initialization - only when actually needed
        return None
    
    def _get_supabase_key(self) -> Optional[str]:
        """Get Supabase key from environment variables or GitHub secrets."""
        # Try environment variable first
        key = os.getenv("SUPABASE_ANON_KEY")
        if key:
            return key
        
        # Try to get from GitHub secrets (if running in GitHub Actions)
        github_key = os.getenv("GITHUB_SUPABASE_ANON_KEY")  # GitHub Actions secret
        if github_key:
            return github_key
        
        # Don't log error during initialization - only when actually needed
        return None
    
    async def initialize(self) -> None:
        """Initialize database connection with fallback support."""
        # Try Supabase first if credentials are available
        if self.supabase_url and self.supabase_key:
            logger.info("🔗 Attempting to connect to Supabase...")
            try:
                self.supabase_db = SupabaseDatabase(self.supabase_url, self.supabase_key)
                connected = await self.supabase_db.connect()
                
                if connected:
                    self.active_db = self.supabase_db
                    logger.info("✅ Successfully connected to Supabase database")
                    return
                else:
                    logger.warning("⚠️ Failed to connect to Supabase, trying fallbacks...")
            except Exception as e:
                logger.warning(f"⚠️ Supabase connection error: {e}, trying fallbacks...")
        else:
            logger.info("ℹ️ Supabase credentials not found, using fallback databases")
        
        # Try MongoDB if connection string is provided
        if self.mongo_connection_string:
            logger.info("🔗 Attempting to connect to MongoDB...")
            try:
                self.mongo_db = MongoDatabase(self.mongo_connection_string)
                connected = await self.mongo_db.connect()
                
                if connected:
                    self.active_db = self.mongo_db
                    logger.info("✅ Successfully connected to MongoDB database")
                    return
                else:
                    logger.warning("⚠️ Failed to connect to MongoDB, falling back to JSON...")
            except Exception as e:
                logger.warning(f"⚠️ MongoDB connection error: {e}, falling back to JSON...")
        
        # Fall back to JSON database
        logger.info("📁 Using JSON file database as fallback")
        try:
            self.json_db = JSONDatabase(self.data_dir)
            self.active_db = self.json_db
            logger.info("✅ Successfully initialized JSON database")
        except Exception as e:
            logger.error(f"❌ Failed to initialize JSON database: {e}")
            raise RuntimeError(f"Failed to initialize any database backend: {e}")
    
    @property
    def db(self):
        """Get the active database instance."""
        if not self.active_db:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.active_db
    
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
    
    async def create_request_log(self, request_log: RequestLog) -> RequestLog:
        """Create a new request log entry."""
        return await self.db.create_request_log(request_log)
    
    async def get_request_logs(self, limit: int = 100, offset: int = 0) -> List[RequestLog]:
        """Get request logs with pagination."""
        return await self.db.get_request_logs(limit, offset)
    
    def get_status(self) -> Dict[str, str]:
        """Get database status."""
        if not self.active_db:
            return {
                "type": "None",
                "status": "not_initialized",
                "message": "Database not initialized"
            }
        
        if isinstance(self.active_db, SupabaseDatabase):
            return {
                "type": "Supabase",
                "status": "connected" if self.active_db._connected else "disconnected",
                "message": "Using Supabase database"
            }
        elif isinstance(self.active_db, MongoDatabase):
            return {
                "type": "MongoDB",
                "status": "connected" if self.active_db._connected else "disconnected",
                "message": "Using MongoDB database"
            }
        elif isinstance(self.active_db, JSONDatabase):
            return {
                "type": "JSON",
                "status": "connected",
                "message": "Using JSON file database"
            }
        else:
            return {
                "type": "Unknown",
                "status": "unknown",
                "message": "Unknown database type"
            }
