# webscout/auth/api_key_manager.py

import secrets
import string
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import hashlib
import logging

from .models import User, APIKey
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API key generation, validation, and lifecycle."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self.key_prefix = "ws_"  # webscout prefix
        self.key_length = 32  # Length of the random part
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        # Generate random string
        alphabet = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(alphabet) for _ in range(self.key_length))
        
        # Add prefix
        api_key = f"{self.key_prefix}{random_part}"
        
        return api_key
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage (optional, for extra security)."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def create_api_key(
        self,
        username: str,
        telegram_id: str,
        name: Optional[str] = None,
        rate_limit: int = 10,
        expires_in_days: Optional[int] = None
    ) -> Tuple[APIKey, User]:
        """Create a new API key and associated user if needed. Only one API key per user allowed."""

        # Check if user already exists by telegram_id
        user = await self.db.get_user_by_telegram_id(telegram_id)

        if user:
            # Check if user already has an API key
            existing_keys = await self.db.get_api_keys_by_user(user.id)
            active_keys = [key for key in existing_keys if key.is_active and not key.is_expired()]

            if active_keys:
                raise ValueError(f"User with Telegram ID {telegram_id} already has an active API key. Only one API key per user is allowed.")
        else:
            # Check if username is already taken
            existing_user = await self.db.get_user_by_username(username)
            if existing_user:
                raise ValueError(f"Username '{username}' is already taken")

            # Create new user
            user = User(
                username=username,
                telegram_id=telegram_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            try:
                user = await self.db.create_user(user)
                logger.info(f"Created new user: {user.username} (Telegram ID: {telegram_id})")
            except ValueError as e:
                # User might already exist, try to get it
                if "already exists" in str(e):
                    user = await self.db.get_user_by_telegram_id(telegram_id)
                    if not user:
                        raise e
                else:
                    raise e
        
        # Generate API key
        api_key_value = self.generate_api_key()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key=api_key_value,
            user_id=user.id,
            name=name,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            rate_limit=rate_limit,
            is_active=True
        )
        
        # Store in database
        try:
            api_key = await self.db.create_api_key(api_key)
            logger.info(f"Created API key for user {user.username}: {api_key.id}")
            return api_key, user
        except ValueError as e:
            # Key collision (very unlikely), try again
            if "already exists" in str(e):
                logger.warning("API key collision detected, regenerating...")
                return await self.create_api_key(username, telegram_id, name, rate_limit, expires_in_days)
            raise e
    
    async def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[APIKey], Optional[str]]:
        """
        Validate an API key.
        
        Returns:
            Tuple of (is_valid, api_key_object, error_message)
        """
        if not api_key:
            return False, None, "API key is required"
        
        if not api_key.startswith(self.key_prefix):
            return False, None, "Invalid API key format"
        
        try:
            # Get API key from database
            key_obj = await self.db.get_api_key(api_key)
            
            if not key_obj:
                return False, None, "API key not found"
            
            if not key_obj.is_active:
                return False, None, "API key is inactive"
            
            if key_obj.is_expired():
                return False, None, "API key has expired"
            
            # Update last used timestamp
            key_obj.last_used_at = datetime.now(timezone.utc)
            key_obj.usage_count += 1
            
            try:
                await self.db.update_api_key(key_obj)
            except Exception as e:
                logger.warning(f"Failed to update API key usage: {e}")
            
            return True, key_obj, None
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return False, None, "Internal error during validation"
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key by marking it as inactive."""
        try:
            key_obj = await self.db.get_api_key(api_key)
            
            if not key_obj:
                return False
            
            key_obj.is_active = False
            key_obj.updated_at = datetime.now(timezone.utc)
            
            await self.db.update_api_key(key_obj)
            logger.info(f"Revoked API key: {key_obj.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return False
    
    async def get_user_api_keys(self, user_id: str) -> list[APIKey]:
        """Get all API keys for a user."""
        try:
            return await self.db.get_api_keys_by_user(user_id)
        except Exception as e:
            logger.error(f"Error getting user API keys: {e}")
            return []
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired API keys (mark as inactive)."""
        # This would require a method to get all API keys, which we haven't implemented
        # For now, we'll just return 0
        # In a production system, you'd want to implement this properly
        logger.info("Cleanup expired keys called (not implemented)")
        return 0
