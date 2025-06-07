# webscout/auth/rate_limiter.py

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
import logging

from .models import APIKey, RateLimitEntry
from .database import DatabaseManager
from .config import AppConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self.default_rate_limit = 10  # requests per minute
        self.window_size = 60  # 1 minute in seconds
    
    async def check_rate_limit(self, api_key: APIKey) -> Tuple[bool, dict]:
        """
        Check if a request is allowed under the rate limit.
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.window_size)
        
        try:
            # Get or create rate limit entry
            entry = await self.db.get_rate_limit_entry(api_key.id)
            
            if not entry:
                entry = RateLimitEntry(api_key_id=api_key.id, requests=[])
            
            # Clean old requests outside the window
            entry.requests = [req for req in entry.requests if req > window_start]
            
            # Check if limit is exceeded
            current_count = len(entry.requests)
            limit = api_key.rate_limit or self.default_rate_limit
            
            if current_count >= limit:
                # Rate limit exceeded
                oldest_request = min(entry.requests) if entry.requests else now
                reset_at = oldest_request + timedelta(seconds=self.window_size)
                retry_after = int((reset_at - now).total_seconds())
                
                rate_limit_info = {
                    "allowed": False,
                    "limit": limit,
                    "remaining": 0,
                    "reset_at": reset_at,
                    "retry_after": max(retry_after, 1)
                }
                
                logger.warning(f"Rate limit exceeded for API key {api_key.id}: {current_count}/{limit}")
                return False, rate_limit_info
            
            # Add current request
            entry.requests.append(now)
            
            # Update database
            await self.db.update_rate_limit_entry(entry)
            
            # Calculate next reset time
            if entry.requests:
                oldest_request = min(entry.requests)
                reset_at = oldest_request + timedelta(seconds=self.window_size)
            else:
                reset_at = now + timedelta(seconds=self.window_size)
            
            rate_limit_info = {
                "allowed": True,
                "limit": limit,
                "remaining": limit - len(entry.requests),
                "reset_at": reset_at,
                "retry_after": None
            }
            
            return True, rate_limit_info
            
        except Exception as e:
            logger.error(f"Error checking rate limit for API key {api_key.id}: {e}")
            # In case of error, allow the request but log the issue
            rate_limit_info = {
                "allowed": True,
                "limit": api_key.rate_limit or self.default_rate_limit,
                "remaining": api_key.rate_limit or self.default_rate_limit,
                "reset_at": now + timedelta(seconds=self.window_size),
                "retry_after": None
            }
            return True, rate_limit_info
    
    async def reset_rate_limit(self, api_key_id: str) -> bool:
        """Reset rate limit for an API key (admin function)."""
        try:
            entry = RateLimitEntry(api_key_id=api_key_id, requests=[])
            await self.db.update_rate_limit_entry(entry)
            logger.info(f"Reset rate limit for API key {api_key_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting rate limit for API key {api_key_id}: {e}")
            return False
    
    async def get_rate_limit_status(self, api_key: APIKey) -> dict:
        """Get current rate limit status without making a request."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.window_size)
        
        try:
            entry = await self.db.get_rate_limit_entry(api_key.id)
            
            if not entry:
                entry = RateLimitEntry(api_key_id=api_key.id, requests=[])
            
            # Clean old requests
            entry.requests = [req for req in entry.requests if req > window_start]
            
            limit = api_key.rate_limit or self.default_rate_limit
            current_count = len(entry.requests)
            
            # Calculate reset time
            if entry.requests:
                oldest_request = min(entry.requests)
                reset_at = oldest_request + timedelta(seconds=self.window_size)
            else:
                reset_at = now + timedelta(seconds=self.window_size)
            
            return {
                "limit": limit,
                "remaining": max(0, limit - current_count),
                "reset_at": reset_at,
                "window_size": self.window_size
            }
            
        except Exception as e:
            logger.error(f"Error getting rate limit status for API key {api_key.id}: {e}")
            return {
                "limit": api_key.rate_limit or self.default_rate_limit,
                "remaining": api_key.rate_limit or self.default_rate_limit,
                "reset_at": now + timedelta(seconds=self.window_size),
                "window_size": self.window_size
            }
    
    async def cleanup_old_entries(self) -> int:
        """Clean up old rate limit entries (maintenance function)."""
        # Remove requests older than the window_size for all rate limit entries
        try:
            # Try to get all rate limit entries from the database
            if hasattr(self.db, 'get_all_rate_limit_entries'):
                entries = await self.db.get_all_rate_limit_entries()
            else:
                logger.warning("Database does not support get_all_rate_limit_entries; cleanup skipped.")
                return 0

            now = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=self.window_size)
            cleaned = 0

            for entry in entries:
                old_count = len(entry.requests)
                entry.requests = [req for req in entry.requests if req > window_start]
                if len(entry.requests) < old_count:
                    await self.db.update_rate_limit_entry(entry)
                    cleaned += 1
            logger.info(f"Cleaned up {cleaned} old rate limit entries.")
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning up old rate limit entries: {e}")
            return 0

    async def check_ip_rate_limit(self, client_ip: str) -> Tuple[bool, dict]:
        """
        Check rate limit for an IP address (used in no-auth mode).

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.window_size)

        # Create a pseudo API key for IP-based rate limiting
        ip_key_id = f"ip_{client_ip}"
        limit = AppConfig.default_rate_limit

        try:
            # Get or create rate limit entry for IP
            entry = await self.db.get_rate_limit_entry(ip_key_id)

            if not entry:
                entry = RateLimitEntry(api_key_id=ip_key_id, requests=[])

            # Clean old requests outside the window
            entry.requests = [req for req in entry.requests if req > window_start]

            # Check if limit is exceeded
            current_count = len(entry.requests)

            if current_count >= limit:
                # Rate limit exceeded
                oldest_request = min(entry.requests) if entry.requests else now
                reset_at = oldest_request + timedelta(seconds=self.window_size)
                retry_after = int((reset_at - now).total_seconds())

                rate_limit_info = {
                    "allowed": False,
                    "limit": limit,
                    "remaining": 0,
                    "reset_at": reset_at,
                    "retry_after": max(retry_after, 1)
                }

                logger.warning(f"Rate limit exceeded for IP {client_ip}: {current_count}/{limit}")
                return False, rate_limit_info

            # Add current request
            entry.requests.append(now)

            # Update database
            await self.db.update_rate_limit_entry(entry)

            # Calculate next reset time
            if entry.requests:
                oldest_request = min(entry.requests)
                reset_at = oldest_request + timedelta(seconds=self.window_size)
            else:
                reset_at = now + timedelta(seconds=self.window_size)

            rate_limit_info = {
                "allowed": True,
                "limit": limit,
                "remaining": limit - len(entry.requests),
                "reset_at": reset_at,
                "retry_after": None
            }

            return True, rate_limit_info

        except Exception as e:
            logger.error(f"Error checking rate limit for IP {client_ip}: {e}")
            # In case of error, allow the request but log the issue
            rate_limit_info = {
                "allowed": True,
                "limit": limit,
                "remaining": limit,
                "reset_at": now + timedelta(seconds=self.window_size),
                "retry_after": None
            }
            return True, rate_limit_info
