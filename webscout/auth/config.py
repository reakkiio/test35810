"""
Configuration management for the Webscout API server.
SUPABASE-ONLY MODE: All data is stored in Supabase database.
"""

import os
import subprocess
from typing import List, Dict, Optional, Any
from webscout.Litlogger import Logger, LogLevel, LogFormat, ConsoleHandler
import sys

# Configuration constants
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"

# Setup logger
logger = Logger(
    name="webscout.api",
    level=LogLevel.INFO,
    handlers=[ConsoleHandler(stream=sys.stdout)],
    fmt=LogFormat.DEFAULT
)


def _get_github_secret(secret_name: str) -> Optional[str]:
    """
    Get GitHub secret from environment or GitHub Actions context.
    
    Args:
        secret_name: Name of the GitHub secret
        
    Returns:
        Secret value or None if not found
    """
    # Try direct environment variable first
    value = os.getenv(secret_name)
    if value:
        logger.info(f"✅ Found {secret_name} in environment")
        return value
    
    # Try GitHub Actions context
    github_value = os.getenv(f"GITHUB_{secret_name}")
    if github_value:
        logger.info(f"✅ Found {secret_name} in GitHub Actions context")
        return github_value
    
    # Try alternative naming patterns
    alt_names = [
        f"GITHUB_SECRET_{secret_name}",
        f"GH_{secret_name}",
        secret_name.replace("_", "-"),
        secret_name.lower(),
        secret_name.upper()
    ]
    
    for alt_name in alt_names:
        alt_value = os.getenv(alt_name)
        if alt_value:
            logger.info(f"✅ Found {secret_name} as {alt_name}")
            return alt_value
    
    logger.warning(f"❌ GitHub secret {secret_name} not found in any location")
    return None


def _get_supabase_url() -> Optional[str]:
    """Get Supabase URL from GitHub secrets or environment."""
    url = _get_github_secret("SUPABASE_URL")
    if not url:
        logger.warning("⚠️  SUPABASE_URL not found in environment or GitHub secrets")
        logger.info("🔄 Supabase database features will be disabled")
        return None
    logger.info("✅ Supabase URL configured from GitHub secrets")
    return url


def _get_supabase_key() -> Optional[str]:
    """Get Supabase anonymous key from GitHub secrets or environment."""
    key = _get_github_secret("SUPABASE_ANON_KEY")
    if not key:
        logger.warning("⚠️  SUPABASE_ANON_KEY not found in environment or GitHub secrets")
        logger.info("🔄 Supabase database features will be disabled")
        return None
    logger.info("✅ Supabase anonymous key configured from GitHub secrets")
    return key


class ServerConfig:
    """
    Centralized configuration management for the API server.
    SUPABASE-ONLY MODE: All authentication and logging uses Supabase database.
    """

    def __init__(self):
        self.api_key: Optional[str] = None
        self.provider_map: Dict[str, Any] = {}
        self.default_provider: str = "ChatGPT"
        self.base_url: Optional[str] = None
        self.host: str = DEFAULT_HOST
        self.port: int = DEFAULT_PORT
        self.debug: bool = False
        self.cors_origins: List[str] = ["*"]
        self.max_request_size: int = 10 * 1024 * 1024  # 10MB
        self.request_timeout: int = 300  # 5 minutes
        
        # FORCE SUPABASE-ONLY CONFIGURATION
        self.auth_required: bool = os.getenv("WEBSCOUT_AUTH_REQUIRED", "true").lower() == "true"  # FORCE auth enabled
        self.rate_limit_enabled: bool = os.getenv("WEBSCOUT_RATE_LIMIT_ENABLED", "true").lower() == "true"  # FORCE rate limiting
        self.default_rate_limit: int = 60  # Default rate limit
        self.request_logging_enabled: bool = True  # ALWAYS enable request logging to Supabase
        
        # SUPABASE DATABASE CONFIGURATION (OPTIONAL)
        # Get Supabase credentials from GitHub secrets or environment
        self.supabase_url: Optional[str] = _get_supabase_url()
        self.supabase_anon_key: Optional[str] = _get_supabase_key()
        
        if self.supabase_url and self.supabase_anon_key:
            logger.info("🎯 Supabase database: Configured successfully")
            self.database_mode = "supabase"
        else:
            logger.info("🔄 Supabase database: Not configured, using fallback mode")
            self.database_mode = "fallback"
        
        # MongoDB fallback option
        self.mongodb_url: Optional[str] = os.getenv("MONGODB_URL")
        
        logger.info("🚀 ServerConfig initialized")
        logger.info("✅ Authentication: ENABLED")
        logger.info("✅ Rate Limiting: ENABLED")
        logger.info(f"✅ Request Logging: ENABLED ({self.database_mode})")
        if self.database_mode == "supabase":
            logger.info("✅ Supabase Database: ENABLED")
            logger.info("❌ MongoDB Fallback: DISABLED")
        else:
            logger.info("⚠️  Supabase Database: DISABLED")
            if self.mongodb_url:
                logger.info("✅ MongoDB Fallback: ENABLED")
            else:
                logger.info("✅ JSON File Fallback: ENABLED")

    def update(self, **kwargs) -> None:
        """Update configuration with provided values."""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                logger.info(f"Config updated: {key} = {value}")

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")

        if self.default_provider not in self.provider_map and self.provider_map:
            available_providers = list(set(v.__name__ for v in self.provider_map.values()))
            logger.warning(f"Default provider '{self.default_provider}' not found. Available: {available_providers}")


class AppConfig:
    """
    Legacy configuration class for backward compatibility.
    SUPABASE-ONLY MODE: All data stored in Supabase database.
    """
    api_key: Optional[str] = None
    provider_map = {}
    tti_provider_map = {}  # Add TTI provider map
    default_provider = "ChatGPT"
    default_tti_provider = "PollinationsAI"  # Add default TTI provider
    base_url: Optional[str] = None
    
    # CONFIGURATION OPTIONS
    auth_required: bool = os.getenv("WEBSCOUT_AUTH_REQUIRED", "true").lower() == "true"
    rate_limit_enabled: bool = os.getenv("WEBSCOUT_RATE_LIMIT_ENABLED", "true").lower() == "true"
    default_rate_limit: int = 60  # Default rate limit
    request_logging_enabled: bool = True  # Enable request logging
    
    # DATABASE CONFIGURATION (OPTIONAL)
    # Get Supabase credentials from GitHub secrets or environment
    supabase_url: Optional[str] = _get_supabase_url()
    supabase_anon_key: Optional[str] = _get_supabase_key()
    
    # MongoDB fallback option
    mongodb_url: Optional[str] = os.getenv("MONGODB_URL")

    @classmethod
    def set_config(cls, **data):
        """Set configuration values."""
        for key, value in data.items():
            setattr(cls, key, value)
        # Sync with new config system
        from .server import config
        config.update(**data)
