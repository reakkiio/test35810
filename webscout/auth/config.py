"""
Configuration management for the Webscout API server.
"""

import os
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


def _get_supabase_url() -> Optional[str]:
    """Get Supabase URL from environment variables or GitHub secrets."""
    # Try environment variable first
    url = os.getenv("SUPABASE_URL")
    if url:
        logger.info("📍 Using SUPABASE_URL from environment")
        return url
    
    # Try to get from GitHub secrets (if running in GitHub Actions)
    github_url = os.getenv("GITHUB_SUPABASE_URL")  # GitHub Actions secret
    if github_url:
        logger.info("📍 Using SUPABASE_URL from GitHub secrets")
        return github_url
    
    # Don't log error during import - only when actually needed
    return None


def _get_supabase_anon_key() -> Optional[str]:
    """Get Supabase anon key from environment variables or GitHub secrets."""
    # Try environment variable first
    key = os.getenv("SUPABASE_ANON_KEY")
    if key:
        logger.info("🔑 Using SUPABASE_ANON_KEY from environment")
        return key
    
    # Try to get from GitHub secrets (if running in GitHub Actions)
    github_key = os.getenv("GITHUB_SUPABASE_ANON_KEY")  # GitHub Actions secret
    if github_key:
        logger.info("🔑 Using SUPABASE_ANON_KEY from GitHub secrets")
        return github_key
    
    # Don't log error during import - only when actually needed
    return None


class ServerConfig:
    """Centralized configuration management for the API server."""

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
        self.auth_required: bool = os.getenv("WEBSCOUT_AUTH_REQUIRED", "false").lower() == "true"  # Default to no auth
        self.rate_limit_enabled: bool = os.getenv("WEBSCOUT_RATE_LIMIT_ENABLED", "false").lower() == "true"  # Default to no rate limit
        self.default_rate_limit: int = 60  # Default rate limit for no-auth mode
        self.request_logging_enabled: bool = os.getenv("WEBSCOUT_REQUEST_LOGGING", "true").lower() == "true"  # Enable request logging by default
        
        # Database configuration - lazy initialization
        self._supabase_url: Optional[str] = None
        self._supabase_anon_key: Optional[str] = None
        self._supabase_url_checked: bool = False
        self._supabase_anon_key_checked: bool = False
        self.mongodb_url: Optional[str] = os.getenv("MONGODB_URL")

    @property
    def supabase_url(self) -> Optional[str]:
        """Get Supabase URL with lazy initialization."""
        if not self._supabase_url_checked:
            self._supabase_url = _get_supabase_url()
            self._supabase_url_checked = True
        return self._supabase_url

    @property
    def supabase_anon_key(self) -> Optional[str]:
        """Get Supabase anon key with lazy initialization."""
        if not self._supabase_anon_key_checked:
            self._supabase_anon_key = _get_supabase_anon_key()
            self._supabase_anon_key_checked = True
        return self._supabase_anon_key

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
    """Legacy configuration class for backward compatibility."""
    api_key: Optional[str] = None
    provider_map = {}
    tti_provider_map = {}  # Add TTI provider map
    default_provider = "ChatGPT"
    default_tti_provider = "PollinationsAI"  # Add default TTI provider
    base_url: Optional[str] = None
    auth_required: bool = os.getenv("WEBSCOUT_AUTH_REQUIRED", "false").lower() == "true"  # Default to no auth
    rate_limit_enabled: bool = os.getenv("WEBSCOUT_RATE_LIMIT_ENABLED", "false").lower() == "true"  # Default to no rate limit
    default_rate_limit: int = 60  # Default rate limit for no-auth mode
    request_logging_enabled: bool = os.getenv("WEBSCOUT_REQUEST_LOGGING", "true").lower() == "true"  # Enable request logging by default
    
    # Database configuration - lazy initialization
    _supabase_url: Optional[str] = None
    _supabase_anon_key: Optional[str] = None
    _supabase_url_checked: bool = False
    _supabase_anon_key_checked: bool = False
    mongodb_url: Optional[str] = os.getenv("MONGODB_URL")

    @classmethod
    def get_supabase_url(cls) -> Optional[str]:
        """Get Supabase URL with lazy initialization."""
        if not cls._supabase_url_checked:
            cls._supabase_url = _get_supabase_url()
            cls._supabase_url_checked = True
        return cls._supabase_url

    @classmethod
    def get_supabase_anon_key(cls) -> Optional[str]:
        """Get Supabase anon key with lazy initialization."""
        if not cls._supabase_anon_key_checked:
            cls._supabase_anon_key = _get_supabase_anon_key()
            cls._supabase_anon_key_checked = True
        return cls._supabase_anon_key

    # For backward compatibility, provide properties that call the methods
    @property
    def supabase_url(self) -> Optional[str]:
        return self.__class__.get_supabase_url()

    @property
    def supabase_anon_key(self) -> Optional[str]:
        return self.__class__.get_supabase_anon_key()

    @classmethod
    def set_config(cls, **data):
        """Set configuration values."""
        for key, value in data.items():
            setattr(cls, key, value)
        # Sync with new config system
        try:
            from .server import get_config
            config = get_config()
            config.update(**data)
        except ImportError:
            # Handle case where server module is not available
            pass
