"""
Configuration management for the Webscout API server.
"""

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
        self.auth_required: bool = True  # New: Enable/disable authentication
        self.rate_limit_enabled: bool = True  # New: Enable/disable rate limiting
        self.default_rate_limit: int = 60  # Default rate limit for no-auth mode

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
    auth_required: bool = True  # New: Enable/disable authentication
    rate_limit_enabled: bool = True  # New: Enable/disable rate limiting
    default_rate_limit: int = 60  # Default rate limit for no-auth mode

    @classmethod
    def set_config(cls, **data):
        """Set configuration values."""
        for key, value in data.items():
            setattr(cls, key, value)
        # Sync with new config system
        from .server import config
        config.update(**data)
