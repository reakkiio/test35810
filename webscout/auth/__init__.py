# webscout/auth/__init__.py

from .models import User, APIKey
from .database import DatabaseManager
from .api_key_manager import APIKeyManager
from .rate_limiter import RateLimiter
from .middleware import AuthMiddleware
from .schemas import (
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyValidationResponse,
    UserCreateRequest,
    UserResponse,
    HealthCheckResponse
)
# Import server functions lazily to avoid module execution issues
def create_app():
    from .server import create_app as _create_app
    return _create_app()

def run_api(*args, **kwargs):
    from .server import run_api as _run_api
    return _run_api(*args, **kwargs)

def start_server(*args, **kwargs):
    from .server import start_server as _start_server
    return _start_server(*args, **kwargs)
from .routes import Api
from .config import ServerConfig, AppConfig
from .exceptions import APIError
from .providers import initialize_provider_map, initialize_tti_provider_map

__all__ = [
    "User",
    "APIKey",
    "DatabaseManager",
    "APIKeyManager",
    "RateLimiter",
    "AuthMiddleware",
    "APIKeyCreateRequest",
    "APIKeyCreateResponse",
    "APIKeyValidationResponse",
    "UserCreateRequest",
    "UserResponse",
    "HealthCheckResponse",
    "create_app",
    "run_api",
    "start_server",
    "Api",
    "ServerConfig",
    "AppConfig",
    "APIError",
    "initialize_provider_map",
    "initialize_tti_provider_map"
]
