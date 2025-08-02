"""
Authentication system initialization for the Webscout API.
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI

from webscout.Litlogger import Logger, LogLevel, LogFormat, ConsoleHandler
from .database import DatabaseManager
from .api_key_manager import APIKeyManager
from .rate_limiter import RateLimiter
from .middleware import AuthMiddleware

# Setup logger
logger = Logger(
    name="webscout.api",
    level=LogLevel.INFO,
    handlers=[ConsoleHandler(stream=sys.stdout)],
    fmt=LogFormat.DEFAULT
)

# Global authentication system instances
db_manager: Optional[DatabaseManager] = None
api_key_manager: Optional[APIKeyManager] = None
rate_limiter: Optional[RateLimiter] = None
auth_middleware: Optional[AuthMiddleware] = None


def initialize_auth_system(app: FastAPI, auth_required: bool = True, rate_limit_enabled: bool = True) -> None:
    """Initialize the authentication system."""
    global db_manager, api_key_manager, rate_limiter, auth_middleware

    if not auth_required:
        logger.info("Auth system is disabled (no-auth mode): skipping DB and API key manager initialization.")
        db_manager = None
        api_key_manager = None
        rate_limiter = None
        auth_middleware = None
        return

    try:
        # Initialize database manager
        mongo_url = os.getenv("MONGODB_URL")
        data_dir = os.getenv("WEBSCOUT_DATA_DIR", "data")

        db_manager = DatabaseManager(mongo_url, data_dir)

        # Initialize API key manager
        api_key_manager = APIKeyManager(db_manager)

        # Initialize rate limiter
        rate_limiter = RateLimiter(db_manager)

        # Initialize auth middleware with configuration
        auth_middleware = AuthMiddleware(
            api_key_manager,
            rate_limiter,
            auth_required=auth_required,
            rate_limit_enabled=rate_limit_enabled
        )

        # Add auth middleware to app
        app.middleware("http")(auth_middleware)

        # Add startup event to initialize database
        async def startup_event():
            if db_manager:
                await db_manager.initialize()
                logger.info("Authentication system initialized successfully")
                logger.info(f"Auth required: {auth_required}, Rate limiting: {rate_limit_enabled}")

        # Store startup function for later use
        app.state.startup_event = startup_event

        logger.info("Authentication system setup completed")

    except Exception as e:
        logger.error(f"Failed to initialize authentication system: {e}")
        # Fall back to legacy auth if new system fails
        logger.warning("Falling back to legacy authentication system")


def get_auth_components():
    """Get the initialized authentication components."""
    if db_manager is None:
        return {
            "db_manager": None,
            "api_key_manager": None,
            "rate_limiter": None,
            "auth_middleware": None
        }

    return {
        "db_manager": db_manager,
        "api_key_manager": api_key_manager,
        "rate_limiter": rate_limiter,
        "auth_middleware": auth_middleware
    }
