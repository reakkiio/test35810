# webscout/auth/middleware.py

import secrets
from datetime import datetime
from typing import Optional, Tuple
import logging

from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse

from .api_key_manager import APIKeyManager
from .rate_limiter import RateLimiter
from .models import APIKey
from .schemas import ErrorResponse, RateLimitStatus

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """Authentication and rate limiting middleware."""
    
    def __init__(self, api_key_manager: APIKeyManager, rate_limiter: RateLimiter, auth_required: bool = True, rate_limit_enabled: bool = True):
        self.api_key_manager = api_key_manager
        self.rate_limiter = rate_limiter
        self.api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
        self.auth_required = auth_required
        self.rate_limit_enabled = rate_limit_enabled

        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/v1/auth/generate-key"  # API key generation endpoint
        }

        # Paths that require authentication (when auth is enabled)
        self.protected_path_prefixes = [
            "/v1/chat/completions",
            "/v1/images/generations",
            "/v1/models",
            "/v1/TTI/models"
        ]
    
    def is_protected_path(self, path: str) -> bool:
        """Check if a path requires authentication."""
        if not self.auth_required:
            return False  # No authentication required in no-auth mode

        if path in self.public_paths:
            return False

        return any(path.startswith(prefix) for prefix in self.protected_path_prefixes)
    
    def extract_api_key(self, authorization_header: Optional[str]) -> Optional[str]:
        """Extract API key from Authorization header."""
        if not authorization_header:
            return None
        
        # Support both "Bearer <key>" and direct key formats
        if authorization_header.startswith("Bearer "):
            return authorization_header[7:]
        elif authorization_header.startswith("ws_"):
            return authorization_header
        else:
            return None
    
    async def authenticate_request(self, request: Request) -> Tuple[bool, Optional[APIKey], Optional[dict]]:
        """
        Authenticate a request.
        
        Returns:
            Tuple of (is_authenticated, api_key_object, error_response)
        """
        path = request.url.path
        
        # Check if path requires authentication
        if not self.is_protected_path(path):
            return True, None, None
        
        # Get authorization header
        auth_header = request.headers.get("authorization")
        api_key = self.extract_api_key(auth_header)
        
        if not api_key:
            error_response = {
                "error": "API key required",
                "code": "missing_api_key",
                "details": {
                    "message": "Please provide a valid API key in the Authorization header",
                    "format": "Authorization: Bearer <your-api-key>"
                }
            }
            return False, None, error_response
        
        # Validate API key
        is_valid, api_key_obj, error_msg = await self.api_key_manager.validate_api_key(api_key)
        
        if not is_valid:
            error_response = {
                "error": error_msg or "Invalid API key",
                "code": "invalid_api_key",
                "details": {
                    "message": "The provided API key is invalid, expired, or inactive"
                }
            }
            return False, None, error_response
        
        return True, api_key_obj, None
    
    async def check_rate_limit(self, api_key: Optional[APIKey], client_ip: str = "unknown") -> Tuple[bool, dict]:
        """
        Check rate limit for an API key or IP address.

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        if not self.rate_limit_enabled:
            # Rate limiting disabled
            return True, {
                "allowed": True,
                "limit": 999999,
                "remaining": 999999,
                "reset_at": None,
                "retry_after": None
            }

        if api_key:
            return await self.rate_limiter.check_rate_limit(api_key)
        else:
            # No-auth mode: use IP-based rate limiting
            return await self.rate_limiter.check_ip_rate_limit(client_ip)
    
    async def process_request(self, request: Request) -> Tuple[bool, Optional[dict], Optional[dict]]:
        """
        Process a request through authentication and rate limiting.
        
        Returns:
            Tuple of (is_allowed, error_response, rate_limit_headers)
        """
        # Authenticate request
        is_authenticated, api_key_obj, auth_error = await self.authenticate_request(request)
        
        if not is_authenticated:
            return False, auth_error, None
        
        # If no API key (public endpoint or no-auth mode), handle rate limiting
        if not api_key_obj:
            if not self.auth_required:
                # No-auth mode: still check rate limiting by IP
                client_ip = request.client.host if request.client else "unknown"
                is_allowed, rate_limit_info = await self.check_rate_limit(None, client_ip)

                if not is_allowed:
                    error_response = {
                        "error": "Rate limit exceeded",
                        "code": "rate_limit_exceeded",
                        "details": {
                            "message": f"Rate limit of {rate_limit_info['limit']} requests per minute exceeded",
                            "retry_after": rate_limit_info["retry_after"],
                            "reset_at": rate_limit_info["reset_at"].isoformat() if rate_limit_info["reset_at"] else None
                        }
                    }
                    rate_limit_headers = {
                        "X-RateLimit-Limit": str(rate_limit_info["limit"]),
                        "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
                        "Retry-After": str(rate_limit_info["retry_after"])
                    }
                    if rate_limit_info["reset_at"]:
                        rate_limit_headers["X-RateLimit-Reset"] = rate_limit_info["reset_at"].isoformat()
                    return False, error_response, rate_limit_headers

                # Store rate limit info for no-auth mode
                request.state.rate_limit_info = rate_limit_info
                rate_limit_headers = {
                    "X-RateLimit-Limit": str(rate_limit_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_limit_info["remaining"])
                }
                if rate_limit_info["reset_at"]:
                    rate_limit_headers["X-RateLimit-Reset"] = rate_limit_info["reset_at"].isoformat()
                return True, None, rate_limit_headers
            else:
                return True, None, None

        # Check rate limit for authenticated requests
        client_ip = request.client.host if request.client else "unknown"
        is_allowed, rate_limit_info = await self.check_rate_limit(api_key_obj, client_ip)
        
        # Prepare rate limit headers
        rate_limit_headers = {
            "X-RateLimit-Limit": str(rate_limit_info["limit"]),
            "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
            "X-RateLimit-Reset": rate_limit_info["reset_at"].isoformat()
        }
        
        if not is_allowed:
            error_response = {
                "error": "Rate limit exceeded",
                "code": "rate_limit_exceeded",
                "details": {
                    "message": f"Rate limit of {rate_limit_info['limit']} requests per minute exceeded",
                    "retry_after": rate_limit_info["retry_after"],
                    "reset_at": rate_limit_info["reset_at"].isoformat()
                }
            }
            rate_limit_headers["Retry-After"] = str(rate_limit_info["retry_after"])
            return False, error_response, rate_limit_headers
        
        # Store API key info in request state for use in endpoints
        request.state.api_key = api_key_obj
        request.state.rate_limit_info = rate_limit_info
        
        return True, None, rate_limit_headers
    
    def create_error_response(self, error_data: dict, status_code: int = 401, headers: Optional[dict] = None) -> JSONResponse:
        """Create a standardized error response."""
        response = JSONResponse(
            status_code=status_code,
            content=error_data
        )
        
        if headers:
            for key, value in headers.items():
                response.headers[key] = value
        
        return response
    
    async def __call__(self, request: Request, call_next):
        """Middleware callable for FastAPI."""
        # Process request
        is_allowed, error_response, rate_limit_headers = await self.process_request(request)
        
        if not is_allowed:
            status_code = 429 if error_response.get("code") == "rate_limit_exceeded" else 401
            return self.create_error_response(error_response, status_code, rate_limit_headers)
        
        # Continue with request
        response = await call_next(request)
        
        # Add rate limit headers to response
        if rate_limit_headers:
            for key, value in rate_limit_headers.items():
                response.headers[key] = value
        
        return response
