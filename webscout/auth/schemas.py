# webscout/auth/schemas.py

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class APIKeyCreateRequest(BaseModel):
    """Request model for creating a new API key."""
    username: str = Field(..., min_length=3, max_length=50, description="Username for the API key owner (required)")
    telegram_id: str = Field(..., min_length=1, description="Telegram user ID (required)")
    name: Optional[str] = Field(None, description="Optional name for the API key")
    rate_limit: Optional[int] = Field(10, description="Rate limit per minute (default: 10)")
    expires_in_days: Optional[int] = Field(None, description="Number of days until expiration")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class APIKeyCreateResponse(BaseModel):
    """Response model for API key creation."""
    api_key: str = Field(..., description="The generated API key")
    key_id: str = Field(..., description="Unique identifier for the API key")
    user_id: str = Field(..., description="User ID associated with the API key")
    name: Optional[str] = Field(None, description="Name of the API key")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    rate_limit: int = Field(..., description="Rate limit per minute")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKeyValidationResponse(BaseModel):
    """Response model for API key validation."""
    valid: bool = Field(..., description="Whether the API key is valid")
    user_id: Optional[str] = Field(None, description="User ID if key is valid")
    key_id: Optional[str] = Field(None, description="Key ID if key is valid")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")
    usage_count: Optional[int] = Field(None, description="Total usage count")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    error: Optional[str] = Field(None, description="Error message if key is invalid")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserCreateRequest(BaseModel):
    """Request model for creating a new user."""
    username: str = Field(..., min_length=3, max_length=50, description="Username for the new user")
    telegram_id: str = Field(..., min_length=1, description="Telegram user ID (required)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class UserResponse(BaseModel):
    """Response model for user information."""
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    telegram_id: str = Field(..., description="Telegram user ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_active: bool = Field(..., description="Whether the user is active")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RateLimitStatus(BaseModel):
    """Response model for rate limit status."""
    allowed: bool = Field(..., description="Whether the request is allowed")
    limit: int = Field(..., description="Rate limit per minute")
    remaining: int = Field(..., description="Remaining requests in current window")
    reset_at: datetime = Field(..., description="When the rate limit resets")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    database: str = Field(..., description="Database status")
    timestamp: datetime = Field(..., description="Check timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
