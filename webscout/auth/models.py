# webscout/auth/models.py

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import uuid
import json


@dataclass
class User:
    """User model for authentication system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    telegram_id: int = field(default_factory=lambda: 0)  # Required Telegram ID as number only

    def validate_telegram_id(self) -> None:
        """Ensure telegram_id is an integer."""
        if not isinstance(self.telegram_id, int):
            raise ValueError("telegram_id must be an integer.")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for storage."""
        return {
            "id": self.id,
            "username": self.username,
            "telegram_id": self.telegram_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create user from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            username=data.get("username", ""),
            telegram_id=int(data.get("telegram_id", 0)),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {})
        )


@dataclass
class APIKey:
    """API Key model for authentication system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key: str = ""
    user_id: str = ""
    name: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 10  # requests per minute
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert API key to dictionary for storage."""
        return {
            "id": self.id,
            "key": self.key,
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "rate_limit": self.rate_limit,
            "usage_count": self.usage_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKey":
        """Create API key from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            key=data.get("key", ""),
            user_id=data.get("user_id", ""),
            name=data.get("name"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            is_active=data.get("is_active", True),
            rate_limit=data.get("rate_limit", 10),
            usage_count=data.get("usage_count", 0),
            metadata=data.get("metadata", {})
        )
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid (active and not expired)."""
        return self.is_active and not self.is_expired()


@dataclass
class RateLimitEntry:
    """Rate limit tracking entry."""
    api_key_id: str
    requests: List[datetime] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "api_key_id": self.api_key_id,
            "requests": [req.isoformat() for req in self.requests]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RateLimitEntry":
        """Create from dictionary."""
        return cls(
            api_key_id=data.get("api_key_id", ""),
            requests=[datetime.fromisoformat(req) for req in data.get("requests", [])]
        )
