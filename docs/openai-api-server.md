# OpenAI-Compatible API Server (`webscout.auth`)

Webscout's [`webscout.auth`](../webscout/auth/__init__.py:1) module provides a comprehensive OpenAI-compatible API server with advanced authentication, rate limiting, and provider management capabilities. This server allows you to use any supported provider with tools and applications designed for OpenAI's API while maintaining enterprise-grade security and performance.

## Table of Contents

1. [Core Components](#core-components)
2. [Server Configuration](#server-configuration)
3. [Authentication System](#authentication-system)
4. [Rate Limiting](#rate-limiting)
5. [Provider Management](#provider-management)
6. [API Endpoints](#api-endpoints)
7. [Starting the Server](#starting-the-server)
8. [Usage Examples](#usage-examples)
9. [Environment Variables](#environment-variables)
10. [Database Integration](#database-integration)
11. [Error Handling](#error-handling)
12. [Security Features](#security-features)
13. [Troubleshooting](#troubleshooting)

## Core Components

### [`server.py`](../webscout/auth/server.py:1)

The main server module that creates and configures the FastAPI application with OpenAI-compatible endpoints.

```python
from webscout.auth.server import create_app, run_api, start_server

# Create FastAPI app
app = create_app()

# Start server programmatically
start_server(port=8000, host="0.0.0.0")
```

**Key Features:**
- OpenAI-compatible API endpoints
- Automatic provider discovery and registration
- Built-in authentication and rate limiting
- Comprehensive error handling and logging
- Interactive API documentation with custom UI

### [`auth_system.py`](../webscout/auth/auth_system.py:1)

Central authentication system that coordinates all security components.

```python
from webscout.auth.auth_system import initialize_auth_system, get_auth_components

# Initialize authentication
initialize_auth_system(app, auth_required=True, rate_limit_enabled=True)

# Get auth components
components = get_auth_components()
```

**Components:**
- [`DatabaseManager`](../webscout/auth/database.py:321): MongoDB with JSON fallback
- [`APIKeyManager`](../webscout/auth/api_key_manager.py:16): API key lifecycle management
- [`RateLimiter`](../webscout/auth/rate_limiter.py:15): Request rate limiting
- [`AuthMiddleware`](../webscout/auth/middleware.py:20): Request authentication and authorization

## Server Configuration

### [`ServerConfig`](../webscout/auth/config.py:22)

Centralized configuration management for the API server.

```python
from webscout.auth.config import ServerConfig

config = ServerConfig()
config.update(
    port=8080,
    host="localhost",
    auth_required=False,
    rate_limit_enabled=True,
    default_rate_limit=60
)
```

**Configuration Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"0.0.0.0"` | Server host address |
| `port` | `int` | `8000` | Server port number |
| `debug` | `bool` | `False` | Enable debug mode |
| `auth_required` | `bool` | `True` | Enable/disable authentication |
| `rate_limit_enabled` | `bool` | `True` | Enable/disable rate limiting |
| `default_rate_limit` | `int` | `60` | Default requests per minute |
| `cors_origins` | `List[str]` | `["*"]` | CORS allowed origins |
| `max_request_size` | `int` | `10MB` | Maximum request size |
| `request_timeout` | `int` | `300` | Request timeout in seconds |

## Authentication System

### [`APIKeyManager`](../webscout/auth/api_key_manager.py:16)

Manages API key generation, validation, and lifecycle with secure storage and comprehensive tracking.

```python
from webscout.auth.api_key_manager import APIKeyManager
from webscout.auth.database import DatabaseManager

# Initialize
db_manager = DatabaseManager()
api_key_manager = APIKeyManager(db_manager)

# Create API key
api_key, user = await api_key_manager.create_api_key(
    username="john_doe",
    telegram_id="123456789",
    name="Production Key",
    rate_limit=100,
    expires_in_days=30
)
```

**Key Features:**
- Secure key generation with [`secrets`](../webscout/auth/api_key_manager.py:24) module
- One API key per user policy
- Automatic expiration handling
- Usage tracking and analytics
- Telegram ID integration for user management

### [`User`](../webscout/auth/models.py:11) and [`APIKey`](../webscout/auth/models.py:53) Models

Data models for user and API key management with comprehensive validation.

```python
from webscout.auth.models import User, APIKey
from datetime import datetime, timezone

# Create user
user = User(
    username="developer",
    telegram_id=123456789,
    created_at=datetime.now(timezone.utc)
)

# Create API key
api_key = APIKey(
    key="ws_abc123...",
    user_id=user.id,
    name="Development Key",
    rate_limit=50,
    expires_at=None  # No expiration
)
```

## Rate Limiting

### [`RateLimiter`](../webscout/auth/rate_limiter.py:15)

Advanced rate limiting system with sliding window algorithm and IP-based fallback.

```python
from webscout.auth.rate_limiter import RateLimiter

rate_limiter = RateLimiter(database_manager)

# Check rate limit
is_allowed, rate_info = await rate_limiter.check_rate_limit(api_key)

# IP-based rate limiting (no-auth mode)
is_allowed, rate_info = await rate_limiter.check_ip_rate_limit("192.168.1.1")
```

**Rate Limiting Features:**
- Sliding window algorithm for accurate rate limiting
- Per-API-key and per-IP rate limiting
- Automatic cleanup of expired entries
- Configurable time windows and limits
- Rate limit status reporting

**Rate Limit Response Headers:**
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 2024-01-15T10:30:00Z
Retry-After: 15
```

## Provider Management

### [`providers.py`](../webscout/auth/providers.py:1)

Automatic provider discovery and management system with intelligent model resolution.

```python
from webscout.auth.providers import (
    initialize_provider_map,
    resolve_provider_and_model,
    get_provider_instance
)

# Initialize providers
initialize_provider_map()

# Resolve provider and model
provider_class, model_name = resolve_provider_and_model("ChatGPT/gpt-4")

# Get cached provider instance
provider = get_provider_instance(provider_class)
```

**Provider Features:**
- Automatic discovery of OpenAI-compatible providers
- Model validation and availability checking
- Provider instance caching for performance
- Support for both chat and image generation providers
- Fallback provider configuration

## API Endpoints

### Chat Completions

**Endpoint:** `POST /v1/chat/completions`

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={
        "model": "ChatGPT/gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 150,
        "stream": False
    }
)
```

### Image Generation

**Endpoint:** `POST /v1/images/generations`

```python
response = requests.post(
    "http://localhost:8000/v1/images/generations",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={
        "prompt": "A futuristic cityscape at sunset",
        "model": "PollinationsAI/flux",
        "n": 1,
        "size": "1024x1024",
        "response_format": "url"
    }
)
```

### Model Listing

**Endpoint:** `GET /v1/models`

```python
response = requests.get(
    "http://localhost:8000/v1/models",
    headers={"Authorization": "Bearer your-api-key"}
)
```

### Authentication Endpoints

**API Key Generation:** `POST /v1/auth/generate-key`

```python
response = requests.post(
    "http://localhost:8000/v1/auth/generate-key",
    json={
        "username": "developer",
        "telegram_id": "123456789",
        "name": "Development Key",
        "rate_limit": 100,
        "expires_in_days": 30
    }
)
```

**API Key Validation:** `GET /v1/auth/validate`

```python
response = requests.get(
    "http://localhost:8000/v1/auth/validate",
    headers={"Authorization": "Bearer your-api-key"}
)
```

## Starting the Server

### Command Line Interface

The server provides a comprehensive CLI with environment variable support:

```bash
# Basic startup
webscout-server

# Custom configuration
webscout-server --port 8080 --host localhost --debug

# Authentication modes
webscout-server --no-auth                    # Disable authentication
webscout-server --api-key "your-secret-key"  # Legacy API key
webscout-server --no-rate-limit             # Disable rate limiting

# Production settings
webscout-server --workers 4 --log-level info
```

### Programmatic Startup

```python
from webscout.auth import start_server, run_api

# Simple startup
start_server()

# Advanced configuration
start_server(
    port=8080,
    host="0.0.0.0",
    debug=False,
    no_auth=False,
    no_rate_limit=False
)

# Full control with run_api
run_api(
    host="0.0.0.0",
    port=8000,
    workers=4,
    log_level="info",
    debug=False
)
```

### Alternative Methods

```bash
# Using UV (no installation required)
uv run --extra api webscout-server

# Using Python module
python -m webscout.auth.server

# Direct module execution
python -m webscout.auth.server --port 8080
```

## Usage Examples

### OpenAI Python Client

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)

# Chat completion
response = client.chat.completions.create(
    model="ChatGPT/gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Streaming Responses

```python
# Streaming chat completion
stream = client.chat.completions.create(
    model="ChatGPT/gpt-4",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### cURL Examples

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "ChatGPT/gpt-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# List models
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-api-key"

# Generate API key
curl http://localhost:8000/v1/auth/generate-key \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "telegram_id": "123456789"
  }'
```

## Environment Variables

The server supports comprehensive environment variable configuration:

### Server Configuration

```bash
# Server settings
export WEBSCOUT_HOST="0.0.0.0"
export WEBSCOUT_PORT="8000"
export WEBSCOUT_WORKERS="4"
export WEBSCOUT_LOG_LEVEL="info"
export WEBSCOUT_DEBUG="false"

# Authentication settings
export WEBSCOUT_NO_AUTH="false"
export WEBSCOUT_NO_RATE_LIMIT="false"
export WEBSCOUT_API_KEY="your-legacy-key"
export WEBSCOUT_DEFAULT_PROVIDER="ChatGPT"

# Database settings
export MONGODB_URL="mongodb://localhost:27017"
export WEBSCOUT_DATA_DIR="./data"
```

### Docker Environment

```bash
# Docker-specific variables
export WEBSCOUT_API_TITLE="My AI API"
export WEBSCOUT_API_DESCRIPTION="Custom AI API Server"
export WEBSCOUT_API_VERSION="1.0.0"
export WEBSCOUT_API_DOCS_URL="/docs"
```

For a complete list of supported environment variables and Docker deployment options, see [DOCKER.md](../DOCKER.md).

## Database Integration

### [`DatabaseManager`](../webscout/auth/database.py:321)

Flexible database system with MongoDB primary and JSON file fallback.

```python
from webscout.auth.database import DatabaseManager

# Initialize with MongoDB
db_manager = DatabaseManager(
    mongo_connection_string="mongodb://localhost:27017",
    data_dir="./data"
)

await db_manager.initialize()

# Database operations
user = await db_manager.create_user(user_obj)
api_key = await db_manager.create_api_key(api_key_obj)
```

**Database Features:**
- MongoDB with automatic JSON fallback
- Atomic operations with transaction support
- Automatic connection management
- Data validation and integrity checks
- Performance optimization with indexing

### JSON Database Fallback

```python
from webscout.auth.database import JSONDatabase

# Direct JSON database usage
json_db = JSONDatabase(data_dir="./data")

# Thread-safe operations
user = await json_db.create_user(user_obj)
api_key = await json_db.get_api_key("ws_key123")
```

## Error Handling

### [`APIError`](../webscout/auth/exceptions.py:26)

Comprehensive error handling with OpenAI-compatible error responses.

```python
from webscout.auth.exceptions import APIError
from starlette.status import HTTP_400_BAD_REQUEST

# Raise API error
raise APIError(
    message="Invalid model specified",
    status_code=HTTP_400_BAD_REQUEST,
    error_type="invalid_request_error",
    param="model",
    code="model_not_found"
)
```

**Error Response Format:**
```json
{
  "error": {
    "message": "Invalid model specified",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found",
    "footer": "If you believe this is a bug, please pull an issue at https://github.com/OEvortex/Webscout."
  }
}
```

### Exception Handling

The server provides comprehensive exception handling with detailed error responses:

```python
# Validation errors
{
  "error": {
    "message": "Request validation error.",
    "details": [
      {
        "loc": ["body", "messages"],
        "message": "field required at body -> messages",
        "type": "value_error.missing"
      }
    ],
    "type": "validation_error"
  }
}

# Rate limit errors
{
  "error": "Rate limit exceeded",
  "code": "rate_limit_exceeded",
  "details": {
    "message": "Rate limit of 60 requests per minute exceeded",
    "retry_after": 15,
    "reset_at": "2024-01-15T10:30:00Z"
  }
}
```

## Security Features

### Authentication Modes

**1. Enhanced API Keys (Default)**
- Telegram ID-based user management
- One API key per user policy
- Automatic expiration and renewal
- Usage tracking and analytics

**2. Legacy API Key**
- Single shared API key
- Backward compatibility
- Simple authentication

**3. No-Auth Mode**
- Development and testing
- IP-based rate limiting
- Public access endpoints

### Security Best Practices

```python
# Secure API key generation
api_key = api_key_manager.generate_api_key()  # Uses secrets.choice()

# Rate limiting configuration
rate_limiter = RateLimiter(database_manager)
await rate_limiter.check_rate_limit(api_key)

# Request validation
from webscout.auth.middleware import AuthMiddleware
auth_middleware = AuthMiddleware(api_key_manager, rate_limiter)
```

**Security Headers:**
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 2024-01-15T10:30:00Z
Authorization: Bearer ws_abc123...
```

*This documentation covers the comprehensive functionality of the [`webscout.auth`](../webscout/auth/__init__.py:1) module. For the most up-to-date information, refer to the source code and inline documentation.*