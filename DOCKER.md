# Docker Setup for Webscout

This Docker configuration is designed to work seamlessly when Webscout is installed via pip or git+pip, without requiring any external docker directory or entrypoint scripts. It supports the new enhanced authentication system with no-auth mode for flexible deployment scenarios.

## Quick Start

### Build and Run

```bash
# Build the image
docker build -t webscout-api .

# Run the container (default port 8000, with authentication)
docker run -p 8000:8000 webscout-api

# Run with no authentication required (great for development/demos)
docker run -p 8000:8000 -e WEBSCOUT_NO_AUTH=true webscout-api

# Run with no authentication and no rate limiting (maximum openness)
docker run -p 8000:8000 -e WEBSCOUT_NO_AUTH=true -e WEBSCOUT_NO_RATE_LIMIT=true webscout-api

# Run with custom port (e.g., 7860)
docker run -p 7860:7860 -e WEBSCOUT_PORT=7860 webscout-api

# Run with MongoDB support
docker run -p 8000:8000 -e MONGODB_URL=mongodb://localhost:27017 webscout-api
```

### Using Docker Compose

```bash
# Basic setup (with authentication)
docker-compose up webscout-api

# No-auth mode for development/demos
docker-compose -f docker-compose.yml -f docker-compose.no-auth.yml up webscout-api

# With custom port
WEBSCOUT_PORT=7860 docker-compose up webscout-api

# Production setup with Gunicorn
docker-compose --profile production up webscout-api-production

# Development setup with hot reload
docker-compose --profile development up webscout-api-dev

# MongoDB setup with authentication
docker-compose --profile mongodb up
```

### Using Makefile

```bash
# Quick start (build + run + test)
make quick-start

# Build image
make build

# Start services
make up

# View logs
make logs

# Run health check
make health

# Clean up
make clean
```

## Configuration

### Environment Variables

All environment variables are now fully supported in the API server:

#### **Core Server Settings**
- `WEBSCOUT_HOST` - Server host (default: 0.0.0.0)
- `WEBSCOUT_PORT` - Server port (default: 8000)
- `WEBSCOUT_WORKERS` - Number of worker processes (default: 1)
- `WEBSCOUT_LOG_LEVEL` - Log level: debug, info, warning, error, critical (default: info)
- `WEBSCOUT_DEBUG` - Enable debug mode (default: false)
- `WEBSCOUT_API_TITLE` - **NEW!** FastAPI app title (default: Webscout OpenAI API)
- `WEBSCOUT_API_DESCRIPTION` - **NEW!** FastAPI app description (default: OpenAI API compatible interface for various LLM providers with enhanced authentication)
- `WEBSCOUT_API_VERSION` - **NEW!** FastAPI app version (default: 0.2.0)
- `WEBSCOUT_API_DOCS_URL` - **NEW!** FastAPI docs URL (default: /docs)
- `WEBSCOUT_API_REDOC_URL` - **NEW!** FastAPI redoc URL (default: /redoc)
- `WEBSCOUT_API_OPENAPI_URL` - **NEW!** FastAPI OpenAPI URL (default: /openapi.json)

#### **Authentication & Security** 🔐
- `WEBSCOUT_NO_AUTH` - **NEW!** Disable authentication (default: false) 🔓
- `WEBSCOUT_NO_RATE_LIMIT` - **NEW!** Disable rate limiting (default: false) ⚡
- `WEBSCOUT_API_KEY` - Legacy API key for authentication (optional)

#### **Database Configuration** 🗄️
- `MONGODB_URL` - **NEW!** MongoDB connection string (optional)
- `WEBSCOUT_DATA_DIR` - **NEW!** Data directory for JSON database (default: /app/data)

#### **Provider Settings**
- `WEBSCOUT_DEFAULT_PROVIDER` - Default LLM provider (default: ChatGPT)
- `WEBSCOUT_BASE_URL` - Base URL for the API (optional)

**Legacy Support**: For backward compatibility, the following legacy environment variables are also supported:
- `PORT` (fallback for `WEBSCOUT_PORT`)
- `API_KEY` (fallback for `WEBSCOUT_API_KEY`)
- `DEFAULT_PROVIDER` (fallback for `WEBSCOUT_DEFAULT_PROVIDER`)
- `BASE_URL` (fallback for `WEBSCOUT_BASE_URL`)
- `DEBUG` (fallback for `WEBSCOUT_DEBUG`)

**Note**: When both WEBSCOUT_* and legacy variables are set, WEBSCOUT_* takes precedence.

### Service Profiles

- **Default**: Basic API server with enhanced authentication system
- **No-Auth**: Development/demo mode with no authentication required 🔓
- **Production**: Gunicorn with multiple workers and optimized settings
- **Development**: Uvicorn with hot reload and debug logging
- **MongoDB**: Full setup with MongoDB database support 🗄️
- **Nginx**: Optional reverse proxy (requires custom nginx.conf)
- **Monitoring**: Optional Prometheus monitoring (requires custom prometheus.yml)

## Features

- ✅ No external docker directory required
- ✅ Works with pip/git installations
- ✅ Multi-stage build for optimized image size
- ✅ Non-root user for security
- ✅ Health checks included
- ✅ Multiple deployment profiles
- ✅ **NEW!** No-auth mode for development/demos 🔓
- ✅ **NEW!** Enhanced authentication system with API key management 🔑
- ✅ **NEW!** MongoDB and JSON database support 🗄️
- ✅ **NEW!** Rate limiting with IP-based fallback 🛡️
- ✅ Comprehensive Makefile for easy management
- ✅ Volume mounts for logs and data persistence

## Health Checks

The setup includes automatic health checks that verify the `/health` endpoint is responding correctly. This endpoint provides comprehensive system status including database connectivity and authentication system status.

## Security

- Runs as non-root user (`webscout:webscout`)
- Minimal runtime dependencies
- Security-optimized container settings
- **Enhanced authentication system** with API key management 🔑
- **Rate limiting** to prevent abuse 🛡️
- **No-auth mode** for development (use with caution in production) 🔓
- **Database encryption** support with MongoDB
- **Secure API key generation** with cryptographic randomness

## Troubleshooting

### Check container status
```bash
make status
```

### View logs
```bash
make logs
```

### Test endpoints
```bash
make test-endpoints
```

### Access container shell
```bash
make shell
```
