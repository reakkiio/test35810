# Docker Setup for Webscout

This Docker configuration is designed to work seamlessly when Webscout is installed via pip or git+pip, without requiring any external docker directory or entrypoint scripts.

## Quick Start

### Build and Run

```bash
# Build the image
docker build -t webscout-api .

# Run the container (default port 8000)
docker run -p 8000:8000 webscout-api

# Run with custom port (e.g., 7860)
docker run -p 7860:7860 -e WEBSCOUT_PORT=7860 webscout-api
```

### Using Docker Compose

```bash
# Basic setup
docker-compose up webscout-api

# With custom port
WEBSCOUT_PORT=7860 docker-compose up webscout-api

# Production setup with Gunicorn
docker-compose --profile production up webscout-api-production

# Development setup with hot reload
docker-compose --profile development up webscout-api-dev
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

- `WEBSCOUT_HOST` - Server host (default: 0.0.0.0)
- `WEBSCOUT_PORT` - Server port (default: 8000) - **Now properly supported!**
- `WEBSCOUT_WORKERS` - Number of worker processes (default: 1) - **New!**
- `WEBSCOUT_LOG_LEVEL` - Log level: debug, info, warning, error, critical (default: info) - **New!**
- `WEBSCOUT_API_KEY` - Optional API key for authentication
- `WEBSCOUT_DEFAULT_PROVIDER` - Default LLM provider
- `WEBSCOUT_BASE_URL` - Base URL for the API (optional)
- `WEBSCOUT_DEBUG` - Enable debug mode (default: false)

**Legacy Support**: For backward compatibility, the following legacy environment variables are also supported:
- `PORT` (fallback for `WEBSCOUT_PORT`)
- `API_KEY` (fallback for `WEBSCOUT_API_KEY`)
- `DEFAULT_PROVIDER` (fallback for `WEBSCOUT_DEFAULT_PROVIDER`)
- `BASE_URL` (fallback for `WEBSCOUT_BASE_URL`)
- `DEBUG` (fallback for `WEBSCOUT_DEBUG`)

**Note**: When both WEBSCOUT_* and legacy variables are set, WEBSCOUT_* takes precedence.

### Service Profiles

- **Default**: Basic API server with single worker
- **Production**: Gunicorn with multiple workers and optimized settings
- **Development**: Uvicorn with hot reload and debug logging
- **Nginx**: Optional reverse proxy (requires custom nginx.conf)
- **Monitoring**: Optional Prometheus monitoring (requires custom prometheus.yml)

## Features

- ✅ No external docker directory required
- ✅ Works with pip/git installations
- ✅ Multi-stage build for optimized image size
- ✅ Non-root user for security
- ✅ Health checks included
- ✅ Multiple deployment profiles
- ✅ Comprehensive Makefile for easy management
- ✅ Volume mounts for logs and data persistence

## Health Checks

The setup includes automatic health checks that verify the `/v1/models` endpoint is responding correctly.

## Security

- Runs as non-root user (`webscout:webscout`)
- Minimal runtime dependencies
- Security-optimized container settings
- Optional API key authentication

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
