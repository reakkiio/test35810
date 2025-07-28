"""
Webscout OpenAI-Compatible API Server

A FastAPI-based server that provides OpenAI-compatible endpoints for various LLM providers.
Supports streaming and non-streaming chat completions with comprehensive error handling,
authentication, and provider management.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.responses import HTMLResponse

from webscout.Litlogger import Logger, LogLevel, LogFormat, ConsoleHandler
from .config import ServerConfig, AppConfig
from .routes import Api
from .providers import initialize_provider_map, initialize_tti_provider_map
from .auth_system import initialize_auth_system

# Configuration constants
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
API_VERSION = "v1"

# Setup Litlogger
logger = Logger(
    name="webscout.api",
    level=LogLevel.INFO,
    handlers=[ConsoleHandler(stream=sys.stdout)],
    fmt=LogFormat.DEFAULT
)

# Global configuration instance - lazy initialization
config = None

def get_config() -> ServerConfig:
    """Get or create the global configuration instance."""
    global config
    if config is None:
        config = ServerConfig()
    return config


def create_app():
    """Create and configure the FastAPI application."""
    import os
    app_title = os.getenv("WEBSCOUT_API_TITLE", "Webscout API")
    app_description = os.getenv("WEBSCOUT_API_DESCRIPTION", "OpenAI API compatible interface for various LLM providers")
    app_version = os.getenv("WEBSCOUT_API_VERSION", "0.2.0")
    app_docs_url = os.getenv("WEBSCOUT_API_DOCS_URL", "/docs")
    app_redoc_url = os.getenv("WEBSCOUT_API_REDOC_URL", "/redoc")
    app_openapi_url = os.getenv("WEBSCOUT_API_OPENAPI_URL", "/openapi.json")

    app = FastAPI(
        title=app_title,
        description=app_description,
        version=app_version,
        docs_url=None,  # Disable default docs
        redoc_url=app_redoc_url,
        openapi_url=app_openapi_url,
    )

    # Simple Custom Swagger UI with WebScout footer
    @app.get(app_docs_url, include_in_schema=False)
    async def custom_swagger_ui_html():
        html = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - API Documentation",
        ).body.decode("utf-8")
        
        # Simple, clean footer with WebScout branding
        footer_html = """
        <div style='
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-top: 1px solid #e5e7eb;
            padding: 12px 20px;
            text-align: center;
            font-size: 14px;
            color: #6b7280;
            z-index: 1000;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        '>
            Powered by <a href='https://github.com/OEvortex/Webscout' target='_blank' style='
                color: #6366f1;
                text-decoration: none;
                font-weight: 600;
                transition: color 0.2s ease;
            ' onmouseover='this.style.color="#4f46e5"' onmouseout='this.style.color="#6366f1"'>
                WebScout
            </a>
        </div>
        <style>
            body {
                padding-bottom: 60px !important;
            }
            .swagger-ui .topbar {
                background-color: #fafafa;
                border-bottom: 1px solid #e5e7eb;
            }
            .swagger-ui .info .title {
                color: #1f2937;
            }
            .swagger-ui .btn.authorize {
                background-color: #6366f1;
                border-color: #6366f1;
            }
            .swagger-ui .btn.authorize:hover {
                background-color: #4f46e5;
                border-color: #4f46e5;
            }
        </style>
        """
        
        html = html.replace("</body>", f"{footer_html}</body>")
        return HTMLResponse(content=html)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize authentication system with configuration
    initialize_auth_system(
        app,
        auth_required=AppConfig.auth_required,
        rate_limit_enabled=AppConfig.rate_limit_enabled
    )
    # Add startup event handler
    @app.on_event("startup")
    async def startup():
        if hasattr(app.state, 'startup_event'):
            await app.state.startup_event()

    # Initialize API routes
    api = Api(app)
    api.register_validation_exception_handler()
    api.register_routes()
    
    # Initialize providers
    initialize_provider_map()
    initialize_tti_provider_map()

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    return app


def create_app_debug():
    """Create app in debug mode."""
    return create_app()


def start_server(
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    api_key: str = None,
    default_provider: str = None,
    base_url: str = None,
    workers: int = 1,
    log_level: str = 'info',
    debug: bool = False,
    no_auth: bool = False,
    no_rate_limit: bool = False
):
    """Start the API server with the given configuration."""
    run_api(
        host=host,
        port=port,
        api_key=api_key,
        default_provider=default_provider,
        base_url=base_url,
        workers=workers,
        log_level=log_level,
        debug=debug,
        no_auth=no_auth,
        no_rate_limit=no_rate_limit,
    )


def run_api(
    host: str = '0.0.0.0',
    port: int = None,
    api_key: str = None,
    default_provider: str = None,
    base_url: str = None,
    debug: bool = False,
    workers: int = 1,
    log_level: str = 'info',
    show_available_providers: bool = True,
    no_auth: bool = False,
    no_rate_limit: bool = False,
) -> None:
    """Run the API server with configuration."""
    print("Starting Webscout OpenAI API server...")
    if port is None:
        port = DEFAULT_PORT
        
    AppConfig.set_config(
        api_key=api_key,
        default_provider=default_provider or AppConfig.default_provider,
        base_url=base_url,
        auth_required=not no_auth,
        rate_limit_enabled=not no_rate_limit
    )

    if show_available_providers:
        if not AppConfig.provider_map:
            initialize_provider_map()
        if not AppConfig.tti_provider_map:
            initialize_tti_provider_map()

        print("\n=== Webscout OpenAI API Server ===")
        print(f"Server URL: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        if AppConfig.base_url:
            print(f"Base Path: {AppConfig.base_url}")
            api_endpoint_base = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}{AppConfig.base_url}"
        else:
            api_endpoint_base = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

        print(f"API Endpoint: {api_endpoint_base}/v1/chat/completions")
        print(f"Docs URL: {api_endpoint_base}/docs")

        # Show authentication status
        if no_auth:
            print(f"Authentication: 🔓 DISABLED (No-Auth Mode)")
        elif api_key:
            print(f"Authentication: 🔐 Legacy API Key")
        else:
            print(f"Authentication: 🔑 Enhanced API Keys")

        # Show rate limiting status
        if no_rate_limit:
            print(f"Rate Limiting: ⚡ DISABLED")
        else:
            print(f"Rate Limiting: 🛡️  Enabled ({AppConfig.default_rate_limit}/min)")

        print(f"Default Provider: {AppConfig.default_provider}")
        print(f"Workers: {workers}")
        print(f"Log Level: {log_level}")
        print(f"Debug Mode: {'Enabled' if debug else 'Disabled'}")

        providers = list(set(v.__name__ for v in AppConfig.provider_map.values()))
        print(f"\n--- Available Providers ({len(providers)}) ---")
        for i, provider_name in enumerate(sorted(providers), 1):
            print(f"{i}. {provider_name}")

        provider_class_names = set(v.__name__ for v in AppConfig.provider_map.values())
        models = sorted([model for model in AppConfig.provider_map.keys() if model not in provider_class_names])
        if models:
            print(f"\n--- Available Models ({len(models)}) ---")
            for i, model_name in enumerate(models, 1):
                print(f"{i}. {model_name} (via {AppConfig.provider_map[model_name].__name__})")
        else:
            print("\nNo specific models registered. Use provider names as models.")

        tti_providers = list(set(v.__name__ for v in AppConfig.tti_provider_map.values()))
        print(f"\n--- Available TTI Providers ({len(tti_providers)}) ---")
        for i, provider_name in enumerate(sorted(tti_providers), 1):
            print(f"{i}. {provider_name}")

        tti_models = sorted([model for model in AppConfig.tti_provider_map.keys() if model not in tti_providers])
        if tti_models:
            print(f"\n--- Available TTI Models ({len(tti_models)}) ---")
            for i, model_name in enumerate(tti_models, 1):
                print(f"{i}. {model_name} (via {AppConfig.tti_provider_map[model_name].__name__})")
        else:
            print("\nNo specific TTI models registered. Use TTI provider names as models.")

        print("\nUse Ctrl+C to stop the server.")
        print("=" * 40 + "\n")

    uvicorn_app_str = "webscout.auth.server:create_app_debug" if debug else "webscout.auth.server:create_app"

    # Configure uvicorn settings
    uvicorn_config = {
        "app": uvicorn_app_str,
        "host": host,
        "port": int(port),
        "factory": True,
        "reload": debug,
        "log_level": log_level.lower() if log_level else ("debug" if debug else "info"),
    }

    # Add workers only if not in debug mode
    if not debug and workers > 1:
        uvicorn_config["workers"] = workers
        print(f"Starting with {workers} workers...")
    elif debug:
        print("Debug mode enabled - using single worker with reload...")

    uvicorn.run(**uvicorn_config)


def main():
    """Main entry point for the webscout-server console script."""
    import argparse

    # Read environment variables with fallbacks
    default_port = int(os.getenv('WEBSCOUT_PORT', os.getenv('PORT', DEFAULT_PORT)))
    default_host = os.getenv('WEBSCOUT_HOST', DEFAULT_HOST)
    default_workers = int(os.getenv('WEBSCOUT_WORKERS', '1'))
    default_log_level = os.getenv('WEBSCOUT_LOG_LEVEL', 'info')
    default_api_key = os.getenv('WEBSCOUT_API_KEY', os.getenv('API_KEY'))
    default_provider = os.getenv('WEBSCOUT_DEFAULT_PROVIDER', os.getenv('DEFAULT_PROVIDER'))
    default_base_url = os.getenv('WEBSCOUT_BASE_URL', os.getenv('BASE_URL'))
    default_debug = os.getenv('WEBSCOUT_DEBUG', os.getenv('DEBUG', 'false')).lower() == 'true'
    default_no_auth = os.getenv('WEBSCOUT_NO_AUTH', 'false').lower() == 'true'
    default_no_rate_limit = os.getenv('WEBSCOUT_NO_RATE_LIMIT', 'false').lower() == 'true'

    parser = argparse.ArgumentParser(description='Start Webscout OpenAI-compatible API server')
    parser.add_argument('--port', type=int, default=default_port, help=f'Port to run the server on (default: {default_port})')
    parser.add_argument('--host', type=str, default=default_host, help=f'Host to bind the server to (default: {default_host})')
    parser.add_argument('--workers', type=int, default=default_workers, help=f'Number of worker processes (default: {default_workers})')
    parser.add_argument('--log-level', type=str, default=default_log_level, choices=['debug', 'info', 'warning', 'error', 'critical'], help=f'Log level (default: {default_log_level})')
    parser.add_argument('--api-key', type=str, default=default_api_key, help='API key for authentication (optional)')
    parser.add_argument('--default-provider', type=str, default=default_provider, help='Default provider to use (optional)')
    parser.add_argument('--base-url', type=str, default=default_base_url, help='Base URL for the API (optional, e.g., /api/v1)')
    parser.add_argument('--debug', action='store_true', default=default_debug, help='Run in debug mode')
    parser.add_argument('--no-auth', action='store_true', default=default_no_auth, help='Disable authentication (no API keys required)')
    parser.add_argument('--no-rate-limit', action='store_true', default=default_no_rate_limit, help='Disable rate limiting (unlimited requests)')
    args = parser.parse_args()

    # Print configuration summary
    print(f"Configuration:")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Workers: {args.workers}")
    print(f"  Log Level: {args.log_level}")
    print(f"  Debug Mode: {args.debug}")
    print(f"  No-Auth Mode: {'🔓 ENABLED' if args.no_auth else '🔐 Disabled'}")
    print(f"  No Rate Limit: {'⚡ ENABLED' if args.no_rate_limit else '🛡️  Disabled'}")
    print(f"  API Key: {'Set' if args.api_key else 'Not set'}")
    print(f"  Default Provider: {args.default_provider or 'Not set'}")
    print(f"  Base URL: {args.base_url or 'Not set'}")
    print()

    run_api(
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        api_key=args.api_key,
        default_provider=args.default_provider,
        base_url=args.base_url,
        debug=args.debug,
        no_auth=args.no_auth,
        no_rate_limit=args.no_rate_limit
    )


if __name__ == "__main__":
    main()
