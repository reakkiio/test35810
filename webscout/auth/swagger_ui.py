"""
Custom Swagger UI implementation for Webscout FastAPI server.
Provides a modern, beautiful, and fully functional API documentation interface.
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

class CustomSwaggerUI:
    """Custom Swagger UI handler for FastAPI applications."""

    def __init__(self, app: FastAPI):
        self.app = app
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")
        self.template_static_dir = os.path.join(self.template_dir, "static")

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )

        # Mount static files - prioritize template static files
        static_mounted = False

        # First try to mount template static files (higher priority)
        if os.path.exists(self.template_static_dir):
            try:
                app.mount("/static", StaticFiles(directory=self.template_static_dir), name="template_static")
                static_mounted = True
                logging.info(f"Mounted template static files from: {self.template_static_dir}")
            except Exception as e:
                logging.warning(f"Failed to mount template static files: {e}")

        # Fallback to regular static directory if template static not available
        if not static_mounted and os.path.exists(self.static_dir):
            try:
                app.mount("/static", StaticFiles(directory=self.static_dir), name="static")
                logging.info(f"Mounted static files from: {self.static_dir}")
            except Exception as e:
                logging.warning(f"Failed to mount static files: {e}")

        # Log static file status
        if not static_mounted:
            logging.warning("No static files mounted - CSS and JS may not load correctly")

        self._setup_routes()
    
    def _setup_routes(self):
        """Setup custom Swagger UI routes."""
        
        @self.app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
        async def custom_swagger_ui(request: Request):
            """Serve the custom Swagger UI."""
            return await self._render_swagger_ui(request)
        
        @self.app.get("/swagger-ui", response_class=HTMLResponse, include_in_schema=False)
        async def custom_swagger_ui_alt(request: Request):
            """Alternative endpoint for custom Swagger UI."""
            return await self._render_swagger_ui(request)
    
    async def _render_swagger_ui(self, request: Request) -> HTMLResponse:
        """Render the custom Swagger UI template."""
        try:
            # Get app metadata
            title = getattr(self.app, 'title', 'Webscout OpenAI API')
            description = getattr(self.app, 'description', 'OpenAI API compatible interface')
            version = getattr(self.app, 'version', '0.2.0')

            # Get base URL
            base_url = str(request.base_url).rstrip('/')

            # Load and count models
            model_count = await self._get_model_count()
            provider_count = await self._get_provider_count()

            # Load the main template using Jinja2 environment
            template = self.jinja_env.get_template("swagger_ui.html")

            # Render with context
            rendered_html = template.render(
                title=title,
                description=description,
                version=version,
                base_url=base_url,
                model_count=model_count,
                provider_count=provider_count
            )

            return HTMLResponse(content=rendered_html, status_code=200)

        except TemplateNotFound:
            # Template file doesn't exist, use fallback
            logging.warning("Template file 'swagger_ui.html' not found, using fallback HTML")
            return HTMLResponse(content=self._get_fallback_html(), status_code=200)

        except Exception as e:
            # Other errors, log and use fallback
            logging.error(f"Error rendering Swagger UI template: {e}")
            return HTMLResponse(content=self._get_fallback_html(), status_code=200)
    
    async def _get_model_count(self) -> int:
        """Get the number of available models."""
        try:
            # Try to get from auth config
            from .config import AppConfig
            if hasattr(AppConfig, 'provider_map') and AppConfig.provider_map:
                # Count models (keys with "/" are model names)
                model_count = len([model for model in AppConfig.provider_map.keys() if "/" in model])
                return model_count if model_count > 0 else 589
            return 589  # Default fallback
        except Exception as e:
            logging.debug(f"Could not get model count: {e}")
            return 589

    async def _get_provider_count(self) -> int:
        """Get the number of available providers."""
        try:
            # Try to get from auth config
            from .config import AppConfig
            if hasattr(AppConfig, 'provider_map') and AppConfig.provider_map:
                # Count unique providers
                providers = set()
                for model_key in AppConfig.provider_map.keys():
                    if "/" in model_key:
                        provider_name = model_key.split("/")[0]
                        providers.add(provider_name)
                return len(providers) if len(providers) > 0 else 42
            return 42  # Default fallback
        except Exception as e:
            logging.debug(f"Could not get provider count: {e}")
            return 42
    
    def _get_fallback_html(self) -> str:
        """Fallback HTML if template loading fails."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Webscout API Documentation</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    margin: 0;
                    padding: 2rem;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                }
                .container {
                    text-align: center;
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    padding: 3rem;
                    border-radius: 1rem;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                h1 { font-size: 3rem; margin-bottom: 1rem; }
                p { font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.9; }
                .btn {
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    padding: 1rem 2rem;
                    border-radius: 0.5rem;
                    text-decoration: none;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    display: inline-block;
                    margin: 0.5rem;
                }
                .btn:hover {
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-2px);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Webscout API</h1>
                <p>OpenAI-Compatible API Documentation</p>
                <a href="/redoc" class="btn">📖 ReDoc Documentation</a>
                <a href="/openapi.json" class="btn">📋 OpenAPI Schema</a>
            </div>
        </body>
        </html>
        """


def setup_custom_swagger_ui(app: FastAPI) -> CustomSwaggerUI:
    """Setup custom Swagger UI for the FastAPI app."""
    return CustomSwaggerUI(app)
