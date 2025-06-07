"""
Custom Swagger UI implementation for Webscout FastAPI server.
Provides a modern, beautiful, and fully functional API documentation interface.
"""

import os
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

class CustomSwaggerUI:
    """Custom Swagger UI handler for FastAPI applications."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")
        
        # Mount static files
        if os.path.exists(self.static_dir):
            app.mount("/static", StaticFiles(directory=self.static_dir), name="static")
        
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
            
            # Load and count models (mock data for now)
            model_count = await self._get_model_count()
            provider_count = await self._get_provider_count()
            
            # Load the HTML template
            template_path = os.path.join(self.template_dir, "swagger_ui.html")
            
            if not os.path.exists(template_path):
                # Fallback to basic HTML if template doesn't exist
                return HTMLResponse(self._get_fallback_html())
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Create Jinja2 template
            template = Template(template_content)
            
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
            
        except Exception as e:
            # Fallback to basic HTML in case of errors
            return HTMLResponse(content=self._get_fallback_html(), status_code=200)
    
    async def _get_model_count(self) -> int:
        """Get the number of available models."""
        try:
            # Try to get from config if available
            from .config import AppConfig
            if hasattr(AppConfig, 'provider_map') and AppConfig.provider_map:
                return len([model for model in AppConfig.provider_map.keys() if "/" in model])
            return 25  # Default fallback
        except:
            return 25
    
    async def _get_provider_count(self) -> int:
        """Get the number of available providers."""
        try:
            # Try to get from config if available
            from .config import AppConfig
            if hasattr(AppConfig, 'provider_map') and AppConfig.provider_map:
                providers = set(v.__name__ for v in AppConfig.provider_map.values())
                return len(providers)
            return 8  # Default fallback
        except:
            return 8
    
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
