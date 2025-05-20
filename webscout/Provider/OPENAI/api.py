"""
OpenAI-Compatible API Server for Webscout

This module provides an OpenAI-compatible API server that allows using
various AI providers through a standardized interface compatible with
OpenAI's API. This enables using Webscout providers with any tool or
application designed to work with OpenAI's API.

Usage:
    # From command line:
    python -m webscout.Provider.OPENAI.api --port 8080 --api-key "your-key"
    
    # From Python code:
    from webscout.Provider.OPENAI.api import start_server
    start_server(port=8080, api_key="your-key")
"""

from __future__ import annotations
import logging
import json
import uvicorn
import secrets
import os
import uuid
import time
import sys
import inspect
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Generator, Callable
from fastapi import FastAPI, Response, Request, Depends, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_ENTITY, 
    HTTP_404_NOT_FOUND,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Literal

# Import provider classes from the OPENAI directory
from webscout.Provider.OPENAI import *
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8000

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use. See the model endpoint for the available models.")
    messages: List[Message] = Field(..., description="A list of messages comprising the conversation so far.")
    temperature: Optional[float] = Field(None, description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.")
    top_p: Optional[float] = Field(None, description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.")
    n: Optional[int] = Field(1, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(False, description="If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message.")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate in the chat completion.")
    presence_penalty: Optional[float] = Field(None, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")
    frequency_penalty: Optional[float] = Field(None, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Modify the likelihood of specified tokens appearing in the completion.")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user, which can help the API to monitor and detect abuse.")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating further tokens.")
    
    class Config:
        extra = "ignore"  # Ignore extra fields that aren't in the model
        schema_extra = {
            "example": {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "temperature": 0.7,
                "max_tokens": 150,
                "stream": False
            }
        }

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]

class ErrorResponse(Response):
    media_type = "application/json"

    @classmethod
    def from_exception(cls, exception: Exception, status_code: int = HTTP_500_INTERNAL_SERVER_ERROR):
        return cls(format_exception(exception), status_code)

    @classmethod
    def from_message(cls, message: str, status_code: int = HTTP_500_INTERNAL_SERVER_ERROR, headers: dict = None):
        return cls(format_exception(message), status_code, headers=headers)

    def render(self, content) -> bytes:
        return str(content).encode(errors="ignore")

class AppConfig:
    api_key: Optional[str] = None
    provider_map = {}
    default_provider = "ChatGPT"
    base_url: Optional[str] = None

    @classmethod
    def set_config(cls, **data):
        for key, value in data.items():
            setattr(cls, key, value)

# Custom route class to handle dynamic base URLs
class DynamicBaseRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            if AppConfig.base_url:
                if not request.url.path.startswith(AppConfig.base_url):
                    pass
            return await original_route_handler(request)
        return custom_route_handler

def create_app():
    app = FastAPI(
        title="Webscout OpenAI API",
        description="OpenAI API compatible interface for various LLM providers",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.router.route_class = DynamicBaseRoute
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api = Api(app)
    api.register_authorization()
    api.register_validation_exception_handler()
    api.register_routes()
    initialize_provider_map()
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
        if "schemas" not in openapi_schema["components"]:
            openapi_schema["components"]["schemas"] = {}
        openapi_schema["components"]["schemas"]["ChatCompletionRequest"] = {
            "type": "object",
            "required": ["model", "messages"],
            "properties": {
                "model": {
                    "type": "string",
                    "description": "ID of the model to use"
                },
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["system", "user", "assistant", "function", "tool"]
                            },
                            "content": {
                                "type": "string"
                            },
                            "name": {
                                "type": "string"
                            }
                        }
                    }
                },
                "temperature": {
                    "type": "number",
                    "format": "float",
                    "nullable": True
                },
                "stream": {
                    "type": "boolean",
                    "default": False
                }
            }
        }
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    app.openapi = custom_openapi
    return app

def create_app_debug():
    logging.basicConfig(level=logging.DEBUG)
    return create_app()

def initialize_provider_map():
    from webscout.Provider.OPENAI.base import OpenAICompatibleProvider
    module = sys.modules["webscout.Provider.OPENAI"]
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, OpenAICompatibleProvider) and obj.__name__ != "OpenAICompatibleProvider":
            AppConfig.provider_map[obj.__name__] = obj
            logger.info(f"Registered provider: {obj.__name__}")
            if hasattr(obj, "AVAILABLE_MODELS") and isinstance(obj.AVAILABLE_MODELS, (list, tuple, set)):
                for model in obj.AVAILABLE_MODELS:
                    if model and isinstance(model, str) and model != obj.__name__:
                        AppConfig.provider_map[model] = obj
                        logger.info(f"Mapped model {model} to provider {obj.__name__}")
    if not AppConfig.provider_map:
        logger.warning("No providers found, using ChatGPT as fallback")
        from webscout.Provider.OPENAI.chatgpt import ChatGPT
        AppConfig.provider_map["ChatGPT"] = ChatGPT
        AppConfig.provider_map["gpt-4"] = ChatGPT
        AppConfig.provider_map["gpt-4o"] = ChatGPT
        AppConfig.provider_map["gpt-4o-mini"] = ChatGPT
        AppConfig.default_provider = "ChatGPT"
    provider_names = list(set(v.__name__ for v in AppConfig.provider_map.values()))
    provider_class_names = set(v.__name__ for v in AppConfig.provider_map.values())
    model_names = [model for model in AppConfig.provider_map.keys() if model not in provider_class_names]
    logger.info(f"Available providers ({len(provider_names)}): {provider_names}")
    logger.info(f"Available models ({len(model_names)}): {sorted(model_names)}")
    logger.info(f"Default provider: {AppConfig.default_provider}")

class Api:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.get_api_key = APIKeyHeader(name="authorization", auto_error=False)

    def register_authorization(self):
        @self.app.middleware("http")
        async def authorization(request: Request, call_next):
            if AppConfig.api_key is not None:
                auth_header = await self.get_api_key(request)
                path = request.url.path
                if path.startswith("/v1"):
                    if auth_header is None:
                        return ErrorResponse.from_message("API key required", HTTP_401_UNAUTHORIZED)
                    if auth_header.startswith("Bearer "):
                        auth_header = auth_header[7:]
                    if AppConfig.api_key is None or not secrets.compare_digest(AppConfig.api_key, auth_header):
                        return ErrorResponse.from_message("Invalid API key", HTTP_403_FORBIDDEN)
            return await call_next(request)

    def register_validation_exception_handler(self):
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            errors = exc.errors()
            error_messages = []
            for error in errors:
                loc = error.get("loc", [])
                loc_str = " -> ".join(str(l) for l in loc if l != "body")
                msg = error.get("msg", "Validation error")
                if "body" in loc:
                    if len(loc) > 1 and loc[1] == "messages":
                        error_messages.append({
                            "loc": loc,
                            "message": "The 'messages' field is required and must be a non-empty array of message objects.",
                            "type": error.get("type", "validation_error")
                        })
                    elif len(loc) > 1 and loc[1] == "model":
                        error_messages.append({
                            "loc": loc,
                            "message": "The 'model' field is required and must be a string.",
                            "type": error.get("type", "validation_error")
                        })
                    else:
                        error_messages.append({
                            "loc": loc,
                            "message": f"{msg} at {loc_str}",
                            "type": error.get("type", "validation_error")
                        })
                else:
                    error_messages.append({
                        "loc": loc,
                        "message": f"{msg} at {loc_str}",
                        "type": error.get("type", "validation_error")
                    })
            if request.url.path == "/v1/chat/completions":
                example = {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"}
                    ]
                }
                return JSONResponse(
                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                    content={"detail": error_messages, "example": example}
                )
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": error_messages}
            )
        @self.app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(request: Request, exc: StarletteHTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger.exception(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Internal server error: {str(exc)}"}
            )

    def register_routes(self):
        @self.app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse(url="/docs")
        @self.app.get("/v1/models", response_model=ModelListResponse)
        async def list_models():
            models = []
            for model_name, provider_class in AppConfig.provider_map.items():
                if any(m["id"] == model_name for m in models):
                    continue
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider_class.__name__
                })
            return {
                "object": "list",
                "data": models
            }
        @self.app.post(
            "/v1/chat/completions", 
            response_model_exclude_none=True,
            response_model_exclude_unset=True,
            openapi_extra={
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/ChatCompletionRequest"
                            },
                            "example": {
                                "model": "gpt-4",
                                "messages": [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": "Hello"}
                                ],
                                "max_tokens": 20,
                                "temperature": 0.8
                            }
                        }
                    }
                }
            }
        )
        async def chat_completions(
            request: Request,
            chat_request: ChatCompletionRequest = Body(...)
        ):
            logger.debug(f"Chat completion request received: {chat_request}")
            try:
                provider_class = None
                model = chat_request.model
                logger.info(f"Chat completion request for model: {model}")
                if model in AppConfig.provider_map:
                    provider_class = AppConfig.provider_map[model]
                    logger.info(f"Found provider class for model {model}: {provider_class.__name__}")
                else:
                    provider_class = AppConfig.provider_map.get(AppConfig.default_provider)
                    logger.info(f"Using default provider {AppConfig.default_provider} for model {model}")
                if not provider_class:
                    logger.error(f"No provider available for model {model}. Available models: {list(AppConfig.provider_map.keys())}")
                    return ErrorResponse.from_message(
                        f"Model '{model}' not supported. Available models: {list(AppConfig.provider_map.keys())}",
                        HTTP_404_NOT_FOUND
                    )
                logger.info(f"Initializing provider {provider_class.__name__}")
                try:
                    provider = provider_class()
                except Exception as e:
                    logger.exception(f"Failed to initialize provider {provider_class.__name__}: {e}")
                    return ErrorResponse.from_message(
                        f"Failed to initialize provider {provider_class.__name__}: {e}",
                        HTTP_500_INTERNAL_SERVER_ERROR
                    )
                messages = []
                for msg in chat_request.messages:
                    message_dict = {
                        "role": msg.role,
                        "content": msg.content
                    }
                    if msg.name:  
                        message_dict["name"] = msg.name
                    messages.append(message_dict)
                params = {
                    "model": model,
                    "messages": messages,
                    "stream": chat_request.stream,
                }
                if chat_request.temperature is not None:
                    params["temperature"] = chat_request.temperature
                if chat_request.max_tokens is not None:
                    params["max_tokens"] = chat_request.max_tokens
                if chat_request.top_p is not None:
                    params["top_p"] = chat_request.top_p
                if chat_request.stream:
                    async def streaming():
                        try:
                            logger.info(f"Creating streaming completion with {provider_class.__name__}")
                            completion_stream = provider.chat.completions.create(**params)
                            logger.info(f"Got streaming response: {type(completion_stream)}")
                            if isinstance(completion_stream, Generator):
                                for chunk in completion_stream:
                                    logger.debug(f"Streaming chunk: {type(chunk)}")
                                    if hasattr(chunk, 'to_dict'):
                                        yield f"data: {json.dumps(chunk.to_dict())}\n\n"
                                    elif hasattr(chunk, 'model_dump'):
                                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                                    else:
                                        yield f"data: {json.dumps(chunk)}\n\n"
                            else:
                                logger.info(f"Provider returned non-streaming response, simulating stream")
                                yield f"data: {json.dumps(completion_stream)}\n\n"
                        except Exception as e:
                            logger.exception(f"Error in streaming: {e}")
                            yield f"data: {format_exception(e)}\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(streaming(), media_type="text/event-stream")
                else:
                    logger.info(f"Creating non-streaming completion with {provider_class.__name__}")
                    try:
                        completion = provider.chat.completions.create(**params)
                        logger.info(f"Got completion response: {type(completion)}")
                        if completion is None:
                            logger.warning(f"Provider {provider_class.__name__} returned None for completion")
                            return {
                                "id": f"chatcmpl-{uuid.uuid4()}",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": "I apologize, but I couldn't generate a response. Please try again or try a different model.",
                                        },
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                },
                            }
                        if isinstance(completion, dict):
                            return completion
                        elif hasattr(completion, "model_dump"):
                            return completion.model_dump()
                        else:
                            return completion
                    except Exception as e:
                        logger.exception(f"Error in completion: {e}")
                        return ErrorResponse.from_exception(e, HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, HTTP_500_INTERNAL_SERVER_ERROR)

def format_exception(e: Union[Exception, str]) -> str:
    if isinstance(e, str):
        message = e
    else:
        message = f"{e.__class__.__name__}: {e}"
    return json.dumps({
        "error": {
            "message": message,
            "type": "server_error",
            "param": None,
            "code": "internal_server_error"
        }
    })

def start_server(port: int = DEFAULT_PORT, api_key: str = None, default_provider: str = None, base_url: str = None):
    run_api(
        host="0.0.0.0",
        port=port,
        api_key=api_key,
        default_provider=default_provider,
        base_url=base_url,
        debug=False,
    )

def run_api(
    host: str = '0.0.0.0',
    port: int = None,
    api_key: str = None,
    default_provider: str = None,
    base_url: str = None,
    debug: bool = False,
    show_available_providers: bool = True,
) -> None:
    print(f"Starting Webscout OpenAI API server...")
    if port is None:
        port = DEFAULT_PORT
    AppConfig.set_config(
        api_key=api_key,
        default_provider=default_provider or AppConfig.default_provider,
        base_url=base_url
    )
    initialize_provider_map()
    if show_available_providers:
        print("\n=== Available Providers ===")
        providers = list(set(v.__name__ for v in AppConfig.provider_map.values()))
        for i, provider in enumerate(providers, 1):
            print(f"{i}. {provider}")
        print("\n=== Available Models ===")
        provider_class_names = set(v.__name__ for v in AppConfig.provider_map.values())
        models = [model for model in AppConfig.provider_map.keys() if model not in provider_class_names]
        if models:
            for i, model in enumerate(sorted(models), 1):
                print(f"{i}. {model}")
        else:
            print("No specific models registered. Use provider names as models.")
        print(f"\nDefault provider: {AppConfig.default_provider}")
        print(f"API Authentication: {'Enabled' if api_key else 'Disabled'}")
        print(f"Base URL: {base_url or '/'}")
        print(f"Server URL: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        print(f"API Endpoint: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/v1/chat/completions")
        print(f"Documentation: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
        print("\nUse Ctrl+C to stop the server")
        print("=" * 30 + "\n")
    uvicorn.run(
        "webscout.Provider.OPENAI.api:create_app_debug" if debug else "webscout.Provider.OPENAI.api:create_app",
        host=host,
        port=int(port),
        factory=True,
        reload=True
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Start Webscout OpenAI-compatible API server')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Port to run the server on (default: {DEFAULT_PORT})')
    parser.add_argument('--api-key', type=str, help='API key for authentication (optional)')
    parser.add_argument('--default-provider', type=str, help='Default provider to use (optional)')
    parser.add_argument('--base-url', type=str, help='Base URL for the API (optional, useful for reverse proxies)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    run_api(
        port=args.port,
        api_key=args.api_key,
        default_provider=args.default_provider,
        base_url=args.base_url,
        debug=args.debug
    )
