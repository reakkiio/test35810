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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_ENTITY, 
    HTTP_404_NOT_FOUND,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, create_model
from typing import List, Optional, Literal, Union

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

# Enhanced middleware for robust request body parsing
class RobustBodyParsingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only process POST requests to specific endpoints
        if request.method == "POST" and any(path in request.url.path for path in ["/v1/chat/completions", "/v1/completions"]):
            try:
                # Try parsing the JSON body manually first to catch JSON errors early
                body = await request.body()
                
                # Handle empty body case explicitly
                if not body or len(body.strip() if isinstance(body, str) else body) == 0:
                    logger.warning("Empty request body received")
                    return JSONResponse(
                        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                        content=jsonable_encoder({
                            "detail": [
                                {
                                    "loc": ["body"],
                                    "message": "Request body is empty. Please provide a valid JSON body with 'model' and 'messages' fields.",
                                    "type": "value_error.missing"
                                }
                            ]
                        }),
                    )
                    
                body_str = body.decode('utf-8', errors='ignore')
                original_body = body_str
                logger.debug(f"Original request body: {body_str}")
                
                # Try to parse the JSON body
                try:
                    # First try normal JSON parsing
                    json_body = json.loads(body_str)
                    logger.debug("JSON parsed successfully")
                    
                    # Validate required fields for chat completions
                    if "/v1/chat/completions" in request.url.path:
                        if "model" not in json_body:
                            json_body["model"] = AppConfig.default_provider
                            logger.info(f"Added default model: {AppConfig.default_provider}")
                        
                        if "messages" not in json_body or not json_body["messages"]:
                            return JSONResponse(
                                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                                content=jsonable_encoder({
                                    "detail": [
                                        {
                                            "loc": ["body", "messages"],
                                            "message": "Field 'messages' is required and cannot be empty.",
                                            "type": "value_error.missing"
                                        }
                                    ]
                                }),
                            )
                    
                    # Update the request body with the validated JSON
                    request._body = json.dumps(json_body).encode('utf-8')
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error, attempting fixes: {str(e)}")
                    
                    # Series of fixes to try for common JSON issues from different clients
                    import re
                    
                    # Fix 1: Remove trailing commas in objects and arrays
                    body_str = re.sub(r',\s*}', '}', body_str)
                    body_str = re.sub(r',\s*]', ']', body_str)
                    
                    # Fix 2: Replace literal backslash+quote with just quote
                    body_str = body_str.replace('\\"', '"')
                    
                    # Fix 3: Replace single quotes with double quotes
                    try:
                        fixed_body = body_str.replace("'", '"')
                        json.loads(fixed_body)
                        body_str = fixed_body
                        logger.info("Fixed JSON by replacing single quotes with double quotes")
                    except json.JSONDecodeError:
                        # If that didn't work, continue with other fixes
                        pass
                            
                    # Fix 4: Add quotes to unquoted property names
                    body_str = re.sub(r'\{([^"\s][^:\s]*)(\s*:)', r'{"\1"\2', body_str)
                    body_str = re.sub(r',\s*([^"\s][^:\s]*)(\s*:)', r', "\1"\2', body_str)
                            
                    # Fix 5: Fix newlines and other control characters
                    body_str = body_str.replace('\n', '\\n')
                    
                    # Fix 6: Handle escaped JSON strings (common in some clients)
                    if body_str.startswith('"') and body_str.endswith('"'):
                        try:
                            # This might be a JSON string that needs to be unescaped
                            import ast
                            unescaped = ast.literal_eval(body_str)
                            if isinstance(unescaped, str):
                                body_str = unescaped
                                logger.info("Unescaped JSON string")
                        except:
                            pass
                            
                    try:
                        # Try to parse with the fixed body
                        json_body = json.loads(body_str)
                        
                        # Add default model if missing
                        if "/v1/chat/completions" in request.url.path:
                            if "model" not in json_body:
                                json_body["model"] = AppConfig.default_provider
                                logger.info(f"Added default model: {AppConfig.default_provider}")
                            
                            # Check for required messages field
                            if "messages" not in json_body or not json_body["messages"]:
                                return JSONResponse(
                                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                                    content=jsonable_encoder({
                                        "detail": [
                                            {
                                                "loc": ["body", "messages"],
                                                "message": "Field 'messages' is required and cannot be empty.",
                                                "type": "value_error.missing"
                                            }
                                        ]
                                    }),
                                )
                        
                        # If successful, modify the request._body for downstream processing
                        logger.info(f"Successfully fixed JSON format\nOriginal: {original_body}\nFixed: {json.dumps(json_body)}")
                        request._body = json.dumps(json_body).encode('utf-8')
                    except json.JSONDecodeError as json_err:
                        # If we still can't parse it, return a helpful error
                        example = json.dumps({
                            "model": "gpt-4",
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "Hello"},
                            ]
                        }, indent=2)
                        return JSONResponse(
                            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                            content=jsonable_encoder({
                                "detail": [
                                    {
                                        "loc": ["body", 0],
                                        "message": f"Invalid JSON format: {str(e)}. Make sure to use double quotes for both keys and values and avoid trailing commas. Valid example: {example}",
                                        "type": "json_invalid"
                                    }
                                ]
                            }),
                        )
                    except Exception as fix_error:
                        logger.error(f"Failed to fix JSON: {str(fix_error)}")
                        return JSONResponse(
                            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                            content=jsonable_encoder({
                                "detail": [
                                    {
                                        "loc": ["body"],
                                        "message": f"Error processing request body: {str(fix_error)}",
                                        "type": "request_validation"
                                    }
                                ]
                            }),
                        )
            except Exception as e:
                logger.error(f"Error in body parsing middleware: {str(e)}")
                # Continue processing even if there's an error in the middleware
        
        # Process the request with the next middleware or route handler
        return await call_next(request)

# Custom route class to handle dynamic base URLs
class DynamicBaseRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        
        async def custom_route_handler(request: Request) -> Response:
            # Check if we need to adjust for base URL
            if AppConfig.base_url:
                # Adjust request URL if needed
                if not request.url.path.startswith(AppConfig.base_url):
                    # Handle base URL adjustments if needed
                    pass
            
            # Call the original route handler
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
    
    # Use custom route class for all routes
    app.router.route_class = DynamicBaseRoute
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add robust body parsing middleware
    app.add_middleware(RobustBodyParsingMiddleware)
    
    api = Api(app)
    api.register_authorization()
    api.register_validation_exception_handler()
    api.register_routes()
    
    # Initialize provider map
    initialize_provider_map()
    
    # Custom OpenAPI schema generator
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
            
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Enhance OpenAPI schema for better Swagger UI compatibility
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
            
        if "schemas" not in openapi_schema["components"]:
            openapi_schema["components"]["schemas"] = {}
        
        # Add explicit schema for chat completions request
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
    """Initialize the provider map with available provider classes"""
    import sys
    import inspect
    from webscout.Provider.OPENAI.base import OpenAICompatibleProvider
    
    # Get all imported modules from OPENAI package
    module = sys.modules["webscout.Provider.OPENAI"]
    
    # Find all provider classes (subclasses of OpenAICompatibleProvider)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, OpenAICompatibleProvider) and obj.__name__ != "OpenAICompatibleProvider":
            # Register the provider class by its name
            AppConfig.provider_map[obj.__name__] = obj
            logger.info(f"Registered provider: {obj.__name__}")
            
            # Also add additional mappings for model names
            if hasattr(obj, "AVAILABLE_MODELS") and isinstance(obj.AVAILABLE_MODELS, (list, tuple, set)):
                for model in obj.AVAILABLE_MODELS:
                    if model and isinstance(model, str) and model != obj.__name__:
                        AppConfig.provider_map[model] = obj
                        logger.info(f"Mapped model {model} to provider {obj.__name__}")
    
    # If no providers were found, add a fallback for testing
    if not AppConfig.provider_map:
        logger.warning("No providers found, using ChatGPT as fallback")
        from webscout.Provider.OPENAI.chatgpt import ChatGPT
        AppConfig.provider_map["ChatGPT"] = ChatGPT
        AppConfig.provider_map["gpt-4"] = ChatGPT
        AppConfig.provider_map["gpt-4o"] = ChatGPT
        AppConfig.provider_map["gpt-4o-mini"] = ChatGPT
        AppConfig.default_provider = "ChatGPT"
    
    # Get distinct provider names
    provider_names = list(set(v.__name__ for v in AppConfig.provider_map.values()))
    
    # Get model names (excluding provider class names)
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
                    # Strip "Bearer " prefix if present
                    if auth_header.startswith("Bearer "):
                        auth_header = auth_header[7:]
                    if AppConfig.api_key is None or not secrets.compare_digest(AppConfig.api_key, auth_header):
                        return ErrorResponse.from_message("Invalid API key", HTTP_403_FORBIDDEN)
            return await call_next(request)

    def register_validation_exception_handler(self):
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            # Enhanced validation error handling
            errors = exc.errors()
            error_messages = []
            
            for error in errors:
                # Create more user-friendly error messages
                loc = error.get("loc", [])
                loc_str = " -> ".join(str(l) for l in loc if l != "body")
                msg = error.get("msg", "Validation error")
                
                if "body" in loc:
                    # Special handling for body validation errors
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
            
            # Add example of correct format for common endpoints
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
            # Enhanced HTTP exception handling
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )
            
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            # General exception handler for unhandled exceptions
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
            """List available models"""
            models = []
            
            # Add all provider class names as models
            for model_name, provider_class in AppConfig.provider_map.items():
                # Skip duplicates
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

        @self.app.post("/v1/chat/completions", 
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
                      })
        async def chat_completions(
            request: Request,
            chat_request: ChatCompletionRequest = Body(...)
        ):
            """Create a chat completion"""
            logger.debug(f"Chat completion request received: {chat_request}")
            
            try:
                # Determine which provider to use based on the model
                provider_class = None
                model = chat_request.model
                logger.info(f"Chat completion request for model: {model}")
                
                if model in AppConfig.provider_map:
                    provider_class = AppConfig.provider_map[model]
                    logger.info(f"Found provider class for model {model}: {provider_class.__name__}")
                else:
                    # Use default provider if specific provider not found
                    provider_class = AppConfig.provider_map.get(AppConfig.default_provider)
                    logger.info(f"Using default provider {AppConfig.default_provider} for model {model}")

                if not provider_class:
                    logger.error(f"No provider available for model {model}. Available models: {list(AppConfig.provider_map.keys())}")
                    return ErrorResponse.from_message(
                        f"Model '{model}' not supported. Available models: {list(AppConfig.provider_map.keys())}",
                        HTTP_404_NOT_FOUND
                    )

                # Initialize provider
                logger.info(f"Initializing provider {provider_class.__name__}")
                try:
                    provider = provider_class()
                except Exception as e:
                    logger.exception(f"Failed to initialize provider {provider_class.__name__}: {e}")
                    return ErrorResponse.from_message(
                        f"Failed to initialize provider {provider_class.__name__}: {e}",
                        HTTP_500_INTERNAL_SERVER_ERROR
                    )

                # Prepare completion parameters
                # Convert Message objects to dictionaries for the provider
                messages = []
                for msg in chat_request.messages:
                    message_dict = {
                        "role": msg.role,
                        "content": msg.content
                    }
                    # Add name field if present
                    if msg.name:  
                        message_dict["name"] = msg.name
                    messages.append(message_dict)
                
                params = {
                    "model": model,
                    "messages": messages,
                    "stream": chat_request.stream,
                }

                # Add optional parameters if provided
                if chat_request.temperature is not None:
                    params["temperature"] = chat_request.temperature
                if chat_request.max_tokens is not None:
                    params["max_tokens"] = chat_request.max_tokens
                if chat_request.top_p is not None:
                    params["top_p"] = chat_request.top_p

                # Create completion
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
                                        # Use to_dict() for our custom dataclasses
                                        yield f"data: {json.dumps(chunk.to_dict())}\n\n"
                                    elif hasattr(chunk, 'model_dump'):
                                        # For Pydantic models
                                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                                    else:
                                        # For dictionaries or other JSON-serializable objects
                                        yield f"data: {json.dumps(chunk)}\n\n"
                            else:
                                # If the provider doesn't implement streaming but stream=True,
                                # simulate streaming with a single chunk
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
                        
                        # If the response is empty or None, create a default response
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
                            
                        # Return the response in the appropriate format
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
    """Format exception into a JSON string"""
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
    """
    Simple helper function to start the OpenAI-compatible API server.
    
    Args:
        port: Port to run the server on (default: 8000)
        api_key: Optional API key for authentication
        default_provider: Default provider to use (e.g., "ChatGPT", "Claude", etc.)
        base_url: Optional base URL for the API (useful for reverse proxies)
        
    Example:
        ```python
        from webscout.Provider.OPENAI.api import start_server
        
        # Start server with default settings
        start_server()
        
        # Start server with custom settings
        start_server(port=8080, api_key="your-api-key", default_provider="Claude", base_url="/api")
        ```
    """
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
    """Run the API server
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        api_key: API key for authentication (optional)
        default_provider: Default provider to use if no provider is specified
        base_url: Base URL for the API (optional, useful for reverse proxies)
        debug: Whether to run in debug mode
        show_available_providers: Whether to display available providers on startup
    """
    print(f"Starting Webscout OpenAI API server...")
    
    if port is None:
        port = DEFAULT_PORT
    
    # Set configuration
    AppConfig.set_config(
        api_key=api_key,
        default_provider=default_provider or AppConfig.default_provider,
        base_url=base_url
    )
    
    # Initialize provider map early to show available providers
    initialize_provider_map()
    
    if show_available_providers:
        print("\n=== Available Providers ===")
        providers = list(set(v.__name__ for v in AppConfig.provider_map.values()))
        for i, provider in enumerate(providers, 1):
            print(f"{i}. {provider}")
        
        print("\n=== Available Models ===")
        # Filter out provider class names from the model list
        provider_class_names = set(v.__name__ for v in AppConfig.provider_map.values())
        models = [model for model in AppConfig.provider_map.keys() if model not in provider_class_names]
        
        # Display models in a more organized way
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
    
    # Run the server
    uvicorn.run(
        "webscout.Provider.OPENAI.api:create_app_debug" if debug else "webscout.Provider.OPENAI.api:create_app",
        host=host,
        port=int(port),
        factory=True,
        reload=True
    )

# Command line interface
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
