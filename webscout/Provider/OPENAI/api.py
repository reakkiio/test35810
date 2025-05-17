from __future__ import annotations

import logging
import json
import uvicorn
import secrets
import os
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Generator
from fastapi import FastAPI, Response, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException
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
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None
    
    class Config:
        extra = "ignore"  # Ignore extra fields that aren't in the model

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

    @classmethod
    def set_config(cls, **data):
        for key, value in data.items():
            setattr(cls, key, value)

def create_app():
    app = FastAPI(
        title="Webscout OpenAI API",
        description="OpenAI API compatible interface for various LLM providers",
        version="0.1.0",
        docs_url=None,
    )
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    api = Api(app)
    api.register_authorization()
    api.register_json_middleware()  # Add custom JSON middleware
    api.register_validation_exception_handler()
    api.register_routes()
    
    # Initialize provider map
    initialize_provider_map()
    
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
            AppConfig.provider_map[obj.__name__] = obj
            logger.info(f"Registered provider: {obj.__name__}")
            # Also add additional mappings for convenience
            if hasattr(obj, "AVAILABLE_MODELS"):
                for model in obj.AVAILABLE_MODELS:
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
    
    logger.info(f"Available providers: {list(set(v.__name__ for v in AppConfig.provider_map.values()))}")
    logger.info(f"Available models: {list(AppConfig.provider_map.keys())}")
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

    def register_json_middleware(self):
        @self.app.middleware("http")
        async def parse_json_middleware(request: Request, call_next):
            if request.method == "POST" and "/v1/chat/completions" in request.url.path:
                try:
                    # Try parsing the JSON body manually first to catch JSON errors early
                    body = await request.body()
                    if body:
                        body_str = body.decode('utf-8', errors='ignore')
                        original_body = body_str
                        logger.debug(f"Original request body: {body_str}")
                        
                        # PowerShell with curl often has formatting issues with JSON
                        try:
                            # First try normal JSON parsing
                            json.loads(body_str)
                            logger.debug("JSON parsed successfully")
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON parse error, attempting fixes: {str(e)}")
                            
                            # Series of fixes to try for common PowerShell JSON issues
                            try:
                                # Fix 1: Try to clean up the JSON string
                                # Replace literal backslash+quote with just quote
                                body_str = body_str.replace("\\\"", "\"")
                                # Add double quotes to unquoted property names and string values
                                # This is a common issue with PowerShell's curl
                                import re
                                
                                # Try a full JSON correction - replace single quotes with double quotes
                                # This is a more aggressive fix that might work in simple cases
                                fixed_body = body_str.replace("'", "\"")
                                try:
                                    json.loads(fixed_body)
                                    body_str = fixed_body
                                    logger.info("Fixed JSON by replacing single quotes with double quotes")
                                except json.JSONDecodeError:
                                    # If that didn't work, try more sophisticated fixes
                                    pass
                                
                                # Check for missing quotes around property names
                                # Look for patterns like {model: instead of {"model":
                                body_str = re.sub(r'\{([^"\s][^:\s]*)(\s*:)', r'{"\1"\2', body_str)
                                body_str = re.sub(r',\s*([^"\s][^:\s]*)(\s*:)', r', "\1"\2', body_str)
                                
                                # Try to parse with the fixed body
                                json.loads(body_str)
                                # If successful, modify the request._body for downstream processing
                                logger.info(f"Successfully fixed JSON format\nOriginal: {original_body}\nFixed: {body_str}")
                                request._body = body_str.encode('utf-8')
                            except Exception as fix_error:
                                logger.error(f"Failed to fix JSON: {str(fix_error)}")
                                
                                # Let's return a helpful error message with the proper format example
                                example = json.dumps({
                                    "model": "gpt-4",
                                    "messages": [{"role": "user", "content": "Hello"}]
                                })
                                return JSONResponse(
                                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                                    content=jsonable_encoder({
                                        "detail": [
                                            {
                                                "loc": ["body", 0],
                                                "message": f"Invalid JSON format: {str(e)}. Make sure to use double quotes for both keys and values. Example: {example}",
                                                "type": "json_invalid"
                                            }
                                        ]
                                    }),
                                )
                except Exception as e:
                    error_detail = str(e)
                    logger.error(f"Request processing error: {error_detail}")
                    return JSONResponse(
                        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                        content=jsonable_encoder({
                            "detail": [
                                {
                                    "loc": ["body", 0],
                                    "message": f"Request processing error: {error_detail}",
                                    "type": "request_invalid"
                                }
                            ]
                        }),
                    )
            return await call_next(request)

    def register_validation_exception_handler(self):
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            details = exc.errors()
            modified_details = []
            for error in details:
                modified_details.append({
                    "loc": error["loc"],
                    "message": error["msg"],
                    "type": error["type"],
                })
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder({"detail": modified_details}),
            )
            
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content=jsonable_encoder({"detail": exc.detail}),
            )
            
        @self.app.exception_handler(json.JSONDecodeError)
        async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder({
                    "detail": [
                        {
                            "loc": ["body", 0],
                            "message": f"Invalid JSON format: {str(exc)}",
                            "type": "json_invalid"
                        }
                    ]
                }),
            )

    def register_routes(self):
        @self.app.get("/")
        async def read_root(request: Request):
            return RedirectResponse(url="/docs")

        @self.app.get("/v1")
        async def read_root_v1(request: Request):
            return RedirectResponse(url="/docs")

        @self.app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui(request: Request):
            from fastapi.openapi.docs import get_swagger_ui_html
            return get_swagger_ui_html(
                openapi_url=self.app.openapi_url,
                title=f"{self.app.title} - Swagger UI"
            )

        @self.app.get("/v1//models", include_in_schema=False)  # Handle double slash case
        async def list_models_double_slash():
            """Redirect double slash models endpoint to the correct one"""
            return RedirectResponse(url="/v1/models")
            
        @self.app.get("/v1/models")
        async def list_models():
            """List available models"""
            from webscout.Provider.OPENAI.utils import ModelData, ModelList
            models_data = []

            # Get current timestamp
            created_time = int(time.time())

            for model_name, provider_class in AppConfig.provider_map.items():
                if not hasattr(provider_class, "AVAILABLE_MODELS") or model_name in provider_class.AVAILABLE_MODELS:
                    # Create a more detailed model data object with proper fields
                    model = ModelData(
                        id=model_name,
                        created=created_time,
                        owned_by=getattr(provider_class, "__name__", "webscout"),
                        permission=[{
                            "id": f"modelperm-{model_name}",
                            "object": "model_permission",
                            "created": created_time,
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": True,
                            "allow_search_indices": hasattr(provider_class, "supports_embeddings") and provider_class.supports_embeddings,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False
                        }]
                    )
                    models_data.append(model)
            
            # Return as ModelList for proper formatting
            response = ModelList(data=models_data)
            return response.to_dict()

        @self.app.get("/v1/models/{model_name}")
        async def get_model(model_name: str):
            """Get information about a specific model"""
            from webscout.Provider.OPENAI.utils import ModelData
            created_time = int(time.time())
            
            # Check if the model exists in our provider map
            if model_name in AppConfig.provider_map:
                provider_class = AppConfig.provider_map[model_name]
                
                # Create a proper OpenAI-compatible model response
                model = ModelData(
                    id=model_name,
                    created=created_time,
                    owned_by=getattr(provider_class, "__name__", "webscout"),
                    permission=[{
                        "id": f"modelperm-{model_name}",
                        "object": "model_permission",
                        "created": created_time,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": hasattr(provider_class, "supports_embeddings") and provider_class.supports_embeddings,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }]
                )
                return model.to_dict()
            
            # If we reached here, the model was not found
            return ErrorResponse.from_message(f"Model '{model_name}' not found", HTTP_404_NOT_FOUND)

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """Create a chat completion"""
            # First manually extract the request body to better handle parsing errors
            try:
                # Note: We don't need to parse JSON here as our middleware already handles that
                # and fixes PowerShell JSON issues
                body = await request.json()
                logger.debug(f"Request body parsed successfully: {body}")
                
                # Check for required fields
                if "model" not in body:
                    return JSONResponse(
                        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                        content=jsonable_encoder({
                            "detail": [
                                {
                                    "loc": ["body", "model"],
                                    "message": "Field 'model' is required",
                                    "type": "missing"
                                }
                            ]
                        }),
                    )
                    
                if "messages" not in body or not isinstance(body["messages"], list) or len(body["messages"]) == 0:
                    return JSONResponse(
                        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                        content=jsonable_encoder({
                            "detail": [
                                {
                                    "loc": ["body", "messages"],
                                    "message": "Field 'messages' must be a non-empty array",
                                    "type": "missing"
                                }
                            ]
                        }),
                    )
                
                # Now parse it through Pydantic model
                try:
                    chat_request = ChatCompletionRequest(**body)
                except Exception as validation_error:
                    logger.warning(f"Validation error: {validation_error}")
                    # Try to provide helpful error messages for common validation issues
                    error_msg = str(validation_error)
                    if "role" in error_msg:
                        return JSONResponse(
                            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                            content=jsonable_encoder({
                                "detail": [
                                    {
                                        "loc": ["body", "messages", 0, "role"],
                                        "message": "Each message must have a 'role' field with one of these values: 'system', 'user', 'assistant'",
                                        "type": "value_error"
                                    }
                                ]
                            }),
                        )
                    elif "content" in error_msg:
                        return JSONResponse(
                            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                            content=jsonable_encoder({
                                "detail": [
                                    {
                                        "loc": ["body", "messages", 0, "content"],
                                        "message": "Each message must have a 'content' field with string value",
                                        "type": "value_error"
                                    }
                                ]
                            }),
                        )
                    else:
                        return JSONResponse(
                            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                            content=jsonable_encoder({
                                "detail": [
                                    {
                                        "loc": ["body"],
                                        "message": f"Validation error: {error_msg}",
                                        "type": "value_error"
                                    }
                                ]
                            }),
                        )
                        
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in chat_completions: {e}")
                example = json.dumps({
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}]
                })
                return JSONResponse(
                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                    content=jsonable_encoder({
                        "detail": [
                            {
                                "loc": ["body", 0],
                                "message": f"Invalid JSON format: {str(e)}. Example of correct format: {example}",
                                "type": "json_invalid"
                            }
                        ]
                    }),
                )
            except Exception as e:
                logger.exception(f"Unexpected error in chat_completions: {e}")
                return ErrorResponse.from_message(
                    f"Invalid request parameters: {str(e)}",
                    HTTP_422_UNPROCESSABLE_ENTITY
                )
            """Create a chat completion"""
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

def run_api(
    host: str = '0.0.0.0',
    port: int = None,
    api_key: str = None,
    default_provider: str = None,
    debug: bool = False,
) -> None:
    """Run the API server
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        api_key: API key for authentication
        default_provider: Default provider to use if no provider is specified
        debug: Whether to run in debug mode
    """
    print(f"Starting Webscout OpenAI API server...")
    
    if port is None:
        port = DEFAULT_PORT
    
    # Set configuration
    AppConfig.set_config(
        api_key=api_key,
        default_provider=default_provider or AppConfig.default_provider
    )
    
    # Run the server
    uvicorn.run(
        "webscout.Provider.OPENAI.api:create_app_debug" if debug else "webscout.Provider.OPENAI.api:create_app",
        host=host,
        port=int(port),
        factory=True,
    )

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Webscout OpenAI-compatible API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind the server to")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--default-provider", help="Default provider to use if no provider is specified")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    run_api(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        default_provider=args.default_provider,
        debug=args.debug,
    )
