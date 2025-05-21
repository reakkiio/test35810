from __future__ import annotations
import json
import uvicorn
import secrets
import os
import uuid
import time
import sys
import inspect
from typing import List, Dict, Optional, Union, Any, Generator, Callable
from fastapi import FastAPI, Response, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY, 
    HTTP_404_NOT_FOUND,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from webscout.Provider.OPENAI.pydantic_imports import * # Imports BaseModel, Field
from typing import Literal


# Import provider classes from the OPENAI directory
from webscout.Provider.OPENAI import *
from webscout.Provider.OPENAI.utils import (
    ChatCompletion, Choice, ChatCompletionMessage, CompletionUsage
)


DEFAULT_PORT = 8000

# Define Pydantic models for multimodal content parts, aligning with OpenAI's API
class TextPart(BaseModel):
    type: Literal["text"]
    text: str

class ImageURL(BaseModel):
    url: str  # Can be http(s) or data URI
    detail: Optional[Literal["auto", "low", "high"]] = Field("auto", description="Specifies the detail level of the image.")

class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL

MessageContentParts = Union[TextPart, ImagePart]

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[Union[str, List[MessageContentParts]]] = Field(None, description="The content of the message. Can be a string, a list of content parts (for multimodal), or null.")
    name: Optional[str] = None
    # To fully support OpenAI's spec, tool_calls and tool_call_id might be needed here
    # tool_calls: Optional[List[Any]] = None # Replace Any with a Pydantic model for ToolCall
    # tool_call_id: Optional[str] = None # For role="tool" messages responding to a tool_call

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
                "model": "ChatGPT/gpt-4",
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
# Note: The /docs 404 issue is likely related to server execution (Werkzeug logs vs. Uvicorn script).
# This DynamicBaseRoute, when AppConfig.base_url is None, should act as a passthrough and not break /docs.
# If AppConfig.base_url is set, this route class has limitations in correctly handling prefixed routes
# without more complex path manipulation or using FastAPI's APIRouter prefixing/mounting features.
class DynamicBaseRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            if AppConfig.base_url:
                if not request.url.path.startswith(AppConfig.base_url):
                    # This logic might need refinement if base_url is used.
                    # For API routes not matching the prefix, a 404 might be appropriate.
                    # Docs routes (/docs, /openapi.json) are usually at the root.
                    # The current 'pass' allows root docs even if base_url is set for APIs.
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
        
        if "components" not in openapi_schema: openapi_schema["components"] = {}
        if "schemas" not in openapi_schema["components"]: openapi_schema["components"]["schemas"] = {}

        # Use Pydantic's schema generation for accuracy
        # Assuming Pydantic v1 .schema() or v2 .model_json_schema() based on pydantic_imports
        # For broader compatibility, trying .schema() first.
        # If using Pydantic v2 primarily, .model_json_schema() is preferred.
        schema_method_name = "model_json_schema" if hasattr(BaseModel, "model_json_schema") else "schema"

        # Add/update schemas derived from Pydantic models to ensure they are correctly defined
        pydantic_models_to_register = {
            "TextPart": TextPart,
            "ImageURL": ImageURL,
            "ImagePart": ImagePart,
            "Message": Message,
            "ChatCompletionRequest": ChatCompletionRequest,
        }
        
        for name, model_cls in pydantic_models_to_register.items():
            schema_data = getattr(model_cls, schema_method_name)()
            # Pydantic might add a "title" to the schema, which is often not desired for component schemas
            if "title" in schema_data:
                del schema_data["title"]
            openapi_schema["components"]["schemas"][name] = schema_data
            
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
    return app

def create_app_debug():
    return create_app()

def initialize_provider_map():
    from webscout.Provider.OPENAI.base import OpenAICompatibleProvider
    module = sys.modules["webscout.Provider.OPENAI"]
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, OpenAICompatibleProvider)
            and obj.__name__ != "OpenAICompatibleProvider"
        ):
            provider_name = obj.__name__
            AppConfig.provider_map[provider_name] = obj
            if hasattr(obj, "AVAILABLE_MODELS") and isinstance(
                obj.AVAILABLE_MODELS, (list, tuple, set)
            ):
                for model in obj.AVAILABLE_MODELS:
                    if model and isinstance(model, str):
                        AppConfig.provider_map[f"{provider_name}/{model}"] = obj
    if not AppConfig.provider_map:
        from webscout.Provider.OPENAI.chatgpt import ChatGPT
        AppConfig.provider_map["ChatGPT"] = ChatGPT
        AppConfig.provider_map["ChatGPT/gpt-4"] = ChatGPT
        AppConfig.provider_map["ChatGPT/gpt-4o"] = ChatGPT
        AppConfig.provider_map["ChatGPT/gpt-4o-mini"] = ChatGPT
        AppConfig.default_provider = "ChatGPT"
    provider_names = list(set(v.__name__ for v in AppConfig.provider_map.values()))
    provider_class_names = set(v.__name__ for v in AppConfig.provider_map.values())
    model_names = [model for model in AppConfig.provider_map.keys() if model not in provider_class_names]

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
                if path.startswith("/v1"): # Only protect /v1 routes
                    # Also allow access to /docs, /openapi.json etc. if AppConfig.base_url is not set or path is not under it
                    # This logic should be fine as it only protects /v1 paths
                    if auth_header is None:
                        return ErrorResponse.from_message("API key required", HTTP_401_UNAUTHORIZED)
                    if auth_header.startswith("Bearer "):
                        auth_header = auth_header[7:]
                    if AppConfig.api_key is None or not secrets.compare_digest(AppConfig.api_key, auth_header): # AppConfig.api_key check is redundant after outer if
                        return ErrorResponse.from_message("Invalid API key", HTTP_403_FORBIDDEN)
            return await call_next(request)

    def register_validation_exception_handler(self):
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            errors = exc.errors()
            error_messages = []
            body = await request.body()
            is_empty_body = not body or body.strip() in (b"", b"null", b"{}")
            for error in errors:
                loc = error.get("loc", [])
                # Ensure loc_str is user-friendly
                loc_str_parts = []
                for item in loc:
                    if item == "body": # Skip "body" part if it's the first element of a longer path
                        if len(loc) > 1: continue 
                    loc_str_parts.append(str(item))
                loc_str = " -> ".join(loc_str_parts)
                
                msg = error.get("msg", "Validation error")

                # Check if this error is for the 'content' field specifically due to multimodal input
                if len(loc) >=3 and loc[0] == 'body' and loc[1] == 'messages' and loc[-1] == 'content':
                     # Check if the error type suggests a string was expected but a list (or vice-versa) was given for content
                    if "Input should be a valid string" in msg and error.get("input_type") == "list":
                         error_messages.append({
                            "loc": loc,
                            "message": f"Invalid message content: {msg}. Ensure content matches the expected format (string or list of content parts). Path: {loc_str}",
                            "type": error.get("type", "validation_error")
                        })
                         continue # Skip default message formatting for this specific case
                    elif "Input should be a valid list" in msg and error.get("input_type") == "string":
                         error_messages.append({
                            "loc": loc,
                            "message": f"Invalid message content: {msg}. Ensure content matches the expected format (string or list of content parts). Path: {loc_str}",
                            "type": error.get("type", "validation_error")
                        })
                         continue

                if "body" in loc:
                    if len(loc) > 1 and loc[1] == "messages":
                        error_messages.append({
                            "loc": loc,
                            "message": "The 'messages' field is required and must be a non-empty array of message objects. " + f"Error: {msg} at {loc_str}",
                            "type": error.get("type", "validation_error")
                        })
                    elif len(loc) > 1 and loc[1] == "model":
                        error_messages.append({
                            "loc": loc,
                            "message": "The 'model' field is required and must be a string. " + f"Error: {msg} at {loc_str}",
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
                example = ChatCompletionRequest.Config.schema_extra["example"]
                if is_empty_body:
                    return JSONResponse(
                        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                        content={
                            "error": {
                                "message": "Request body is required and must include 'model' and 'messages'.",
                                "type": "invalid_request_error",
                                "param": None,
                                "code": "body_missing"
                            },
                            "example": example
                        }
                    )
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
            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Internal server error: {str(exc)}"}
            )

    def register_routes(self):
        @self.app.get("/", include_in_schema=False)
        async def root():
            # Note: If /docs is 404ing, check if server is Uvicorn (expected) or Werkzeug (from logs).
            # Werkzeug logs suggest possible execution of a Flask app or WSGI misconfiguration.
            # This api.py file is intended for Uvicorn.
            return RedirectResponse(url="/docs")
        
        @self.app.get("/v1/models", response_model=ModelListResponse)
        async def list_models():
            models = []
            for model_name, provider_class in AppConfig.provider_map.items():
                if "/" not in model_name:
                    continue  # Skip provider names
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
            openapi_extra={ # This ensures the example is shown in docs
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/ChatCompletionRequest" # Relies on custom_openapi
                            },
                            "example": ChatCompletionRequest.Config.schema_extra["example"]
                        }
                    }
                }
            }
        )
        async def chat_completions(
            request: Request, # Keep request for raw body or other request properties if needed
            chat_request: ChatCompletionRequest = Body(...)
        ):
            # raw_body = await request.body() # Already read by validation_exception_handler if error
            try:
                start_time = time.time()
                provider_class = None
                model_identifier = chat_request.model
                model_name = None

                # Check for explicit provider/model syntax
                if model_identifier in AppConfig.provider_map and "/" in model_identifier:
                    provider_class = AppConfig.provider_map[model_identifier]
                    _, model_name = model_identifier.split("/", 1)
                elif "/" in model_identifier:
                    provider_name, model_name = model_identifier.split("/", 1)
                    provider_class = AppConfig.provider_map.get(provider_name)
                else:
                    provider_class = AppConfig.provider_map.get(AppConfig.default_provider)
                    model_name = model_identifier

                if not provider_class:
                    return ErrorResponse.from_message(
                        f"Provider for model '{model_identifier}' not found.",
                        HTTP_404_NOT_FOUND
                    )

                if hasattr(provider_class, "AVAILABLE_MODELS") and model_name is not None:
                    available = getattr(provider_class, "AVAILABLE_MODELS", [])
                    if available and model_name not in available:
                        return ErrorResponse.from_message(
                            f"Model '{model_name}' not supported by provider '{provider_class.__name__}'.",
                            HTTP_404_NOT_FOUND,
                        )
                try:
                    provider = provider_class()
                except Exception as e:
                    return ErrorResponse.from_message(
                        f"Failed to initialize provider {provider_class.__name__}: {e}",
                        HTTP_500_INTERNAL_SERVER_ERROR
                    )

                processed_messages = []
                for msg_in in chat_request.messages:
                    message_dict_out = {"role": msg_in.role}
                    
                    if msg_in.content is None:
                        message_dict_out["content"] = None
                    elif isinstance(msg_in.content, str):
                        message_dict_out["content"] = msg_in.content
                    else:  # It's List[MessageContentParts]
                        message_dict_out["content"] = [part.model_dump(exclude_none=True) for part in msg_in.content]
                    
                    if msg_in.name:
                        message_dict_out["name"] = msg_in.name
                    
                    # Add tool_calls processing if/when Message model supports it
                    # if hasattr(msg_in, 'tool_calls') and msg_in.tool_calls:
                    #    message_dict_out["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in msg_in.tool_calls]
                    # if hasattr(msg_in, 'tool_call_id') and msg_in.tool_call_id:
                    #    message_dict_out["tool_call_id"] = msg_in.tool_call_id
                        
                    processed_messages.append(message_dict_out)

                params = {
                    "model": model_name,
                    "messages": processed_messages, # Use processed messages
                    "stream": chat_request.stream,
                }
                # Add other optional parameters if present
                if chat_request.temperature is not None: params["temperature"] = chat_request.temperature
                if chat_request.max_tokens is not None: params["max_tokens"] = chat_request.max_tokens
                if chat_request.top_p is not None: params["top_p"] = chat_request.top_p

                if chat_request.stream:
                    async def streaming():
                        try:
                            completion_stream = provider.chat.completions.create(**params)
                            
                            if isinstance(completion_stream, Generator):
                                for chunk in completion_stream:
                                    # Standardize chunk format before sending
                                    if hasattr(chunk, 'model_dump'): # Pydantic v2
                                        chunk_data = chunk.model_dump(exclude_none=True)
                                    elif hasattr(chunk, 'dict'): # Pydantic v1
                                        chunk_data = chunk.dict(exclude_none=True)
                                    elif isinstance(chunk, dict):
                                        chunk_data = chunk
                                    else: # Fallback for unknown chunk types
                                        chunk_data = chunk 
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                            else: # Non-generator, might be a full response or an async iterable
                                # This branch might need more robust handling for different async iterable types or full responses
                                if hasattr(completion_stream, 'model_dump'):
                                    yield f"data: {json.dumps(completion_stream.model_dump(exclude_none=True))}\n\n"
                                elif hasattr(completion_stream, 'dict'):
                                     yield f"data: {json.dumps(completion_stream.dict(exclude_none=True))}\n\n"
                                else:
                                    yield f"data: {json.dumps(completion_stream)}\n\n"

                        except Exception as e:
                            yield f"data: {format_exception(e)}\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(streaming(), media_type="text/event-stream")
                else: # Non-streaming
                    try:
                        completion = provider.chat.completions.create(**params)
                        if completion is None:
                            # Return a valid OpenAI-compatible error or empty response
                            return ChatCompletion( # Assuming ChatCompletion is a Pydantic model for the response
                                id=f"chatcmpl-{uuid.uuid4()}",
                                created=int(time.time()),
                                model=model_name,
                                choices=[Choice( # Assuming Choice model
                                    index=0,
                                    message=ChatCompletionMessage(role="assistant", content="Apology: No response generated."), # Assuming ChatCompletionMessage
                                    finish_reason="error"
                                )],
                                usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0) # Assuming CompletionUsage
                            ).model_dump(exclude_none=True)

                        # Standardize response format
                        if hasattr(completion, "model_dump"): # Pydantic v2
                            response_data = completion.model_dump(exclude_none=True)
                        elif hasattr(completion, "dict"): # Pydantic v1
                            response_data = completion.dict(exclude_none=True)
                        elif isinstance(completion, dict):
                            response_data = completion
                        else:
                            return ErrorResponse.from_message("Invalid response format from provider", HTTP_500_INTERNAL_SERVER_ERROR)
                        
                        return response_data
                    except Exception as e:
                        return ErrorResponse.from_exception(e, HTTP_500_INTERNAL_SERVER_ERROR)
                
                elapsed = time.time() - start_time

            except Exception as e:
                return ErrorResponse.from_exception(e, HTTP_500_INTERNAL_SERVER_ERROR)

def format_exception(e: Union[Exception, str]) -> str:
    if isinstance(e, str):
        message = e
    else:
        message = f"{e.__class__.__name__}: {str(e)}" # Keep it concise
    return json.dumps({
        "error": {
            "message": message,
            "type": "server_error", # Or more specific if possible
            "param": None,
            "code": "internal_server_error" # Or more specific
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
    print("Starting Webscout OpenAI API server...")
    if port is None:
        port = DEFAULT_PORT
    AppConfig.set_config(
        api_key=api_key,
        default_provider=default_provider or AppConfig.default_provider,
        base_url=base_url
    )
    # initialize_provider_map() # This is called inside create_app now.
                              # Call here if create_app doesn't exist yet or for early info.
                              # For showing providers, it needs to be called before printing.
    if show_available_providers: # Initialize map if needed for display before app creation
        if not AppConfig.provider_map: # Avoid re-initializing if already done by app creation logic path
            initialize_provider_map()

        print("\n=== Webscout OpenAI API Server ===")
        print(f"Server URL: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        if AppConfig.base_url:
            print(f"Base Path: {AppConfig.base_url}")
            api_endpoint_base = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}{AppConfig.base_url}"
        else:
            api_endpoint_base = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
        
        print(f"API Endpoint: {api_endpoint_base}/v1/chat/completions")
        print(f"Docs URL: {api_endpoint_base}/docs") # Adjusted for potential base_url in display
        print(f"API Authentication: {'Enabled' if api_key else 'Disabled'}")
        print(f"Default Provider: {AppConfig.default_provider}")

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
        
        print("\nUse Ctrl+C to stop the server.")
        print("=" * 40 + "\n")

    uvicorn_app_str = "webscout.Provider.OPENAI.api:create_app_debug" if debug else "webscout.Provider.OPENAI.api:create_app"
    
    # Note: Logs show "werkzeug". If /docs 404s persist, ensure Uvicorn is the actual server running.
    # The script uses uvicorn.run, so "werkzeug" logs are unexpected for this file.
    uvicorn.run(
        uvicorn_app_str,
        host=host,
        port=int(port),
        factory=True,
        reload=debug, # Enable reload only in debug mode for stability
        # log_level="debug" if debug else "info"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Start Webscout OpenAI-compatible API server')
    parser.add_argument('--port', type=int, default=os.getenv('PORT', DEFAULT_PORT), help=f'Port to run the server on (default: {DEFAULT_PORT})')
    parser.add_argument('--api-key', type=str, default=os.getenv('API_KEY'), help='API key for authentication (optional)')
    parser.add_argument('--default-provider', type=str, default=os.getenv('DEFAULT_PROVIDER'), help='Default provider to use (optional)')
    parser.add_argument('--base-url', type=str, default=os.getenv('BASE_URL'), help='Base URL for the API (optional, e.g., /api/v1)')
    parser.add_argument('--debug', action='store_true', default=os.getenv('DEBUG', 'false').lower() == 'true', help='Run in debug mode')
    args = parser.parse_args()
    

    run_api(
        host="0.0.0.0", # Host configurable via env if needed
        port=args.port,
        api_key=args.api_key,
        default_provider=args.default_provider,
        base_url=args.base_url,
        debug=args.debug
    )
