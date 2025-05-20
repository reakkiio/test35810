from __future__ import annotations
import json
import uvicorn
import secrets
import os
import uuid
import time
import sys
import inspect
import asyncio # Added for asyncio.sleep
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
from webscout.Provider.OPENAI.pydantic_imports import * # Imports BaseModel, Field
from typing import Literal


# Import provider classes from the OPENAI directory
from webscout.Provider.OPENAI import *
# We will redefine some of these utility models locally to include tool calling fields
# So, we alias them if we need to refer to originals, or just let our definitions shadow them.
# from webscout.Provider.OPENAI.utils import (
#     ChatCompletionChunk as OriginalChatCompletionChunk,
#     ChatCompletion as OriginalChatCompletion,
#     Choice as OriginalChoice,
#     ChoiceDelta as OriginalChoiceDelta,
#     ChatCompletionMessage as OriginalChatCompletionMessage,
#     CompletionUsage as OriginalCompletionUsage
# )


DEFAULT_PORT = 8000

# --- Pydantic Models for Tool Calling & Overridden/Extended Core Models ---

class FunctionDescription(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict) # JSON Schema

class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDescription

class FunctionCallName(BaseModel): # For tool_choice
    name: str

class ToolChoiceFunction(BaseModel): # For tool_choice
    type: Literal["function"]
    function: FunctionCallName

# For tool_choice: "none", "auto", a specific function name string, or a ToolChoiceFunction object.
ToolChoiceOption = Union[Literal["none", "auto"], str, ToolChoiceFunction]

class ToolCallFunction(BaseModel): # In assistant message's tool_calls
    name: Optional[str] = None # Name of the function called
    arguments: Optional[str] = None # JSON string of arguments

class ToolCall(BaseModel): # In assistant message
    id: str # Unique ID for this tool call
    type: Literal["function"]
    function: ToolCallFunction

class DeltaFunctionCall(BaseModel): # For streaming delta
    name: Optional[str] = None
    arguments: Optional[str] = None # Part of the JSON string of arguments

class DeltaToolCall(BaseModel): # For streaming delta
    index: int
    id: Optional[str] = None
    type: Optional[Literal["function"]] = "function"
    function: Optional[DeltaFunctionCall] = None

# Define Pydantic models for multimodal content parts, aligning with OpenAI's API
class TextPart(BaseModel):
    type: Literal["text"]
    text: str

class ImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = Field("auto")

class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL

MessageContentParts = Union[TextPart, ImagePart]

# Redefined Message model to include tool_calls and tool_call_id
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[Union[str, List[MessageContentParts]]] = Field(None, description="Content of the message.")
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None # For assistant messages that call tools
    tool_call_id: Optional[str] = None # For tool messages responding to a tool_call

# Redefined/Extended Pydantic models for OpenAI responses to support tool calling
class ChatCompletionMessage(BaseModel): # Used in Choice.message
    role: Literal["assistant", "user", "system", "tool"] # OpenAI spec for choice message role
    content: Optional[Union[str, List[MessageContentParts]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    # function_call: Optional[Dict[str, str]] = None # Legacy, prefer tool_calls

class ChoiceDelta(BaseModel): # Used in Choice.delta (streaming)
    role: Optional[Literal["assistant", "user", "system", "tool"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall]] = None
    # function_call: Optional[Dict[str, str]] = None # Legacy

class Choice(BaseModel):
    index: int
    message: Optional[ChatCompletionMessage] = None # For non-streaming
    delta: Optional[ChoiceDelta] = None # For streaming
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    logprobs: Optional[Any] = None

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletion(BaseModel): # Non-streaming response
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[CompletionUsage] = None
    system_fingerprint: Optional[str] = None

class ChatCompletionChunk(BaseModel): # Streaming response
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Choice] # Each choice here will use 'delta'
    usage: Optional[CompletionUsage] = None # Typically null in chunks until maybe the last one
    system_fingerprint: Optional[str] = None


# Redefined ChatCompletionRequest to include tools and tool_choice
class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use.")
    messages: List[Message] = Field(..., description="A list of messages.")
    temperature: Optional[float] = Field(None)
    top_p: Optional[float] = Field(None)
    n: Optional[int] = Field(1, description="How many chat completion choices to generate.") # Fake response will only generate 1
    stream: Optional[bool] = Field(False, description="If set, partial message deltas will be sent.")
    max_tokens: Optional[int] = Field(None)
    presence_penalty: Optional[float] = Field(None)
    frequency_penalty: Optional[float] = Field(None)
    logit_bias: Optional[Dict[str, float]] = Field(None)
    user: Optional[str] = Field(None)
    stop: Optional[Union[str, List[str]]] = Field(None)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoiceOption] = None
    # response_format: Optional[Dict[str, str]] = None # e.g., {"type": "json_object"}
    # seed: Optional[int] = None

    class Config:
        extra = "ignore"
        schema_extra = {
            "example_text_completion": {
                "model": "gpt-4-fake",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "temperature": 0.7,
                "max_tokens": 150,
                "stream": False
            },
            "example_tool_call": {
                "model": "gpt-4-fake",
                "messages": [
                    {"role": "user", "content": "What's the weather like in Boston?"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                },
                                "required": ["location"],
                            },
                        }
                    }
                ],
                "tool_choice": "auto",
                "stream": False
            },
             "example_tool_call_stream": { # Added example for streaming tool call
                "model": "gpt-4-fake",
                "messages": [
                    {"role": "user", "content": "What's the weather like in Boston?"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                },
                                "required": ["location"],
                            },
                        }
                    }
                ],
                "tool_choice": "auto",
                "stream": True
            }
        }
# --- End of Pydantic Model Definitions ---


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
    default_provider = "ChatGPT_Fake" # Changed default to avoid confusion if real ChatGPT is also configured
    base_url: Optional[str] = None

    @classmethod
    def set_config(cls, **data):
        for key, value in data.items():
            setattr(cls, key, value)

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
        title="Webscout OpenAI API (Fake Stream/Tool)",
        description="OpenAI API compatible interface with fake streaming and tool calling support",
        version="0.2.0", # Incremented version
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
    initialize_provider_map() # This now also sets up the fake provider

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

        schema_method_name = "model_json_schema" if hasattr(BaseModel, "model_json_schema") else "schema"
        
        pydantic_models_to_register = {
            "TextPart": TextPart,
            "ImageURL": ImageURL,
            "ImagePart": ImagePart,
            "Message": Message, # Already part of ChatCompletionRequest schema effectively
            "ChatCompletionRequest": ChatCompletionRequest, # Main request model
            # Tool related models
            "FunctionDescription": FunctionDescription,
            "Tool": Tool,
            "FunctionCallName": FunctionCallName,
            "ToolChoiceFunction": ToolChoiceFunction,
            # ToolChoiceOption is a Union, harder to register directly, but will be part of ChatCompletionRequest
            "ToolCallFunction": ToolCallFunction,
            "ToolCall": ToolCall,
            "DeltaFunctionCall": DeltaFunctionCall,
            "DeltaToolCall": DeltaToolCall,
            # Response models
            "ChatCompletionMessage": ChatCompletionMessage,
            "ChoiceDelta": ChoiceDelta,
            "Choice": Choice,
            "CompletionUsage": CompletionUsage,
            "ChatCompletion": ChatCompletion,
            "ChatCompletionChunk": ChatCompletionChunk,
        }
        
        for name, model_cls in pydantic_models_to_register.items():
            # Check if schema for this model name already exists, possibly from ChatCompletionRequest.
            # The get_openapi function might have already processed ChatCompletionRequest and its submodels.
            if name not in openapi_schema["components"]["schemas"] or name in ["ChatCompletionRequest", "Message"]: # Always update these main ones
                try:
                    schema_data = getattr(model_cls, schema_method_name)()
                    if "title" in schema_data: # Pydantic v2 might add "title"
                         del schema_data["title"]
                    openapi_schema["components"]["schemas"][name] = schema_data
                except Exception as e:
                    print(f"Warning: Could not generate schema for {name}: {e}")

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
    return app

def create_app_debug():
    return create_app()

# A dummy provider class for the fake responses
class FakeStreamingProvider:
    def __init__(self, model=None): # model arg to match potential instantiation patterns
        self.model = model or "gpt-4-fake"
    
    # The chat_completions endpoint will handle logic directly,
    # so this provider doesn't need complex methods.
    # It's mainly for listing in /v1/models.
    @property
    def AVAILABLE_MODELS(self):
        return ["gpt-4-fake", "gpt-3.5-turbo-fake"]


def initialize_provider_map():
    from webscout.Provider.OPENAI.base import OpenAICompatibleProvider
    module = sys.modules.get("webscout.Provider.OPENAI") # Use .get for safety
    if module:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, OpenAICompatibleProvider) and obj.__name__ != "OpenAICompatibleProvider":
                AppConfig.provider_map[obj.__name__] = obj
                if hasattr(obj, "AVAILABLE_MODELS") and isinstance(obj.AVAILABLE_MODELS, (list, tuple, set)):
                    for model in obj.AVAILABLE_MODELS:
                        if model and isinstance(model, str) and model != obj.__name__:
                            AppConfig.provider_map[model] = obj
    
    # Add our Fake Provider
    AppConfig.provider_map["FakeStreamingProvider"] = FakeStreamingProvider
    for model_name in FakeStreamingProvider().AVAILABLE_MODELS:
        AppConfig.provider_map[model_name] = FakeStreamingProvider
    
    if not AppConfig.provider_map: # Fallback if no other providers were found
        from webscout.Provider.OPENAI.chatgpt import ChatGPT # Assuming this exists for a non-fake default
        # AppConfig.provider_map["ChatGPT"] = ChatGPT # Example, might not be desired if only fake is wanted
        AppConfig.default_provider = "FakeStreamingProvider" # Default to our fake one
    elif AppConfig.default_provider == "ChatGPT": # If default was ChatGPT, maybe change to fake
         AppConfig.default_provider = "FakeStreamingProvider"


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
                    if not secrets.compare_digest(AppConfig.api_key, auth_header):
                        return ErrorResponse.from_message("Invalid API key", HTTP_403_FORBIDDEN)
            return await call_next(request)

    def register_validation_exception_handler(self):
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            errors = exc.errors()
            # ... (rest of the validation handler, assumed to be fine) ...
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder({"detail": errors, "body": exc.body}) # simplified for brevity
            ) # Original handler was more detailed, can be restored if needed. For this change, focusing on core logic.

        @self.app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(request: Request, exc: StarletteHTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": f"Internal server error: {str(exc)}"})

    def register_routes(self):
        @self.app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse(url="/docs")
        
        @self.app.get("/v1/models", response_model=ModelListResponse)
        async def list_models():
            models_data = []
            seen_model_ids = set()
            for model_id, provider_class in AppConfig.provider_map.items():
                # If model_id is actually a provider class name, skip if its models are listed
                if inspect.isclass(provider_class) and model_id == provider_class.__name__:
                    # Check if this provider has AVAILABLE_MODELS to avoid double listing
                    if hasattr(provider_class, "AVAILABLE_MODELS"):
                        # This model_id (provider name) might be listed if it's also an alias for a specific model
                        # or if it's intended to be selectable directly.
                        # For now, let's assume provider names are not models unless explicitly listed.
                        pass # Handled by specific model names

                if model_id in seen_model_ids:
                    continue
                
                # Heuristic: if model_id is the same as provider_class.__name__, it might be a generic provider entry.
                # We are more interested in specific model names.
                # However, some providers might be used by their class name as a model.
                # Let's ensure we get the correct "owned_by".
                owner = provider_class.__name__ if inspect.isclass(provider_class) else "UnknownProvider"

                models_data.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": owner
                })
                seen_model_ids.add(model_id)
            
            # Ensure unique models by id
            unique_models_data = []
            final_seen_ids = set()
            for model_entry in models_data:
                if model_entry["id"] not in final_seen_ids:
                    unique_models_data.append(model_entry)
                    final_seen_ids.add(model_entry["id"])

            return ModelListResponse(data=unique_models_data)

        @self.app.post(
            "/v1/chat/completions",
            # response_model is tricky with StreamingResponse, FastAPI handles it.
            # For non-streaming, it would be ChatCompletion.
            response_model_exclude_none=True,
            response_model_exclude_unset=True,
            openapi_extra={
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ChatCompletionRequest"},
                            # Provide multiple examples if possible or a general one
                            "examples": {
                                "text_completion": {
                                    "summary": "Simple text completion",
                                    "value": ChatCompletionRequest.Config.schema_extra["example_text_completion"]
                                },
                                "tool_call": {
                                    "summary": "Tool call request",
                                    "value": ChatCompletionRequest.Config.schema_extra["example_tool_call"]
                                },
                                "tool_call_stream": {
                                    "summary": "Streaming tool call request",
                                    "value": ChatCompletionRequest.Config.schema_extra["example_tool_call_stream"]
                                }
                            }
                        }
                    }
                }
            }
        )
        async def chat_completions(
            request_fastapi: Request, # Renamed to avoid clash with ChatCompletionRequest model
            chat_request: ChatCompletionRequest = Body(...)
        ):
            # Determine if a tool call should be faked or if we're responding to one
            should_fake_tool_call = False
            is_reply_after_tool_call = False

            if chat_request.tools and chat_request.tool_choice != "none":
                if not chat_request.messages or chat_request.messages[-1].role != "tool":
                    should_fake_tool_call = True
            
            if chat_request.messages and chat_request.messages[-1].role == "tool":
                is_reply_after_tool_call = True
                should_fake_tool_call = False # Don't make a new tool call if we just got a tool response

            completion_id = f"chatcmpl-fake-{uuid.uuid4()}"
            created_time = int(time.time())
            model_name = chat_request.model # Use the requested model name

            if chat_request.stream:
                async def streaming_generator():
                    try:
                        if should_fake_tool_call and chat_request.tools:
                            # Fake a tool call stream
                            tool_to_call = chat_request.tools[0].function # Assuming at least one tool
                            tool_call_id = f"call_fake_{uuid.uuid4().hex[:10]}"
                            
                            # 1. First chunk: role and initial tool_call structure
                            delta_choice = ChoiceDelta(
                                role="assistant",
                                tool_calls=[
                                    DeltaToolCall(
                                        index=0,
                                        id=tool_call_id,
                                        type="function",
                                        function=DeltaFunctionCall(
                                            name=tool_to_call.name,
                                            arguments="" # Start with empty arguments
                                        )
                                    )
                                ]
                            )
                            chunk = ChatCompletionChunk(
                                id=completion_id, created=created_time, model=model_name,
                                choices=[Choice(index=0, delta=delta_choice, finish_reason=None)]
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                            await asyncio.sleep(0.05)

                            # 2. Stream arguments
                            fake_arguments_dict = {"location": "Fake Boston", "unit": "fake_celsius"}
                            if tool_to_call.parameters and "location" not in tool_to_call.parameters.get("properties", {}):
                                # If the function doesn't expect location, send some generic params
                                fake_arguments_dict = {"param1": "fake_value1", "param2": 123}
                                if tool_to_call.parameters.get("properties"):
                                    first_param_name = list(tool_to_call.parameters["properties"].keys())[0]
                                    fake_arguments_dict = {first_param_name: "fake_dynamic_value"}


                            fake_arguments_str = json.dumps(fake_arguments_dict)
                            for char_idx, char_to_send in enumerate(fake_arguments_str):
                                delta_args_choice = ChoiceDelta(
                                    tool_calls=[DeltaToolCall(
                                        index=0,
                                        function=DeltaFunctionCall(arguments=char_to_send)
                                    )]
                                )
                                chunk = ChatCompletionChunk(
                                    id=completion_id, created=created_time, model=model_name,
                                    choices=[Choice(index=0, delta=delta_args_choice, finish_reason=None)]
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                                await asyncio.sleep(0.02)

                            # 3. Final chunk for this choice: finish_reason = tool_calls
                            final_delta_choice = ChoiceDelta() # Empty delta
                            chunk = ChatCompletionChunk(
                                id=completion_id, created=created_time, model=model_name,
                                choices=[Choice(index=0, delta=final_delta_choice, finish_reason="tool_calls")]
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                        else: # Regular text streaming or text response after tool call
                            # 1. First chunk: role
                            delta_role_choice = ChoiceDelta(role="assistant")
                            chunk = ChatCompletionChunk(
                                id=completion_id, created=created_time, model=model_name,
                                choices=[Choice(index=0, delta=delta_role_choice, finish_reason=None)]
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                            await asyncio.sleep(0.05)

                            # 2. Stream content
                            text_to_stream = "This is a fake streaming response from the assistant. "
                            if is_reply_after_tool_call:
                                tool_msg = chat_request.messages[-1]
                                tool_name = "unknown_tool"
                                # Try to find the original tool call to get its name
                                for msg_idx in range(len(chat_request.messages) - 2, -1, -1):
                                    prev_msg = chat_request.messages[msg_idx]
                                    if prev_msg.role == "assistant" and prev_msg.tool_calls:
                                        for tc in prev_msg.tool_calls:
                                            if tc.id == tool_msg.tool_call_id:
                                                tool_name = tc.function.name or tool_name
                                                break
                                        break
                                text_to_stream = f"Okay, I have processed the output from tool '{tool_name}' (id: {tool_msg.tool_call_id}). Result: '{tool_msg.content}'. Now, here's a summary. "
                            
                            words = text_to_stream.split(" ")
                            for i, word in enumerate(words):
                                content_to_send = word + (" " if i < len(words) - 1 else "")
                                delta_content_choice = ChoiceDelta(content=content_to_send)
                                chunk = ChatCompletionChunk(
                                    id=completion_id, created=created_time, model=model_name,
                                    choices=[Choice(index=0, delta=delta_content_choice, finish_reason=None)]
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                                await asyncio.sleep(0.1)

                            # 3. Final chunk for this choice: finish_reason = stop
                            final_delta_choice = ChoiceDelta()
                            chunk = ChatCompletionChunk(
                                id=completion_id, created=created_time, model=model_name,
                                choices=[Choice(index=0, delta=final_delta_choice, finish_reason="stop")]
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        error_payload = format_exception(e) # format_exception returns a JSON string
                        yield f"data: {error_payload}\n\n" # Send error as SSE
                        yield "data: [DONE]\n\n" # Still need to terminate stream
                return StreamingResponse(streaming_generator(), media_type="text/event-stream")
            
            else: # Non-streaming
                if should_fake_tool_call and chat_request.tools:
                    tool_to_call = chat_request.tools[0].function
                    tool_call_id = f"call_fake_{uuid.uuid4().hex[:10]}"
                    fake_arguments_dict = {"location": "Fake San Francisco", "unit": "fake_fahrenheit"}
                    # (similar logic for dynamic fake_arguments_dict as in streaming)
                    fake_arguments_str = json.dumps(fake_arguments_dict)

                    message = ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=tool_call_id,
                                type="function",
                                function=ToolCallFunction(
                                    name=tool_to_call.name,
                                    arguments=fake_arguments_str
                                )
                            )
                        ]
                    )
                    choice = Choice(index=0, message=message, finish_reason="tool_calls")
                    response = ChatCompletion(
                        id=completion_id, created=created_time, model=model_name,
                        choices=[choice],
                        usage=CompletionUsage(prompt_tokens=15, completion_tokens=30, total_tokens=45)
                    )
                    return response # FastAPI will handle .model_dump() for Pydantic models

                else: # Regular text response or text response after tool call
                    response_text = "This is a fake non-streaming response from the assistant."
                    if is_reply_after_tool_call:
                        tool_msg = chat_request.messages[-1]
                        tool_name = "unknown_tool"
                        # (similar logic to find tool_name as in streaming)
                        response_text = f"Understood. Tool '{tool_name}' (id: {tool_msg.tool_call_id}) returned: '{tool_msg.content}'. I will now proceed based on this (non-streaming)."

                    message = ChatCompletionMessage(role="assistant", content=response_text)
                    choice = Choice(index=0, message=message, finish_reason="stop")
                    response = ChatCompletion(
                        id=completion_id, created=created_time, model=model_name,
                        choices=[choice],
                        usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
                    )
                    return response # FastAPI will handle .model_dump()

def format_exception(e: Union[Exception, str]) -> str:
    if isinstance(e, str):
        message = e
    else:
        message = f"{e.__class__.__name__}: {str(e)}"
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
    print(f"Starting Webscout OpenAI API server (with Fake Streaming/Tool Support)...")
    if port is None:
        port = DEFAULT_PORT
    
    # Set default_provider to our fake one if none is specified
    AppConfig.set_config(
        api_key=api_key,
        default_provider=default_provider or AppConfig.default_provider, # AppConfig.default_provider is now "FakeStreamingProvider"
        base_url=base_url
    )

    if show_available_providers:
        if not AppConfig.provider_map:
            initialize_provider_map() # Ensures map is populated for display

        print("\n=== Webscout OpenAI API Server (Fake Stream/Tool) ===")
        # ... (rest of the server info print statements, should be fine) ...
        print(f"Server URL: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        # ... (rest of info print) ...
        print(f"Default Provider: {AppConfig.default_provider}")

        providers = list(set(
            (v.__name__ if hasattr(v, '__name__') else str(v)) 
            for v in AppConfig.provider_map.values()
        ))
        print(f"\n--- Available Provider Aliases/Classes ({len(providers)}) ---")
        for i, provider_name in enumerate(sorted(providers), 1):
            print(f"{i}. {provider_name}")
        
        model_names = sorted([
            model for model in AppConfig.provider_map.keys() 
            if not (hasattr(AppConfig.provider_map[model], '__name__') and model == AppConfig.provider_map[model].__name__)
        ]) # Filter out class names if they are not also model names
        
        # Or, more simply, list all keys that don't look like class names themselves
        model_names_display = []
        provider_class_names_set = {p.__name__ for p in AppConfig.provider_map.values() if hasattr(p, '__name__')}

        for model_id, provider_cls in AppConfig.provider_map.items():
            if model_id not in provider_class_names_set: # Only list if key is a model alias
                 model_names_display.append(f"{model_id} (via {provider_cls.__name__ if hasattr(provider_cls, '__name__') else 'Unknown'})")

        if model_names_display:
            print(f"\n--- Available Models ({len(model_names_display)}) ---")
            for i, model_name_str in enumerate(sorted(model_names_display), 1):
                print(f"{i}. {model_name_str}")
        else:
            print("\nNo specific models registered beyond provider names. Use provider names (if applicable) or specific fake models like 'gpt-4-fake'.")
        print("\nUse Ctrl+C to stop the server.")
        print("=" * 40 + "\n")


    uvicorn_app_str = "webscout.Provider.OPENAI.api:create_app_debug" if debug else "webscout.Provider.OPENAI.api:create_app"
    
    uvicorn.run(
        uvicorn_app_str, # Assuming this file is named api.py inside webscout/Provider/OPENAI/
        host=host,
        port=int(port),
        factory=True,
        reload=debug,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Start Webscout OpenAI-compatible API server with Fake Streaming/Tool support')
    parser.add_argument('--port', type=int, default=os.getenv('PORT', DEFAULT_PORT), help=f'Port (default: {DEFAULT_PORT})')
    parser.add_argument('--api-key', type=str, default=os.getenv('API_KEY'), help='API key (optional)')
    # Default provider will now typically be FakeStreamingProvider due to initialize_provider_map logic
    parser.add_argument('--default-provider', type=str, default=os.getenv('DEFAULT_PROVIDER'), help='Default provider (optional)')
    parser.add_argument('--base-url', type=str, default=os.getenv('BASE_URL'), help='Base URL (optional, e.g., /api/v1)')
    parser.add_argument('--debug', action='store_true', default=os.getenv('DEBUG', 'false').lower() == 'true', help='Debug mode')
    args = parser.parse_args()
    
    # The uvicorn app string needs to correctly point to this file's create_app function.
    # If you save this modified code as, e.g., `fake_api.py` in the current directory,
    # you might run it as `python fake_api.py` and the uvicorn_app_str should be:
    # `fake_api:create_app` if `factory=True` is used and create_app is in fake_api.py
    # The original `webscout.Provider.OPENAI.api:create_app` implies a specific package structure.
    # For simplicity if running this standalone, you might change uvicorn_app_str to something like:
    # `__main__:create_app` if this script is run directly and `factory=True`.
    # However, I will keep the original uvicorn_app_str assuming the file structure is maintained.

    run_api(
        host="0.0.0.0",
        port=args.port,
        api_key=args.api_key,
        default_provider=args.default_provider, # This will be passed to AppConfig
        base_url=args.base_url,
        debug=args.debug
    )
