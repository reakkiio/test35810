"""
API server with OpenAI compatibility
"""

import json
import time
from typing import Dict, Any, List, Optional, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import config
from .model_manager import ModelManager
from .llm import LLMInterface

app = FastAPI(title="webscout.local API", description="OpenAI-compatible API for webscout.local")

# Models
model_manager: ModelManager = ModelManager()
loaded_models: Dict[str, LLMInterface] = {}  # Cache for loaded models

class ChatMessage(BaseModel):
    """
    Represents a single chat message for the chat completion endpoint.
    """
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """
    Request model for chat completions.
    """
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    """
    Request model for text completions.
    """
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None

class ModelInfo(BaseModel):
    """
    Model information for listing available models.
    """
    id: str
    object: str = "model"
    created: int
    owned_by: str = "webscout.local"

class ModelList(BaseModel):
    """
    List of available models.
    """
    object: str = "list"
    data: List[ModelInfo]

def get_model(model_name: str) -> LLMInterface:
    """
    Get or load a model by name, using a cache for efficiency.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        LLMInterface: Loaded model interface.
    Raises:
        HTTPException: If the model cannot be loaded.
    """
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = LLMInterface(model_name)
            loaded_models[model_name].load_model()
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")
    return loaded_models[model_name]

@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """
    List available models.
    Returns:
        ModelList: List of available models.
    """
    models = model_manager.list_models()
    model_list: List[ModelInfo] = []
    for model in models:
        model_list.append(
            ModelInfo(
                id=model["name"],
                created=int(time.time()),
            )
        )
    return ModelList(object="list", data=model_list)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> Any:
    """
    Create a chat completion.
    Args:
        request (ChatCompletionRequest): Chat completion request.
    Returns:
        StreamingResponse or dict: Streaming or regular response.
    """
    model = get_model(request.model)
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    if request.stream:
        async def generate() -> AsyncGenerator[str, None]:
            stream = model.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
                stop=request.stop,
            )
            for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
            stop=request.stop,
        )
        return response

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> Any:
    """
    Create a text completion.
    Args:
        request (CompletionRequest): Completion request.
    Returns:
        StreamingResponse or dict: Streaming or regular response.
    """
    model = get_model(request.model)
    if request.stream:
        async def generate() -> AsyncGenerator[str, None]:
            stream = model.create_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
                stop=request.stop,
            )
            for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        response = model.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
            stop=request.stop,
        )
        return response

def start_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """
    Start the API server.
    Args:
        host (Optional[str]): Host to bind the server to.
        port (Optional[int]): Port to bind the server to.
    """
    host = host or config.get("api_host", "127.0.0.1")
    port = port or config.get("api_port", 8000)
    uvicorn.run(app, host=host, port=port)
