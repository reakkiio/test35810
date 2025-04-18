"""
API endpoints for webscout.Local
"""

import time
import json
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import base64
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .llm import ModelManager
from .config import Config

logger = logging.getLogger(__name__)

# API Models
class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    suffix: Optional[str] = None
    images: Optional[List[str]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    raw: bool = False
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    tools: Optional[List[Dict[str, Any]]] = None
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    truncate: bool = True
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"

class PullModelRequest(BaseModel):
    model: str
    insecure: bool = False
    stream: bool = True

class DeleteModelRequest(BaseModel):
    model: str

class ModelResponse(BaseModel):
    name: str
    modified_at: str
    size: int
    details: Optional[Dict[str, Any]] = None

class ModelsResponse(BaseModel):
    models: List[ModelResponse]

# API Router
router = APIRouter()

# Dependency to get model manager
def get_model_manager(config: Config = Depends(lambda: Config.from_env())):
    return ModelManager(config)

@router.post("/api/generate")
async def generate(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Generate a completion for a given prompt"""
    try:
        # Parse keep_alive
        keep_alive_seconds = 300  # Default 5 minutes
        if request.keep_alive:
            if request.keep_alive.endswith("ms"):
                keep_alive_seconds = int(request.keep_alive[:-2]) / 1000
            elif request.keep_alive.endswith("s"):
                keep_alive_seconds = int(request.keep_alive[:-1])
            elif request.keep_alive.endswith("m"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 60
            elif request.keep_alive.endswith("h"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 3600
            elif request.keep_alive == "0":
                # Special case: unload immediately after completion
                keep_alive_seconds = 0
            else:
                try:
                    keep_alive_seconds = int(request.keep_alive)
                except ValueError:
                    pass
        
        # If prompt is empty, just load the model and return
        if not request.prompt:
            model = model_manager.load_model(
                request.model,
                **(request.options or {})
            )
            
            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                background_tasks.add_task(model_manager.unload_model, request.model)
                
            return {
                "model": request.model,
                "created_at": datetime.now().isoformat(),
                "response": "",
                "done": True,
                "done_reason": "load" if keep_alive_seconds > 0 else "unload"
            }
        
        # Load the model
        model = model_manager.load_model(
            request.model,
            **(request.options or {})
        )
        
        # Process images if provided
        image_data = None
        if request.images and len(request.images) > 0:
            # For now, we only support the first image
            image_base64 = request.images[0]
            if image_base64.startswith("data:"):
                # Handle data URI
                image_base64 = image_base64.split(",", 1)[1]
            image_data = base64.b64decode(image_base64)
        
        # Prepare generation parameters
        generation_params = {
            "prompt": request.prompt,
            "suffix": request.suffix,
            "max_tokens": request.options.get("num_predict", 128) if request.options else 128,
            "temperature": request.options.get("temperature", 0.8) if request.options else 0.8,
            "top_p": request.options.get("top_p", 0.95) if request.options else 0.95,
            "echo": False,
        }
        
        # Add system prompt if provided
        if request.system:
            generation_params["system_prompt"] = request.system
            
        # Add format if provided
        if request.format:
            generation_params["response_format"] = request.format
            
        # Add images if provided
        if image_data:
            generation_params["image_data"] = image_data
            
        # Add context if provided
        if request.context:
            generation_params["context"] = request.context
            
        # Stream the response
        if request.stream:
            async def generate_stream():
                start_time = time.time()
                load_time = 0  # We don't track this separately
                
                # Start generation
                completion_id = str(uuid4())
                
                # Initial response
                yield json.dumps({
                    "model": request.model,
                    "created_at": datetime.now().isoformat(),
                    "response": "",
                    "done": False
                }) + "\n"
                
                # Generate completion
                completion = model.create_completion(**generation_params)
                
                # Final response with stats
                end_time = time.time()
                total_duration = int((end_time - start_time) * 1e9)  # Convert to nanoseconds
                
                yield json.dumps({
                    "model": request.model,
                    "created_at": datetime.now().isoformat(),
                    "response": completion["choices"][0]["text"],
                    "done": True,
                    "context": completion.get("context", []),
                    "total_duration": total_duration,
                    "load_duration": load_time,
                    "prompt_eval_count": completion.get("usage", {}).get("prompt_tokens", 0),
                    "prompt_eval_duration": 0,  # Not tracked
                    "eval_count": completion.get("usage", {}).get("completion_tokens", 0),
                    "eval_duration": 0  # Not tracked
                }) + "\n"
                
                # Schedule unloading if keep_alive is 0
                if keep_alive_seconds == 0:
                    background_tasks.add_task(model_manager.unload_model, request.model)
            
            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            # Non-streaming response
            start_time = time.time()
            
            # Generate completion
            completion = model.create_completion(**generation_params)
            
            # Calculate durations
            end_time = time.time()
            total_duration = int((end_time - start_time) * 1e9)  # Convert to nanoseconds
            
            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                background_tasks.add_task(model_manager.unload_model, request.model)
            
            return {
                "model": request.model,
                "created_at": datetime.now().isoformat(),
                "response": completion["choices"][0]["text"],
                "done": True,
                "context": completion.get("context", []),
                "total_duration": total_duration,
                "load_duration": 0,  # Not tracked separately
                "prompt_eval_count": completion.get("usage", {}).get("prompt_tokens", 0),
                "prompt_eval_duration": 0,  # Not tracked
                "eval_count": completion.get("usage", {}).get("completion_tokens", 0),
                "eval_duration": 0  # Not tracked
            }
    
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Generate a chat completion"""
    try:
        # Parse keep_alive
        keep_alive_seconds = 300  # Default 5 minutes
        if request.keep_alive:
            if request.keep_alive.endswith("ms"):
                keep_alive_seconds = int(request.keep_alive[:-2]) / 1000
            elif request.keep_alive.endswith("s"):
                keep_alive_seconds = int(request.keep_alive[:-1])
            elif request.keep_alive.endswith("m"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 60
            elif request.keep_alive.endswith("h"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 3600
            elif request.keep_alive == "0":
                # Special case: unload immediately after completion
                keep_alive_seconds = 0
            else:
                try:
                    keep_alive_seconds = int(request.keep_alive)
                except ValueError:
                    pass
        
        # If messages is empty, just load the model and return
        if not request.messages:
            model = model_manager.load_model(
                request.model,
                **(request.options or {})
            )
            
            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                background_tasks.add_task(model_manager.unload_model, request.model)
                
            return {
                "model": request.model,
                "created_at": datetime.now().isoformat(),
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done_reason": "load" if keep_alive_seconds > 0 else "unload",
                "done": True
            }
        
        # Load the model
        model = model_manager.load_model(
            request.model,
            **(request.options or {})
        )
        
        # Convert messages to the format expected by llama-cpp-python
        messages = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            else:
                # Handle multimodal content
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Prepare chat parameters
        chat_params = {
            "messages": messages,
            "temperature": request.options.get("temperature", 0.8) if request.options else 0.8,
            "top_p": request.options.get("top_p", 0.95) if request.options else 0.95,
        }
        
        # Add tools if provided
        if request.tools:
            chat_params["tools"] = request.tools
            
        # Add format if provided
        if request.format:
            chat_params["response_format"] = request.format
            
        # Stream the response
        if request.stream:
            async def generate_stream():
                start_time = time.time()
                
                # Start generation
                completion_id = str(uuid4())
                
                # Initial response
                yield json.dumps({
                    "model": request.model,
                    "created_at": datetime.now().isoformat(),
                    "message": {
                        "role": "assistant",
                        "content": ""
                    },
                    "done": False
                }) + "\n"
                
                # Generate chat completion
                completion = model.create_chat_completion(**chat_params)
                
                # Final response with stats
                end_time = time.time()
                total_duration = int((end_time - start_time) * 1e9)  # Convert to nanoseconds
                
                response_message = completion["choices"][0]["message"]
                
                yield json.dumps({
                    "model": request.model,
                    "created_at": datetime.now().isoformat(),
                    "message": response_message,
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": total_duration,
                    "load_duration": 0,  # Not tracked separately
                    "prompt_eval_count": completion.get("usage", {}).get("prompt_tokens", 0),
                    "prompt_eval_duration": 0,  # Not tracked
                    "eval_count": completion.get("usage", {}).get("completion_tokens", 0),
                    "eval_duration": 0  # Not tracked
                }) + "\n"
                
                # Schedule unloading if keep_alive is 0
                if keep_alive_seconds == 0:
                    background_tasks.add_task(model_manager.unload_model, request.model)
            
            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            # Non-streaming response
            start_time = time.time()
            
            # Generate chat completion
            completion = model.create_chat_completion(**chat_params)
            
            # Calculate durations
            end_time = time.time()
            total_duration = int((end_time - start_time) * 1e9)  # Convert to nanoseconds
            
            response_message = completion["choices"][0]["message"]
            
            # Schedule unloading if keep_alive is 0
            if keep_alive_seconds == 0:
                background_tasks.add_task(model_manager.unload_model, request.model)
            
            return {
                "model": request.model,
                "created_at": datetime.now().isoformat(),
                "message": response_message,
                "done": True,
                "done_reason": "stop",
                "total_duration": total_duration,
                "load_duration": 0,  # Not tracked separately
                "prompt_eval_count": completion.get("usage", {}).get("prompt_tokens", 0),
                "prompt_eval_duration": 0,  # Not tracked
                "eval_count": completion.get("usage", {}).get("completion_tokens", 0),
                "eval_duration": 0  # Not tracked
            }
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/embed")
async def embed(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Generate embeddings from a model"""
    try:
        # Parse keep_alive
        keep_alive_seconds = 300  # Default 5 minutes
        if request.keep_alive:
            if request.keep_alive.endswith("ms"):
                keep_alive_seconds = int(request.keep_alive[:-2]) / 1000
            elif request.keep_alive.endswith("s"):
                keep_alive_seconds = int(request.keep_alive[:-1])
            elif request.keep_alive.endswith("m"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 60
            elif request.keep_alive.endswith("h"):
                keep_alive_seconds = int(request.keep_alive[:-1]) * 3600
            elif request.keep_alive == "0":
                # Special case: unload immediately after completion
                keep_alive_seconds = 0
            else:
                try:
                    keep_alive_seconds = int(request.keep_alive)
                except ValueError:
                    pass
        
        # Load the model with embedding=True
        model_options = {**(request.options or {}), "embedding": True}
        model = model_manager.load_model(
            request.model,
            **model_options
        )
        
        # Generate embeddings
        start_time = time.time()
        
        if isinstance(request.input, str):
            # Single input
            embedding = model.create_embedding(request.input)
            embeddings = [embedding["embedding"]]
        else:
            # Multiple inputs
            embeddings = []
            for text in request.input:
                embedding = model.create_embedding(text)
                embeddings.append(embedding["embedding"])
        
        # Calculate durations
        end_time = time.time()
        total_duration = int((end_time - start_time) * 1e9)  # Convert to nanoseconds
        
        # Schedule unloading if keep_alive is 0
        if keep_alive_seconds == 0:
            background_tasks.add_task(model_manager.unload_model, request.model)
        
        return {
            "model": request.model,
            "embeddings": embeddings,
            "total_duration": total_duration,
            "load_duration": 0,  # Not tracked separately
            "prompt_eval_count": 0  # Not tracked
        }
    
    except Exception as e:
        logger.error(f"Error in embed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/pull")
async def pull_model(
    request: PullModelRequest,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Pull a model from Hugging Face Hub"""
    try:
        if request.stream:
            async def generate_stream():
                # Initial response
                yield json.dumps({"status": "pulling manifest"}) + "\n"
                
                # Pull the model
                result = await model_manager.pull_model(request.model, request.insecure)
                
                if result["status"] == "error":
                    yield json.dumps({"status": "error", "error": result["message"]}) + "\n"
                    return
                
                # Success response
                yield json.dumps({"status": "downloading model"}) + "\n"
                yield json.dumps({"status": "verifying sha256 digest"}) + "\n"
                yield json.dumps({"status": "writing manifest"}) + "\n"
                yield json.dumps({"status": "success"}) + "\n"
            
            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            # Non-streaming response
            result = await model_manager.pull_model(request.model, request.insecure)
            
            if result["status"] == "error":
                return {"status": "error", "error": result["message"]}
            
            return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error in pull_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/delete")
async def delete_model(
    request: DeleteModelRequest,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Delete a model"""
    try:
        success = model_manager.delete_model(request.model)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")
        
        return {"status": "success"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/tags")
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
):
    """List all available models"""
    try:
        models = model_manager.list_models()
        
        # Convert to response format
        response_models = []
        for model in models:
            response_models.append(ModelResponse(
                name=model["name"],
                modified_at=model["modified_at"],
                size=model["size"],
                details={
                    "format": "gguf",
                    "family": "llama",  # Default, could be improved with model metadata
                    "parameter_size": "Unknown",
                    "quantization_level": "Unknown"
                }
            ))
        
        return ModelsResponse(models=response_models)
    
    except Exception as e:
        logger.error(f"Error in list_models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/version")
async def version():
    """Get the version of webscout.Local"""
    return {"version": "0.1.0"}
