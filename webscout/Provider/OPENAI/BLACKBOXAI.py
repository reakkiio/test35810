import requests
import random
import string
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Generator, Any
import json
import uuid
import time

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)


def to_data_uri(image_data):
    """Convert image data to a data URI format"""
    if isinstance(image_data, str):
        # Assume it's already a data URI
        return image_data

    # Encode binary data to base64
    encoded = base64.b64encode(image_data).decode('utf-8')

    # Determine MIME type (simplified)
    mime_type = "image/jpeg"  # Default
    if image_data.startswith(b'\x89PNG'):
        mime_type = "image/png"
    elif image_data.startswith(b'\xff\xd8'):
        mime_type = "image/jpeg"
    elif image_data.startswith(b'GIF'):
        mime_type = "image/gif"

    return f"data:{mime_type};base64,{encoded}"


class Completions(BaseCompletions):
    def __init__(self, client: 'BLACKBOXAI'):
        self._client = client
    
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Create a chat completion with BlackboxAI API.
        
        Args:
            model: The model to use (from AVAILABLE_MODELS)
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            If stream=False, returns a ChatCompletion object
            If stream=True, returns a Generator yielding ChatCompletionChunk objects
        """
        # Generate request ID and timestamp
        request_id = str(uuid.uuid4())
        created_time = int(time.time())
        
        # Extract system message if present
        system_message = "You are a helpful AI assistant."
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
                break
        
        # Look for any image content
        media = []
        for msg in messages:
            if msg.get("role") == "user":
                # Check for image attachments in content
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict) and "url" in image_url:
                                url = image_url["url"]
                                if url.startswith("data:"):
                                    # It's already a data URI
                                    image_name = f"image_{len(media)}.png"
                                    media.append((url, image_name))
                                else:
                                    # Need to fetch and convert to data URI
                                    try:
                                        image_response = requests.get(url)
                                        if image_response.ok:
                                            image_name = f"image_{len(media)}.png"
                                            media.append((image_response.content, image_name))
                                    except Exception as e:
                                        pass
        
        # Use streaming implementation if requested
        if stream:
            return self._create_streaming(
                request_id=request_id,
                created_time=created_time,
                model=model,
                messages=messages,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                media=media
            )
        
        # Otherwise use non-streaming implementation
        return self._create_non_streaming(
            request_id=request_id,
            created_time=created_time,
            model=model,
            messages=messages,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            media=media
        )
    
    def _create_streaming(
        self,
        *,
        request_id: str,
        created_time: int,
        model: str,
        messages: List[Dict[str, Any]],
        system_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        media: List = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Implementation for streaming chat completions."""
        try:
            # Prepare user messages for BlackboxAI API format
            blackbox_messages = []
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    continue  # System message handled separately
                    
                msg_id = self._client.generate_id() if i > 0 else request_id
                blackbox_messages.append({
                    "id": msg_id,
                    "content": msg["content"],
                    "role": msg["role"]
                })
            
            # Add image data if provided
            if media and blackbox_messages:
                blackbox_messages[-1]['data'] = {
                    "imagesData": [
                        {
                            "filePath": f"/",
                            "contents": to_data_uri(image)
                        } for image in media
                    ],
                    "fileText": "",
                    "title": ""
                }
            
            # Generate request payload with session
            request_email = f"{self._client.generate_random_string(8)}@blackbox.ai"
            session_data = self._client.generate_session(request_email)
            
            # Create the API request payload
            payload = self._client.create_request_payload(
                messages=blackbox_messages,
                chat_id=request_id,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                session_data=session_data,
                model=model
            )
            
            # Make the API request with cookies
            response = self._client.session.post(
                self._client.api_endpoint,
                json=payload,
                headers=self._client.headers,
                cookies=self._client.cookies,  # Add cookies to the request
                stream=True,
                timeout=self._client.timeout
            )
            
            # Process the streaming response
            accumulated_content = ""
            for line in response.iter_lines(decode_unicode=True, delimiter="\n"):
                if not line:
                    continue
                    
                # Skip error messages but don't raise exceptions
                if "service has been suspended" in line.lower() or "API request failed" in line:
                    continue
                
                if "You have reached your request limit" in line:
                    continue
                    
                # Handle SSE formatted lines
                if line.startswith("data: "):
                    line = line[6:]
                
                # Remove any special formatting that might be in the response
                line = line.strip()
                
                # Skip empty lines after processing
                if not line:
                    continue
                    
                # Create and yield a chunk
                delta = ChoiceDelta(content=line)
                choice = Choice(index=0, delta=delta, finish_reason=None)
                accumulated_content += line
                
                chunk = ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=created_time,
                    model=model
                )
                
                yield chunk
            
            # Final chunk with finish_reason
            delta = ChoiceDelta(content=None)
            choice = Choice(index=0, delta=delta, finish_reason="stop")
            chunk = ChatCompletionChunk(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model
            )
            
            yield chunk
            
        except Exception as e:
            raise IOError(f"BlackboxAI streaming request failed: {str(e)}") from e
    
    def _create_non_streaming(
        self,
        *,
        request_id: str,
        created_time: int,
        model: str,
        messages: List[Dict[str, Any]],
        system_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        media: List = None
    ) -> ChatCompletion:
        """Implementation for non-streaming chat completions."""
        try:
            # Collect all content from streaming response
            response_chunks = []
            for chunk in self._create_streaming(
                request_id=request_id,
                created_time=created_time,
                model=model,
                messages=messages,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                media=media
            ):
                # Only collect chunks with content
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    response_chunks.append(chunk.choices[0].delta.content)
            
            # Combine all chunks into full content
            full_content = "".join(response_chunks)
            
            # Create the completion message
            message = ChatCompletionMessage(
                role="assistant",
                content=full_content
            )
            
            # Create the choice with the message
            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )
            
            # Estimate token usage
            prompt_tokens = sum(len(str(msg.get("content", ""))) // 4 for msg in messages)
            completion_tokens = len(full_content) // 4
            
            # Create the final completion object
            completion = ChatCompletion(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            
            return completion
            
        except Exception as e:
            raise IOError(f"BlackboxAI request failed: {str(e)}") from e


class Chat(BaseChat):
    def __init__(self, client: 'BLACKBOXAI'):
        self.completions = Completions(client)


class BLACKBOXAI(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for BlackboxAI API.
    
    Usage:
        client = BLACKBOXAI()
        response = client.chat.completions.create(
            model="GPT-4.1",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    """
    # Default model
    default_model = "GPT-4.1"
    default_vision_model = default_model
    api_endpoint = "https://www.blackbox.ai/api/chat"
    timeout = 30
    

    # Default model (remains the same as per original class)
    default_model = "GPT-4.1"
    default_vision_model = default_model

    # New OpenRouter models list
    openrouter_models = [
        "Deepcoder 14B Preview",
        "DeepHermes 3 Llama 3 8B Preview",
        "DeepSeek R1 Zero",
        "Dolphin3.0 Mistral 24B",
        "Dolphin3.0 R1 Mistral 24B",
        "Flash 3",
        "Gemini 2.0 Flash Experimental",
        "Gemma 2 9B",
        "Gemma 3 12B",
        "Gemma 3 1B",
        "Gemma 3 27B",
        "Gemma 3 4B",
        "Kimi VL A3B Thinking",
        "Llama 3.1 8B Instruct",
        "Llama 3.1 Nemotron Ultra 253B v1",
        "Llama 3.2 11B Vision Instruct",
        "Llama 3.2 1B Instruct",
        "Llama 3.2 3B Instruct",
        "Llama 3.3 70B Instruct",
        "Llama 3.3 Nemotron Super 49B v1",
        "Llama 4 Maverick",
        "Llama 4 Scout",
        "Mistral 7B Instruct",
        "Mistral Nemo",
        "Mistral Small 3",
        "Mistral Small 3.1 24B",
        "Molmo 7B D",
        "Moonlight 16B A3B Instruct",
        "Qwen2.5 72B Instruct",
        "Qwen2.5 7B Instruct",
        "Qwen2.5 Coder 32B Instruct",
        "Qwen2.5 VL 32B Instruct",
        "Qwen2.5 VL 3B Instruct",
        "Qwen2.5 VL 72B Instruct",
        "Qwen2.5-VL 7B Instruct",
        "Qwerky 72B",
        "QwQ 32B",
        "QwQ 32B Preview",
        "QwQ 32B RpR v1",
        "R1",
        "R1 Distill Llama 70B",
        "R1 Distill Qwen 14B",
        "R1 Distill Qwen 32B",
    ]

    # New base models list
    models = [
        default_model,
        "o3-mini",
        "gpt-4.1-nano",
        "Claude-sonnet-3.7",
        "Claude-sonnet-3.5",
        "DeepSeek-R1",
        "Mistral-Small-24B-Instruct-2501",
        *openrouter_models,
        # Trending agent modes (names)
        'Python Agent', 'HTML Agent', 'Builder Agent', 'Java Agent', 'JavaScript Agent',
        'React Agent', 'Android Agent', 'Flutter Agent', 'Next.js Agent', 'AngularJS Agent',
        'Swift Agent', 'MongoDB Agent', 'PyTorch Agent', 'Xcode Agent', 'Azure Agent',
        'Bitbucket Agent', 'DigitalOcean Agent', 'Docker Agent', 'Electron Agent',
        'Erlang Agent', 'FastAPI Agent', 'Firebase Agent', 'Flask Agent', 'Git Agent',
        'Gitlab Agent', 'Go Agent', 'Godot Agent', 'Google Cloud Agent', 'Heroku Agent'
    ]

    # Models that support vision capabilities
    vision_models = [default_vision_model, 'o3-mini', "Llama 3.2 11B Vision Instruct"] # Added Llama vision

    # Models that can be directly selected by users
    userSelectedModel = ['o3-mini','Claude-sonnet-3.7', 'Claude-sonnet-3.5', 'DeepSeek-R1', 'Mistral-Small-24B-Instruct-2501'] + openrouter_models

    # Agent mode configurations
    agentMode = {
        # OpenRouter Free
        'Deepcoder 14B Preview': {'mode': True, 'id': "agentica-org/deepcoder-14b-preview:free", 'name': "Deepcoder 14B Preview"},
        'DeepHermes 3 Llama 3 8B Preview': {'mode': True, 'id': "nousresearch/deephermes-3-llama-3-8b-preview:free", 'name': "DeepHermes 3 Llama 3 8B Preview"},
        'DeepSeek R1 Zero': {'mode': True, 'id': "deepseek/deepseek-r1-zero:free", 'name': "DeepSeek R1 Zero"},
        'Dolphin3.0 Mistral 24B': {'mode': True, 'id': "cognitivecomputations/dolphin3.0-mistral-24b:free", 'name': "Dolphin3.0 Mistral 24B"},
        'Dolphin3.0 R1 Mistral 24B': {'mode': True, 'id': "cognitivecomputations/dolphin3.0-r1-mistral-24b:free", 'name': "Dolphin3.0 R1 Mistral 24B"},
        'Flash 3': {'mode': True, 'id': "rekaai/reka-flash-3:free", 'name': "Flash 3"},
        'Gemini 2.0 Flash Experimental': {'mode': True, 'id': "google/gemini-2.0-flash-exp:free", 'name': "Gemini 2.0 Flash Experimental"},
        'Gemma 2 9B': {'mode': True, 'id': "google/gemma-2-9b-it:free", 'name': "Gemma 2 9B"},
        'Gemma 3 12B': {'mode': True, 'id': "google/gemma-3-12b-it:free", 'name': "Gemma 3 12B"},
        'Gemma 3 1B': {'mode': True, 'id': "google/gemma-3-1b-it:free", 'name': "Gemma 3 1B"},
        'Gemma 3 27B': {'mode': True, 'id': "google/gemma-3-27b-it:free", 'name': "Gemma 3 27B"},
        'Gemma 3 4B': {'mode': True, 'id': "google/gemma-3-4b-it:free", 'name': "Gemma 3 4B"},
        'Kimi VL A3B Thinking': {'mode': True, 'id': "moonshotai/kimi-vl-a3b-thinking:free", 'name': "Kimi VL A3B Thinking"},
        'Llama 3.1 8B Instruct': {'mode': True, 'id': "meta-llama/llama-3.1-8b-instruct:free", 'name': "Llama 3.1 8B Instruct"},
        'Llama 3.1 Nemotron Ultra 253B v1': {'mode': True, 'id': "nvidia/llama-3.1-nemotron-ultra-253b-v1:free", 'name': "Llama 3.1 Nemotron Ultra 253B v1"},
        'Llama 3.2 11B Vision Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-11b-vision-instruct:free", 'name': "Llama 3.2 11B Vision Instruct"},
        'Llama 3.2 1B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-1b-instruct:free", 'name': "Llama 3.2 1B Instruct"},
        'Llama 3.2 3B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-3b-instruct:free", 'name': "Llama 3.2 3B Instruct"},
        'Llama 3.3 70B Instruct': {'mode': True, 'id': "meta-llama/llama-3.3-70b-instruct:free", 'name': "Llama 3.3 70B Instruct"},
        'Llama 3.3 Nemotron Super 49B v1': {'mode': True, 'id': "nvidia/llama-3.3-nemotron-super-49b-v1:free", 'name': "Llama 3.3 Nemotron Super 49B v1"},
        'Llama 4 Maverick': {'mode': True, 'id': "meta-llama/llama-4-maverick:free", 'name': "Llama 4 Maverick"},
        'Llama 4 Scout': {'mode': True, 'id': "meta-llama/llama-4-scout:free", 'name': "Llama 4 Scout"},
        'Mistral 7B Instruct': {'mode': True, 'id': "mistralai/mistral-7b-instruct:free", 'name': "Mistral 7B Instruct"},
        'Mistral Nemo': {'mode': True, 'id': "mistralai/mistral-nemo:free", 'name': "Mistral Nemo"},
        'Mistral Small 3': {'mode': True, 'id': "mistralai/mistral-small-24b-instruct-2501:free", 'name': "Mistral Small 3"}, # Matches Mistral-Small-24B-Instruct-2501
        'Mistral Small 3.1 24B': {'mode': True, 'id': "mistralai/mistral-small-3.1-24b-instruct:free", 'name': "Mistral Small 3.1 24B"},
        'Molmo 7B D': {'mode': True, 'id': "allenai/molmo-7b-d:free", 'name': "Molmo 7B D"},
        'Moonlight 16B A3B Instruct': {'mode': True, 'id': "moonshotai/moonlight-16b-a3b-instruct:free", 'name': "Moonlight 16B A3B Instruct"},
        'Qwen2.5 72B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-72b-instruct:free", 'name': "Qwen2.5 72B Instruct"},
        'Qwen2.5 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-7b-instruct:free", 'name': "Qwen2.5 7B Instruct"},
        'Qwen2.5 Coder 32B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-coder-32b-instruct:free", 'name': "Qwen2.5 Coder 32B Instruct"},
        'Qwen2.5 VL 32B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-32b-instruct:free", 'name': "Qwen2.5 VL 32B Instruct"},
        'Qwen2.5 VL 3B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-3b-instruct:free", 'name': "Qwen2.5 VL 3B Instruct"},
        'Qwen2.5 VL 72B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-72b-instruct:free", 'name': "Qwen2.5 VL 72B Instruct"},
        'Qwen2.5-VL 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-vl-7b-instruct:free", 'name': "Qwen2.5-VL 7B Instruct"},
        'Qwerky 72B': {'mode': True, 'id': "featherless/qwerky-72b:free", 'name': "Qwerky 72B"},
        'QwQ 32B': {'mode': True, 'id': "qwen/qwq-32b:free", 'name': "QwQ 32B"},
        'QwQ 32B Preview': {'mode': True, 'id': "qwen/qwq-32b-preview:free", 'name': "QwQ 32B Preview"},
        'QwQ 32B RpR v1': {'mode': True, 'id': "arliai/qwq-32b-arliai-rpr-v1:free", 'name': "QwQ 32B RpR v1"},
        'R1': {'mode': True, 'id': "deepseek/deepseek-r1:free", 'name': "R1"}, # Matches DeepSeek-R1
        'R1 Distill Llama 70B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-llama-70b:free", 'name': "R1 Distill Llama 70B"},
        'R1 Distill Qwen 14B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-14b:free", 'name': "R1 Distill Qwen 14B"},
        'R1 Distill Qwen 32B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-32b:free", 'name': "R1 Distill Qwen 32B"},
        # Default models from the new list
        'Claude-sonnet-3.7': {'mode': True, 'id': "Claude-sonnet-3.7", 'name': "Claude-sonnet-3.7"},
        'Claude-sonnet-3.5': {'mode': True, 'id': "Claude-sonnet-3.5", 'name': "Claude-sonnet-3.5"},
        'DeepSeek-R1': {'mode': True, 'id': "deepseek-reasoner", 'name': "DeepSeek-R1"}, # This is 'R1' in openrouter, but 'DeepSeek-R1' in base models
        'Mistral-Small-24B-Instruct-2501': {'mode': True, 'id': "mistralai/Mistral-Small-24B-Instruct-2501", 'name': "Mistral-Small-24B-Instruct-2501"},
        # Add default_model if it's not covered and has an agent mode
        default_model: {'mode': True, 'id': "openai/gpt-4.1", 'name': default_model}, # Assuming GPT-4.1 is agent-compatible
        'o3-mini': {'mode': True, 'id': "o3-mini", 'name': "o3-mini"}, # Assuming o3-mini is agent-compatible
        'gpt-4.1-nano': {'mode': True, 'id': "gpt-4.1-nano", 'name': "gpt-4.1-nano"}, # Assuming gpt-4.1-nano is agent-compatible
    }

    # Trending agent modes
    trendingAgentMode = {
        'Python Agent': {'mode': True, 'id': "python"},
        'HTML Agent': {'mode': True, 'id': "html"},
        'Builder Agent': {'mode': True, 'id': "builder"},
        'Java Agent': {'mode': True, 'id': "java"},
        'JavaScript Agent': {'mode': True, 'id': "javascript"},
        'React Agent': {'mode': True, 'id': "react"},
        'Android Agent': {'mode': True, 'id': "android"},
        'Flutter Agent': {'mode': True, 'id': "flutter"},
        'Next.js Agent': {'mode': True, 'id': "next.js"},
        'AngularJS Agent': {'mode': True, 'id': "angularjs"},
        'Swift Agent': {'mode': True, 'id': "swift"},
        'MongoDB Agent': {'mode': True, 'id': "mongodb"},
        'PyTorch Agent': {'mode': True, 'id': "pytorch"},
        'Xcode Agent': {'mode': True, 'id': "xcode"},
        'Azure Agent': {'mode': True, 'id': "azure"},
        'Bitbucket Agent': {'mode': True, 'id': "bitbucket"},
        'DigitalOcean Agent': {'mode': True, 'id': "digitalocean"},
        'Docker Agent': {'mode': True, 'id': "docker"},
        'Electron Agent': {'mode': True, 'id': "electron"},
        'Erlang Agent': {'mode': True, 'id': "erlang"},
        'FastAPI Agent': {'mode': True, 'id': "fastapi"},
        'Firebase Agent': {'mode': True, 'id': "firebase"},
        'Flask Agent': {'mode': True, 'id': "flask"},
        'Git Agent': {'mode': True, 'id': "git"},
        'Gitlab Agent': {'mode': True, 'id': "gitlab"},
        'Go Agent': {'mode': True, 'id': "go"},
        'Godot Agent': {'mode': True, 'id': "godot"},
        'Google Cloud Agent': {'mode': True, 'id': "googlecloud"},
        'Heroku Agent': {'mode': True, 'id': "heroku"},
    }

    # Complete list of all models (for authorized users) - used for AVAILABLE_MODELS
    _all_models = list(dict.fromkeys([
        *models,  # Includes default_model, o3-mini, etc., and openrouter_models and agent names
        *list(agentMode.keys()), # Ensure all agentMode keys are included
        *list(trendingAgentMode.keys()) # Ensure all trendingAgentMode keys are included
    ]))

    AVAILABLE_MODELS = {name: name for name in _all_models}
    # Update AVAILABLE_MODELS to use names from agentMode if available
    for model_name_key in agentMode:
        if model_name_key in AVAILABLE_MODELS: # Check if the key from agentMode is in _all_models
            AVAILABLE_MODELS[model_name_key] = agentMode[model_name_key].get('name', model_name_key)


    # Model aliases for easier reference
    model_aliases = {
        "gpt-4": default_model, # default_model is "GPT-4.1"
        "gpt-4.1": default_model,
        "gpt-4o": default_model, # Defaulting to GPT-4.1 as per previous logic if specific GPT-4o handling isn't defined elsewhere
        "gpt-4o-mini": default_model, # Defaulting
        "claude-3.7-sonnet": "Claude-sonnet-3.7",
        "claude-3.5-sonnet": "Claude-sonnet-3.5",
        # "deepseek-r1": "DeepSeek-R1", # This is in base models, maps to R1 or DeepSeek R1 Zero in agentMode
        #
        "deepcoder-14b": "Deepcoder 14B Preview",
        "deephermes-3-8b": "DeepHermes 3 Llama 3 8B Preview",
        "deepseek-r1-zero": "DeepSeek R1 Zero",
        "deepseek-r1": "R1", # Alias for R1 (which is deepseek/deepseek-r1:free)
        "dolphin-3.0-24b": "Dolphin3.0 Mistral 24B",
        "dolphin-3.0-r1-24b": "Dolphin3.0 R1 Mistral 24B",
        "reka-flash": "Flash 3",
        "gemini-2.0-flash": "Gemini 2.0 Flash Experimental",
        "gemma-2-9b": "Gemma 2 9B",
        "gemma-3-12b": "Gemma 3 12B",
        "gemma-3-1b": "Gemma 3 1B",
        "gemma-3-27b": "Gemma 3 27B",
        "gemma-3-4b": "Gemma 3 4B",
        "kimi-vl-a3b-thinking": "Kimi VL A3B Thinking",
        "llama-3.1-8b": "Llama 3.1 8B Instruct",
        "nemotron-253b": "Llama 3.1 Nemotron Ultra 253B v1",
        "llama-3.2-11b": "Llama 3.2 11B Vision Instruct",
        "llama-3.2-1b": "Llama 3.2 1B Instruct",
        "llama-3.2-3b": "Llama 3.2 3B Instruct",
        "llama-3.3-70b": "Llama 3.3 70B Instruct",
        "nemotron-49b": "Llama 3.3 Nemotron Super 49B v1",
        "llama-4-maverick": "Llama 4 Maverick",
        "llama-4-scout": "Llama 4 Scout",
        "mistral-7b": "Mistral 7B Instruct",
        "mistral-nemo": "Mistral Nemo",
        "mistral-small-24b": "Mistral Small 3", # Alias for "Mistral Small 3"
        "mistral-small-24b-instruct-2501": "Mistral-Small-24B-Instruct-2501", # Specific name
        "mistral-small-3.1-24b": "Mistral Small 3.1 24B",
        "molmo-7b": "Molmo 7B D",
        "moonlight-16b": "Moonlight 16B A3B Instruct",
        "qwen-2.5-72b": "Qwen2.5 72B Instruct",
        "qwen-2.5-7b": "Qwen2.5 7B Instruct",
        "qwen-2.5-coder-32b": "Qwen2.5 Coder 32B Instruct",
        "qwen-2.5-vl-32b": "Qwen2.5 VL 32B Instruct",
        "qwen-2.5-vl-3b": "Qwen2.5 VL 3B Instruct",
        "qwen-2.5-vl-72b": "Qwen2.5 VL 72B Instruct",
        "qwen-2.5-vl-7b": "Qwen2.5-VL 7B Instruct",
        "qwerky-72b": "Qwerky 72B",
        "qwq-32b": "QwQ 32B",
        "qwq-32b-preview": "QwQ 32B Preview",
        "qwq-32b-arliai": "QwQ 32B RpR v1",
        "deepseek-r1-distill-llama-70b": "R1 Distill Llama 70B",
        "deepseek-r1-distill-qwen-14b": "R1 Distill Qwen 14B",
        "deepseek-r1-distill-qwen-32b": "R1 Distill Qwen 32B",
    }
    
    def __init__(
        self,
        proxies: dict = {},
        **kwargs: Any
    ):
        """
        Initialize the BlackboxAI provider with OpenAI compatibility.
        
        Args:
            api_key: Optional API key (included for compatibility but not used)
            tools: Optional list of tools (included for compatibility but not used)
            proxies: Optional proxy configuration
            **kwargs: Additional parameters
        """
        # Initialize session
        self.session = requests.Session()
        
        # Set headers based on GitHub reference
        self.headers = {
            'Accept': 'text/event-stream',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9,en-IN;q=0.8',
            'Content-Type': 'application/json',
            'DNT': '1',
            'Origin': 'https://www.blackbox.ai',
            'Referer': 'https://www.blackbox.ai/',
            'Sec-CH-UA': '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            'Sec-CH-UA-Mobile': '?0',
            'Sec-CH-UA-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
        }
        
        # Set cookies for the session
        self.cookies = {
            'cfzs_amplitude': self.generate_id(32),
            'cfz_amplitude': self.generate_id(32),
            '__cf_bm': self.generate_id(32),
        }
        
        # Set proxies if provided
        self.session.proxies = proxies
        
        # Initialize chat interface with completions
        self.chat = Chat(self)
    
    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()

    
    @classmethod
    def get_model(cls, model: str) -> str:
        """Resolve model name from alias."""
        # Convert to lowercase for case-insensitive matching
        model_lower = model.lower()

        # Check aliases (case-insensitive)
        for alias, target in cls.model_aliases.items():
            if model_lower == alias.lower():
                return target

        # If the model is directly in available models, return it
        for available_model in cls.AVAILABLE_MODELS:
            if model_lower == available_model.lower():
                return available_model

        # If we get here, use the default model
        return cls.default_model

    @classmethod
    def generate_random_string(cls, length: int = 8) -> str:
        """Generate a random string of specified length."""
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    
    @classmethod
    def generate_id(cls, length: int = 7) -> str:
        """Generate a random ID of specified length."""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    @classmethod
    def generate_session(cls, email: str, id_length: int = 21, days_ahead: int = 30) -> dict:
        """
        Generate a dynamic session with proper ID and expiry format using a specific email.

        Args:
            email: The email to use for this session
            id_length: Length of the numeric ID (default: 21)
            days_ahead: Number of days ahead for expiry (default: 30)

        Returns:
            dict: A session dictionary with user information and expiry
        """
        # Generate a random name
        first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Skyler", "Dakota"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
        name = f"{random.choice(first_names)} {random.choice(last_names)}"

        # Generate numeric ID - using Google-like ID format
        numeric_id = ''.join(random.choice('0123456789') for _ in range(id_length))

        # Generate future expiry date
        future_date = datetime.now() + timedelta(days=days_ahead)
        expiry = future_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        # Generate random image ID for the new URL format
        chars = string.ascii_letters + string.digits + "-"
        random_img_id = ''.join(random.choice(chars) for _ in range(48))
        image_url = f"https://lh3.googleusercontent.com/a/ACg8oc{random_img_id}=s96-c"

        return {
            "user": {
                "name": name,
                "email": email,
                "image": image_url,
                "id": numeric_id
            },
            "expires": expiry,
            "isNewUser": False
        }
        
    def create_request_payload(
        self,
        messages: List[Dict[str, Any]],
        chat_id: str,
        system_message: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        session_data: Dict[str, Any] = None,
        model: str = None
    ) -> Dict[str, Any]:
        """Create the full request payload for the BlackboxAI API."""
        # Get the correct model ID and agent mode
        model_name = self.get_model(model or self.default_model)
        agent_mode = self.agentMode.get(model_name, {})
        
        # Generate a random customer ID for the subscription
        customer_id = "cus_" + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(14))
        
        # Create the full request payload
        return {
            "messages": messages,
            "agentMode": agent_mode,
            "id": chat_id,
            "previewToken": None,
            "userId": None,
            "codeModelMode": True,
            "trendingAgentMode": {},
            "isMicMode": False,
            "userSystemPrompt": system_message,
            "maxTokens": max_tokens,
            "playgroundTopP": top_p,
            "playgroundTemperature": temperature,
            "isChromeExt": False,
            "githubToken": "",
            "clickedAnswer2": False,
            "clickedAnswer3": False,
            "clickedForceWebSearch": False,
            "visitFromDelta": False,
            "isMemoryEnabled": False,
            "mobileClient": False,
            "userSelectedModel": model_name if model_name in self.userSelectedModel else None,
            "validated": "00f37b34-a166-4efb-bce5-1312d87f2f94",
            "imageGenerationMode": False,
            "webSearchModePrompt": False,
            "deepSearchMode": False,
            "designerMode": False,
            "domains": None,
            "vscodeClient": False,
            "codeInterpreterMode": False,
            "customProfile": {
                "name": "",
                "occupation": "",
                "traits": [],
                "additionalInfo": "",
                "enableNewChats": False
            },
            "webSearchModeOption": {
                "autoMode": True,
                "webMode": False,
                "offlineMode": False
            },
            "session": session_data,
            "isPremium": True,
            "subscriptionCache": {
                "status": "PREMIUM",
                "customerId": customer_id,
                "expiryTimestamp": int((datetime.now() + timedelta(days=30)).timestamp()),
                "lastChecked": int(datetime.now().timestamp() * 1000),
                "isTrialSubscription": True
            },
            "beastMode": False,
            "reasoningMode": False,
            "designerMode": False,
            "workspaceId": ""
        }
