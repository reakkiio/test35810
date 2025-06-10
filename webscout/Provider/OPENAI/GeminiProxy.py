import json
import time
import uuid
import base64
from typing import List, Dict, Optional, Union, Generator, Any

import requests
from uuid import uuid4

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage, count_tokens
)

from webscout.litagent import LitAgent
from webscout import exceptions


class Completions(BaseCompletions):
    def __init__(self, client: 'GeminiProxy'):
        self._client = client
    
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[dict] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Create a chat completion with GeminiProxy API.
        
        Args:
            model: The model to use (from AVAILABLE_MODELS)
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response (not supported by GeminiProxy)
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            timeout: Request timeout in seconds
            proxies: Proxy configuration
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            If stream=False, returns a ChatCompletion object
            If stream=True, returns a Generator yielding ChatCompletionChunk objects
        """
        # Generate request ID and timestamp
        request_id = str(uuid.uuid4())
        created_time = int(time.time())
        
        # Extract image URL from kwargs if present
        img_url = kwargs.get('img_url')
        
        # Convert messages to GeminiProxy format
        conversation_prompt = self._format_messages(messages)
        
        # Prepare parts for the request
        parts = []
        if img_url:
            parts.append({"inline_data": self._get_image(img_url, proxies, timeout)})
        parts.append({"text": conversation_prompt})
        
        # Prepare the payload
        payload = {
            "model": model,
            "contents": [{"parts": parts}]
        }
        
        # GeminiProxy doesn't support streaming, so we always return non-streaming
        if stream:
            return self._create_streaming_fallback(
                request_id=request_id,
                created_time=created_time,
                model=model,
                payload=payload,
                timeout=timeout,
                proxies=proxies
            )
        
        # Non-streaming implementation
        return self._create_non_streaming(
            request_id=request_id,
            created_time=created_time,
            model=model,
            payload=payload,
            timeout=timeout,
            proxies=proxies
        )
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI messages format to a single conversation prompt."""
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n".join(formatted_parts)
    
    def _get_image(self, img_url: str, proxies: Optional[dict] = None, timeout: Optional[int] = None) -> Dict[str, str]:
        """Fetch and encode image from URL."""
        try:
            session = requests.Session()
            if proxies:
                session.proxies.update(proxies)
            
            response = session.get(
                img_url, 
                stream=True, 
                timeout=timeout or self._client.timeout
            )
            response.raise_for_status()
            
            mime_type = response.headers.get("content-type", "application/octet-stream")
            data = base64.b64encode(response.content).decode("utf-8")
            return {"mime_type": mime_type, "data": data}
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Error fetching image: {e}")
    
    def _create_non_streaming(
        self,
        *,
        request_id: str,
        created_time: int,
        model: str,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        proxies: Optional[dict] = None
    ) -> ChatCompletion:
        """Implementation for non-streaming chat completions."""
        original_proxies = self._client.session.proxies.copy()
        if proxies is not None:
            self._client.session.proxies.update(proxies)
        
        try:
            response = self._client.session.post(
                self._client.base_url,
                json=payload,
                headers=self._client.headers,
                timeout=timeout if timeout is not None else self._client.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract content from GeminiProxy response
            content = self._extract_content(data)
            
            # Create the completion message
            message = ChatCompletionMessage(
                role="assistant",
                content=content
            )
            
            # Create the choice
            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )
            
            # Estimate token usage
            prompt_tokens = count_tokens([msg.get("content", "") for msg in payload.get("contents", [{}])[0].get("parts", [{}])])
            completion_tokens = count_tokens(content)
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            # Create the completion object
            completion = ChatCompletion(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
                usage=usage,
            )
            
            return completion
            
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"GeminiProxy request failed: {e}")
        finally:
            if proxies is not None:
                self._client.session.proxies = original_proxies
    
    def _create_streaming_fallback(
        self,
        *,
        request_id: str,
        created_time: int,
        model: str,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        proxies: Optional[dict] = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Fallback streaming implementation that simulates streaming from non-streaming response."""
        # Get the full response first
        completion = self._create_non_streaming(
            request_id=request_id,
            created_time=created_time,
            model=model,
            payload=payload,
            timeout=timeout,
            proxies=proxies
        )
        
        # Simulate streaming by yielding chunks
        content = completion.choices[0].message.content
        if content:
            # Split content into chunks (simulate streaming)
            chunk_size = max(1, len(content) // 10)  # Split into ~10 chunks
            for i in range(0, len(content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                
                delta = ChoiceDelta(content=chunk_content)
                choice = Choice(index=0, delta=delta, finish_reason=None)
                
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
    
    def _extract_content(self, response: dict) -> str:
        """Extract content from GeminiProxy response."""
        try:
            return response['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError, TypeError):
            return str(response)


class Chat(BaseChat):
    def __init__(self, client: 'GeminiProxy'):
        self.completions = Completions(client)


class GeminiProxy(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for GeminiProxy API.
    
    Usage:
        client = GeminiProxy()
        response = client.chat.completions.create(
            model="gemini-2.0-flash-lite",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    """
    
    AVAILABLE_MODELS = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash-preview-05-20",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,  # Not used but included for compatibility
        browser: str = "chrome",
        **kwargs: Any
    ):
        """
        Initialize the GeminiProxy client.
        
        Args:
            api_key: Not used but included for compatibility with OpenAI interface
            browser: Browser type for fingerprinting
            **kwargs: Additional parameters
        """
        super().__init__(api_key=api_key, **kwargs)
        
        self.timeout = 30
        self.base_url = "https://us-central1-infinite-chain-295909.cloudfunctions.net/gemini-proxy-staging-v1"
        
        # Initialize LitAgent for fingerprinting
        self.agent = LitAgent()
        self.fingerprint = self.agent.generate_fingerprint(browser)
        
        # Initialize session
        self.session = requests.Session()
        self.headers = self.fingerprint.copy()
        self.session.headers.update(self.headers)
        self.session.proxies = {}
        
        # Initialize chat interface
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()
