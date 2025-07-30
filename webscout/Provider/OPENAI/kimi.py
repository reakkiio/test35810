import requests
import json
import time
import uuid
import random
from typing import List, Dict, Optional, Union, Generator, Any

# Import curl_cffi for improved request handling
from curl_cffi.requests import Session
from curl_cffi import CurlError

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChatCompletionMessage,
    ChoiceDelta,
    CompletionUsage,
    format_prompt,
    get_system_prompt,
    count_tokens
)

# Attempt to import LitAgent, fallback if not available
try:
    from webscout.litagent import LitAgent
except ImportError:
    pass

from webscout import exceptions


class Completions(BaseCompletions):
    def __init__(self, client: 'Kimi'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 4000,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Create a completion using the Kimi API.
        
        Args:
            model: The model to use (k1.5, k2, k1.5-thinking)
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens for response
            stream: Whether to stream the response
            temperature: Sampling temperature (not used by Kimi)
            top_p: Top-p sampling (not used by Kimi)
            timeout: Request timeout
            proxies: Proxy configuration
            
        Returns:
            ChatCompletion or generator of ChatCompletionChunk
        """
        # Validate model
        if model not in self._client.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self._client.AVAILABLE_MODELS}")

        # Ensure authentication and chat creation
        self._client._authenticate()
        self._client._create_chat()

        # Format messages exactly like the original Kimi.py
        # Use the first user message content directly, no formatting needed
        user_content = ""
        system_content = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
        
        # If we have system content, prepend it to user content
        if system_content:
            final_content = f"{system_content}\n\n{user_content}"
        else:
            final_content = user_content

        # Create payload exactly like the original Kimi.py
        payload = {
            "kimiplus_id": "kimi",
            "extend": {"sidebar": True},
            "model": model,
            "use_search": self._client.web_search,
            "messages": [
                {
                    "role": "user",
                    "content": final_content
                }
            ],
            "refs": [],
            "history": [],
            "scene_labels": [],
            "use_semantic_memory": False,
            "use_deep_research": False
        }

        # Generate request ID and timestamp
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, messages, payload, timeout, proxies)
        else:
            return self._create_non_stream(request_id, created_time, model, messages, payload, timeout, proxies)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, messages: List[Dict[str, str]], 
        payload: Dict[str, Any], timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._client.session.post(
                self._client.chat_completion_endpoint.format(chat_id=self._client.chat_id),
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                impersonate="chrome110"
            )
            response.raise_for_status()

            # Calculate prompt tokens using the messages parameter
            prompt_tokens = count_tokens(messages)
            completion_tokens = 0
            total_tokens = prompt_tokens

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str == "[DONE]":
                            break

                        try:
                            data = json.loads(json_str)
                            if data.get("event") == "cmpl":
                                content = data.get("text")
                                if content:
                                    completion_tokens += count_tokens(content)
                                    total_tokens = prompt_tokens + completion_tokens

                                    delta = ChoiceDelta(content=content, role=None, tool_calls=None)
                                    choice = Choice(index=0, delta=delta, finish_reason=None, logprobs=None)
                                    chunk = ChatCompletionChunk(
                                        id=request_id,
                                        choices=[choice],
                                        created=created_time,
                                        model=model,
                                        system_fingerprint=None
                                    )

                                    # Add usage information
                                    chunk.usage = {
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": total_tokens,
                                        "estimated_cost": None
                                    }

                                    yield chunk
                        except json.JSONDecodeError:
                            continue

            # Final chunk with finish_reason="stop"
            delta = ChoiceDelta(content=None, role=None, tool_calls=None)
            choice = Choice(index=0, delta=delta, finish_reason="stop", logprobs=None)
            chunk = ChatCompletionChunk(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
                system_fingerprint=None
            )
            chunk.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": None
            }
            yield chunk

        except CurlError as e:
            print(f"Error during Kimi stream request: {e}")
            raise IOError(f"Kimi request failed: {e}") from e
        except Exception as e:
            print(f"Error processing Kimi stream: {e}")
            raise

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, messages: List[Dict[str, str]],
        payload: Dict[str, Any], timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> ChatCompletion:
        try:
            response = self._client.session.post(
                self._client.chat_completion_endpoint.format(chat_id=self._client.chat_id),
                json=payload,
                timeout=timeout or self._client.timeout,
                impersonate="chrome110",
                stream=True
            )
            response.raise_for_status()

            # Collect all streaming data
            full_text = ""
            for line in response.iter_lines():
                if line:
                    # Decode bytes to string if needed
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str == "[DONE]":
                            break

                        try:
                            data = json.loads(json_str)
                            if data.get("event") == "cmpl":
                                content = data.get("text")
                                if content:
                                    full_text += content
                        except json.JSONDecodeError:
                            continue

            # Create the message object
            message = ChatCompletionMessage(
                role="assistant",
                content=full_text
            )

            # Create the choice object
            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )

            # Create usage object with proper token counting
            prompt_tokens = count_tokens(messages)
            completion_tokens = count_tokens(full_text) if full_text else 0
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

        except CurlError as e:
            print(f"Error during Kimi request: {e}")
            raise IOError(f"Kimi request failed: {e}") from e
        except Exception as e:
            print(f"Error processing Kimi response: {e}")
            raise


class Chat(BaseChat):
    def __init__(self, client: 'Kimi'):
        self._client = client
        self.completions = Completions(client)


class Kimi(OpenAICompatibleProvider):
    """
    OpenAI-compatible Kimi provider.
    
    This provider implements the OpenAI API interface for Kimi models.
    It supports the following models:
    - k1.5
    - k2
    - k1.5-thinking
    
    Examples:
        >>> from webscout.Provider.OPENAI.kimi import Kimi
        >>> client = Kimi()
        >>> response = client.chat.completions.create(
        ...     model="k2",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.choices[0].message.content)
    """
    
    AVAILABLE_MODELS = ["k1.5", "k2", "k1.5-thinking"]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        proxies: Optional[Dict[str, str]] = None,
        browser: str = "chrome",
        web_search: bool = False,
        **kwargs
    ):
        """
        Initialize the Kimi provider.
        
        Args:
            api_key: Not used for Kimi (authentication is handled via device registration)
            base_url: Not used for Kimi
            timeout: Request timeout in seconds
            proxies: Proxy configuration
            browser: Browser to impersonate
            web_search: Whether to enable web search
            **kwargs: Additional arguments
        """
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        
        self.timeout = timeout
        self.proxies = proxies or {}
        self.web_search = web_search
        
        # Kimi API endpoints
        self.register_endpoint = "https://www.kimi.com/api/device/register"
        self.chat_create_endpoint = "https://www.kimi.com/api/chat"
        self.chat_completion_endpoint = "https://www.kimi.com/api/chat/{chat_id}/completion/stream"
        
        # Initialize session
        self.session = Session()
        self.session.proxies = self.proxies
        
        # Initialize LitAgent for browser fingerprinting
        try:
            self.agent = LitAgent()
            self.fingerprint = self.agent.generate_fingerprint(browser)
        except:
            self.fingerprint = {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "accept_language": "en-US,en;q=0.9",
                "sec_ch_ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "platform": "Windows"
            }
        
        # Generate device ID
        self.device_id = str(random.randint(1000000000000000, 9999999999999999))
        
        # Headers for Kimi API
        self.headers = {
            "Accept": "text/event-stream",
            "Accept-Language": self.fingerprint["accept_language"],
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "DNT": "1",
            "Origin": "https://www.kimi.com",
            "Pragma": "no-cache",
            "Referer": "https://www.kimi.com/",
            "Sec-CH-UA": self.fingerprint["sec_ch_ua"],
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{self.fingerprint["platform"]}"',
            "User-Agent": self.fingerprint["user_agent"],
            "x-msh-device-id": self.device_id,
            "x-msh-platform": "web",
            "x-traffic-id": self.device_id,
        }
        
        # Initialize authentication
        self.access_token = None
        self.chat_id = None
        
        # Update session headers
        self.session.headers.update(self.headers)
        
        # Initialize chat and completions
        self.chat = Chat(self)
        self.completions = Completions(self)
    
    @property
    def models(self):
        """Property that returns an object with a .list() method returning available models."""
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()
    
    def _authenticate(self) -> str:
        """Authenticate with Kimi API and get access token."""
        if self.access_token:
            return self.access_token
        
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.register_endpoint,
                    json={},
                    timeout=self.timeout,
                    impersonate="chrome110"
                )
                response.raise_for_status()
                
                data = response.json()
                if not data.get("access_token"):
                    raise exceptions.FailedToGenerateResponseError("No access token received")
                    
                self.access_token = data["access_token"]
                self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                return self.access_token
                
            except CurlError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    continue
                raise exceptions.FailedToGenerateResponseError(f"Authentication failed after {max_retries} attempts (CurlError): {e}")
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    continue
                raise exceptions.FailedToGenerateResponseError(f"Authentication failed after {max_retries} attempts: {e}")
    
    def _create_chat(self) -> str:
        """Create a new chat session and return chat ID."""
        if self.chat_id:
            return self.chat_id
            
        self._authenticate()
        
        try:
            response = self.session.post(
                self.chat_create_endpoint,
                json={
                    "name": "Unnamed Chat",
                    "born_from": "home",
                    "kimiplus_id": "kimi",
                    "is_example": False,
                    "source": "web",
                    "tags": []
                },
                timeout=self.timeout,
                impersonate="chrome110"
            )
            response.raise_for_status()
            
            data = response.json()
            self.chat_id = data.get("id")
            if not self.chat_id:
                raise exceptions.FailedToGenerateResponseError("No chat ID received")
                
            return self.chat_id
            
        except CurlError as e:
            raise exceptions.FailedToGenerateResponseError(f"Chat creation failed (CurlError): {e}")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Chat creation failed: {e}")