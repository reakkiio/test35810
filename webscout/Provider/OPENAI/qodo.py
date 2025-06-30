"""
QodoAI Provider for OpenAI-compatible API

This module provides a QodoAI implementation that follows the OpenAI API interface.
QodoAI offers access to various models including GPT-4, Claude, and others through a unified API.

Usage:
    from webscout.Provider.OPENAI.qodo import QodoAI
    
    # Initialize with API key
    client = QodoAI(api_key="your_qodo_api_key_here")
    
    # List available models
    models = client.models
    print("Available models:", models)
    
    # Create chat completion
    response = client.chat.completions.create(
        model="claude-4-sonnet",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=False
    )
    print(response.choices[0].message.content)
    
    # Streaming example
    for chunk in client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True
    ):
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

Getting API Key:
    To get a QodoAI API key, follow the instructions at:
    https://docs.qodo.ai/qodo-documentation/qodo-gen-cli/getting-started/setup-and-quickstart
"""

import json
import time
import uuid
from typing import List, Dict, Optional, Union, Generator, Any

# Import curl_cffi for improved request handling
from curl_cffi.requests import Session
from curl_cffi import CurlError

# Import base classes and utility structures
from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions, Tool
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage, ModelData, ModelList
)

# Import webscout utilities
from webscout.AIutel import sanitize_stream
from webscout import exceptions

# Attempt to import LitAgent, fallback if not available
try:
    from webscout.litagent import LitAgent
except ImportError:
    LitAgent = None


class Completions(BaseCompletions):
    def __init__(self, client: 'QodoAI'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = 2049,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        proxies: Optional[dict] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Creates a model response for the given chat conversation.
        Mimics openai.chat.completions.create
        """
        # Get the last user message for the prompt
        user_prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_prompt = message.get("content", "")
                break

        if not user_prompt:
            raise ValueError("No user message found in messages")

        # Build payload for Qodo API
        payload = self._client._build_payload(user_prompt, model)
        payload["stream"] = stream
        payload["custom_model"] = model

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload, user_prompt)
        else:
            return self._create_non_stream(request_id, created_time, model, payload, user_prompt)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any], user_prompt: str
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._client.session.post(
                self._client.url,
                json=payload,
                stream=True,
                timeout=self._client.timeout,
                impersonate=self._client.fingerprint.get("browser_type", "chrome110")
            )

            if response.status_code == 401:
                raise exceptions.FailedToGenerateResponseError(
                    "Invalid API key. You need to provide your own API key.\n"
                    "Usage: QodoAI(api_key='your_api_key_here')\n"
                    "To get an API key, install Qodo CLI via: https://docs.qodo.ai/qodo-documentation/qodo-gen-cli/getting-started/setup-and-quickstart"
                )
            elif response.status_code != 200:
                raise IOError(f"Qodo request failed with status code {response.status_code}: {response.text}")

            # Track token usage
            prompt_tokens = len(user_prompt.split())
            completion_tokens = 0
            
            processed_stream = sanitize_stream(
                data=response.iter_content(chunk_size=None),
                intro_value="",
                to_json=True,
                skip_markers=["[DONE]"],
                content_extractor=QodoAI._qodo_extractor,
                yield_raw_on_error=True,
                raw=False
            )

            for content_chunk in processed_stream:
                if content_chunk:
                    completion_tokens += len(content_chunk.split())
                    
                    # Create the delta object
                    delta = ChoiceDelta(
                        content=content_chunk,
                        role="assistant"
                    )

                    # Create the choice object
                    choice = Choice(
                        index=0,
                        delta=delta,
                        finish_reason=None
                    )

                    # Create the chunk object
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        choices=[choice],
                        created=created_time,
                        model=model,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    )

                    yield chunk

            # Send final chunk with finish_reason
            final_choice = Choice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop"
            )

            final_chunk = ChatCompletionChunk(
                id=request_id,
                choices=[final_choice],
                created=created_time,
                model=model,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )

            yield final_chunk

        except CurlError as e:
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}")

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any], user_prompt: str
    ) -> ChatCompletion:
        try:
            payload["stream"] = False
            response = self._client.session.post(
                self._client.url,
                json=payload,
                timeout=self._client.timeout,
                impersonate=self._client.fingerprint.get("browser_type", "chrome110")
            )

            if response.status_code == 401:
                raise exceptions.FailedToGenerateResponseError(
                    "Invalid API key. You need to provide your own API key.\n"
                    "Usage: QodoAI(api_key='your_api_key_here')\n"
                    "To get an API key, install Qodo CLI via: https://docs.qodo.ai/qodo-documentation/qodo-gen-cli/getting-started/setup-and-quickstart"
                )
            elif response.status_code != 200:
                raise IOError(f"Qodo request failed with status code {response.status_code}: {response.text}")

            response_text = response.text
            
            # Parse multiple JSON objects from the response
            full_response = ""
            
            # Try to split by line breaks and parse each potential JSON object
            lines = response_text.replace('}\n{', '}\n{').split('\n')
            json_objects = []
            
            current_json = ""
            brace_count = 0
            
            for line in lines:
                line = line.strip()
                if line:
                    current_json += line
                    
                    # Count braces to detect complete JSON objects
                    brace_count += line.count('{') - line.count('}')
                    
                    if brace_count == 0 and current_json:
                        json_objects.append(current_json)
                        current_json = ""
            
            # Add any remaining JSON
            if current_json and brace_count == 0:
                json_objects.append(current_json)
            
            for json_str in json_objects:
                if json_str.strip():
                    try:
                        json_obj = json.loads(json_str)
                        content = QodoAI._qodo_extractor(json_obj)
                        if content:
                            full_response += content
                    except json.JSONDecodeError:
                        # Silently skip malformed JSON
                        pass

            # Calculate token usage
            prompt_tokens = len(user_prompt.split())
            completion_tokens = len(full_response.split())
            total_tokens = prompt_tokens + completion_tokens

            # Create the message object
            message = ChatCompletionMessage(
                role="assistant",
                content=full_response
            )

            # Create the choice object
            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )

            # Create the usage object
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )

            # Create the completion object
            completion = ChatCompletion(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
                usage=usage
            )

            return completion

        except CurlError as e:
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {e}")


class Chat(BaseChat):
    def __init__(self, client: 'QodoAI'):
        self.completions = Completions(client)


class QodoAI(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for Qodo AI API.
    
    Usage:
        client = QodoAI(api_key="your_api_key")
        response = client.chat.completions.create(
            model="claude-4-sonnet",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """

    AVAILABLE_MODELS = [
        "gpt-4.1",
        "gpt-4o",
        "o3",
        "o4-mini", 
        "claude-4-sonnet",
        "gemini-2.5-pro"
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        proxies: Optional[dict] = None,
        disable_auto_proxy: bool = False,
        timeout: int = 30,
        browser: str = "chrome",
        **kwargs: Any
    ):
        # Initialize parent class
        super().__init__(api_key=api_key, tools=tools, proxies=proxies, disable_auto_proxy=disable_auto_proxy, **kwargs)
        
        self.url = "https://api.cli.qodo.ai/v2/agentic/start-task"
        self.info_url = "https://api.cli.qodo.ai/v2/info/get-things"
        self.timeout = timeout
        # Store API key
        self.api_key = api_key or "sk-dS7U-extxMWUxc8SbYYOuncqGUIE8-y2OY8oMCpu0eI-qnSUyH9CYWO_eAMpqwfMo7pXU3QNrclfZYMO0M6BJTM"
        
        # Initialize LitAgent for user agent generation
        if LitAgent:
            self.agent = LitAgent()
            self.fingerprint = self.agent.generate_fingerprint(browser)
        else:
            self.fingerprint = {"user_agent": "axios/1.10.0", "browser_type": "chrome"}
        
        # Generate session ID dynamically from API
        self.session_id = self._get_session_id()
        self.request_id = str(uuid.uuid4())
        
        # Use the fingerprint for headers
        self.headers = {
            "Accept": "text/plain",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": self.fingerprint.get("accept_language", "en-US,en;q=0.9"),
            "Authorization": f"Bearer {self.api_key}",
            "Connection": "close",
            "Content-Type": "application/json",
            "host": "api.cli.qodo.ai",
            "Request-id": self.request_id,
            "Session-id": self.session_id,
            "User-Agent": self.fingerprint["user_agent"],
        }
        
        # Initialize curl_cffi Session
        self.session = Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        
        # Initialize OpenAI-compatible interface
        self.chat = Chat(self)
        
    @property
    def models(self):
        """Property that returns an object with a .list() method returning available models."""
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()

    @staticmethod
    def _qodo_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from Qodo stream JSON objects."""
        if isinstance(chunk, dict):
            # Check for Qodo's specific response format
            data = chunk.get("data", {})
            if isinstance(data, dict):
                tool_args = data.get("tool_args", {})
                if isinstance(tool_args, dict):
                    content = tool_args.get("content")
                    if content:
                        return content
                
                # Check for direct content in data
                if "content" in data:
                    return data["content"]
            
            # Check for OpenAI-like format (choices)
            if "choices" in chunk:
                choices = chunk["choices"]
                if isinstance(choices, list) and len(choices) > 0:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        # Check delta content
                        delta = choice.get("delta", {})
                        if isinstance(delta, dict) and "content" in delta:
                            return delta["content"]
                        
                        # Check message content
                        message = choice.get("message", {})
                        if isinstance(message, dict) and "content" in message:
                            return message["content"]
        
        elif isinstance(chunk, str):
            # Try to parse as JSON if it's a string
            try:
                parsed = json.loads(chunk)
                return QodoAI._qodo_extractor(parsed)
            except json.JSONDecodeError:
                # If it's not JSON, it might be direct content
                if chunk.strip():
                    return chunk.strip()
        
        return None

    def _get_session_id(self) -> str:
        """Get session ID from Qodo API."""
        try:
            # Create temporary session for the info request
            temp_session = Session()
            temp_headers = {
                "Accept": "text/plain",
                "Accept-Encoding": "gzip, deflate, br",
                "Authorization": f"Bearer {self.api_key}",
                "Connection": "close",
                "Content-Type": "application/json",
                "host": "api.cli.qodo.ai",
                "Request-id": str(uuid.uuid4()),
                "User-Agent": self.fingerprint.get("user_agent", "axios/1.10.0"),
            }
            temp_session.headers.update(temp_headers)
            
            response = temp_session.get(
                self.info_url,
                timeout=self.timeout,
                impersonate="chrome110"
            )
            
            if response.status_code == 200:
                data = response.json()
                session_id = data.get("session-id")
                if session_id:
                    return session_id
            elif response.status_code == 401:
                # API key is invalid
                raise exceptions.FailedToGenerateResponseError(
                    "Invalid API key. You need to provide your own API key.\n"
                    "Usage: QodoAI(api_key='your_api_key_here')\n"
                    "To get an API key, install Qodo CLI via: https://docs.qodo.ai/qodo-documentation/qodo-gen-cli/getting-started/setup-and-quickstart"
                )
            else:
                # Other HTTP errors
                raise exceptions.FailedToGenerateResponseError(
                    f"Failed to authenticate with Qodo API (HTTP {response.status_code}). "
                    "You may need to provide your own API key.\n"
                    "Usage: QodoAI(api_key='your_api_key_here')\n"
                    "To get an API key, install Qodo CLI via: https://docs.qodo.ai/qodo-documentation/qodo-gen-cli/getting-started/setup-and-quickstart"
                )
                    
            # Fallback to generated session ID if API call fails
            return f"20250630-{str(uuid.uuid4())}"
            
        except exceptions.FailedToGenerateResponseError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # For other errors, show the API key message
            raise exceptions.FailedToGenerateResponseError(
                f"Failed to connect to Qodo API: {e}\n"
                "You may need to provide your own API key.\n"
                "Usage: QodoAI(api_key='your_api_key_here')\n"
                "To get an API key, install Qodo CLI via: https://docs.qodo.ai/qodo-documentation/qodo-gen-cli/getting-started/setup-and-quickstart"
            )

    def _build_payload(self, prompt: str, model: str = "claude-4-sonnet"):
        """Build the payload for Qodo AI API."""
        return {
            "agent_type": "cli",
            "session_id": self.session_id,
            "user_data": {
                "extension_version": "0.7.2",
                "os_platform": "win32",
                "os_version": "v23.9.0",
                "editor_type": "cli"
            },
            "tools": {
                "web_search": [
                    {
                        "name": "web_search",
                        "description": "Searches the web and returns results based on the user's query (Powered by Nimble).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "llm_description": {
                                    "default": "Searches the web and returns results based on the user's query.",
                                    "title": "Llm Description",
                                    "type": "string"
                                },
                                "query": {
                                    "description": "The search query to execute",
                                    "title": "Query",
                                    "type": "string"
                                }
                            },
                            "required": ["query"],
                            "title": "NimbleWebSearch"
                        },
                        "be_tool": True,
                        "autoApproved": True
                    },
                    {
                        "name": "web_fetch",
                        "description": "Fetches content from a given URL (Powered by Nimble).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "llm_description": {
                                    "default": "Fetches content from a given URL.",
                                    "title": "Llm Description",
                                    "type": "string"
                                },
                                "url": {
                                    "description": "The URL to fetch content from",
                                    "title": "Url",
                                    "type": "string"
                                }
                            },
                            "required": ["url"],
                            "title": "NimbleWebFetch"
                        },
                        "be_tool": True,
                        "autoApproved": True
                    }
                ]
            },
            "user_request": prompt,
            "execution_strategy": "act",
            "custom_model": model,
            "stream": True
        }

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get available models and info from Qodo API.
        
        Returns:
            Dict containing models, default_model, version, and session info
        """
        try:
            response = self.session.get(
                self.info_url,
                timeout=self.timeout,
                impersonate=self.fingerprint.get("browser_type", "chrome110")
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise exceptions.FailedToGenerateResponseError(
                    "Invalid API key. You need to provide your own API key.\n"
                    "Usage: QodoAI(api_key='your_api_key_here')\n"
                    "To get an API key, install Qodo CLI via: https://docs.qodo.ai/qodo-documentation/qodo-gen-cli/getting-started/setup-and-quickstart"
                )
            else:
                raise exceptions.FailedToGenerateResponseError(f"Failed to get models: HTTP {response.status_code}")
                
        except CurlError as e:
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Failed to get models ({type(e).__name__}): {e}")


if __name__ == "__main__":
    # Example usage
    client = QodoAI()  # You will need to provide your API key here
    
    # List available models
    models = client.models.list()
    print("Available models:")
    for model in models:
        print(f"  - {model}")
    
    # Create a chat completion
    response = client.chat.completions.create(
        model="claude-4-sonnet",
        messages=[
            {"role": "user", "content": "Write a short poem about AI"}
        ],
        stream=False
    )
    
    print(f"\nResponse: {response}")
    
    # Example with streaming
    print("\nStreaming response:")
    stream = client.chat.completions.create(
        model="claude-4-sonnet",
        messages=[
            {"role": "user", "content": "Count from 1 to 5"}
        ],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print()
