from datetime import datetime
import time
import uuid
import requests
import json
import random
from typing import List, Dict, Optional, Union, Generator, Any

# Import base classes and utility structures
from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage, count_tokens
)
from webscout.litagent import LitAgent
# ANSI escape codes for formatting
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"

class Completions(BaseCompletions):
    def __init__(self, client: 'OPKFC'):
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
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Create a chat completion with OPKFC API.

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
        # Use streaming implementation if requested
        if stream:
            return self._create_streaming(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                proxies=proxies,
                **kwargs
            )

        # Otherwise use non-streaming implementation
        return self._create_non_streaming(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            proxies=proxies,
            **kwargs
        )

    def _create_streaming(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Implementation for streaming chat completions."""
        try:
            # Generate request ID and timestamp
            request_id = str(uuid.uuid4())
            created_time = int(time.time())

            # Generate a random 6-digit auth token
            auth_token = str(random.randint(0, 999999)).zfill(6)

            # Prepare headers exactly as in the original script
            headers = {
                "Accept": "text/event-stream",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8",
                "Authorization": f"Bearer {auth_token}",
                "Cache-Control": "no-cache",
                "Content-Type": "application/json",
                "Cookie": self._client.cookie,
                "DNT": "1",
                "Origin": "https://www.opkfc.com",
                "Pragma": "no-cache",
                "Referer": "https://www.opkfc.com/",
                "Sec-CH-UA": "\"Microsoft Edge\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
                "Sec-CH-UA-Mobile": "?0",
                "Sec-CH-UA-Platform": "\"Windows\"",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Sec-GPC": "1",
                "User-Agent": self._client.user_agent,
                "openai-sentinel-chat-requirements-token": "0cb55714-5810-47d4-a9c0-648406004279"
            }

            # Prepare payload with individual messages
            payload = {
                "action": "next",
                "messages": [
                    {
                        "id": str(uuid.uuid4()),
                        "author": {"role": msg["role"]},
                        "content": {"content_type": "text", "parts": [msg["content"]]},
                        "create_time": time.time()
                    }
                    for msg in messages
                ],
                "parent_message_id": str(uuid.uuid4()),
                "model": model,
                "timezone_offset_min": -330,
                "timezone": "Asia/Calcutta"
            }

            # Add optional parameters if provided
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p

            # Make the streaming request
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None)
            )
            response.raise_for_status()

            # Process the streaming response
            content_buffer = ""
            response_started = False
            assistant_message_found = False

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if line.startswith("data:"):
                    part = line[len("data:"):].strip()

                    if part == "[DONE]":
                        break

                    try:
                        # Skip the delta_encoding event
                        if part == '"v1"':
                            continue

                        obj = json.loads(part)
                        if isinstance(obj, dict):
                            # Check if this is an assistant message
                            if isinstance(obj.get("v"), dict) and obj.get("v", {}).get("message", {}).get("author", {}).get("role") == "assistant":
                                assistant_message_found = True
                                # Reset content buffer when we find a new assistant message
                                content_buffer = ""
                                response_started = False
                                continue

                            # Skip until we find an assistant message
                            if not assistant_message_found:
                                continue

                            # Handle different response formats
                            content_to_add = None

                            # Format 1: Direct content in 'v' field
                            if isinstance(obj.get("v"), str):
                                content_to_add = obj["v"]

                            # Format 2: Path-based content with append operation
                            elif obj.get("p") == "/message/content/parts/0" and obj.get("o") == "append" and isinstance(obj.get("v"), str):
                                content_to_add = obj["v"]

                            # Format 3: Nested content in complex structure
                            elif isinstance(obj.get("v"), dict) and obj.get("v", {}).get("message", {}).get("content", {}).get("parts"):
                                parts = obj["v"]["message"]["content"]["parts"]
                                if parts and isinstance(parts[0], str):
                                    content_to_add = parts[0]

                            # Format 4: Patch operation with append to content
                            elif obj.get("o") == "patch" and isinstance(obj.get("v"), list):
                                for patch in obj["v"]:
                                    if patch.get("p") == "/message/content/parts/0" and patch.get("o") == "append" and isinstance(patch.get("v"), str):
                                        content_to_add = patch["v"]

                            # If we found content to add
                            if content_to_add:
                                # Skip the first part if it's repeating the user's message
                                if not response_started and content_buffer == "" and any(msg["content"] in content_to_add for msg in messages if msg["role"] == "user"):
                                    # This is likely the user's message being echoed back, skip it
                                    continue

                                response_started = True
                                content_buffer += content_to_add

                                # Create and yield a chunk
                                delta = ChoiceDelta(content=content_to_add)
                                choice = Choice(index=0, delta=delta, finish_reason=None)
                                chunk = ChatCompletionChunk(
                                    id=request_id,
                                    choices=[choice],
                                    created=created_time,
                                    model=model
                                )

                                yield chunk
                    except (ValueError, json.JSONDecodeError) as e:
                        print(f"{RED}Error parsing streaming response: {e} - {part}{RESET}")
                        pass

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
            print(f"{RED}Error during OPKFC streaming request: {e}{RESET}")
            raise IOError(f"OPKFC streaming request failed: {e}") from e

    def _create_non_streaming(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> ChatCompletion:
        """Implementation for non-streaming chat completions."""
        try:
            # Generate request ID and timestamp
            request_id = str(uuid.uuid4())
            created_time = int(time.time())

            # Generate a random 6-digit auth token
            auth_token = str(random.randint(0, 999999)).zfill(6)

            # Prepare headers exactly as in the original script
            headers = {
                "Accept": "text/event-stream",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8",
                "Authorization": f"Bearer {auth_token}",
                "Cache-Control": "no-cache",
                "Content-Type": "application/json",
                "Cookie": self._client.cookie,
                "DNT": "1",
                "Origin": "https://www.opkfc.com",
                "Pragma": "no-cache",
                "Referer": "https://www.opkfc.com/",
                "Sec-CH-UA": "\"Microsoft Edge\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
                "Sec-CH-UA-Mobile": "?0",
                "Sec-CH-UA-Platform": "\"Windows\"",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Sec-GPC": "1",
                "User-Agent": self._client.user_agent,
                "openai-sentinel-chat-requirements-token": "0cb55714-5810-47d4-a9c0-648406004279"
            }

            # Prepare payload with individual messages
            payload = {
                "action": "next",
                "messages": [
                    {
                        "id": str(uuid.uuid4()),
                        "author": {"role": msg["role"]},
                        "content": {"content_type": "text", "parts": [msg["content"]]},
                        "create_time": time.time()
                    }
                    for msg in messages
                ],
                "parent_message_id": str(uuid.uuid4()),
                "model": model,
                "timezone_offset_min": -330,
                "timezone": "Asia/Calcutta"
            }

            # Add optional parameters if provided
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p

            # Make the non-streaming request but process it as streaming
            # since the API only supports streaming responses
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None)
            )
            response.raise_for_status()

            # Process the streaming response to collect the full content
            full_content = ""
            response_started = False
            assistant_message_found = False

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if line.startswith("data:"):
                    part = line[len("data:"):].strip()

                    if part == "[DONE]":
                        break

                    try:
                        # Skip the delta_encoding event
                        if part == '"v1"':
                            continue

                        obj = json.loads(part)
                        if isinstance(obj, dict):
                            # Check if this is an assistant message
                            if isinstance(obj.get("v"), dict) and obj.get("v", {}).get("message", {}).get("author", {}).get("role") == "assistant":
                                assistant_message_found = True
                                # Reset content buffer when we find a new assistant message
                                full_content = ""
                                response_started = False
                                continue

                            # Skip until we find an assistant message
                            if not assistant_message_found:
                                continue

                            # Handle different response formats
                            content_to_add = None

                            # Format 1: Direct content in 'v' field
                            if isinstance(obj.get("v"), str):
                                content_to_add = obj["v"]

                            # Format 2: Path-based content with append operation
                            elif obj.get("p") == "/message/content/parts/0" and obj.get("o") == "append" and isinstance(obj.get("v"), str):
                                content_to_add = obj["v"]

                            # Format 3: Nested content in complex structure
                            elif isinstance(obj.get("v"), dict) and obj.get("v", {}).get("message", {}).get("content", {}).get("parts"):
                                parts = obj["v"]["message"]["content"]["parts"]
                                if parts and isinstance(parts[0], str):
                                    content_to_add = parts[0]

                            # Format 4: Patch operation with append to content
                            elif obj.get("o") == "patch" and isinstance(obj.get("v"), list):
                                for patch in obj["v"]:
                                    if patch.get("p") == "/message/content/parts/0" and patch.get("o") == "append" and isinstance(patch.get("v"), str):
                                        content_to_add = patch["v"]

                            # If we found content to add
                            if content_to_add:
                                # Skip the first part if it's repeating the user's message
                                if not response_started and full_content == "" and any(msg["content"] in content_to_add for msg in messages if msg["role"] == "user"):
                                    # This is likely the user's message being echoed back, skip it
                                    continue

                                response_started = True
                                full_content += content_to_add
                    except (ValueError, json.JSONDecodeError) as e:
                        print(f"{RED}Error parsing non-streaming response: {e} - {part}{RESET}")
                        pass

            # Create the completion message
            message = ChatCompletionMessage(
                role="assistant",
                content=full_content
            )

            # Create the choice
            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )

            # Estimate token usage using count_tokens
            prompt_tokens = count_tokens([msg.get("content", "") for msg in messages])
            completion_tokens = count_tokens(full_content)
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
            print(f"{RED}Error during OPKFC non-stream request: {e}{RESET}")
            raise IOError(f"OPKFC request failed: {e}") from e

class Chat(BaseChat):
    def __init__(self, client: 'OPKFC'):
        self.completions = Completions(client)

class OPKFC(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for OPKFC API.

    Usage:
        client = OPKFC()
        response = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    """

    AVAILABLE_MODELS = [
        "auto",
        "o4-mini",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-1-mini",
        
    ]

    def __init__(
        self,
        timeout: int = 30,
        proxies: dict = {}
    ):
        """
        Initialize the OPKFC client.

        Args:
            timeout: Request timeout in seconds
            proxies: Optional proxy configuration
        """
        self.timeout = timeout
        self.api_endpoint = "https://www.opkfc.com/backend-api/conversation"
        self.proxies = proxies

        # Initialize session
        self.session = requests.Session()
        if proxies:
            self.session.proxies.update(proxies)

        # Set the user agent to match the original script
        self.user_agent = LitAgent().random()

        # Set the cookie from the original script
        self.cookie = f"__vtins__KUc0LhjVWFNXQv11=%7B%22sid%22%3A%20%{uuid.uuid4().hex}%22%2C%20%22vd%22%3A%201%2C%20%22stt%22%3A%200%2C%20%22dr%22%3A%200%2C%20%22expires%22%3A%201744896723481%2C%20%22ct%22%3A%201744894923481%7D; __51uvsct__KUc0LhjVWFNXQv11=1; __51vcke__KUc0LhjVWFNXQv11=06da852c-bb56-547c-91a8-43a0d485ffed; __51vuft__KUc0LhjVWFNXQv11=1744894923504; gfsessionid=1ochrgv17vy4sbd98xmwt6crpmkxwlqf; oai-nav-state=1; p_uv_id=ad86646801bc60d6d95f6098e4ee7450; _dd_s=rum=0&expire=1744895920821&logs=1&id={uuid.uuid4().hex}&created={int(datetime.utcnow().timestamp() * 1000)}"

        # Initialize chat interface
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()

if __name__ == "__main__":
    # Example usage
    client = OPKFC()
    response = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)