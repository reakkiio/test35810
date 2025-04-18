import time
import uuid
import requests
import json
from typing import List, Dict, Optional, Union, Generator, Any

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)

# Import LitAgent for browser fingerprinting
from webscout.litagent import LitAgent

# ANSI escape codes for formatting
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"

class Completions(BaseCompletions):
    def __init__(self, client: 'TextPollinations'):
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
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Creates a model response for the given chat conversation.
        Mimics openai.chat.completions.create
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        payload.update(kwargs)

        request_id = str(uuid.uuid4())
        created_time = int(time.time())

        if stream:
            return self._create_streaming(request_id, created_time, model, payload)
        else:
            return self._create_non_streaming(request_id, created_time, model, payload)

    def _create_streaming(
        self,
        request_id: str,
        created_time: int,
        model: str,
        payload: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Implementation for streaming chat completions."""
        try:

            # Make the streaming request
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=self._client.timeout
            )

            if not response.ok:
                raise IOError(f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}")

            # Process the streaming response
            full_response = ""

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8').strip()
                    if line == "data: [DONE]":
                        break
                    if line.startswith('data: '):
                        try:
                            json_data = json.loads(line[6:])
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                choice = json_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    full_response += content

                                    # Create and yield a chunk
                                    delta = ChoiceDelta(content=content)
                                    choice = Choice(index=0, delta=delta, finish_reason=None)
                                    chunk = ChatCompletionChunk(
                                        id=request_id,
                                        choices=[choice],
                                        created=created_time,
                                        model=model
                                    )

                                    yield chunk
                        except json.JSONDecodeError:
                            continue

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
            print(f"{RED}Error during TextPollinations streaming request: {e}{RESET}")
            raise IOError(f"TextPollinations streaming request failed: {e}") from e

    def _create_non_streaming(
        self,
        request_id: str,
        created_time: int,
        model: str,
        payload: Dict[str, Any]
    ) -> ChatCompletion:
        """Implementation for non-streaming chat completions."""
        try:

            # Make the non-streaming request
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                timeout=self._client.timeout
            )

            if not response.ok:
                raise IOError(f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}")

            # Parse the response
            response_json = response.json()

            # Extract the content
            if 'choices' in response_json and len(response_json['choices']) > 0:
                if 'message' in response_json['choices'][0]:
                    full_content = response_json['choices'][0]['message']['content']
                else:
                    full_content = ""
            else:
                full_content = ""

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

            # Estimate token usage (very rough estimate)
            prompt_tokens = sum(len(msg.get("content", "")) // 4 for msg in payload.get("messages", []))
            completion_tokens = len(full_content) // 4
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
            print(f"{RED}Error during TextPollinations non-stream request: {e}{RESET}")
            raise IOError(f"TextPollinations request failed: {e}") from e

class Chat(BaseChat):
    def __init__(self, client: 'TextPollinations'):
        self.completions = Completions(client)

class TextPollinations(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for TextPollinations API.

    Usage:
        client = TextPollinations()
        response = client.chat.completions.create(
            model="openai-large",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    """

    AVAILABLE_MODELS = [
        "openai",
        "openai-large",
        "openai-reasoning",
        "qwen-coder",
        "llama",
        "llamascout",
        "mistral",
        "unity",
        "midijourney",
        "rtist",
        "searchgpt",
        "evil",
        "deepseek-reasoning",
        "deepseek-reasoning-large",
        "llamalight",
        "phi",
        "llama-vision",
        "pixtral",
        "gemini",
        "hormoz",
        "hypnosis-tracy",
        "mistral-roblox",
        "roblox-rp",
        "deepseek",
        "sur",
        "llama-scaleway",
        "openai-audio",
    ]

    def __init__(
        self,
        timeout: int = 30,
        proxies: dict = {}
    ):
        """
        Initialize the TextPollinations client.

        Args:
            timeout: Request timeout in seconds
            proxies: Optional proxy configuration
        """
        self.timeout = timeout
        self.api_endpoint = "https://text.pollinations.ai/openai"
        self.proxies = proxies

        # Initialize session
        self.session = requests.Session()
        if proxies:
            self.session.proxies.update(proxies)

        # Initialize LitAgent for user agent generation
        agent = LitAgent()
        self.user_agent = agent.random()

        # Set headers
        self.headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': self.user_agent,
            'Content-Type': 'application/json',
        }

        self.session.headers.update(self.headers)

        # Initialize chat interface
        self.chat = Chat(self)
