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

# Attempt to import LitAgent, fallback if not available
try:
    from webscout.litagent import LitAgent
except ImportError:
    pass

# --- FreeAIChat Client ---

class Completions(BaseCompletions):
    def __init__(self, client: 'FreeAIChat'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 2049,
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
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        payload.update(kwargs)

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload)
        else:
            return self._create_non_stream(request_id, created_time, model, payload)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=self._client.timeout
            )

            # Handle non-200 responses
            if not response.ok:
                raise IOError(
                    f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                )

            # Track token usage across chunks
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            # Estimate prompt tokens based on message length
            for msg in payload.get("messages", []):
                prompt_tokens += len(msg.get("content", "").split())

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8').strip()

                if line_str.startswith("data: "):
                    json_str = line_str[6:]  # Remove "data: " prefix
                    if json_str == "[DONE]":
                        break

                    try:
                        data = json.loads(json_str)
                        choice_data = data.get('choices', [{}])[0]
                        delta_data = choice_data.get('delta', {})
                        finish_reason = choice_data.get('finish_reason')

                        # Update token counts if available
                        usage_data = data.get('usage', {})
                        if usage_data:
                            prompt_tokens = usage_data.get('prompt_tokens', prompt_tokens)
                            completion_tokens = usage_data.get('completion_tokens', completion_tokens)
                            total_tokens = usage_data.get('total_tokens', total_tokens)

                        # Create the delta object
                        delta = ChoiceDelta(
                            content=delta_data.get('content'),
                            role=delta_data.get('role'),
                            tool_calls=delta_data.get('tool_calls')
                        )

                        # Create the choice object
                        choice = Choice(
                            index=choice_data.get('index', 0),
                            delta=delta,
                            finish_reason=finish_reason,
                            logprobs=choice_data.get('logprobs')
                        )

                        # Create the chunk object
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            choices=[choice],
                            created=created_time,
                            model=model,
                            system_fingerprint=data.get('system_fingerprint')
                        )

                        # Return the chunk object
                        yield chunk
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line: {json_str}")
                        continue

            # Final chunk with finish_reason="stop"
            delta = ChoiceDelta(
                content=None,
                role=None,
                tool_calls=None
            )

            choice = Choice(
                index=0,
                delta=delta,
                finish_reason="stop",
                logprobs=None
            )

            chunk = ChatCompletionChunk(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
                system_fingerprint=None
            )

            yield chunk

        except Exception as e:
            print(f"Error during FreeAIChat stream request: {e}")
            raise IOError(f"FreeAIChat request failed: {e}") from e

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any]
    ) -> ChatCompletion:
        try:
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                timeout=self._client.timeout
            )

            # Handle non-200 responses
            if not response.ok:
                raise IOError(
                    f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                )

            # Parse the response
            data = response.json()

            choices_data = data.get('choices', [])
            usage_data = data.get('usage', {})

            choices = []
            for choice_d in choices_data:
                message_d = choice_d.get('message', {})
                message = ChatCompletionMessage(
                    role=message_d.get('role', 'assistant'),
                    content=message_d.get('content', '')
                )
                choice = Choice(
                    index=choice_d.get('index', 0),
                    message=message,
                    finish_reason=choice_d.get('finish_reason', 'stop')
                )
                choices.append(choice)

            usage = CompletionUsage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0)
            )

            completion = ChatCompletion(
                id=request_id,
                choices=choices,
                created=created_time,
                model=data.get('model', model),
                usage=usage,
            )
            return completion

        except Exception as e:
            print(f"Error during FreeAIChat non-stream request: {e}")
            raise IOError(f"FreeAIChat request failed: {e}") from e

class Chat(BaseChat):
    def __init__(self, client: 'FreeAIChat'):
        self.completions = Completions(client)

class FreeAIChat(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for FreeAIChat API.

    Usage:
        client = FreeAIChat()
        response = client.chat.completions.create(
            model="GPT 4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """

    AVAILABLE_MODELS = [
        # OpenAI Models
        "GPT 4o",
        "GPT 4.5 Preview",
        "GPT 4o Latest",
        "GPT 4o mini",
        "GPT 4o Search Preview",
        "O1",
        "O1 Mini",
        "O3 Mini",
        "O3 Mini High",
        "O3 Mini Low",

        # Anthropic Models
        "Claude 3.5 haiku",
        "claude 3.5 sonnet",
        "Claude 3.7 Sonnet",
        "Claude 3.7 Sonnet (Thinking)",

        # Deepseek Models
        "Deepseek R1",
        "Deepseek R1 Fast",
        "Deepseek V3",
        "Deepseek v3 0324",

        # Google Models
        "Gemini 1.5 Flash",
        "Gemini 1.5 Pro",
        "Gemini 2.0 Flash",
        "Gemini 2.0 Pro",
        "Gemini 2.5 Pro",

        # Llama Models
        "Llama 3.1 405B",
        "Llama 3.1 70B Fast",
        "Llama 3.3 70B",
        "Llama 3.2 90B Vision",
        "Llama 4 Scout",
        "Llama 4 Maverick",

        # Mistral Models
        "Mistral Large",
        "Mistral Nemo",
        "Mixtral 8x22B",

        # Qwen Models
        "Qwen Max",
        "Qwen Plus",
        "Qwen Turbo",
        "QwQ 32B",
        "QwQ Plus",

        # XAI Models
        "Grok 2",
        "Grok 3",
    ]

    def __init__(
        self,
        timeout: Optional[int] = None,
        browser: str = "chrome"
    ):
        """
        Initialize the FreeAIChat client.

        Args:
            timeout: Request timeout in seconds (None for no timeout)
            browser: Browser to emulate in user agent
        """
        self.timeout = timeout
        self.api_endpoint = "https://freeaichatplayground.com/api/v1/chat/completions"
        self.session = requests.Session()

        # Initialize LitAgent for user agent generation
        agent = LitAgent()
        self.fingerprint = agent.generate_fingerprint(browser)

        # Initialize headers
        self.headers = {
            'User-Agent': self.fingerprint["user_agent"],
            'Accept': '*/*',
            'Content-Type': 'application/json',
            'Origin': 'https://freeaichatplayground.com',
            'Referer': 'https://freeaichatplayground.com/',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        }

        self.session.headers.update(self.headers)

        # Initialize the chat interface
        self.chat = Chat(self)

    def convert_model_name(self, model: str) -> str:
        """
        Convert model names to ones supported by FreeAIChat.

        Args:
            model: Model name to convert

        Returns:
            FreeAIChat model name
        """
        # If the model is already a valid FreeAIChat model, return it
        if model in self.AVAILABLE_MODELS:
            return model

        # Default to GPT 4o if model not found
        print(f"Warning: Unknown model '{model}'. Using 'GPT 4o' instead.")
        return "GPT 4o"
