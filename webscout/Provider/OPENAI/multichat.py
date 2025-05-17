import time
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Union, Generator, Any

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage,
    format_prompt
)

# Import curl_cffi for Cloudflare bypass
from curl_cffi.requests import Session
from curl_cffi import CurlError

# Import LitAgent for user agent generation
from webscout.litagent import LitAgent

# ANSI escape codes for formatting
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"

# Model configurations
MODEL_CONFIGS = {
    "llama": {
        "endpoint": "https://www.multichatai.com/api/chat/meta",
        "models": {
            "llama-3.3-70b-versatile": {"contextLength": 131072},
            "llama-3.2-11b-vision-preview": {"contextLength": 32768},
            "deepseek-r1-distill-llama-70b": {"contextLength": 128000},
        },
    },
    "cohere": {
        "endpoint": "https://www.multichatai.com/api/chat/cohere",
        "models": {
            "command-r": {"contextLength": 128000},
            "command": {"contextLength": 4096},
        },
    },
    "google": {
        "endpoint": "https://www.multichatai.com/api/chat/google",
        "models": {
            "gemini-1.5-flash-002": {"contextLength": 1048576},
            "gemma2-9b-it": {"contextLength": 8192},
            "gemini-2.0-flash": {"contextLength": 128000},
        },
        "message_format": "parts",
    },
    "deepinfra": {
        "endpoint": "https://www.multichatai.com/api/chat/deepinfra",
        "models": {
            "Sao10K/L3.1-70B-Euryale-v2.2": {"contextLength": 8192},
            "Gryphe/MythoMax-L2-13b": {"contextLength": 8192},
            "nvidia/Llama-3.1-Nemotron-70B-Instruct": {"contextLength": 131072},
            "deepseek-ai/DeepSeek-V3": {"contextLength": 32000},
            "meta-llama/Meta-Llama-3.1-405B-Instruct": {"contextLength": 131072},
            "NousResearch/Hermes-3-Llama-3.1-405B": {"contextLength": 131072},
            "gemma-2-27b-it": {"contextLength": 8192},
        },
    },
    "mistral": {
        "endpoint": "https://www.multichatai.com/api/chat/mistral",
        "models": {
            "mistral-small-latest": {"contextLength": 32000},
            "codestral-latest": {"contextLength": 32000},
            "open-mistral-7b": {"contextLength": 8000},
            "open-mixtral-8x7b": {"contextLength": 8000},
        },
    },
    "alibaba": {
        "endpoint": "https://www.multichatai.com/api/chat/alibaba",
        "models": {
            "Qwen/Qwen2.5-72B-Instruct": {"contextLength": 32768},
            "Qwen/Qwen2.5-Coder-32B-Instruct": {"contextLength": 32768},
            "Qwen/QwQ-32B-Preview": {"contextLength": 32768},
        },
    },
}

class Completions(BaseCompletions):
    def __init__(self, client: 'MultiChatAI'):
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
        Create a chat completion using the MultiChatAI API.

        Args:
            model: The model to use
            messages: A list of messages in the conversation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            temperature: Temperature for response generation
            top_p: Top-p sampling parameter

        Returns:
            Either a ChatCompletion object or a generator of ChatCompletionChunk objects
        """
        try:
            # Set client parameters based on function arguments
            self._client.model = model
            if temperature is not None:
                self._client.temperature = temperature
            if max_tokens is not None:
                self._client.max_tokens_to_sample = max_tokens

            # Extract system messages and set as system prompt
            for message in messages:
                if message.get("role") == "system":
                    self._client.system_prompt = message.get("content", "")
                    break

            # Format all messages into a single prompt
            user_message = format_prompt(messages)

            # Generate a unique request ID
            request_id = f"multichat-{str(uuid.uuid4())}"
            created_time = int(time.time())

            # Make the API request
            response_text = self._client._make_api_request(user_message)

            # If streaming is requested, simulate streaming with the full response
            if stream:
                def generate_chunks():
                    # Create a single chunk with the full response
                    delta = ChoiceDelta(content=response_text)
                    choice = Choice(index=0, delta=delta, finish_reason="stop")
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        choices=[choice],
                        created=created_time,
                        model=model,
                    )
                    yield chunk

                return generate_chunks()

            # For non-streaming, create a complete response
            message = ChatCompletionMessage(role="assistant", content=response_text)
            choice = Choice(index=0, message=message, finish_reason="stop")

            # Estimate token usage (this is approximate)
            prompt_tokens = len(user_message) // 4  # Rough estimate
            completion_tokens = len(response_text) // 4  # Rough estimate
            total_tokens = prompt_tokens + completion_tokens

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
                usage=usage,
            )

            return completion

        except Exception as e:
            print(f"{RED}Error during MultiChatAI request: {e}{RESET}")
            raise IOError(f"MultiChatAI request failed: {e}") from e

class Chat(BaseChat):
    def __init__(self, client: 'MultiChatAI'):
        self.completions = Completions(client)

class MultiChatAI(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for MultiChatAI API.

    Usage:
        client = MultiChatAI()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    """

    AVAILABLE_MODELS = [
        # Llama Models
        "llama-3.3-70b-versatile",
        "llama-3.2-11b-vision-preview",
        "deepseek-r1-distill-llama-70b",

        # Google Models
        "gemma2-9b-it",
        "gemini-2.0-flash",

        # DeepInfra Models
        "Sao10K/L3.1-70B-Euryale-v2.2",
        "Gryphe/MythoMax-L2-13b",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct",
        "deepseek-ai/DeepSeek-V3",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "NousResearch/Hermes-3-Llama-3.1-405B",

        # Alibaba Models
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/QwQ-32B-Preview"
    ]

    def __init__(
        self,
        timeout: int = 30,
        proxies: dict = {},
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = "You are a friendly, helpful AI assistant.",
        temperature: float = 0.5,
        max_tokens: int = 4000
    ):
        """
        Initialize the MultiChatAI client.

        Args:
            timeout: Request timeout in seconds
            proxies: Optional proxy configuration
            model: Default model to use
            system_prompt: System prompt to use
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens to generate
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        # Initialize curl_cffi Session
        self.session = Session()
        self.timeout = timeout
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens_to_sample = max_tokens

        # Initialize LitAgent for user agent generation
        self.agent = LitAgent()

        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "text/plain;charset=UTF-8",
            "origin": "https://www.multichatai.com",
            "referer": "https://www.multichatai.com/",
            "user-agent": self.agent.random(),
        }

        # Update curl_cffi session headers, proxies, and cookies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies
        self.session.cookies.set("session", uuid.uuid4().hex)

        # Initialize the provider based on the model
        self.provider = self._get_provider_from_model(self.model)
        self.model_name = self.model

        # Initialize the chat interface
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()


    def _get_endpoint(self) -> str:
        """Get the API endpoint for the current provider."""
        return MODEL_CONFIGS[self.provider]["endpoint"]

    def _get_chat_settings(self) -> Dict[str, Any]:
        """Get chat settings for the current model."""
        base_settings = MODEL_CONFIGS[self.provider]["models"][self.model_name]
        return {
            "model": self.model,
            "prompt": self.system_prompt,
            "temperature": self.temperature,
            "contextLength": base_settings["contextLength"],
            "includeProfileContext": True,
            "includeWorkspaceInstructions": True,
            "embeddingsProvider": "openai"
        }

    def _get_system_message(self) -> str:
        """Generate system message with current date."""
        current_date = datetime.now().strftime("%d/%m/%Y")
        return f"Today is {current_date}.\n\nUser Instructions:\n{self.system_prompt}"

    def _build_messages(self, conversation_prompt: str) -> list:
        """Build messages array based on provider type."""
        if self.provider == "google":
            return [
                {"role": "user", "parts": self._get_system_message()},
                {"role": "model", "parts": "I will follow your instructions."},
                {"role": "user", "parts": conversation_prompt}
            ]
        else:
            return [
                {"role": "system", "content": self._get_system_message()},
                {"role": "user", "content": conversation_prompt}
            ]

    def _get_provider_from_model(self, model: str) -> str:
        """Determine the provider based on the model name."""
        for provider, config in MODEL_CONFIGS.items():
            if model in config["models"]:
                return provider

        available_models = []
        for provider, config in MODEL_CONFIGS.items():
            for model_name in config["models"].keys():
                available_models.append(f"{provider}/{model_name}")

        error_msg = f"Invalid model: {model}\nAvailable models: {', '.join(available_models)}"
        raise ValueError(error_msg)

    def _make_api_request(self, prompt: str) -> str:
        """Make the API request with proper error handling."""
        try:
            payload = {
                "chatSettings": self._get_chat_settings(),
                "messages": self._build_messages(prompt),
                "customModelId": "",
            }

            # Use curl_cffi session post with impersonate
            response = self.session.post(
                self._get_endpoint(),
                json=payload,
                timeout=self.timeout,
                impersonate="chrome110"
            )
            response.raise_for_status()

            # Return the response text
            return response.text.strip()

        except CurlError as e:
            raise IOError(f"API request failed (CurlError): {e}") from e
        except Exception as e:
            err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
            raise IOError(f"API request failed ({type(e).__name__}): {e} - {err_text}") from e

if __name__ == "__main__":
    print(f"{BOLD}Testing MultiChatAI OpenAI-compatible provider{RESET}")

    client = MultiChatAI()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello' in one word"}
        ]
    )

    print(f"Response: {response.choices[0].message.content}")
