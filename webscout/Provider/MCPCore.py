import json
import uuid
import random
import string
from typing import Any, Dict, Generator, Union

# Use curl_cffi for requests
from curl_cffi.requests import Session
from curl_cffi import CurlError

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream 
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class MCPCore(Provider):
    """
    A class to interact with the chat.mcpcore.xyz API.
    Supports streaming responses.
    """

    # Add more models if known, starting with the one from the example
    AVAILABLE_MODELS = [
        "@cf/deepseek-ai/deepseek-math-7b-instruct",
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "@cf/defog/sqlcoder-7b-2",
        "@cf/fblgit/una-cybertron-7b-v2-bf16",
        "@cf/google/gemma-3-12b-it",
        "@cf/meta/llama-2-7b-chat-int8",
        "@hf/thebloke/llama-2-13b-chat-awq",
        "@hf/thebloke/llamaguard-7b-awq",
        "@hf/thebloke/mistral-7b-instruct-v0.1-awq",
        "@hf/thebloke/neural-chat-7b-v3-1-awq",
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.7-sonnet:thinking",
        "anthropic/claude-opus-4",
        "anthropic/claude-sonnet-4",
        "openai/chatgpt-4o-latest",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "openai/gpt-4o-mini-search-preview",
        "openai/gpt-4o-search-preview",
        "openai/o1-pro",
        "openai/o3-mini",
        "sarvam-m",
        "x-ai/grok-3-beta",
    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2048,
        timeout: int = 60,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "qwen3-32b",
        system_prompt: str = "You are a helpful assistant.",
    ):
        """Initializes the MCPCore API client."""
        if model not in self.AVAILABLE_MODELS:
            print(f"Warning: Model '{model}' is not listed in AVAILABLE_MODELS. Proceeding with the provided model.")

        self.api_endpoint = "https://chat.mcpcore.xyz/api/chat/completions"
        
        self.model = model
        self.system_prompt = system_prompt

        # Initialize curl_cffi Session
        self.session = Session()

        # Set up headers based on the provided request
        self.headers = {
            **LitAgent().generate_fingerprint(),
            'origin': 'https://chat.mcpcore.xyz',
            'referer': 'https://chat.mcpcore.xyz/',
        }

        # Apply headers, proxies, and cookies to the session
        self.session.headers.update(self.headers)
        self.session.proxies = proxies

        # Provider settings
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}

        # Initialize optimizers
        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method))
            and not method.startswith("__")
        )
        Conversation.intro = (
            AwesomePrompts().get_act(
                act, raise_not_found=True, default=None, case_insensitive=True
            )
            if act
            else intro or Conversation.intro
        )
        self.conversation = Conversation(
            is_conversation, self.max_tokens_to_sample, filepath, update_file
        )
        self.conversation.history_offset = history_offset

        # Token handling: always auto-fetch token, no cookies logic
        self.token = self._auto_fetch_token()

        # Set the Authorization header for the session
        self.session.headers.update({
            'authorization': f'Bearer {self.token}',
        })

    def _auto_fetch_token(self):
        """Automatically fetch a token from the signup endpoint."""
        session = Session()
        def random_string(length=8):
            return ''.join(random.choices(string.ascii_lowercase, k=length))
        name = random_string(6)
        email = f"{random_string(8)}@gmail.com"
        password = email
        profile_image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAAAXNSR0IArs4c6QAAAkRJREFUeF7tmDFOw0AUBdcSiIaKM3CKHIQ7UHEISq5AiUTFHYC0XADoTRsJEZFEjhFIaYAim92fjGFS736/zOTZzjavl0d98oMh0CgE4+IriEJYPhQC86EQhdAIwPL4DFEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg2BCfkAIqwAA94KZ/EAAAAASUVORK5CYII="
        payload = {
            "name": name,
            "email": email,
            "password": password,
            "profile_image_url": profile_image_url
        }
        headers = {
            **LitAgent().generate_fingerprint(),
            'origin': 'https://chat.mcpcore.xyz',
            'referer': 'https://chat.mcpcore.xyz/auth',
        }
        try:
            resp = session.post(
                "https://chat.mcpcore.xyz/api/v1/auths/signup",
                headers=headers,
                json=payload,
                timeout=30,
                impersonate="chrome110"
            )
            if resp.ok:
                data = resp.json()
                token = data.get("token")
                if token:
                    return token
                # fallback: try to get from set-cookie
                set_cookie = resp.headers.get("set-cookie", "")
                if "token=" in set_cookie:
                    return set_cookie.split("token=")[1].split(";")[0]
            raise exceptions.FailedToGenerateResponseError(f"Failed to auto-fetch token: {resp.status_code} {resp.text}")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Token auto-fetch failed: {e}")

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator]:
        """Sends a prompt to the MCPCore API and returns the response."""

        conversation_prompt = self.conversation.gen_complete_prompt(prompt)

        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise exceptions.InvalidOptimizerError(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )

        chat_id = kwargs.get("chat_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        payload = {
            "stream": stream,
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt}
            ],
            "params": kwargs.get("params", {}),
            "tool_servers": kwargs.get("tool_servers", []),
            "features": kwargs.get("features", {"web_search": False}),
            "chat_id": chat_id,
            "id": message_id,
            "stream_options": kwargs.get("stream_options", {"include_usage": True})
        }

        def for_stream():
            streaming_text = ""
            try:
                response = self.session.post(
                    self.api_endpoint,
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    impersonate="chrome110"
                )
                response.raise_for_status()

                # Use sanitize_stream
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value="data:",
                    to_json=True,     # Stream sends JSON
                    skip_markers=["[DONE]"],
                    content_extractor=lambda chunk: chunk.get('choices', [{}])[0].get('delta', {}).get('content') if isinstance(chunk, dict) else None,
                    yield_raw_on_error=False # Skip non-JSON or lines where extractor fails
                )

                for content_chunk in processed_stream:
                    # content_chunk is the string extracted by the content_extractor
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        yield dict(text=content_chunk) if not raw else content_chunk

                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, self.get_message(self.last_response))

            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e:
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e} - {err_text}") from e

        def for_non_stream():
            full_text = ""
            try:
                stream_generator = self.ask(
                    prompt, stream=True, raw=False, optimizer=optimizer, conversationally=conversationally, **kwargs
                )
                for chunk_data in stream_generator:
                    if isinstance(chunk_data, dict):
                        full_text += chunk_data["text"]
                    elif isinstance(chunk_data, str):
                        full_text += chunk_data
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Failed to aggregate non-stream response: {str(e)}") from e

            return full_text if raw else self.last_response

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """Generates a response from the MCPCore API."""

        def for_stream_chat() -> Generator[str, None, None]:
            gen = self.ask(
                prompt, stream=True, raw=False,
                optimizer=optimizer, conversationally=conversationally, **kwargs
            )
            for response_dict in gen:
                yield self.get_message(response_dict)

        def for_non_stream_chat() -> str:
            response_data = self.ask(
                prompt, stream=False, raw=False,
                optimizer=optimizer, conversationally=conversationally, **kwargs
            )
            return self.get_message(response_data)

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: Dict[str, Any]) -> str:
        """Extracts the message from the API response."""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "")

# Example usage (no cookies file needed)
if __name__ == "__main__":
    from rich import print

    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in MCPCore.AVAILABLE_MODELS:
        try:
            test_ai = MCPCore(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            # Accumulate the response text without printing in the loop
            for chunk in response:
                response_text += chunk

            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            # Print the final status and response, overwriting the "Testing..." line
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            # Print error, overwriting the "Testing..." line
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
