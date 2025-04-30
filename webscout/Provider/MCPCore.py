import json
import uuid
from typing import Any, Dict, Generator, Union

# Use curl_cffi for requests
from curl_cffi.requests import Session
from curl_cffi import CurlError

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
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
        "google/gemma-7b-it",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-v3-0324",
        "fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-4-Maverick-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "qwen-max-latest",
        "qwen-plus-latest",
        "qwen2.5-coder-32b-instruct",
        "qwen-turbo-latest",
        "qwen2.5-14b-instruct-1m",
        "GLM-4-32B",
        "Z1-32B",
        "Z1-Rumination",
        "arena-model",
        "qvq-72b-preview-0310",
        "qwq-32b",
        "qwen3-235b-a22b",
        "qwen3-30b-a3b",
        "qwen3-32b",
        "deepseek-flash",
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        "任务专用",
    ]

    def __init__(
        self,
        cookies_path: str,
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
            print(f"Warning: Model '{model}' not in known AVAILABLE_MODELS. Attempting to use anyway.")

        self.api_endpoint = "https://chat.mcpcore.xyz/api/chat/completions"
        self.model = model
        self.system_prompt = system_prompt
        self.cookies_path = cookies_path
        self.cookie_string, self.token = self._load_cookies()

        # Initialize curl_cffi Session
        self.session = Session()

        # Set up headers based on the provided request
        self.headers = {
            'authority': 'chat.mcpcore.xyz',
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9,en-IN;q=0.8',
            'authorization': f'Bearer {self.token}' if self.token else '',
            'content-type': 'application/json',
            'dnt': '1',
            'origin': 'https://chat.mcpcore.xyz',
            'referer': 'https://chat.mcpcore.xyz/',
            'priority': 'u=1, i',
            'sec-ch-ua': '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'sec-gpc': '1',
            'user-agent': LitAgent().random(),
        }

        # Apply headers, proxies, and cookies to the session
        self.session.headers.update(self.headers)
        self.session.proxies = proxies
        self.cookies = {
            'token': self.token,
        }
        for name, value in self.cookies.items():
            self.session.cookies.set(name, value, domain="chat.mcpcore.xyz")

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

    def _load_cookies(self) -> tuple[str, str]:
        """Load cookies from a JSON file and build a cookie header string."""
        try:
            with open(self.cookies_path, "r") as f:
                cookies = json.load(f)
            cookie_string = "; ".join(
                f"{cookie['name']}={cookie['value']}" for cookie in cookies if 'name' in cookie and 'value' in cookie
            )
            token = next(
                (cookie.get("value") for cookie in cookies if cookie.get("name") == "token"),
                "",
            )
            return cookie_string, token
        except FileNotFoundError:
            raise exceptions.InvalidAuthenticationError(
                f"Error: Cookies file not found at {self.cookies_path}!"
            )
        except json.JSONDecodeError:
            raise exceptions.InvalidAuthenticationError(
                f"Error: Invalid JSON format in cookies file: {self.cookies_path}!"
            )

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

                for line_bytes in response.iter_lines():
                    if line_bytes:
                        try:
                            line = line_bytes.decode('utf-8').strip()
                            if line.startswith("data: "):
                                json_str = line[6:]
                                if json_str == "[DONE]":
                                    break
                                json_data = json.loads(json_str)
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    delta = json_data['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    if content:
                                        streaming_text += content
                                        resp = dict(text=content)
                                        yield resp if not raw else content
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue

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

# Example usage (remember to create a cookies.json file)
if __name__ == "__main__":
    from rich import print

    cookies_file_path = "cookies.json"

    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in MCPCore.AVAILABLE_MODELS:
        try:
            test_ai = MCPCore(cookies_path=cookies_file_path, model=model, timeout=60)
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
