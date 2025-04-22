import re
import requests
import uuid
import base64
import json
import random
import string
import time
from datetime import datetime
from typing import Any, Dict, Optional, Generator, Union, List

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions

class Toolbaz(Provider):
    """
    A class to interact with the Toolbaz API. Supports streaming responses.
    """

    AVAILABLE_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash-thinking",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gpt-4o-latest",
        "gpt-4o",
        "deepseek-r1",
        "Llama-4-Maverick",
        "Llama-4-Scout",
        "Llama-3.3-70B",
        "Qwen2.5-72B",
        "Qwen2-72B",
        "grok-2-1212",
        "grok-3-beta",
        "toolbaz_v3.5_pro",
        "toolbaz_v3",
        "mixtral_8x22b",
        "L3-70B-Euryale-v2.1",
        "midnight-rose",
        "unity",
        "unfiltered_x"
    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 600,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "gemini-2.0-flash",
        system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initializes the Toolbaz API with given parameters.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.session = requests.Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.system_prompt = system_prompt
        self.model = model
        self.proxies = proxies

        # Set up headers
        self.session.headers.update({
            "user-agent": "Mozilla/5.0 (Linux; Android 10)",
            "accept": "*/*",
            "accept-language": "en-US",
            "cache-control": "no-cache",
            "connection": "keep-alive",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "origin": "https://toolbaz.com",
            "pragma": "no-cache",
            "referer": "https://toolbaz.com/",
            "sec-fetch-mode": "cors"
        })

        # Initialize conversation history
        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
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

    def random_string(self, length):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def generate_token(self):
        payload = {
            "bR6wF": {
                "nV5kP": "Mozilla/5.0 (Linux; Android 10)",
                "lQ9jX": "en-US",
                "sD2zR": "431x958",
                "tY4hL": time.tzname[0] if time.tzname else "UTC",
                "pL8mC": "Linux armv81",
                "cQ3vD": datetime.now().year,
                "hK7jN": datetime.now().hour
            },
            "uT4bX": {
                "mM9wZ": [],
                "kP8jY": []
            },
            "tuTcS": int(time.time()),
            "tDfxy": None,
            "RtyJt": str(uuid.uuid4())
        }
        return "d8TW0v" + base64.b64encode(json.dumps(payload).encode()).decode()

    def get_auth(self):
        try:
            session_id = self.random_string(36)
            token = self.generate_token()
            data = {
                "session_id": session_id,
                "token": token
            }
            resp = self.session.post("https://data.toolbaz.com/token.php", data=data)
            resp.raise_for_status()
            result = resp.json()
            if result.get("success"):
                return {"token": result["token"], "session_id": session_id}
            return None
        except Exception:
            return None

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,  # Kept for compatibility with other providers
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        """Sends a prompt to the Toolbaz API and returns the response."""
        if optimizer and optimizer not in self.__available_optimizers:
            raise exceptions.FailedToGenerateResponseError(f"Optimizer is not one of {self.__available_optimizers}")

        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            conversation_prompt = getattr(Optimizers, optimizer)(
                conversation_prompt if conversationally else prompt
            )

        auth = self.get_auth()
        if not auth:
            raise exceptions.ProviderConnectionError("Failed to authenticate with Toolbaz API")

        data = {
            "text": conversation_prompt,
            "capcha": auth["token"],
            "model": self.model,
            "session_id": auth["session_id"]
        }

        def for_stream():
            try:
                resp = self.session.post(
                    "https://data.toolbaz.com/writing.php",
                    data=data,
                    stream=True,
                    proxies=self.proxies,
                    timeout=self.timeout
                )
                resp.raise_for_status()

                buffer = ""
                tag_start = "[model:"
                streaming_text = ""

                for chunk in resp.iter_content(chunk_size=1):
                    if chunk:
                        text = chunk.decode(errors="ignore")
                        buffer += text
                        # Remove all complete [model: ...] tags in buffer
                        while True:
                            match = re.search(r"\[model:.*?\]", buffer)
                            if not match:
                                break
                            buffer = buffer[:match.start()] + buffer[match.end():]
                        # Only yield up to the last possible start of a tag
                        last_tag = buffer.rfind(tag_start)
                        if last_tag == -1 or last_tag + len(tag_start) > len(buffer):
                            if buffer:
                                streaming_text += buffer
                                yield {"text": buffer}
                                buffer = ""
                        else:
                            if buffer[:last_tag]:
                                streaming_text += buffer[:last_tag]
                                yield {"text": buffer[:last_tag]}
                            buffer = buffer[last_tag:]

                # Remove any remaining [model: ...] tag in the buffer
                buffer = re.sub(r"\[model:.*?\]", "", buffer)
                if buffer:
                    streaming_text += buffer
                    yield {"text": buffer}

                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)

            except requests.exceptions.RequestException as e:
                raise exceptions.ProviderConnectionError(f"Network error: {str(e)}") from e
            except Exception as e:
                raise exceptions.ProviderConnectionError(f"Unexpected error: {str(e)}") from e

        def for_non_stream():
            try:
                resp = self.session.post(
                    "https://data.toolbaz.com/writing.php",
                    data=data,
                    proxies=self.proxies,
                    timeout=self.timeout
                )
                resp.raise_for_status()

                text = resp.text
                # Remove [model: ...] tags
                text = re.sub(r"\[model:.*?\]", "", text)

                self.last_response = {"text": text}
                self.conversation.update_chat_history(prompt, text)

                return self.last_response

            except requests.exceptions.RequestException as e:
                raise exceptions.FailedToGenerateResponseError(f"Network error: {str(e)}") from e
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Unexpected error: {str(e)}") from e

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generates a response from the Toolbaz API."""
        def for_stream():
            for response in self.ask(
                prompt,
                stream=True,
                optimizer=optimizer,
                conversationally=conversationally
            ):
                yield self.get_message(response)

        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt,
                    stream=False,
                    optimizer=optimizer,
                    conversationally=conversationally,
                )
            )

        return for_stream() if stream else for_non_stream()

    def get_message(self, response: Dict[str, Any]) -> str:
        """Extract the message from the response.

        Args:
            response: Response dictionary

        Returns:
            str: Message extracted
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "")

# Example usage
if __name__ == "__main__":
    # Test the provider with different models
    for model in Toolbaz.AVAILABLE_MODELS:
        try:
            test_ai = Toolbaz(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            for chunk in response:
                response_text += chunk
                print(f"\r{model:<50} {'Testing...':<10}", end="", flush=True)

            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")