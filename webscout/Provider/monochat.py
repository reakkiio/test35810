from typing import Generator, Optional, Union, Any, Dict
from uuid import uuid4
from curl_cffi import CurlError
from curl_cffi.requests import Session
import re

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class MonoChat(Provider):
    """
    MonoChat provider for interacting with the gg.is-a-furry.dev API (OpenAI-compatible).
    """
    AVAILABLE_MODELS = [
        "deepseek-r1",
        "deepseek-v3",
        "uncensored-r1-32b",
        "o3-pro",
        "o4-mini",
        "o3",
        "gpt-4.5-preview",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-search-preview",
        "gpt-4o-mini-search-preview",
        "gpt-4-turbo"
    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2049,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "gpt-4.1",
        system_prompt: str = "You are a helpful assistant.",
        browser: str = "chrome"
    ):
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://gg.is-a-furry.dev/api/chat"
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt
        self.agent = LitAgent()
        self.fingerprint = self.agent.generate_fingerprint(browser)
        self.headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": self.fingerprint["accept_language"],
            "content-type": "application/json",
            "origin": "https://gg.is-a-furry.dev",
            "referer": "https://gg.is-a-furry.dev/",
            "user-agent": self.fingerprint["user_agent"]
        }
        self.session.headers.update(self.headers)
        self.session.proxies = proxies
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

    def refresh_identity(self, browser: str = None):
        browser = browser or self.fingerprint.get("browser_type", "chrome")
        self.fingerprint = self.agent.generate_fingerprint(browser)
        self.headers.update({
            "accept-language": self.fingerprint["accept_language"],
            "user-agent": self.fingerprint["user_agent"]
        })
        self.session.headers.update(self.headers)
        return self.fingerprint

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        """
        Sends a prompt to the gg.is-a-furry.dev API and returns the response.

        Args:
            prompt (str): The prompt to send to the API.
            stream (bool): Whether to stream the response.
            raw (bool): Whether to return the raw response.
            optimizer (str): Optimizer to use for the prompt.
            conversationally (bool): Whether to generate the prompt conversationally.

        Returns:
            Dict[str, Any]: The API response.
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt}
            ],
            "model": self.model,
            "max_tokens": self.max_tokens_to_sample
        }

        def for_stream():
            try:
                response = self.session.post(
                    self.api_endpoint,
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )
                if not response.ok:
                    raise exceptions.FailedToGenerateResponseError(
                        f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                    )
                streaming_response = ""
                # Use sanitize_stream with regex-based extraction and filtering (like x0gpt)
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value=None,
                    to_json=False,
                    extract_regexes=[r'0:"(.*?)"'],
                    skip_regexes=[
                        r'^f:',
                        r'^e:',
                        r'^d:',
                        r'^\s*$',
                        r'data:\s*\[DONE\]',
                        r'event:\s*',
                        r'^\d+:\s*$',
                        r'^:\s*$',
                        r'^\s*[\x00-\x1f]+\s*$',
                    ],
                    raw=raw
                )

                for content_chunk in processed_stream:
                    if isinstance(content_chunk, bytes):
                        content_chunk = content_chunk.decode('utf-8', errors='ignore')
                    if raw:
                        yield content_chunk
                    else:
                        if content_chunk and isinstance(content_chunk, str):
                            try:
                                clean_content = content_chunk.encode().decode('unicode_escape')
                                clean_content = clean_content.replace('\\\\', '\\').replace('\\"', '"')
                                streaming_response += clean_content
                                yield dict(text=clean_content)
                            except (UnicodeDecodeError, UnicodeEncodeError):
                                streaming_response += content_chunk
                                yield dict(text=content_chunk)

                self.last_response.update(dict(text=streaming_response))
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}")

        def for_non_stream():
            if stream:
                return for_stream()
            for _ in for_stream():
                pass
            return self.last_response

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        raw: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generates a response from the MonoChat API.

        Args:
            prompt (str): The prompt to send to the API.
            stream (bool): Whether to stream the response.
            optimizer (str): Optimizer to use for the prompt.
            conversationally (bool): Whether to generate the prompt conversationally.
            raw (bool): Whether to return raw response chunks.

        Returns:
            str: The API response.
        """

        def for_stream():
            for response in self.ask(
                prompt, True, raw=raw, optimizer=optimizer, conversationally=conversationally
            ):
                if raw:
                    yield response
                else:
                    yield self.get_message(response)

        def for_non_stream():
            result = self.ask(
                prompt,
                False,
                raw=raw,
                optimizer=optimizer,
                conversationally=conversationally,
            )
            if raw:
                return result
            else:
                return self.get_message(result)

        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        """
        Extracts the message from the API response.

        Args:
            response (dict): The API response.

        Returns:
            str: The message content.
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        text = response.get("text", "")
        return text

if __name__ == "__main__":
    from rich import print
    ai = MonoChat(timeout=60)
    response = ai.chat("In points tell me about humans", stream=True, raw=False)
    for chunk in response:
        print(chunk, end="", flush=True)
