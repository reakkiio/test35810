import requests
import re
from typing import Optional, Union, Any, Dict, Generator
from uuid import uuid4

from webscout.AIutel import Optimizers, Conversation, AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions

class KOALA(Provider):
    """
    A class to interact with the Koala.sh API, X0GPT-style, without sanitize_stream.
    """
    AVAILABLE_MODELS = [
        "gpt-4.1-mini",
        "gpt-4.1",
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
        model: str = "gpt-4.1",
        web_search: bool = True,
    ):
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
        self.session = requests.Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://koala.sh/api/gpt/"
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.headers = {
            "accept": "text/event-stream",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json",
            "dnt": "1",
            "flag-real-time-data": "true" if web_search else "false",
            "origin": "https://koala.sh",
            "referer": "https://koala.sh/chat",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0",
        }
        self.session.headers.update(self.headers)
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
        self.session.proxies = proxies

    @staticmethod
    def _koala_extractor(line: str) -> Optional[str]:
        # Koala returns lines like: data: "Hello" or data: "..."
        match = re.match(r'data:\s*"(.*)"', line)
        if match:
            return match.group(1)
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator[Any, None, None]]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if hasattr(Optimizers, optimizer):
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not valid.")
        payload = {
            "input": conversation_prompt,
            "inputHistory": [],
            "outputHistory": [],
            "model": self.model
        }
        response = self.session.post(
            self.api_endpoint, json=payload, headers=self.headers, stream=True, timeout=self.timeout
        )
        if not response.ok:
            raise exceptions.FailedToGenerateResponseError(
                f"Failed to generate response - ({response.status_code}, {response.reason})"
            )
        # Use sanitize_stream with content_extractor and intro_value like YEPCHAT/X0GPT
        processed_stream = sanitize_stream(
            data=response.iter_lines(decode_unicode=True),
            intro_value="data:",
            to_json=False,
            content_extractor=self._koala_extractor,
            raw=raw
        )
        if stream:
            streaming_response = ""
            for content_chunk in processed_stream:
                if raw:
                    if content_chunk and isinstance(content_chunk, str) and content_chunk.strip():
                        streaming_response += content_chunk
                    yield content_chunk
                else:
                    if content_chunk and isinstance(content_chunk, str) and content_chunk.strip():
                        streaming_response += content_chunk
                        yield dict(text=content_chunk)
            if streaming_response.strip():
                self.last_response = dict(text=streaming_response)
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
        else:
            full_text = ""
            for content_chunk in processed_stream:
                if raw:
                    if content_chunk and isinstance(content_chunk, str):
                        full_text += content_chunk
                else:
                    if content_chunk and isinstance(content_chunk, str):
                        full_text += content_chunk
            if full_text.strip():
                self.last_response = dict(text=full_text)
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
            return self.last_response

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
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
                return result.get("text", "") if isinstance(result, dict) else str(result)
            return self.get_message(result)
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "")

if __name__ == "__main__":
    from rich import print
    ai = KOALA(timeout=60)
    response = ai.chat("tell me about humans", stream=True, raw=False)
    for chunk in response:
        print(chunk)