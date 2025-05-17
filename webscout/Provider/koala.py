import requests
import re
from typing import Optional, Union, Any, Dict, Generator
from uuid import uuid4

from webscout.AIutel import Optimizers, Conversation, AwesomePrompts
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
    ) -> Dict[str, Any]:
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
        def for_stream():
            response = self.session.post(
                self.api_endpoint, json=payload, headers=self.headers, stream=True, timeout=self.timeout
            )
            if not response.ok:
                raise exceptions.FailedToGenerateResponseError(
                    f"Failed to generate response - ({response.status_code}, {response.reason})"
                )
            streaming_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                # Only process lines starting with data:
                if line.startswith("data:"):
                    content = self._koala_extractor(line)
                    if content and content.strip():
                        streaming_response += content
                        yield dict(text=content) if not raw else content
            # Only update chat history if response is not empty
            if streaming_response.strip():
                self.last_response = dict(text=streaming_response)
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
        def for_non_stream():
            # Use streaming logic to collect the full response
            full_text = ""
            for chunk in for_stream():
                if isinstance(chunk, dict):
                    full_text += chunk.get("text", "")
                elif isinstance(chunk, str):
                    full_text += chunk
            # Only update chat history if response is not empty
            if full_text.strip():
                self.last_response = dict(text=full_text)
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
            return self.last_response
        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream():
            for response in self.ask(
                prompt, True, optimizer=optimizer, conversationally=conversationally
            ):
                yield self.get_message(response)
        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt,
                    False,
                    optimizer=optimizer,
                    conversationally=conversationally,
                )
            )
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "")

if __name__ == "__main__":
    from rich import print
    ai = KOALA(timeout=60)
    response = ai.chat("Say 'Hello' in one word", stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)