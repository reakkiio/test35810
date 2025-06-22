import os
import json
import requests
from typing import Any, Dict, Optional, Union, Generator
from webscout.AIutel import sanitize_stream, Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions

class MiniMax(Provider):
    """
    Provider for MiniMax-Reasoning-01 API, following the standard provider interface.
    """
    AVAILABLE_MODELS = ["MiniMax-Reasoning-01"]
    API_URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
    # TODO: Move API_KEY to env/config for security
    API_KEY = os.environ.get("MINIMAX_API_KEY") or """eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJtbyBuaSIsIlVzZXJOYW1lIjoibW8gbmkiLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTg3NjIwMDY0ODA2NDYzNTI0MiIsIlBob25lIjoiIiwiR3JvdXBJRCI6IjE4NzYyMDA2NDgwNjA0NDA5MzgiLCJQYWdlTmFtZSI6IiIsIk1haWwiOiJuaW1vQHN1YnN1cC52aXAiLCJDcmVhdGVUaW1lIjoiMjAyNS0wMS0wNyAxMToyNzowNyIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.Ge1ZnpFPUfXVdMini0P_qXbP_9VYwzXiffG9DsNQck4GtYEOs33LDeAiwrVsrrLZfvJ2icQZ4sRZS54wmPuWua_Dav6pYJty8ZtahmUX1IuhlUX5YErhhCRAIy3J1xB8FkLHLyylChuBHpkNz6O6BQLmPqmoa-cOYK9Qrc6IDeu8SX1iMzO9-MSkcWNvkvpCF2Pf9tekBVWNKMDK6IZoMEPbtkaPXdDyP6l0M0e2AlL_E0oM9exg3V-ohAi8OTPFyqM6dcd4TwF-b9DULxfIsRFw401mvIxcTDWa42u2LULewdATVRD2BthU65tuRqEiWeFWMvFlPj2soMze_QIiUA"""
    MODEL_CONTROL_DEFAULTS = {"tokens_to_generate": 40000, "temperature": 1, "top_p": 0.95}

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
        model: str = "MiniMax-Reasoning-01",
        system_prompt: str = "You are a helpful assistant, always respond in english",
    ):
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
        self.model = model
        self.api_url = self.API_URL
        self.api_key = self.API_KEY
        self.timeout = timeout
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.last_response = {}
        self.system_prompt = system_prompt
        self.proxies = proxies
        self.__available_optimizers = tuple(
            method for method in dir(Optimizers)
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

    @staticmethod
    def _extract_content(chunk: Any) -> Optional[dict]:
        if not isinstance(chunk, dict):
            return None
        choice = chunk.get('choices', [{}])[0]
        delta = choice.get('delta', {})
        content = delta.get('content')
        reasoning = delta.get('reasoning_content')
        result = {}
        if content:
            result['content'] = content
        if reasoning:
            result['reasoning_content'] = reasoning
        return result if result else None

    def ask(
        self,
        prompt: str,
        stream: bool = True,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': conversation_prompt}
        ]
        data = {
            'model': self.model,
            'messages': messages,
            'stream': True,
            'max_tokens': self.MODEL_CONTROL_DEFAULTS.get('tokens_to_generate', 512),
            'temperature': self.MODEL_CONTROL_DEFAULTS.get('temperature', 1.0),
            'top_p': self.MODEL_CONTROL_DEFAULTS.get('top_p', 1.0),
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        def for_stream():
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(data),
                    stream=True,
                    timeout=self.timeout,
                    proxies=self.proxies if self.proxies else None
                )
                if not response.ok:
                    raise exceptions.FailedToGenerateResponseError(
                        f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                    )
                streaming_response = ""
                last_content = ""
                last_reasoning = ""
                in_think = False
                processed_stream = sanitize_stream(
                    response.iter_lines(),
                    intro_value="data:",
                    to_json=True,
                    content_extractor=self._extract_content,
                    raw=False # always process as dict for logic
                )
                for chunk in processed_stream:
                    if not chunk:
                        continue
                    content = chunk.get('content') if isinstance(chunk, dict) else None
                    reasoning = chunk.get('reasoning_content') if isinstance(chunk, dict) else None
                    # Handle reasoning_content with <think> tags
                    if reasoning and reasoning != last_reasoning:
                        if not in_think:
                            yield "<think>\n\n"
                            in_think = True
                        yield reasoning
                        last_reasoning = reasoning
                    # If we were in <think> and now have new content, close <think>
                    if in_think and content and content != last_content:
                        yield "</think>\n\n"
                        in_think = False
                    # Handle normal content
                    if content and content != last_content:
                        yield content
                        streaming_response += content
                        last_content = content
                if not raw:
                    self.last_response = {"text": streaming_response}
                    self.conversation.update_chat_history(prompt, streaming_response)
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")
        def for_non_stream():
            full_response = ""
            for chunk in for_stream():
                if isinstance(chunk, dict) and "text" in chunk:
                    full_response += chunk["text"]
                elif isinstance(chunk, str):
                    full_response += chunk
            if not raw:
                self.last_response = {"text": full_response}
                self.conversation.update_chat_history(prompt, full_response)
                return {"text": full_response}
            else:
                return full_response
        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = True,
        optimizer: str = None,
        conversationally: bool = False,
        raw: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream():
            for response in self.ask(
                prompt, stream=True, raw=raw, optimizer=optimizer, conversationally=conversationally
            ):
                if raw:
                    yield response
                else:
                    yield response
        def for_non_stream():
            result = self.ask(
                prompt, stream=False, raw=raw, optimizer=optimizer, conversationally=conversationally
            )
            if raw:
                return result
            else:
                return self.get_message(result)
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "")

if __name__ == "__main__":
    ai = MiniMax()
    resp = ai.chat("What is the capital of France?", stream=True, raw=False)
    for chunk in resp:
        print(chunk, end="", flush=True)