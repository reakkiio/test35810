
import requests
from typing import Any, Dict, Generator, Optional, Union, List
from webscout.litagent import LitAgent
from webscout.AIutel import sanitize_stream, Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions

class GptOss(Provider):
    """
    Provider for GPT-OSS API.
    """
    AVAILABLE_MODELS = ["gpt-oss-20b", "gpt-oss-120b"]

    def __init__(
        self,
        model: str = "gpt-oss-120b",
        is_conversation: bool = True,
        max_tokens: int = 600,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        system_prompt: str = "You are a helpful assistant.",
        reasoning_effort: str = "high"
    ):
        self.api_endpoint = "https://api.gpt-oss.com/chatkit"
        self.model = model if model in self.AVAILABLE_MODELS else self.AVAILABLE_MODELS[0]
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self.agent = LitAgent()
        self.proxies = proxies
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

    def ask(
        self,
        prompt: str,
        stream: bool = False,
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
                raise Exception(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )

        data = {
            "op": "threads.create",
            "params": {
                "input": {
                    "text": conversation_prompt,
                    "content": [{"type": "input_text", "text": conversation_prompt}],
                    "quoted_text": "",
                    "attachments": []
                }
            }
        }
        headers = self.agent.generate_fingerprint()
        headers.update({
            "accept": "text/event-stream",
            "x-reasoning-effort": self.reasoning_effort,
            "x-selected-model": self.model,
            "x-show-reasoning": "true"
        })
        cookies = {}

        def for_stream():
            full_response_content = ""
            try:
                with requests.post(
                    self.api_endpoint,
                    headers=headers,
                    cookies=cookies,
                    json=data,
                    stream=True,
                    proxies=self.proxies if self.proxies else None,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    for chunk in sanitize_stream(
                        response.iter_lines(),
                        intro_value="data: ",
                        to_json=True,
                        skip_markers=["[DONE]"],
                        strip_chars=None,
                        content_extractor=lambda d: d.get('update', {}).get('delta') if d.get('type') == 'thread.item_updated' and d.get('update', {}).get('type') == 'assistant_message.content_part.text_delta' else None,
                        yield_raw_on_error=False,
                        encoding="utf-8",
                        raw=raw
                    ):
                        if chunk:
                            yield chunk
                            full_response_content += chunk
                self.last_response.update(dict(text=full_response_content))
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")

        def for_non_stream():
            result = ""
            try:
                with requests.post(
                    self.api_endpoint,
                    headers=headers,
                    cookies=cookies,
                    json=data,
                    stream=False,
                    proxies=self.proxies if self.proxies else None,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    # The API is event-stream only, so we simulate non-stream by joining all chunks
                    for chunk in sanitize_stream(
                        response.iter_lines(),
                        intro_value="data: ",
                        to_json=True,
                        skip_markers=["[DONE]"],
                        strip_chars=None,
                        content_extractor=lambda d: d.get('update', {}).get('delta') if d.get('type') == 'thread.item_updated' and d.get('update', {}).get('type') == 'assistant_message.content_part.text_delta' else None,
                        yield_raw_on_error=False,
                        encoding="utf-8",
                        raw=raw
                    ):
                        if chunk:
                            result += chunk
                self.last_response.update(dict(text=result))
                self.conversation.update_chat_history(
                    prompt, self.get_message(self.last_response)
                )
                return self.last_response
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        raw: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream():
            for response in self.ask(
                prompt, True, raw=raw, optimizer=optimizer, conversationally=conversationally
            ):
                yield response

        def for_non_stream():
            result = self.ask(
                prompt,
                False,
                raw=raw,
                optimizer=optimizer,
                conversationally=conversationally,
            )
            return self.get_message(result)

        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        text = response.get("text", "")
        return text

if __name__ == "__main__":
    from webscout.AIutel import timeIt
    from rich import print
    ai = GptOss(timeout=30)
    @timeIt
    def get_response():
        response = ai.chat("write a poem about AI", stream=True, raw=False)
        for chunk in response:
            print(chunk, end="", flush=True)
    get_response()