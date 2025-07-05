import time
import uuid
import requests
import json
import re
from typing import List, Dict, Optional, Union, Generator, Any

from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage, count_tokens
)

from webscout.litagent import LitAgent

# --- MonoChat Client ---

class Completions(BaseCompletions):
    def __init__(self, client: 'MonoChat'):
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
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Creates a model response for the given chat conversation.
        Mimics openai.chat.completions.create
        """
        # Prepare the payload for MonoChat API
        payload = {
            "messages": messages,
            "model": model
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        payload.update(kwargs)

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload, timeout, proxies)
        else:
            return self._create_non_stream(request_id, created_time, model, payload, timeout, proxies)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any], timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None)
            )
            if not response.ok:
                raise IOError(
                    f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                )

            prompt_tokens = count_tokens([msg.get("content", "") for msg in payload.get("messages", [])])
            completion_tokens = 0
            total_tokens = 0

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    # MonoChat returns lines like: 0:"Hello" or 0:"!" etc.
                    match = re.search(r'0:"(.*?)"', decoded_line)
                    if match:
                        content = match.group(1)
                        content = self._client.format_text(content)
                        completion_tokens += count_tokens(content)
                        total_tokens = prompt_tokens + completion_tokens

                        delta = ChoiceDelta(
                            content=content,
                            role="assistant",
                            tool_calls=None
                        )
                        choice = Choice(
                            index=0,
                            delta=delta,
                            finish_reason=None,
                            logprobs=None
                        )
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            choices=[choice],
                            created=created_time,
                            model=model,
                            system_fingerprint=None
                        )
                        chunk.usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "estimated_cost": None
                        }
                        yield chunk

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
            chunk.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": None
            }
            yield chunk

        except Exception as e:
            print(f"Error during MonoChat stream request: {e}")
            raise IOError(f"MonoChat request failed: {e}") from e

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any], timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> ChatCompletion:
        try:
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None)
            )
            if not response.ok:
                raise IOError(
                    f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                )

            full_text = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    match = re.search(r'0:"(.*?)"', line)
                    if match:
                        content = match.group(1)
                        full_text += content

            full_text = self._client.format_text(full_text)

            prompt_tokens = count_tokens([msg.get("content", "") for msg in payload.get("messages", [])])
            completion_tokens = count_tokens(full_text)
            total_tokens = prompt_tokens + completion_tokens

            message = ChatCompletionMessage(
                role="assistant",
                content=full_text
            )
            choice = Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            completion = ChatCompletion(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model,
                usage=usage,
            )
            return completion

        except Exception as e:
            print(f"Error during MonoChat non-stream request: {e}")
            raise IOError(f"MonoChat request failed: {e}") from e

class Chat(BaseChat):
    def __init__(self, client: 'MonoChat'):
        self.completions = Completions(client)

class MonoChat(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for MonoChat API.

    Usage:
        client = MonoChat()
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Hello!"}]
        )
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
        browser: str = "chrome"
    ):
        """
        Initialize the MonoChat client.

        Args:
            browser: Browser to emulate in user agent
        """
        self.timeout = None
        self.api_endpoint = "https://gg.is-a-furry.dev/api/chat"
        self.session = requests.Session()

        agent = LitAgent()
        self.fingerprint = agent.generate_fingerprint(browser)

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
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return MonoChat.AVAILABLE_MODELS
        return _ModelList()

    def format_text(self, text: str) -> str:
        """
        Format text by replacing escaped newlines with actual newlines.

        Args:
            text: Text to format

        Returns:
            Formatted text
        """
        try:
            text = text.replace('\\\\', '\\')
            text = text.replace('\\n', '\n')
            text = text.replace('\\r', '\r')
            text = text.replace('\\t', '\t')
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")
            try:
                json_str = f'"{text}"'
                decoded = json.loads(json_str)
                return decoded
            except json.JSONDecodeError:
                return text
        except Exception as e:
            print(f"Warning: Error formatting text: {e}")
            return text

    def convert_model_name(self, model: str) -> str:
        """
        Convert model names to ones supported by MonoChat.

        Args:
            model: Model name to convert

        Returns:
            MonoChat model name
        """
        return model

if __name__ == "__main__":
    client = MonoChat()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "tell me about humans"}],
        max_tokens=1000,
        stream=True
    )
    for chunk in response:
        if chunk.choices and hasattr(chunk.choices[0], "delta") and getattr(chunk.choices[0].delta, "content", None):
            print(chunk.choices[0].delta.content, end="", flush=True)
