import requests
import json
import time
import uuid
from typing import List, Dict, Optional, Union, Generator, Any

from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)

try:
    from webscout.litagent import LitAgent
except ImportError:
    LitAgent = None

class Completions(BaseCompletions):
    def __init__(self, client: 'Friendli'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 81920,
        min_tokens: Optional[int] = 0,
        stream: bool = False,
        temperature: Optional[float] = 1,
        top_p: Optional[float] = 0.8,
        frequency_penalty: Optional[float] = 0,
        stop: Optional[List[str]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        payload = {
            "model": model,
            "messages": messages,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "stop": stop or [],
            "stream": stream,
            "stream_options": stream_options or {"include_usage": True},
        }
        payload.update(kwargs)
        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())
        if stream:
            return self._create_stream(request_id, created_time, model, payload, timeout, proxies)
        else:
            return self._create_non_stream(request_id, created_time, model, payload, timeout, proxies)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any],
        timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._client.session.post(
                self._client.base_url,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        if json_str == "[DONE]":
                            break
                        try:
                            data = json.loads(json_str)
                            choices = data.get('choices', [])
                            if not choices:
                                continue  # Skip if choices is empty
                            choice_data = choices[0]
                            delta_data = choice_data.get('delta', {})
                            finish_reason = choice_data.get('finish_reason')
                            delta = ChoiceDelta(
                                content=delta_data.get('content'),
                                role=delta_data.get('role'),
                                tool_calls=delta_data.get('tool_calls')
                            )
                            choice = Choice(
                                index=choice_data.get('index', 0),
                                delta=delta,
                                finish_reason=finish_reason,
                                logprobs=choice_data.get('logprobs')
                            )
                            chunk = ChatCompletionChunk(
                                id=data.get('id', request_id),
                                choices=[choice],
                                created=data.get('created', created_time),
                                model=data.get('model', model),
                                system_fingerprint=data.get('system_fingerprint'),
                            )
                            yield chunk
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.RequestException as e:
            print(f"Error during Friendli stream request: {e}")
            raise IOError(f"Friendli request failed: {e}") from e
        except Exception as e:
            print(f"Error processing Friendli stream: {e}")
            raise

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any],
        timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> ChatCompletion:
        try:
            response = self._client.session.post(
                self._client.base_url,
                headers=self._client.headers,
                json=payload,
                timeout=timeout or self._client.timeout,
                proxies=proxies
            )
            response.raise_for_status()
            data = response.json()
            choices_data = data.get('choices', [])
            usage_data = data.get('usage', {})
            choices = []
            for choice_d in choices_data:
                message_d = choice_d.get('message', {})
                message = ChatCompletionMessage(
                    role=message_d.get('role', 'assistant'),
                    content=message_d.get('content', '')
                )
                choice = Choice(
                    index=choice_d.get('index', 0),
                    message=message,
                    finish_reason=choice_d.get('finish_reason', 'stop')
                )
                choices.append(choice)
            usage = CompletionUsage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0)
            )
            completion = ChatCompletion(
                id=data.get('id', request_id),
                choices=choices,
                created=data.get('created', created_time),
                model=data.get('model', model),
                usage=usage,
            )
            return completion
        except requests.exceptions.RequestException as e:
            print(f"Error during Friendli non-stream request: {e}")
            raise IOError(f"Friendli request failed: {e}") from e
        except Exception as e:
            print(f"Error processing Friendli response: {e}")
            raise

class Chat(BaseChat):
    def __init__(self, client: 'Friendli'):
        self.completions = Completions(client)

class Friendli(OpenAICompatibleProvider):
    AVAILABLE_MODELS = [
        "deepseek-r1",
        # Add more as needed
    ]
    def __init__(self, browser: str = "chrome"):
        self.timeout = None
        self.base_url = "https://friendli.ai/serverless/v1/chat/completions"
        self.session = requests.Session()
        agent = LitAgent()
        fingerprint = agent.generate_fingerprint(browser)
        self.headers = {
            "Accept": fingerprint["accept"],
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": fingerprint["accept_language"],
            "Content-Type": "application/json",
            "Origin": "https://friendli.ai",
            "Referer": "https://friendli.ai/",
            "Sec-CH-UA": fingerprint["sec_ch_ua"] or '"Not)A;Brand";v="99", "Microsoft Edge";v="137", "Chromium";v="137"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{fingerprint["platform"]}"',
            "User-Agent": fingerprint["user_agent"],
            # Improved formatting for cookie header
            "cookie": (
                f"Next-Locale=en; "
                f"cookie-consent-state=rejected; "
                f"_gcl_au=1.1.2030343227.1749659739; "
                f"st-last-access-token-update=1749659740085; "
                f"_ga=GA1.1.912258413.1749659740; "
                f"AMP_MKTG_26fe53b9aa=JTdCJTdE; "
                f"pfTmpSessionVisitorContext=eb4334fe9f7540c7828d3ba71bab1fa7; "
                f"_fuid=MGVkY2IzZTItNDExNC00OTMxLWIyYjMtMDlhM2QyZDkwMTlj; "
                f"g_state={{\"i_p\":1749666944837,\"i_l\":1}}; "
                f"__stripe_mid={str(uuid.uuid4())}; "
                f"__stripe_sid={str(uuid.uuid4())}; "
                f"intercom-id-hcnpxbkh={str(uuid.uuid4())}; "
                f"intercom-session-hcnpxbkh=; "
                f"intercom-device-id-hcnpxbkh={str(uuid.uuid4())}; "
                f"AMP_26fe53b9aa=JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjJjOTJkMDYxYy0yYzBkLTQ4YTYtOGYzMy1kMjIzZTNjMzA1MzMlMjIlMkMlMjJzZXNzaW9uSWQlMjIlM0ExNzQ5NjU5NzQxMDkxJTJDJTIyb3B0T3V0JTIyJTNBZmFsc2UlMkMlMjJsYXN0RXZlbnRUaW1lJTIyJTNBMTc0OTY1OTc1NzQ5NiUyQyUyMmxhc3RFdmVudElkJTIyJTNBNCUyQyUyMnBhZ2VDb3VudGVyJTIyJTNBMiU3RA==; "
                f"_ga_PS0FM9F67K=GS2.1.s1749659740$o1$g1$t1749659771$j29$l0$h644129183"
            ),  # Replace with actual cookie
            "rid": "anti-csrf", # Replace with actual rid token if dynamic, otherwise keep as is
            "Sec-Fetch-Dest": "empty", # Keep existing headers
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }
        self.session.headers.update(self.headers)
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()

if __name__ == "__main__":
    client = Friendli()
    resp = client.chat.completions.create(
        model="deepseek-r1",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=True
    )
    for chunk in resp:
        print(chunk.choices[0].delta.content, end='', flush=True)  # Print each chunk as it arrives