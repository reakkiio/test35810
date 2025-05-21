from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import time
import uuid
import re
import urllib.parse
from typing import List, Dict, Optional, Union, Generator, Any

from webscout.Extra.tempmail import get_random_email
from webscout.litagent import LitAgent

# Import base classes and utilities from OPENAI provider stack
from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)

# Attempt to import LitAgent for browser fingerprinting
try:
    from webscout.litagent import LitAgent
except ImportError:  # pragma: no cover - LitAgent optional
    LitAgent = None


class Completions(BaseCompletions):
    """TwoAI chat completions compatible with OpenAI format."""

    def __init__(self, client: 'TwoAI'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = 2049,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Create a chat completion using TwoAI."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        payload.update(kwargs)

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload)
        return self._create_non_stream(request_id, created_time, model, payload)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            response = self._client.session.post(
                self._client.base_url,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=self._client.timeout,
            )
            response.raise_for_status()

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                json_str = decoded[6:]
                if json_str == "[DONE]":
                    break
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                choice_data = data.get("choices", [{}])[0]
                delta_data = choice_data.get("delta", {})
                finish_reason = choice_data.get("finish_reason")

                usage_data = data.get("usage", {})
                if usage_data:
                    prompt_tokens = usage_data.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage_data.get(
                        "completion_tokens", completion_tokens
                    )
                    total_tokens = usage_data.get("total_tokens", total_tokens)

                delta = ChoiceDelta(
                    content=delta_data.get("content"),
                    role=delta_data.get("role"),
                    tool_calls=delta_data.get("tool_calls"),
                )

                choice = Choice(
                    index=choice_data.get("index", 0),
                    delta=delta,
                    finish_reason=finish_reason,
                    logprobs=choice_data.get("logprobs"),
                )

                chunk = ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=created_time,
                    model=model,
                    system_fingerprint=data.get("system_fingerprint"),
                )

                yield chunk
        except Exception as e:
            raise IOError(f"TwoAI request failed: {e}") from e
        except Exception as e:
            raise IOError(f"Error processing TwoAI stream: {e}") from e

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any]
    ) -> ChatCompletion:
        try:
            response = self._client.session.post(
                self._client.base_url,
                headers=self._client.headers,
                json=payload,
                timeout=self._client.timeout,
            )
            response.raise_for_status()
            data = response.json()

            choices_data = data.get("choices", [])
            usage_data = data.get("usage", {})

            choices = []
            for choice_d in choices_data:
                message_d = choice_d.get("message", {})
                message = ChatCompletionMessage(
                    role=message_d.get("role", "assistant"),
                    content=message_d.get("content", ""),
                    tool_calls=message_d.get("tool_calls"),
                )
                choice = Choice(
                    index=choice_d.get("index", 0),
                    message=message,
                    finish_reason=choice_d.get("finish_reason", "stop"),
                )
                choices.append(choice)

            usage = CompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            completion = ChatCompletion(
                id=request_id,
                choices=choices,
                created=created_time,
                model=data.get("model", model),
                usage=usage,
            )
            return completion
        except Exception as e:
            raise IOError(f"TwoAI request failed: {e}") from e
        except Exception as e:
            raise IOError(f"Error processing TwoAI response: {e}") from e


class Chat(BaseChat):
    def __init__(self, client: 'TwoAI'):
        self.completions = Completions(client)


class TwoAI(OpenAICompatibleProvider):
    """OpenAI-compatible client for the TwoAI API."""

    AVAILABLE_MODELS = ["sutra-v2", "sutra-r0"]

    @staticmethod
    def generate_api_key() -> str:
        """
        Generate a new Two AI API key using a temporary email.
        """
        email, provider = get_random_email("tempmailio")
        loops_url = "https://app.loops.so/api/newsletter-form/cm7i4o92h057auy1o74cxbhxo"

        session = Session()
        session.headers.update({
            'User-Agent': LitAgent().random(),
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://www.two.ai',
            'Referer': 'https://app.loops.so/',
        })

        form_data = {
            'email': email,
            'userGroup': 'Via Framer',
            'mailingLists': 'cm8ay9cic00x70kjv0bd34k66'
        }

        encoded_data = urllib.parse.urlencode(form_data)
        response = session.post(loops_url, data=encoded_data, impersonate="chrome120")

        if response.status_code != 200:
            raise RuntimeError(f"Failed to register for Two AI: {response.status_code} - {response.text}")

        max_attempts = 15
        attempt = 0
        api_key = None
        wait_time = 2

        while attempt < max_attempts and not api_key:
            messages = provider.get_messages()
            for message in messages:
                subject = message.get('subject', '')
                sender = ''
                if 'from' in message:
                    if isinstance(message['from'], dict):
                        sender = message['from'].get('address', '')
                    else:
                        sender = str(message['from'])
                elif 'sender' in message:
                    if isinstance(message['sender'], dict):
                        sender = message['sender'].get('address', '')
                    else:
                        sender = str(message['sender'])
                subject_match = any(keyword in subject.lower() for keyword in
                                    ['welcome', 'confirm', 'verify', 'api', 'key', 'sutra', 'two.ai', 'loops'])
                sender_match = any(keyword in sender.lower() for keyword in
                                   ['two.ai', 'sutra', 'loops.so', 'loops', 'no-reply', 'noreply'])
                is_confirmation = subject_match or sender_match

                content = None
                if 'body' in message:
                    content = message['body']
                elif 'content' in message and 'text' in message['content']:
                    content = message['content']['text']
                elif 'html' in message:
                    content = message['html']
                elif 'text' in message:
                    content = message['text']
                if not content:
                    continue

                # Robust API key extraction with multiple regex patterns
                patterns = [
                    r'sutra_[A-Za-z0-9]{60,70}',
                    r'sutra_[A-Za-z0-9]{30,}',
                    r'sutra_\S+',
                ]
                api_key_match = None
                for pat in patterns:
                    api_key_match = re.search(pat, content)
                    if api_key_match:
                        break
                # Also try to extract from labeled section
                if not api_key_match:
                    key_section_match = re.search(r'🔑 SUTRA API Key\s*([^\s]+)', content)
                    if key_section_match:
                        api_key_match = re.search(r'sutra_[A-Za-z0-9]+', key_section_match.group(1))
                if api_key_match:
                    api_key = api_key_match.group(0)
                    break
            if not api_key:
                attempt += 1
                time.sleep(wait_time)
        if not api_key:
            raise RuntimeError("Failed to get API key from confirmation email")
        return api_key

    def __init__(self, timeout: Optional[int] = None, browser: str = "chrome"):
        api_key = self.generate_api_key()
        self.timeout = timeout
        self.base_url = "https://api.two.ai/v2/chat/completions"
        self.api_key = api_key
        self.session = Session()

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if LitAgent is not None:
            try:
                agent = LitAgent()
                fingerprint = agent.generate_fingerprint(browser)
                headers.update({
                    "Accept": fingerprint["accept"],
                    "Accept-Encoding": "gzip, deflate, br, zstd",
                    "Accept-Language": fingerprint["accept_language"],
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Origin": "https://chat.two.ai",
                    "Pragma": "no-cache",
                    "Referer": "https://chat.two.ai/",
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "same-site",
                    "Sec-CH-UA": fingerprint["sec_ch_ua"] or '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
                    "Sec-CH-UA-Mobile": "?0",
                    "Sec-CH-UA-Platform": f'"{fingerprint["platform"]}"',
                    "User-Agent": fingerprint["user_agent"],
                })
            except Exception:
                # Fallback minimal headers if fingerprinting fails
                headers.update({
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept-Language": "en-US,en;q=0.9",
                    "User-Agent": "Mozilla/5.0",
                })
        else:
            headers.update({
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": "Mozilla/5.0",
            })

        self.headers = headers
        self.session.headers.update(headers)
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()

if __name__ == "__main__":
    from rich import print
    two_ai = TwoAI()
    resp = two_ai.chat.completions.create(
        model="sutra-v2",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=True
    )
    for chunk in resp:
        print(chunk, end="")
