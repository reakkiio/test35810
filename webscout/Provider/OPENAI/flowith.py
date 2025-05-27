from typing import List, Dict, Optional, Union, Generator, Any
import time
import json

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)

# Import requests for HTTP requests (instead of curl_cffi)
import requests
import zstandard as zstd
import uuid

# Attempt to import LitAgent, fallback if not available
try:
    from webscout.litagent import LitAgent
except ImportError:
    class LitAgent:
        def generate_fingerprint(self, browser):
            return {"user_agent": "Mozilla/5.0"}

# --- Flowith OpenAI-Compatible Client ---

class Completions(BaseCompletions):
    def __init__(self, client: 'Flowith'):
        self.client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 2048,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Implements OpenAI-compatible chat/completions endpoint for Flowith.
        """
        url = "https://edge.flowith.net/ai/chat?mode=general"
        agent = LitAgent()
        fingerprint = agent.generate_fingerprint("chrome")
        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json",
            "origin": "https://flowith.io",
            "referer": "https://edge.flowith.net/",
            "user-agent": fingerprint["user_agent"],
            "dnt": "1",
            "sec-gpc": "1"
        }
        session = requests.Session()
        session.headers.update(headers)
        node_id = str(uuid.uuid4())
        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "nodeId": node_id
        }
        payload.update(kwargs)

        def for_stream():
            try:
                print(f"[DEBUG] Sending streaming request to {url} with payload: {payload}")
                response = session.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=timeout or 30,
                    proxies=proxies
                )
                print(f"[DEBUG] Response status: {response.status_code}")
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        break
                    text = chunk.decode('utf-8', errors='replace')
                    print(f"[DEBUG] Stream chunk: {repr(text)}")
                    delta = ChoiceDelta(content=text, role="assistant")
                    choice = Choice(index=0, delta=delta)
                    chunk_obj = ChatCompletionChunk(
                        id=request_id,
                        choices=[choice],
                        created=created_time,
                        model=model,
                        system_fingerprint=None
                    )
                    yield chunk_obj
            except Exception as e:
                print(f"[DEBUG] Streaming error: {e}")
                raise RuntimeError(f"Flowith streaming request failed: {e}")

        def for_non_stream():
            try:
                print(f"[DEBUG] Sending non-stream request to {url} with payload: {payload}")
                response = session.post(
                    url,
                    json=payload,
                    timeout=timeout or 30,
                    proxies=proxies
                )
                print(f"[DEBUG] Response status: {response.status_code}")
                response.raise_for_status()
                encoding = response.headers.get('Content-Encoding', '').lower()
                print(f"[DEBUG] Response encoding: {encoding}")
                if encoding == 'zstd':
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(response.raw) as reader:
                        decompressed = reader.read()
                        text = decompressed.decode('utf-8', errors='replace')
                else:
                    text = response.text
                print(f"[DEBUG] Raw response text: {repr(text)}")
                # Flowith returns raw text, not JSON
                content = text.strip()
                print(f"[DEBUG] Final content for ChatCompletion: {repr(content)}")
                message = ChatCompletionMessage(role="assistant", content=content)
                choice = Choice(index=0, message=message, finish_reason="stop")
                usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                completion = ChatCompletion(
                    id=request_id,
                    choices=[choice],
                    created=created_time,
                    model=model,
                    usage=usage
                )
                print(f"[DEBUG] Returning ChatCompletion: {completion}")
                return completion
            except Exception as e:
                print(f"[DEBUG] Non-streaming error: {e}")
                raise RuntimeError(f"Flowith request failed: {e}")

        return for_stream() if stream else for_non_stream()

class Chat(BaseChat):
    def __init__(self, client: 'Flowith'):
        self.completions = Completions(client)

class Flowith(OpenAICompatibleProvider):
    AVAILABLE_MODELS = [
        "gpt-4.1-mini", "deepseek-chat", "deepseek-reasoner", "claude-3.5-haiku",
        "gemini-2.0-flash", "gemini-2.5-flash", "grok-3-mini"
    ]

    chat: Chat
    def __init__(self):
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()
