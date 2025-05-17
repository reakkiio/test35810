from typing import List, Dict, Optional, Union, Generator, Any
import time
import json
from webscout.litagent import LitAgent
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChatCompletionMessage,
    ChoiceDelta,
    CompletionUsage,
    format_prompt
)
import requests

# ANSI escape codes for formatting
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"

class Completions(BaseCompletions):
    def __init__(self, client: 'ChatSandbox'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        OpenAI-compatible chat/completions endpoint for ChatSandbox.
        """
        # Use model name conversion for compatibility
        model = self._client.convert_model_name(model)
        # Compose the conversation prompt using format_prompt
        question = format_prompt(messages, add_special_tokens=True)
        payload = {
            "messages": [question],
            "character": model
        }
        request_id = f"chatcmpl-{int(time.time() * 1000)}"
        created_time = int(time.time())
        url = "https://chatsandbox.com/api/chat"
        agent = LitAgent()
        headers = {
            'authority': 'chatsandbox.com',
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': 'https://chatsandbox.com',
            'referer': f'https://chatsandbox.com/chat/{model}',
            'sec-ch-ua': '"Microsoft Edge";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': agent.random(),
            'dnt': '1',
            'sec-gpc': '1',
        }
        session = requests.Session()
        session.headers.update(headers)
        def for_stream():
            try:
                response = session.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=30
                )
                response.raise_for_status()
                streaming_text = ""
                for chunk in response.iter_content(chunk_size=None):
                    if not chunk:
                        continue
                    text = chunk.decode('utf-8', errors='replace')
                    try:
                        data = json.loads(text)
                        content = data.get("reasoning_content", text)
                    except Exception:
                        content = text
                    streaming_text += content
                    delta = ChoiceDelta(content=content)
                    choice = Choice(index=0, delta=delta, finish_reason=None)
                    chunk_obj = ChatCompletionChunk(
                        id=request_id,
                        choices=[choice],
                        created=created_time,
                        model=model,
                    )
                    yield chunk_obj
                # Final chunk
                delta = ChoiceDelta(content=None)
                choice = Choice(index=0, delta=delta, finish_reason="stop")
                chunk_obj = ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=created_time,
                    model=model,
                )
                yield chunk_obj
            except Exception as e:
                raise RuntimeError(f"ChatSandbox streaming request failed: {e}")
        def for_non_stream():
            try:
                response = session.post(
                    url,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                text = response.text
                try:
                    data = json.loads(text)
                    content = data.get("reasoning_content", text)
                except Exception:
                    content = text
                prompt_tokens = len(question) // 4
                completion_tokens = len(content) // 4
                total_tokens = prompt_tokens + completion_tokens
                usage = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
                message = ChatCompletionMessage(role="assistant", content=content)
                choice = Choice(index=0, message=message, finish_reason="stop")
                completion = ChatCompletion(
                    id=request_id,
                    choices=[choice],
                    created=created_time,
                    model=model,
                    usage=usage,
                )
                return completion
            except Exception as e:
                raise RuntimeError(f"ChatSandbox request failed: {e}")
        return for_stream() if stream else for_non_stream()

class Chat(BaseChat):
    def __init__(self, client: 'ChatSandbox'):
        self.completions = Completions(client)

class ChatSandbox(OpenAICompatibleProvider):
    AVAILABLE_MODELS = ["openai", "deepseek", "llama", "gemini", "mistral-large"]
    chat: Chat
    def __init__(self):
        self.chat = Chat(self)
    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()
    def convert_model_name(self, model: str) -> str:
        if model in self.AVAILABLE_MODELS:
            return model
        for available_model in self.AVAILABLE_MODELS:
            if model.lower() in available_model.lower():
                return available_model
        # Default to openai if no match
        print(f"{RED}{BOLD}Warning: Model '{model}' not found, using default model 'openai'{RESET}")
        return "openai"
