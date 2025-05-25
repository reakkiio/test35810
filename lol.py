import requests
import json
import time
import uuid
from typing import List, Dict, Optional, Union, Generator, Any

from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage,
    get_last_user_message, get_system_prompt
)

class Completions(BaseCompletions):
    def __init__(self, client: 'QwenQwen3'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 2048,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        payload = {
            "data": [
                get_last_user_message(messages),
                {
                    "thinking_budget": kwargs.get("thinking_budget", 38),
                    "model": self._client.get_model(model),
                    "sys_prompt": get_system_prompt(messages)
                },
                None, None
            ],
            "event_data": None,
            "fn_index": 13,
            "trigger_id": 31,
            "session_hash": str(uuid.uuid4()).replace('-', '')
        }

        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload)
        else:
            return self._create_non_stream(request_id, created_time, model, payload)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        session = self._client.session
        headers = self._client.headers
        # Step 1: Join the queue
        join_resp = session.post(self._client.api_endpoint, headers=headers, json=payload, timeout=self._client.timeout)
        join_resp.raise_for_status()
        event_id = join_resp.json().get('event_id')
        session_hash = payload["session_hash"]

        # Step 2: Stream data
        params = {'session_hash': session_hash}
        stream_resp = session.get(self._client.url + "/gradio_api/queue/data", headers=self._client.stream_headers, params=params, stream=True, timeout=self._client.timeout)
        stream_resp.raise_for_status()

        # --- New logic to yield all content, tool reasoning, and status, similar to Reasoning class ---
        is_thinking = False
        label = None
        for line in stream_resp.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    try:
                        json_data = json.loads(decoded_line[6:])
                        if json_data.get('msg') == 'process_generating':
                            if 'output' in json_data and 'data' in json_data['output'] and len(json_data['output']['data']) > 5:
                                updates = json_data['output']['data'][5]
                                for update in updates:
                                    # Tool reasoning as dict
                                    if isinstance(update[2], dict):
                                        if update[2].get('type') == 'tool':
                                            content = update[2].get('content')
                                            status = update[2].get('options', {}).get('title')
                                            label = status
                                            is_thinking = True
                                            delta = ChoiceDelta(content=content, role="tool")
                                            choice = Choice(index=0, delta=delta)
                                            chunk = ChatCompletionChunk(
                                                id=request_id,
                                                choices=[choice],
                                                created=created_time,
                                                model=model
                                            )
                                            yield chunk
                                        elif update[2].get('type') == 'text':
                                            content = update[2].get('content')
                                            is_thinking = False
                                            delta = ChoiceDelta(content=content, role="assistant")
                                            choice = Choice(index=0, delta=delta)
                                            chunk = ChatCompletionChunk(
                                                id=request_id,
                                                choices=[choice],
                                                created=created_time,
                                                model=model
                                            )
                                            yield chunk
                                    # Plain text content
                                    elif isinstance(update, list) and isinstance(update[1], list) and len(update[1]) > 4:
                                        if update[1][4] == "content":
                                            content = update[2]
                                            delta = ChoiceDelta(content=content, role="assistant")
                                            choice = Choice(index=0, delta=delta)
                                            chunk = ChatCompletionChunk(
                                                id=request_id,
                                                choices=[choice],
                                                created=created_time,
                                                model=model
                                            )
                                            yield chunk
                                        elif update[1][4] == "options":
                                            if update[2] != "done":
                                                status = update[2]
                                                delta = ChoiceDelta(content=None, role="tool")
                                                choice = Choice(index=0, delta=delta)
                                                chunk = ChatCompletionChunk(
                                                    id=request_id,
                                                    choices=[choice],
                                                    created=created_time,
                                                    model=model
                                                )
                                                yield chunk
                                            is_thinking = False
                        if json_data.get('msg') == 'process_completed':
                            break
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        # Log or handle other potential exceptions
                        continue

    def _extract_content(self, json_data):
        # Extracts the text content from the Qwen Qwen-3 streaming response
        try:
            if 'output' in json_data and 'data' in json_data['output'] and len(json_data['output']['data']) > 5:
                updates = json_data['output']['data'][5]
                for update in updates:
                    if isinstance(update, list) and len(update) > 2 and isinstance(update[2], str):
                        return update[2]
        except Exception:
            pass
        return None

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any]
    ) -> ChatCompletion:
        # For non-streaming, just call the join endpoint and parse the result
        session = self._client.session
        headers = self._client.headers
        resp = session.post(self._client.api_endpoint, headers=headers, json=payload, timeout=self._client.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Return the full content as a single message, including all tool and text reasoning if present
        output = ""
        if 'output' in data and 'data' in data['output'] and len(data['output']['data']) > 5:
            updates = data['output']['data'][5]
            parts = []
            for update in updates:
                if isinstance(update, list) and len(update) > 2 and isinstance(update[2], str):
                    parts.append(update[2])
                elif isinstance(update, list) and isinstance(update[1], list) and len(update[1]) > 4:
                    if update[1][4] == "content":
                        parts.append(update[2])
                    elif update[1][4] == "options" and update[2] != "done":
                        parts.append(str(update[2]))
                elif isinstance(update, dict):
                    if update.get('type') == 'tool':
                        parts.append(update.get('content', ''))
                    elif update.get('type') == 'text':
                        parts.append(update.get('content', ''))
            output = "\n".join([str(p) for p in parts if p])
        else:
            output = data.get('output', {}).get('data', ["", "", "", "", "", [["", "", ""]]])[5][0][2]
        message = ChatCompletionMessage(role="assistant", content=output)
        choice = Choice(index=0, message=message, finish_reason="stop")
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=len(output), total_tokens=len(output))
        completion = ChatCompletion(
            id=request_id,
            choices=[choice],
            created=created_time,
            model=model,
            usage=usage,
        )
        return completion

class Chat(BaseChat):
    def __init__(self, client: 'QwenQwen3'):
        self.completions = Completions(client)

class QwenQwen3(OpenAICompatibleProvider):
    url = "https://qwen-qwen3-demo.hf.space"
    api_endpoint = "https://qwen-qwen3-demo.hf.space/gradio_api/queue/join?__theme=system"
    AVAILABLE_MODELS = [
        "qwen3-235b-a22b",
        "qwen3-32b",
        "qwen3-30b-a3b",
        "qwen3-14b",
        "qwen3-8b",
        "qwen3-4b",
        "qwen3-1.7b",
        "qwen3-0.6b",
    ]
    MODEL_ALIASES = {
        "qwen-3-235b": "qwen3-235b-a22b",
        "qwen-3-30b": "qwen3-30b-a3b",
        "qwen-3-32b": "qwen3-32b",
        "qwen-3-14b": "qwen3-14b",
        "qwen-3-4b": "qwen3-4b",
        "qwen-3-1.7b": "qwen3-1.7b",
        "qwen-3-0.6b": "qwen3-0.6b"
    }

    def __init__(self, timeout: Optional[int] = None):
        self.timeout = timeout
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Referer': f'{self.url}/?__theme=system',
            'content-type': 'application/json',
            'Origin': self.url,
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }
        self.stream_headers = {
            'Accept': 'text/event-stream',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': f'{self.url}/?__theme=system',
            'User-Agent': self.headers['User-Agent'],
        }
        self.session.headers.update(self.headers)
        self.chat = Chat(self)

    def get_model(self, model):
        return self.MODEL_ALIASES.get(model, model)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()

if __name__ == "__main__":
    client = QwenQwen3()
    resp = client.chat.completions.create(
        model="qwen3-14b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello "}
        ],
        stream=True
    )
    for chunk in resp:
        print(chunk.choices[0].delta.content, end="", flush=True)