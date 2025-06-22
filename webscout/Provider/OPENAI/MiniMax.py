import os
import requests
import json
import time
import uuid
from typing import List, Dict, Optional, Union, Generator, Any

from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage, count_tokens
)

class Completions(BaseCompletions):
    def __init__(self, client: 'MiniMax'):
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
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Creates a model response for the given chat conversation.
        Mimics openai.chat.completions.create
        """
        api_key = self._client.api_key
        if not api_key:
            raise Exception("MINIMAX_API_KEY not set in environment.")
        model_name = self._client.convert_model_name(model)
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        payload.update(kwargs)
        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())
        if stream:
            return self._create_stream(request_id, created_time, model_name, payload, timeout, proxies)
        else:
            return self._create_non_stream(request_id, created_time, model_name, payload, timeout, proxies)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any], timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self._client.api_key}',
            }
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies
            )
            response.raise_for_status()
            prompt_tokens = count_tokens([msg.get("content", "") for msg in payload.get("messages", [])])
            completion_tokens = 0
            total_tokens = prompt_tokens
            streaming_response = ""
            last_content = ""
            last_reasoning = ""
            in_think = False
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(line)
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice_data = chunk_data['choices'][0]
                                delta = choice_data.get('delta', {})
                                content = delta.get('content')
                                reasoning_content = delta.get('reasoning_content')
                                finish_reason = choice_data.get('finish_reason')
                                # Only yield <think> and reasoning_content if reasoning_content is not empty
                                if reasoning_content and reasoning_content.strip() and reasoning_content != last_reasoning:
                                    if not in_think:
                                        yield ChatCompletionChunk(
                                            id=request_id,
                                            choices=[Choice(index=0, delta=ChoiceDelta(content='<think>\n\n', role=None, tool_calls=None), finish_reason=None, logprobs=None)],
                                            created=created_time,
                                            model=model
                                        )
                                        in_think = True
                                    yield ChatCompletionChunk(
                                        id=request_id,
                                        choices=[Choice(index=0, delta=ChoiceDelta(content=reasoning_content, role=None, tool_calls=None), finish_reason=None, logprobs=None)],
                                        created=created_time,
                                        model=model
                                    )
                                    last_reasoning = reasoning_content
                                # Only yield </think> if we were in <think> and now have new content
                                if in_think and content and content.strip() and content != last_content:
                                    yield ChatCompletionChunk(
                                        id=request_id,
                                        choices=[Choice(index=0, delta=ChoiceDelta(content='</think>\n\n', role=None, tool_calls=None), finish_reason=None, logprobs=None)],
                                        created=created_time,
                                        model=model
                                    )
                                    in_think = False
                                # Only yield content if it is not empty
                                if content and content.strip() and content != last_content:
                                    completion_tokens += count_tokens(content)
                                    total_tokens = prompt_tokens + completion_tokens
                                    choice_delta = ChoiceDelta(
                                        content=content,
                                        role=delta.get('role', 'assistant'),
                                        tool_calls=delta.get('tool_calls')
                                    )
                                    choice = Choice(
                                        index=0,
                                        delta=choice_delta,
                                        finish_reason=finish_reason,
                                        logprobs=None
                                    )
                                    chunk = ChatCompletionChunk(
                                        id=request_id,
                                        choices=[choice],
                                        created=created_time,
                                        model=model
                                    )
                                    chunk.usage = {
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": total_tokens,
                                        "estimated_cost": None
                                    }
                                    yield chunk
                                    streaming_response += content
                                    last_content = content
                        except Exception:
                            continue
            # Final chunk with finish_reason="stop"
            delta = ChoiceDelta(content=None, role=None, tool_calls=None)
            choice = Choice(index=0, delta=delta, finish_reason="stop", logprobs=None)
            chunk = ChatCompletionChunk(
                id=request_id,
                choices=[choice],
                created=created_time,
                model=model
            )
            chunk.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": None
            }
            yield chunk
        except Exception as e:
            raise IOError(f"MiniMax stream request failed: {e}") from e

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any], timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> ChatCompletion:
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self._client.api_key}',
            }
            payload_copy = payload.copy()
            payload_copy["stream"] = False
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=headers,
                data=json.dumps(payload_copy),
                timeout=timeout or self._client.timeout,
                proxies=proxies
            )
            response.raise_for_status()
            data = response.json()
            full_text = ""
            finish_reason = "stop"
            if 'choices' in data and data['choices']:
                choice_data = data['choices'][0]
                # MiniMax returns content in 'message' or directly in 'delta' for streaming
                reasoning_content = ""
                if 'message' in choice_data and choice_data['message']:
                    full_text = choice_data['message'].get('content', '')
                    reasoning_content = choice_data['message'].get('reasoning_content', '')
                elif 'delta' in choice_data and choice_data['delta']:
                    full_text = choice_data['delta'].get('content', '')
                    reasoning_content = choice_data['delta'].get('reasoning_content', '')
                finish_reason = choice_data.get('finish_reason', 'stop')
                # If both are present, concatenate with <think> ... </think>
                if reasoning_content and reasoning_content.strip():
                    if full_text and full_text.strip():
                        full_text = f"<think>\n\n{reasoning_content}</think>\n\n{full_text}"
                    else:
                        full_text = f"<think>\n\n{reasoning_content}</think>\n\n"
            message = ChatCompletionMessage(
                role="assistant",
                content=full_text
            )
            choice = Choice(
                index=0,
                message=message,
                finish_reason=finish_reason
            )
            prompt_tokens = count_tokens([msg.get("content", "") for msg in payload.get("messages", [])])
            completion_tokens = count_tokens(full_text)
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
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
            raise IOError(f"MiniMax non-stream request failed: {e}") from e

class Chat(BaseChat):
    def __init__(self, client: 'MiniMax'):
        self.completions = Completions(client)

class MiniMax(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for MiniMax API.
    """
    AVAILABLE_MODELS = [
        "MiniMax-Reasoning-01"
    ]
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.api_endpoint = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
        self.session = requests.Session()
        self.api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJtbyBuaSIsIlVzZXJOYW1lIjoibW8gbmkiLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTg3NjIwMDY0ODA2NDYzNTI0MiIsIlBob25lIjoiIiwiR3JvdXBJRCI6IjE4NzYyMDA2NDgwNjA0NDA5MzgiLCJQYWdlTmFtZSI6IiIsIk1haWwiOiJuaW1vQHN1YnN1cC52aXAiLCJDcmVhdGVUaW1lIjoiMjAyNS0wMS0wNyAxMToyNzowNyIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.Ge1ZnpFPUfXVdMini0P_qXbP_9VYwzXiffG9DsNQck4GtYEOs33LDeAiwrVsrrLZfvJ2icQZ4sRZS54wmPuWua_Dav6pYJty8ZtahmUX1IuhlUX5YErhhCRAIy3J1xB8FkLHLyylChuBHpkNz6O6BQLmPqmoa-cOYK9Qrc6IDeu8SX1iMzO9-MSkcWNvkvpCF2Pf9tekBVWNKMDK6IZoMEPbtkaPXdDyP6l0M0e2AlL_E0oM9exg3V-ohAi8OTPFyqM6dcd4TwF-b9DULxfIsRFw401mvIxcTDWa42u2LULewdATVRD2BthU65tuRqEiWeFWMvFlPj2soMze_QIiUA"
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return MiniMax.AVAILABLE_MODELS
        return _ModelList()

    def convert_model_name(self, model: str) -> str:
        if model in self.AVAILABLE_MODELS:
            return model
        return self.AVAILABLE_MODELS[0]

if __name__ == "__main__":
    from rich import print
    client = MiniMax()
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    # Non-streaming example
    response = client.chat.completions.create(
        model="MiniMax-Reasoning-01",
        messages=messages,
        max_tokens=5000,
        stream=False
    )
    print("Non-streaming response:")
    print(response)
    # Streaming example
    print("\nStreaming response:")
    stream = client.chat.completions.create(
        model="MiniMax-Reasoning-01",
        messages=messages,
        max_tokens=5000,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    print()