import requests
import json
import time
import uuid
from typing import List, Dict, Optional, Union, Generator, Any

from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage, format_prompt, count_tokens
)

try:
    from webscout.litagent import LitAgent
except ImportError:
    pass

class Completions(BaseCompletions):
    def __init__(self, client: 'GptOss'):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 600,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        proxies: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        # Format messages into conversation prompt
        conversation_prompt = format_prompt(messages, add_special_tokens=False, do_continue=True)
        
        # Count tokens for usage tracking
        prompt_tokens = count_tokens(conversation_prompt)
        
        payload = {
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
        
        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())
        
        if stream:
            return self._create_stream(request_id, created_time, model, payload, timeout, proxies, prompt_tokens)
        else:
            return self._create_non_stream(request_id, created_time, model, payload, timeout, proxies, prompt_tokens)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any],
        timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None,
        prompt_tokens: int = 0
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
            
            completion_tokens = 0
            total_tokens = prompt_tokens
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    json_str = line[6:]
                    if json_str == "[DONE]":
                        break
                    try:
                        data = json.loads(json_str)
                        
                        # Extract content from GptOss response format
                        content = None
                        if (data.get('type') == 'thread.item_updated' and 
                            data.get('update', {}).get('type') == 'assistant_message.content_part.text_delta'):
                            content = data.get('update', {}).get('delta')
                        
                        if content:
                            # Count tokens in the content chunk
                            chunk_tokens = count_tokens(content)
                            completion_tokens += chunk_tokens
                            total_tokens = prompt_tokens + completion_tokens
                            
                            delta = ChoiceDelta(
                                content=content,
                                role="assistant"
                            )
                            choice = Choice(
                                index=0,
                                delta=delta,
                                finish_reason=None
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
                    except json.JSONDecodeError:
                        continue
            
            # Final chunk with finish_reason="stop"
            delta = ChoiceDelta(content=None, role=None)
            choice = Choice(index=0, delta=delta, finish_reason="stop")
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
            print(f"Error during GptOss stream request: {e}")
            raise IOError(f"GptOss request failed: {e}") from e

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any],
        timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None,
        prompt_tokens: int = 0
    ) -> ChatCompletion:
        try:
            response = self._client.session.post(
                self._client.base_url,
                headers=self._client.headers,
                json=payload,
                stream=True,  # GptOss API is event-stream only
                timeout=timeout or self._client.timeout,
                proxies=proxies
            )
            response.raise_for_status()
            
            # Collect all chunks to form complete response
            full_content = ""
            completion_tokens = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    json_str = line[6:]
                    if json_str == "[DONE]":
                        break
                    try:
                        data = json.loads(json_str)
                        
                        # Extract content from GptOss response format
                        if (data.get('type') == 'thread.item_updated' and 
                            data.get('update', {}).get('type') == 'assistant_message.content_part.text_delta'):
                            content = data.get('update', {}).get('delta')
                            if content:
                                full_content += content
                    except json.JSONDecodeError:
                        continue
            
            # Count tokens in the complete response
            completion_tokens = count_tokens(full_content)
            total_tokens = prompt_tokens + completion_tokens
            
            message = ChatCompletionMessage(
                role="assistant",
                content=full_content
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
                usage=usage
            )
            return completion
            
        except Exception as e:
            print(f"Error during GptOss non-stream request: {e}")
            raise IOError(f"GptOss request failed: {e}") from e

class Chat(BaseChat):
    def __init__(self, client: 'GptOss'):
        self.completions = Completions(client)

class GptOss(OpenAICompatibleProvider):
    AVAILABLE_MODELS = ["gpt-oss-20b", "gpt-oss-120b"]
    
    def __init__(
        self, 
        browser: str = "chrome", 
        api_key: str = None,
        model: str = "gpt-oss-120b",
        reasoning_effort: str = "high",
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(api_key=api_key, **kwargs)
        self.timeout = timeout
        self.base_url = "https://api.gpt-oss.com/chatkit"
        self.model = model if model in self.AVAILABLE_MODELS else self.AVAILABLE_MODELS[0]
        self.reasoning_effort = reasoning_effort
        self.session = requests.Session()
        
        # Generate headers using LitAgent
        try:
            agent = LitAgent()
            fingerprint = agent.generate_fingerprint(browser)
            self.headers = {
                "Accept": "text/event-stream",
                "Accept-Encoding": fingerprint.get("accept_encoding", "gzip, deflate, br"),
                "Accept-Language": fingerprint.get("accept_language", "en-US,en;q=0.9"),
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Pragma": "no-cache",
                "User-Agent": fingerprint.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
                "x-reasoning-effort": self.reasoning_effort,
                "x-selected-model": self.model,
                "x-show-reasoning": "true"
            }
        except:
            # Fallback headers if LitAgent fails
            self.headers = {
                "Accept": "text/event-stream",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Pragma": "no-cache",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "x-reasoning-effort": self.reasoning_effort,
                "x-selected-model": self.model,
                "x-show-reasoning": "true"
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
    client = GptOss()
    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=100,
        stream=False
    )
    print(response)