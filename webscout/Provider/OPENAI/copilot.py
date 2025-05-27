import json
import uuid
import time
from typing import List, Dict, Optional, Union, Generator, Any
from urllib.parse import quote
from curl_cffi.requests import Session, CurlWsFlag

# Import base classes and utility structures
from .base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from .utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage, format_prompt, count_tokens
)

# Attempt to import LitAgent, fallback if not available
try:
    from webscout.litagent import LitAgent
except ImportError:
    pass

# --- Microsoft Copilot Client ---

class Completions(BaseCompletions):
    def __init__(self, client: 'Copilot'):
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
        proxies: Optional[dict] = None,
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Creates a model response for the given chat conversation.
        Mimics openai.chat.completions.create
        """
        # Format the entire conversation using the utility function
        formatted_prompt = format_prompt(messages, add_special_tokens=True, include_system=True, do_continue=True)
        
        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        # Handle image if provided
        image = kwargs.get("image")

        if stream:
            return self._create_stream(request_id, created_time, model, formatted_prompt, image, timeout=timeout, proxies=proxies)
        else:
            return self._create_non_stream(request_id, created_time, model, formatted_prompt, image, timeout=timeout, proxies=proxies)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, prompt_text: str, image: Optional[bytes] = None,
        timeout: Optional[int] = None, proxies: Optional[dict] = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        original_proxies = self._client.session.proxies
        if proxies is not None:
            self._client.session.proxies = proxies
        else:
            self._client.session.proxies = {}
        try:
            timeout_val = timeout if timeout is not None else self._client.timeout
            s = self._client.session
            # Create a new conversation if needed
            r = s.post(self._client.conversation_url, timeout=timeout_val)
            if r.status_code != 200:
                raise RuntimeError(f"Failed to create conversation: {r.text}")
            conv_id = r.json().get("id")

            # Handle image upload if provided
            images = []
            if image:
                r = s.post(
                    f"{self._client.url}/c/api/attachments",
                    headers={"content-type": "image/jpeg"},
                    data=image,
                    timeout=timeout_val
                )
                if r.status_code != 200:
                    raise RuntimeError(f"Image upload failed: {r.text}")
                images.append({"type": "image", "url": r.json().get("url")})

            # Connect to websocket
            # Note: ws_connect might not use timeout in the same way as POST/GET
            ws = s.ws_connect(self._client.websocket_url)

            # Use model to set mode ("reasoning" for Think Deeper)
            mode = "reasoning" if "Think" in model else "chat"

            # Send the message to Copilot
            ws.send(json.dumps({
                "event": "send",
                "conversationId": conv_id,
                "content": images + [{"type": "text", "text": prompt_text}],
                "mode": mode
            }).encode(), CurlWsFlag.TEXT)

            # Track token usage using count_tokens
            prompt_tokens = count_tokens(prompt_text)
            completion_tokens = 0
            total_tokens = prompt_tokens

            started = False
            while True:
                try:
                    msg = json.loads(ws.recv()[0])
                except Exception:
                    break

                if msg.get("event") == "appendText":
                    started = True
                    content = msg.get("text", "")
                    
                    # Update token counts using count_tokens
                    content_tokens = count_tokens(content)
                    completion_tokens += content_tokens
                    total_tokens = prompt_tokens + completion_tokens

                    # Create the delta object
                    delta = ChoiceDelta(
                        content=content,
                        role="assistant"
                    )

                    # Create the choice object
                    choice = Choice(
                        index=0,
                        delta=delta,
                        finish_reason=None
                    )

                    # Create the chunk object
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        choices=[choice],
                        created=created_time,
                        model=model
                    )

                    yield chunk
                elif msg.get("event") == "done":
                    # Final chunk with finish_reason
                    delta = ChoiceDelta(
                        content=None,
                        role=None
                    )

                    choice = Choice(
                        index=0,
                        delta=delta,
                        finish_reason="stop"
                    )

                    chunk = ChatCompletionChunk(
                        id=request_id,
                        choices=[choice],
                        created=created_time,
                        model=model
                    )

                    yield chunk
                    break
                elif msg.get("event") == "error":
                    raise RuntimeError(f"Copilot error: {msg}")

            ws.close()

            if not started:
                raise RuntimeError("No response received from Copilot")

        except Exception as e:
            raise RuntimeError(f"Stream error: {e}") from e
        finally:
            self._client.session.proxies = original_proxies

    def _create_non_stream(
        self, request_id: str, created_time: int, model: str, prompt_text: str, image: Optional[bytes] = None,
        timeout: Optional[int] = None, proxies: Optional[dict] = None
    ) -> ChatCompletion:
        result = ""
        # Pass timeout and proxies to the underlying _create_stream call
        for chunk in self._create_stream(request_id, created_time, model, prompt_text, image, timeout=timeout, proxies=proxies):
            if hasattr(chunk, 'choices') and chunk.choices and hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content

        # Create the message object
        message = ChatCompletionMessage(
            role="assistant",
            content=result
        )

        # Create the choice object
        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop"
        )

        # Estimate token usage using count_tokens
        prompt_tokens = count_tokens(prompt_text)
        completion_tokens = count_tokens(result)
        total_tokens = prompt_tokens + completion_tokens

        # Create usage object
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

        # Create the completion object
        completion = ChatCompletion(
            id=request_id,
            choices=[choice],
            created=created_time,
            model=model,
            usage=usage
        )

        return completion

class Chat(BaseChat):
    def __init__(self, client: 'Copilot'):
        self.completions = Completions(client)

class Copilot(OpenAICompatibleProvider):
    
    url = "https://copilot.microsoft.com"
    conversation_url = f"{url}/c/api/conversations"
    websocket_url = "wss://copilot.microsoft.com/c/api/chat?api-version=2"
    
    AVAILABLE_MODELS = ["Copilot", "Think Deeper"]

    def __init__(self, browser: str = "chrome", tools: Optional[List] = None, **kwargs):
        self.timeout = 900
        self.session = Session(impersonate=browser)
        self.session.proxies = {}

        # Initialize tools
        self.available_tools = {}
        if tools:
            self.register_tools(tools)
            
        # Set up the chat interface
        self.chat = Chat(self)

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return self.AVAILABLE_MODELS
        return _ModelList()
