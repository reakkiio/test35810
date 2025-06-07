import time
import uuid
import json
import random
import string
from typing import List, Dict, Optional, Union, Generator, Any

# Use curl_cffi for requests
from curl_cffi.requests import Session
from curl_cffi import CurlError

# Import base classes and utility structures
from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)

# Import LitAgent for user agent generation
try:
    from webscout.litagent import LitAgent
except ImportError:
    # Define a dummy LitAgent if webscout is not installed or accessible
    class LitAgent:
        def random(self) -> str:
            # Return a default user agent if LitAgent is unavailable
            print("Warning: LitAgent not found. Using default minimal headers.")
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"

# ANSI escape codes for formatting
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"

class Completions(BaseCompletions):
    def __init__(self, client: 'MCPCore'):
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
        **kwargs: Any
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """
        Creates a model response for the given chat conversation using MCPCore API.
        Mimics openai.chat.completions.create
        """
        if model not in self._client.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model}' not supported by MCPCore. Available: {self._client.AVAILABLE_MODELS}")

        # Construct the MCPCore-specific payload
        payload = {
            "stream": stream,
            "model": model,
            "messages": messages,
            "params": kwargs.get("params", {}),
            "tool_servers": kwargs.get("tool_servers", []),
            "features": kwargs.get("features", {"web_search": False}),
            "chat_id": kwargs.get("chat_id", str(uuid.uuid4())),
            "id": str(uuid.uuid4()), # Message ID
            "stream_options": kwargs.get("stream_options", {"include_usage": True})
        }

        # Add optional OpenAI params to MCPCore's 'params' field if provided
        if temperature is not None: payload["params"]["temperature"] = temperature
        if top_p is not None: payload["params"]["top_p"] = top_p
        if max_tokens is not None: payload["params"]["max_tokens"] = max_tokens

        # Generate standard OpenAI-compatible IDs
        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        if stream:
            return self._create_stream(request_id, created_time, model, payload, timeout, proxies)
        else:
            return self._create_non_stream_from_stream(request_id, created_time, model, payload, timeout, proxies)

    def _create_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any], timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Handles the streaming response from MCPCore."""
        final_usage_data = None # To store usage if received
        try:
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None),
                impersonate="chrome110" # Impersonation often helps
            )

            if not response.ok:
                try:
                    error_text = response.text
                except Exception:
                    error_text = "<Failed to read error response>"
                raise IOError(
                    f"MCPCore API Error: {response.status_code} {response.reason} - {error_text}"
                 )

            for line_bytes in response.iter_lines():
                if line_bytes:
                    try:
                        line = line_bytes.decode('utf-8').strip()
                        if line.startswith("data: "):
                            json_str = line[6:]
                            if json_str == "[DONE]":
                                break # End of stream signal

                            json_data = json.loads(json_str)

                            # Check for usage data in the chunk (often comes near the end)
                            if 'usage' in json_data and json_data['usage']:
                                final_usage_data = json_data['usage']
                                # Don't yield a chunk just for usage, wait for content or final chunk

                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                choice_data = json_data['choices'][0]
                                delta_data = choice_data.get('delta', {})
                                finish_reason = choice_data.get('finish_reason')
                                content = delta_data.get('content')
                                role = delta_data.get('role', 'assistant') # Default role

                                # Only yield chunks with content or finish reason
                                if content is not None or finish_reason:
                                    delta = ChoiceDelta(content=content, role=role)
                                    choice = Choice(index=0, delta=delta, finish_reason=finish_reason)
                                    chunk = ChatCompletionChunk(
                                        id=request_id,
                                        choices=[choice],
                                        created=created_time,
                                        model=model,
                                        system_fingerprint=json_data.get('system_fingerprint')
                                    )
                                    yield chunk

                    except (json.JSONDecodeError, UnicodeDecodeError):
                        print(f"{RED}Warning: Could not decode JSON line: {line}{RESET}")
                        continue
                    except Exception as e:
                         print(f"{RED}Error processing stream line: {e} - Line: {line}{RESET}")
                         continue

            # Final chunk to ensure stream termination is signaled correctly
            # (even if [DONE] was received, this confirms the generator end)
            final_delta = ChoiceDelta()
            # Include usage in the final chunk if available
            usage_obj = None
            if final_usage_data:
                 usage_obj = CompletionUsage(
                     prompt_tokens=final_usage_data.get('prompt_tokens', 0),
                     completion_tokens=final_usage_data.get('completion_tokens', 0),
                     total_tokens=final_usage_data.get('total_tokens', 0),
                 )

            final_choice = Choice(index=0, delta=final_delta, finish_reason="stop")
            final_chunk = ChatCompletionChunk(
                id=request_id,
                choices=[final_choice],
                created=created_time,
                model=model,
                # system_fingerprint=..., # Can be added if available in final event
            )
            # Add usage to the final chunk dictionary representation if available
            if hasattr(final_chunk, "model_dump"):
                final_chunk_dict = final_chunk.model_dump(exclude_none=True)
            else:
                final_chunk_dict = final_chunk.dict(exclude_none=True)
            if usage_obj:
                if hasattr(usage_obj, "model_dump"):
                    final_chunk_dict["usage"] = usage_obj.model_dump(exclude_none=True)
                else:
                    final_chunk_dict["usage"] = usage_obj.dict(exclude_none=True)

            # Yield the final dictionary or object as needed by downstream consumers
            # Yielding the object aligns better with the generator type hint
            yield final_chunk


        except CurlError as e:
            print(f"{RED}CurlError during MCPCore stream request: {e}{RESET}")
            raise IOError(f"MCPCore request failed due to network/curl issue: {e}") from e
        except Exception as e:
            print(f"{RED}Unexpected error during MCPCore stream: {e}{RESET}")
            error_details = ""
            if hasattr(e, 'response') and e.response is not None:
                error_details = f" - Status: {e.response.status_code}, Response: {e.response.text}"
            raise IOError(f"MCPCore stream processing failed: {e}{error_details}") from e

    def _create_non_stream_from_stream(
        self, request_id: str, created_time: int, model: str, payload: Dict[str, Any], timeout: Optional[int] = None, proxies: Optional[Dict[str, str]] = None
    ) -> ChatCompletion:
        """Handles the non-streaming response by making a single POST request (like deepinfra)."""
        try:
            # Ensure stream is False for non-streaming
            payload = dict(payload)
            payload["stream"] = False

            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None),
                impersonate="chrome110"
            )
            if not response.ok:
                try:
                    error_text = response.text
                except Exception:
                    error_text = "<Failed to read error response>"
                raise IOError(
                    f"MCPCore API Error: {response.status_code} {response.reason} - {error_text}"
                )

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
                id=request_id,
                choices=choices,
                created=created_time,
                model=data.get('model', model),
                usage=usage,
            )
            return completion

        except CurlError as e:
            print(f"{RED}CurlError during MCPCore non-stream request: {e}{RESET}")
            raise IOError(f"MCPCore request failed due to network/curl issue: {e}") from e
        except Exception as e:
            print(f"{RED}Unexpected error during MCPCore non-stream: {e}{RESET}")
            error_details = ""
            if hasattr(e, 'response') and e.response is not None:
                error_details = f" - Status: {e.response.status_code}, Response: {e.response.text}"
            raise IOError(f"MCPCore non-stream processing failed: {e}{error_details}") from e


class Chat(BaseChat):
    def __init__(self, client: 'MCPCore'):
        self.completions = Completions(client)


class MCPCore(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for the MCPCore API (chat.mcpcore.xyz).

    Usage:
        client = MCPCore()
        response = client.chat.completions.create(
            model="google/gemma-7b-it",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    """
    AVAILABLE_MODELS = [
        "@cf/deepseek-ai/deepseek-math-7b-instruct",
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "@cf/defog/sqlcoder-7b-2",
        "@cf/fblgit/una-cybertron-7b-v2-bf16",
        "@cf/google/gemma-3-12b-it",
        "@cf/meta/llama-2-7b-chat-int8",
        "@hf/thebloke/llama-2-13b-chat-awq",
        "@hf/thebloke/llamaguard-7b-awq",
        "@hf/thebloke/mistral-7b-instruct-v0.1-awq",
        "@hf/thebloke/neural-chat-7b-v3-1-awq",
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.7-sonnet:thinking",
        "anthropic/claude-opus-4",
        "anthropic/claude-sonnet-4",
        "openai/chatgpt-4o-latest",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "openai/gpt-4o-mini-search-preview",
        "openai/gpt-4o-search-preview",
        "openai/o1-pro",
        "openai/o3-mini",
        "sarvam-m",
        "x-ai/grok-3-beta",
    ]

    def _auto_fetch_token(self):
        """Automatically fetch a token from the signup endpoint."""
        session = Session()
        def random_string(length=8):
            return ''.join(random.choices(string.ascii_lowercase, k=length))
        name = random_string(6)
        email = f"{random_string(8)}@gmail.com"
        password = email
        profile_image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAAAXNSR0IArs4c6QAAAkRJREFUeF7tmDFOw0AUBdcSiIaKM3CKHIQ7UHEISq5AiUTFHYC0XADoTRsJEZFEjhFIaYAim92fjGFS736/zOTZzjavl0d98oMh0CgE4+IriEJYPhQC86EQhdAIwPL4DFEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg1RCIwALI4NUQiMACyODVEIjAAsjg2BCfkAIqwAA94KZ/EAAAAASUVORK5CYII="
        payload = {
            "name": name,
            "email": email,
            "password": password,
            "profile_image_url": profile_image_url
        }
        headers = {
            **LitAgent().generate_fingerprint(),
            'origin': 'https://chat.mcpcore.xyz',
            'referer': 'https://chat.mcpcore.xyz/auth',
        }
        try:
            resp = session.post(
                "https://chat.mcpcore.xyz/api/v1/auths/signup",
                headers=headers,
                json=payload,
                timeout=30,
                impersonate="chrome110"
            )
            if resp.ok:
                data = resp.json()
                token = data.get("token")
                if token:
                    return token
                # fallback: try to get from set-cookie
                set_cookie = resp.headers.get("set-cookie", "")
                if "token=" in set_cookie:
                    return set_cookie.split("token=")[1].split(";")[0]
            raise RuntimeError(f"Failed to auto-fetch token: {resp.status_code} {resp.text}")
        except Exception as e:
            raise RuntimeError(f"Token auto-fetch failed: {e}")

    def __init__(
        self,
        timeout: int = 60,
    ):
        """
        Initializes the MCPCore OpenAI-compatible client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.api_endpoint = "https://chat.mcpcore.xyz/api/chat/completions"
        self.timeout = timeout
        self.token = self._auto_fetch_token()
        self.session = Session() # Use curl_cffi Session

        # Basic headers + Authorization
        self.headers = {
            **LitAgent().generate_fingerprint(),
            'origin': 'https://chat.mcpcore.xyz',
            'referer': 'https://chat.mcpcore.xyz/auth',
        }
        # Add more headers mimicking browser behavior if needed, e.g., sec-ch-ua, etc.
        # Example:
        # self.headers.update({
        #     'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        #     'sec-ch-ua-mobile': '?0',
        #     'sec-ch-ua-platform': '"Windows"',
        #     'sec-fetch-dest': 'empty',
        #     'sec-fetch-mode': 'cors',
        #     'sec-fetch-site': 'same-origin',
        # })

        self.session.headers.update(self.headers)
        self.chat = Chat(self) # Initialize chat interface

    @property
    def models(self):
        class _ModelList:
            def list(inner_self):
                return type(self).AVAILABLE_MODELS
        return _ModelList()

if __name__ == "__main__":
    print("-" * 100)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 100)

    test_prompt = "Say 'Hello' in one word"

    client = MCPCore()
    for model in client.models.list():
        print(f"\rTesting {model}...", end="")
        try:
            presp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test_prompt}]
            )
            # Try to get the response text (truncate to 100 chars)
            if hasattr(presp, "choices") and presp.choices and hasattr(presp.choices[0], "message"):
                content = presp.choices[0].message.content or ""
                clean_text = content.strip().encode('utf-8', errors='ignore').decode('utf-8')
                display_text = clean_text[:100] + "..." if len(clean_text) > 100 else clean_text
                status = "✓" if clean_text else "✗"
                if not clean_text:
                    display_text = "Empty or invalid response"
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 100:
                error_msg = error_msg[:97] + "..."
            print(f"\r{model:<50} {'✗':<10} Error: {error_msg}")