import time
import uuid
import json
import random
import string
from typing import List, Dict, Optional, Union, Generator, Any
import warnings

# Use curl_cffi for requests
from curl_cffi.requests import Session
from curl_cffi import CurlError
import urllib3

# Import base classes and utility structures
from webscout.Provider.OPENAI.base import OpenAICompatibleProvider, BaseChat, BaseCompletions
from webscout.Provider.OPENAI.utils import (
    ChatCompletionChunk, ChatCompletion, Choice, ChoiceDelta,
    ChatCompletionMessage, CompletionUsage
)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
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
    def __init__(self, client: 'XenAI'):
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
        Creates a model response for the given chat conversation using XenAI API.
        Mimics openai.chat.completions.create
        """
        if model not in self._client.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model}' not supported by XenAI. Available: {self._client.AVAILABLE_MODELS}")

        # Construct the XenAI-specific payload
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

        # Add optional OpenAI params to XenAI's 'params' field if provided
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
        """Handles the streaming response from XenAI."""
        final_usage_data = None # To store usage if received
        try:
            response = self._client.session.post(
                self._client.api_endpoint,
                headers=self._client.headers,
                json=payload,
                stream=True,
                timeout=timeout or self._client.timeout,
                proxies=proxies or getattr(self._client, "proxies", None),
                impersonate="chrome110",
                verify=False
            )

            if not response.ok:
                try:
                    error_text = response.text
                except Exception:
                    error_text = "<Failed to read error response>"
                raise IOError(
                    f"XenAI API Error: {response.status_code} {response.reason} - {error_text}"
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
            print(f"{RED}CurlError during XenAI stream request: {e}{RESET}")
            raise IOError(f"XenAI request failed due to network/curl issue: {e}") from e
        except Exception as e:
            print(f"{RED}Unexpected error during XenAI stream: {e}{RESET}")
            error_details = ""
            if hasattr(e, 'response') and e.response is not None:
                error_details = f" - Status: {e.response.status_code}, Response: {e.response.text}"
            raise IOError(f"XenAI stream processing failed: {e}{error_details}") from e

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
                impersonate="chrome110",
                verify=False
            )
            if not response.ok:
                try:
                    error_text = response.text
                except Exception:
                    error_text = "<Failed to read error response>"
                raise IOError(
                    f"XenAI API Error: {response.status_code} {response.reason} - {error_text}"
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
            print(f"{RED}CurlError during XenAI non-stream request: {e}{RESET}")
            raise IOError(f"XenAI request failed due to network/curl issue: {e}") from e
        except Exception as e:
            print(f"{RED}Unexpected error during XenAI non-stream: {e}{RESET}")
            error_details = ""
            if hasattr(e, 'response') and e.response is not None:
                error_details = f" - Status: {e.response.status_code}, Response: {e.response.text}"
            raise IOError(f"XenAI non-stream processing failed: {e}{error_details}") from e


class Chat(BaseChat):
    def __init__(self, client: 'XenAI'):
        self.completions = Completions(client)


class XenAI(OpenAICompatibleProvider):
    """
    OpenAI-compatible client for the XenAI API (chat.XenAI.xyz).

    Usage:
        client = XenAI()
        response = client.chat.completions.create(
            model="google/gemma-7b-it",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    """

    AVAILABLE_MODELS = [
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-preview-05-20",
        "o4-mini-high",
        "grok-3-mini-fast-beta",
        "grok-3-fast-beta",
        "gpt-4.1",
        "o3-high",
        "gpt-4o-search-preview",
        "gpt-4o",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-20250514-thinking",
        "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-ai/DeepSeek-R1-0528",
        "groq/deepseek-r1-distill-llama-70b",
        "deepseek-ai/DeepSeek-Prover-V2-671B",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "cognitivecomputations/Dolphin3.0-Mistral-24B",
        "sonar-pro",
        "gpt-4o-mini",
        "gemini-2.0-flash-lite-preview-02-05",
        "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet-20250219-thinking",
        "claude-opus-4-20250514",
        "claude-opus-4-20250514-thinking",
        "chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "chutesai/Llama-4-Scout-17B-16E-Instruct",
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
        # Add more detailed browser-like headers
        try:
            # First try with fingerprint from LitAgent
            headers = {
                **LitAgent().generate_fingerprint(),
                'origin': 'https://chat.xenai.tech',
                'referer': 'https://chat.xenai.tech/auth',
                'sec-ch-ua': '"Google Chrome";v="127", "Chromium";v="127", "Not=A?Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'accept-language': 'en-US,en;q=0.9'
            }
        except:
            # Fallback to basic Chrome user agent if LitAgent fails
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
                'origin': 'https://chat.xenai.tech',
                'referer': 'https://chat.xenai.tech/auth',
                'sec-ch-ua': '"Google Chrome";v="127", "Chromium";v="127", "Not=A?Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'accept-language': 'en-US,en;q=0.9'
            }
        try:
            # Try signup with newer Chrome version
            resp = session.post(
                "https://chat.xenai.tech/api/v1/auths/signup",
                headers=headers,
                json=payload,
                timeout=30,
                impersonate="chrome120",  # Try newer Chrome version
                verify=False
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
            
            # If signup fails, try login (account might already exist)
            login_resp = session.post(
                "https://chat.xenai.tech/api/v1/auths/login",
                headers=headers,
                json={"email": email, "password": password},
                timeout=30,
                impersonate="chrome120",
                verify=False
            )
            
            if login_resp.ok:
                data = login_resp.json()
                token = data.get("token")
                if token:
                    return token
                
            # Try guest authentication as last resort
            guest_resp = session.post(
                "https://chat.xenai.tech/api/v1/auths/guest",
                headers=headers,
                json={},
                timeout=30,
                impersonate="chrome120",
                verify=False
            )
            
            if guest_resp.ok:
                data = guest_resp.json()
                token = data.get("token")
                if token:
                    return token
                    
            raise RuntimeError(f"Failed to auto-fetch token: {resp.status_code} {resp.text}")
        except Exception as e:
            raise RuntimeError(f"Token auto-fetch failed: {e}")

    def __init__(
        self,
        timeout: int = 60,
    ):
        """
        Initializes the XenAI OpenAI-compatible client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.api_endpoint = "https://chat.xenai.tech/api/chat/completions"
        self.timeout = timeout
        self.token = self._auto_fetch_token()
        self.session = Session() # Use curl_cffi Session

        # Enhanced headers with browser impersonation
        try:
            self.headers = {
                **LitAgent().generate_fingerprint(),
                'origin': 'https://chat.xenai.tech',
                'referer': 'https://chat.xenai.tech/',
                'sec-ch-ua': '"Google Chrome";v="127", "Chromium";v="127", "Not=A?Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'accept-language': 'en-US,en;q=0.9'
            }
        except:
            self.headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
                'origin': 'https://chat.xenai.tech',
                'referer': 'https://chat.xenai.tech/',
                'sec-ch-ua': '"Google Chrome";v="127", "Chromium";v="127", "Not=A?Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'accept-language': 'en-US,en;q=0.9'
            }

        # Update headers and set authorization token
        self.headers['authorization'] = f'Bearer {self.token}'
        self.session.headers.update(self.headers)
        
        # Configure session
        self.session.impersonate = "chrome120"
        self.session.verify = False
        
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

    client = XenAI()
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