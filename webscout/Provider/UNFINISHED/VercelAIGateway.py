from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import random
import string
from typing import Any, Dict, Optional, Generator, Union, List

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions
from webscout.litagent import LitAgent
# Using LitProxy for intelligent proxy management
try:
    from litproxy import (
        get_auto_proxy, get_proxy_dict, test_proxy, get_working_proxy,
        refresh_proxy_cache, get_proxy_stats, set_proxy_cache_duration,
        patch, use_proxy, proxyify, list_proxies, test_all_proxies,
        current_proxy, make_request_with_auto_retry, create_auto_retry_session
    )
    LITPROXY_AVAILABLE = True
except ImportError:
    LITPROXY_AVAILABLE = False

import requests

class VercelAIGateway(Provider):
    """
    A class to interact with the Vercel AI SDK Gateway Demo API with intelligent proxy management using LitProxy.
    
    Install LitProxy for advanced proxy features:
        pip install litproxy
    
    Features:
    - Intelligent proxy rotation and health monitoring
    - Automatic retry with proxy fallback on failures
    - Support for multiple proxy sources (Webshare, NordVPN, Remote lists)
    - Seamless curl_cffi session integration
    - Comprehensive proxy diagnostics and statistics
    """

    AVAILABLE_MODELS = [
        "amazon/nova-lite",
        "amazon/nova-micro",
        "anthropic/claude-3.5-haiku",
        "google/gemini-2.0-flash",
        "meta/llama-3.1-8b",
        "mistral/ministral-3b",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4o-mini",
        "xai/grok-3"
    ]

    @staticmethod
    def _vercel_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from Vercel AI Gateway stream JSON objects."""
        if isinstance(chunk, dict):
            if chunk.get("type") == "text-delta":
                return chunk.get("delta")
        return None

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2049,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        history_offset: int = 10250,
        act: str = None,
        model: str = "openai/gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant.",
        browser: str = "chrome",
        use_proxy: bool = True,
        max_proxy_attempts: int = 3,
        proxy_cache_duration: int = 300
    ):
        """
        Initializes the Vercel AI Gateway API client with LitProxy integration.
        
        Args:
            use_proxy (bool): Enable proxy usage via LitProxy (default: True)
            max_proxy_attempts (int): Maximum proxy retry attempts (default: 3)
            proxy_cache_duration (int): Proxy cache duration in seconds (default: 300)
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.url = "https://ai-sdk-gateway-demo.labs.vercel.dev/api/chat"

        # Initialize LitAgent
        self.agent = LitAgent()
        self.fingerprint = self.agent.generate_fingerprint(browser)
        
        self.headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": self.fingerprint["accept_language"],
            "Content-Type": "application/json",
            "DNT": "1",
            "Origin": "https://ai-sdk-gateway-demo.labs.vercel.dev",
            "Priority": "u=1, i",
            "Referer": f"https://ai-sdk-gateway-demo.labs.vercel.dev/?modelId={model.replace('/', '%2F')}",
            "Sec-CH-UA": self.fingerprint.get("sec_ch_ua", ""),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{self.fingerprint.get("platform", "")}"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-GPC": "1",
            "User-Agent": self.fingerprint.get("user_agent", ""),
            "X-Forwarded-For": self.fingerprint.get("x-forwarded-for", ""),
            "X-Real-IP": self.fingerprint.get("x-real-ip", ""),
            "X-Client-IP": self.fingerprint.get("x-client-ip", ""),
        }

        # Initialize curl_cffi Session
        self.session = Session()
        self.session.headers.update(self.headers)
        
        # Configure proxy settings
        self.use_proxy = use_proxy
        self.max_proxy_attempts = max_proxy_attempts
        self.proxy_cache_duration = proxy_cache_duration
        
        # Integrate LitProxy for intelligent proxy management
        if use_proxy and LITPROXY_AVAILABLE:
            try:
                # Configure proxy cache duration
                set_proxy_cache_duration(proxy_cache_duration)
                # Patch the session with proxy support
                patch(self.session)
                self.proxy_enabled = True
            except Exception as e:
                self.proxy_enabled = False
        else:
            self.proxy_enabled = False
            if use_proxy and not LITPROXY_AVAILABLE:
                # Silently disable proxy if LitProxy not available
                pass
        
        self.system_prompt = system_prompt
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model

        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        
        Conversation.intro = (
            AwesomePrompts().get_act(
                act, raise_not_found=True, default=None, case_insensitive=True
            )
            if act
            else intro or Conversation.intro
        )

        self.conversation = Conversation(
            is_conversation, self.max_tokens_to_sample, filepath, update_file
        )
        self.conversation.history_offset = history_offset

    def refresh_identity(self, browser: str = None):
        """
        Refreshes the browser identity fingerprint.

        Args:
            browser: Specific browser to use for the new fingerprint
        """
        browser = browser or self.fingerprint.get("browser_type", "chrome")
        self.fingerprint = self.agent.generate_fingerprint(browser)

        # Update headers with new fingerprint
        self.headers.update({
            "Accept-Language": self.fingerprint["accept_language"],
            "User-Agent": self.fingerprint.get("user_agent", ""),
            "Sec-CH-UA": self.fingerprint.get("sec_ch_ua", ""),
            "Sec-CH-UA-Platform": f'"{self.fingerprint.get("platform", "")}"',
        })

        # Update session headers
        self.session.headers.update(self.headers)
        return self.fingerprint

    def _make_request(self, payload: dict, stream: bool = False):
        """
        Make a request to the API. The session is already patched with LitProxy auto-retry if enabled.
        
        Args:
            payload: Request payload
            stream: Whether to stream the response
            
        Returns:
            Response object
        """
        # Use the session directly - it's already patched with proxy auto-retry if enabled
        response = self.session.post(
            self.url,
            data=json.dumps(payload),
            stream=stream,
            timeout=self.timeout,
            impersonate="chrome110"
        )
        response.raise_for_status()
        return response

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        # Generate random IDs
        conversation_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        message_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        # Payload construction
        payload = {
            "modelId": self.model,
            "id": conversation_id,
            "messages": [
                {
                    "parts": [{"type": "text", "text": conversation_prompt}],
                    "id": message_id,
                    "role": "user"
                }
            ],
            "trigger": "submit-message"
        }

        def for_stream():
            streaming_text = ""
            try:
                response = self._make_request(payload, stream=True)

                # Use sanitize_stream for SSE format
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value="data:",
                    to_json=True,
                    skip_markers=["[DONE]"],
                    content_extractor=self._vercel_extractor,
                    yield_raw_on_error=False
                )

                for content_chunk in processed_stream:
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        resp = dict(text=content_chunk)
                        yield resp if not raw else content_chunk

            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {str(e)}") from e
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {str(e)}") from e
            finally:
                if streaming_text:
                    self.last_response = {"text": streaming_text}
                    self.conversation.update_chat_history(prompt, streaming_text)

        def for_non_stream():
            try:
                response = self._make_request(payload, stream=False)

                # Collect all streaming chunks for non-stream mode
                full_text = ""
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value="data:",
                    to_json=True,
                    skip_markers=["[DONE]"],
                    content_extractor=self._vercel_extractor,
                    yield_raw_on_error=False
                )

                for content_chunk in processed_stream:
                    if content_chunk and isinstance(content_chunk, str):
                        full_text += content_chunk

                self.last_response = {"text": full_text}
                self.conversation.update_chat_history(prompt, full_text)
                return self.last_response if not raw else full_text

            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e:
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {e} - {err_text}") from e

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream_chat():
            gen = self.ask(
                prompt, stream=True, raw=False,
                optimizer=optimizer, conversationally=conversationally
            )
            for response_dict in gen:
                yield self.get_message(response_dict)

        def for_non_stream_chat():
            response_data = self.ask(
                prompt, stream=False, raw=False,
                optimizer=optimizer, conversationally=conversationally
            )
            return self.get_message(response_data)

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]
if __name__ == "__main__":
    test_ai = VercelAIGateway(use_proxy=True, max_proxy_attempts=3, proxy_cache_duration=300)
    print(test_ai.chat("Hello, how are you?"))