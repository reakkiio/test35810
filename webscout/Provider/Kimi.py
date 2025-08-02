from curl_cffi import CurlError
from curl_cffi.requests import Session
import json
import random
from typing import Any, Dict, Optional, Generator, Union, List
import uuid

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class Kimi(Provider):
    """
    A class to interact with the Kimi API (kimi.com).
    
    This provider uses the Kimi web interface API endpoints to provide
    access to Kimi's AI models.
    
    Examples:
        >>> from webscout.Provider.Kimi import Kimi
        >>> ai = Kimi()
        >>> response = ai.chat("What's the weather today?")
        >>> print(response)
        'The weather today is sunny...'
    """
    
    AVAILABLE_MODELS = ["k1.5", "k2", "k1.5-thinking"]
    
    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 4000,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "k2",
        system_prompt: str = "You are a helpful assistant.",
        browser: str = "chrome",
        web_search: bool = False,
    ):
        """
        Initializes the Kimi API client with given parameters.
        
        Args:
            is_conversation: Whether to maintain conversation history
            max_tokens: Maximum tokens for response
            timeout: Request timeout in seconds
            intro: Introduction message
            filepath: Path to conversation history file
            update_file: Whether to update conversation file
            proxies: Proxy configuration
            history_offset: History offset for conversation
            act: Act/persona for the assistant
            model: Model to use (k1.5, k2, kimi, kimi-plus)
            system_prompt: System prompt for the assistant
            browser: Browser to impersonate
            web_search: Whether to enable web search
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}"
            )
            
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt
        self.web_search = web_search
        
        # Kimi API endpoints
        self.register_endpoint = "https://www.kimi.com/api/device/register"
        self.chat_create_endpoint = "https://www.kimi.com/api/chat"
        self.chat_completion_endpoint = "https://www.kimi.com/api/chat/{chat_id}/completion/stream"
        
        # Initialize LitAgent for browser fingerprinting
        self.agent = LitAgent()
        self.fingerprint = self.agent.generate_fingerprint(browser)
        
        # Generate device ID
        self.device_id = str(random.randint(1000000000000000, 9999999999999999))
        
        # Headers for Kimi API
        self.headers = {
            "Accept": "text/event-stream",
            "Accept-Language": self.fingerprint["accept_language"],
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "DNT": "1",
            "Origin": "https://www.kimi.com",
            "Pragma": "no-cache",
            "Referer": "https://www.kimi.com/",
            "Sec-CH-UA": self.fingerprint["sec_ch_ua"],
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{self.fingerprint["platform"]}"',
            "User-Agent": self.fingerprint["user_agent"],
            "x-msh-device-id": self.device_id,
            "x-msh-platform": "web",
            "x-traffic-id": self.device_id,
        }
        
        # Initialize authentication
        self.access_token = None
        self.chat_id = None
        
        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method))
            and not method.startswith("__")
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
        
        # Update session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies
        
    def _authenticate(self) -> str:
        """Authenticate with Kimi API and get access token."""
        if self.access_token:
            return self.access_token
        
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.register_endpoint,
                    json={},
                    timeout=self.timeout,
                    impersonate="chrome110"
                )
                response.raise_for_status()
                
                data = response.json()
                if not data.get("access_token"):
                    raise exceptions.FailedToGenerateResponseError("No access token received")
                    
                self.access_token = data["access_token"]
                self.session.headers["Authorization"] = f"Bearer {self.access_token}"
                return self.access_token
                
            except CurlError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    continue
                raise exceptions.FailedToGenerateResponseError(f"Authentication failed after {max_retries} attempts (CurlError): {e}")
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    continue
                raise exceptions.FailedToGenerateResponseError(f"Authentication failed after {max_retries} attempts: {e}")
        
        # This should never be reached, but just in case
        raise exceptions.FailedToGenerateResponseError(f"Authentication failed after {max_retries} attempts: {last_exception}")
    
    def _create_chat(self) -> str:
        """Create a new chat session and return chat ID."""
        if self.chat_id:
            return self.chat_id
            
        self._authenticate()
        
        try:
            response = self.session.post(
                self.chat_create_endpoint,
                json={
                    "name": "Unnamed Chat",
                    "born_from": "home",
                    "kimiplus_id": "kimi",
                    "is_example": False,
                    "source": "web",
                    "tags": []
                },
                timeout=self.timeout,
                impersonate="chrome110"
            )
            response.raise_for_status()
            
            data = response.json()
            self.chat_id = data.get("id")
            if not self.chat_id:
                raise exceptions.FailedToGenerateResponseError("No chat ID received")
                
            return self.chat_id
            
        except CurlError as e:
            raise exceptions.FailedToGenerateResponseError(f"Chat creation failed (CurlError): {e}")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Chat creation failed: {e}")
    
    @staticmethod
    def _kimi_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extract content from Kimi SSE stream."""
        if isinstance(chunk, dict):
            if chunk.get("event") == "cmpl":
                return chunk.get("text")
        return None
    
    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        """
        Send a prompt to Kimi API and return the response.
        
        Args:
            prompt: The prompt to send
            stream: Whether to stream the response
            raw: Whether to return raw response
            optimizer: Optimizer to use
            conversationally: Whether to generate conversationally
            
        Returns:
            Dict or Generator with the response
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )
        
        self._create_chat()
        
        # Fixed payload structure based on actual Kimi API requirements
        payload = {
            "kimiplus_id": "kimi",
            "extend": {"sidebar": True},
            "model": self.model,
            "use_search": self.web_search,
            "messages": [
                {
                    "role": "user",
                    "content": conversation_prompt
                }
            ],
            "refs": [],
            "history": [],
            "scene_labels": [],
            "use_semantic_memory": False,
            "use_deep_research": False
        }
        
        def for_stream():
            try:
                response = self.session.post(
                    self.chat_completion_endpoint.format(chat_id=self.chat_id),
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    impersonate="chrome110"
                )
                response.raise_for_status()
                
                streaming_text = ""
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value="data:",
                    to_json=True,
                    skip_markers=["[DONE]"],
                    content_extractor=self._kimi_extractor,
                    yield_raw_on_error=False
                )
                
                for content_chunk in processed_stream:
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        resp = dict(text=content_chunk)
                        yield resp if not raw else content_chunk
                
                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)
                
            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")
        
        def for_non_stream():
            try:
                response = self.session.post(
                    self.chat_completion_endpoint.format(chat_id=self.chat_id),
                    json=payload,
                    timeout=self.timeout,
                    impersonate="chrome110"
                )
                response.raise_for_status()
                
                # Collect all streaming data
                full_text = ""
                processed_stream = sanitize_stream(
                    data=response.text,
                    to_json=True,
                    intro_value="data:",
                    skip_markers=["[DONE]"],
                    content_extractor=self._kimi_extractor,
                    yield_raw_on_error=False
                )
                
                for content_chunk in processed_stream:
                    if content_chunk and isinstance(content_chunk, str):
                        full_text += content_chunk
                
                self.last_response = {"text": full_text}
                self.conversation.update_chat_history(prompt, full_text)
                return self.last_response if not raw else full_text
                
            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")
        
        return for_stream() if stream else for_non_stream()
    
    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        raw: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Chat with Kimi API.
        
        Args:
            prompt: The prompt to send
            stream: Whether to stream the response
            optimizer: Optimizer to use
            conversationally: Whether to generate conversationally
            raw: Whether to return raw response
            
        Returns:
            str or Generator with the response
        """
        def for_stream():
            for response in self.ask(
                prompt, True, raw=raw, optimizer=optimizer, conversationally=conversationally
            ):
                if raw:
                    yield response
                else:
                    yield self.get_message(response)
        
        def for_non_stream():
            result = self.ask(
                prompt,
                False,
                raw=raw,
                optimizer=optimizer,
                conversationally=conversationally,
            )
            if raw:
                return result
            else:
                return self.get_message(result)
        
        return for_stream() if stream else for_non_stream()
    
    def get_message(self, response: dict) -> str:
        """Extract message from response."""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]
    
    def refresh_identity(self, browser: str = None):
        """
        Refresh browser identity fingerprint.
        
        Args:
            browser: Specific browser to use
        """
        browser = browser or self.fingerprint.get("browser_type", "chrome")
        self.fingerprint = self.agent.generate_fingerprint(browser)
        
        self.headers.update({
            "Accept-Language": self.fingerprint["accept_language"],
            "Sec-CH-UA": self.fingerprint["sec_ch_ua"],
            "Sec-CH-UA-Platform": f'"{self.fingerprint["platform"]}"',
            "User-Agent": self.fingerprint["user_agent"],
        })
        
        self.session.headers.update(self.headers)
        
        # Generate new device ID
        self.device_id = str(random.randint(1000000000000000, 9999999999999999))
        self.session.headers.update({
            "x-msh-device-id": self.device_id,
            "x-traffic-id": self.device_id,
        })
        
        return self.fingerprint

if __name__ == "__main__":
    # Test the Kimi provider
    print("-" * 80)
    print(f"{'Model':<20} {'Status':<10} {'Response'}")
    print("-" * 80)
    
    for model in Kimi.AVAILABLE_MODELS:
        try:
            ai = Kimi(model=model, timeout=30)
            response = ai.chat("Say 'Hello' in one word")
            
            if response and len(response.strip()) > 0:
                status = "✓"
                display_text = response.strip()[:50] + "..." if len(response.strip()) > 50 else response.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"{model:<20} {status:<10} {display_text}")
        except Exception as e:
            print(f"{model:<20} {'✗':<10} {str(e)}")