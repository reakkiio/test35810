import time
import uuid
import json
from typing import Any, Dict, Optional, Generator, Union
from dataclasses import dataclass, asdict
from datetime import date
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent
# Replace requests with curl_cffi
from curl_cffi.requests import Session # Import Session
from curl_cffi import CurlError # Import CurlError

class Netwrck(Provider):
    """
    A class to interact with the Netwrck.com API. Supports streaming.
    """
    greeting = """Hello! I'm a helpful assistant. How can I help you today?"""

    AVAILABLE_MODELS = [
        "neversleep/llama-3-lumimaid-8b:extended",
        "x-ai/grok-2",
        "anthropic/claude-3-7-sonnet-20250219",
        "sao10k/l3-euryale-70b",
        "openai/gpt-4.1-mini",
        "gryphe/mythomax-l2-13b",
        "google/gemini-pro-1.5",
        "google/gemini-2.5-flash-preview-04-17",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat"

    ]

    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-20250219",
        is_conversation: bool = True,
        max_tokens: int = 4096, # Note: max_tokens is not used by this API
        timeout: int = 30,
        intro: Optional[str] = None,
        filepath: Optional[str] = None,
        update_file: bool = False,
        proxies: Optional[dict] = None,
        history_offset: int = 0,
        act: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7, # Note: temperature is not used by this API
        top_p: float = 0.8 # Note: top_p is not used by this API
    ):
        """Initializes the Netwrck API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.model = model
        self.model_name = model
        self.system_prompt = system_prompt
        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response: Dict[str, Any] = {}
        self.temperature = temperature
        self.top_p = top_p
        
        self.agent = LitAgent() # Keep for potential future use or other headers
        self.headers = {
            'authority': 'netwrck.com',
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': 'https://netwrck.com',
            'referer': 'https://netwrck.com/',
            'user-agent': self.agent.random() 
            # Add sec-ch-ua headers if needed for impersonation consistency
        }
        
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.proxies = proxies or {}
        self.session.proxies = self.proxies # Assign proxies directly

        Conversation.intro = (
            AwesomePrompts().get_act(act, raise_not_found=True, default=None, case_insensitive=True)
            if act
            else intro or Conversation.intro
        )
        
        self.conversation = Conversation(is_conversation, max_tokens, filepath, update_file)
        self.conversation.history_offset = history_offset
        self.__available_optimizers = (
            method for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False, # Keep raw param for interface consistency
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        """Sends a prompt to the Netwrck API and returns the response."""
        if optimizer and optimizer not in self.__available_optimizers:
            raise exceptions.FailedToGenerateResponseError(f"Optimizer is not one of {self.__available_optimizers}")

        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            conversation_prompt = getattr(Optimizers, optimizer)(
                conversation_prompt if conversationally else prompt
            )

        payload = {
            "query": prompt,
            "context": self.system_prompt,
            "examples": [],
            "model_name": self.model_name,
            "greeting": self.greeting
        }

        def for_stream():
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    "https://netwrck.com/api/chatpred_or",
                    json=payload,
                    # headers are set on the session
                    # proxies are set on the session
                    timeout=self.timeout,
                    stream=True,
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors

                streaming_text = ""
                # Iterate over bytes and decode manually
                for line_bytes in response.iter_lines():
                    if line_bytes:
                        try:
                            decoded_line = line_bytes.decode('utf-8').strip('"')
                            # Handle potential escape sequences if necessary
                            # decoded_line = decoded_line.encode().decode('unicode_escape') # Uncomment if needed
                            streaming_text += decoded_line
                            resp = {"text": decoded_line}
                            # Yield dict or raw string
                            yield resp if not raw else decoded_line
                        except UnicodeDecodeError:
                            # Handle potential decoding errors if chunks split mid-character
                            continue 

                # Update history after stream finishes
                self.last_response = {"text": streaming_text} # Store aggregated text
                self.conversation.update_chat_history(payload["query"], streaming_text)

            except CurlError as e: # Catch CurlError
                raise exceptions.ProviderConnectionError(f"Network error (CurlError): {str(e)}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.ProviderConnectionError(f"Unexpected error ({type(e).__name__}): {str(e)} - {err_text}") from e

        def for_non_stream():
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    "https://netwrck.com/api/chatpred_or",
                    json=payload,
                    # headers are set on the session
                    # proxies are set on the session
                    timeout=self.timeout,
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors
                
                # Use response.text which is already decoded
                text = response.text.strip('"')
                # Handle potential escape sequences if necessary
                # text = text.encode().decode('unicode_escape') # Uncomment if needed
                self.last_response = {"text": text}
                self.conversation.update_chat_history(prompt, text)

                # Return dict or raw string
                return text if raw else self.last_response

            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Network error (CurlError): {str(e)}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"Unexpected error ({type(e).__name__}): {str(e)} - {err_text}") from e

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> str:
        """Generates a response from the Netwrck API."""
        def for_stream_chat():
            # ask() yields dicts or strings when streaming
            gen = self.ask(
                prompt,
                stream=True,
                raw=False, # Ensure ask yields dicts for get_message
                optimizer=optimizer,
                conversationally=conversationally
            )
            for response_dict in gen:
                yield self.get_message(response_dict) # get_message expects dict

        def for_non_stream_chat():
            # ask() returns dict or str when not streaming
            response_data = self.ask(
                prompt,
                stream=False,
                raw=False, # Ensure ask returns dict for get_message
                optimizer=optimizer,
                conversationally=conversationally,
            )
            return self.get_message(response_data) # get_message expects dict

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: Dict[str, Any]) -> str:
        """Retrieves message only from response"""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"].replace('\\n', '\n').replace('\\n\\n', '\n\n')

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)
    
    # Test all available models
    working = 0
    total = len(Netwrck.AVAILABLE_MODELS)
    
    for model in Netwrck.AVAILABLE_MODELS:
        try:
            test_ai = Netwrck(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            for chunk in response:
                response_text += chunk
                print(f"\r{model:<50} {'Testing...':<10}", end="", flush=True)
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
