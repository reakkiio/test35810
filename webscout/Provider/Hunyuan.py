from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import os
from typing import Any, Dict, Optional, Generator, Union
import time
import uuid
import re

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions
from webscout.litagent import LitAgent

class Hunyuan(Provider):
    """
    A class to interact with the Tencent Hunyuan API with LitAgent user-agent.
    """

    AVAILABLE_MODELS = [
        "hunyuan-t1-latest",
        # Add more models as they become available
    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2048, # Note: max_tokens is not used by this API
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "hunyuan-t1-latest",
        browser: str = "chrome", # Note: browser fingerprinting might be less effective with impersonate
        api_key: str = None,
        system_prompt: str = "You are a helpful assistant.",
    ):

        """Initializes the Hunyuan API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
            
        self.url = "https://llm.hunyuan.tencent.com/aide/api/v2/triton_image/demo_text_chat/"
        
        # Initialize LitAgent (keep if needed for other headers or logic)
        self.agent = LitAgent()
        # Fingerprint generation might be less relevant with impersonate
        self.fingerprint = self.agent.generate_fingerprint(browser) 
        
        # Use the fingerprint for headers (keep relevant ones)
        self.headers = {
            "Accept": "*/*",
            "Accept-Language": self.fingerprint["accept_language"], # Keep Accept-Language
            "Content-Type": "application/json",
            "DNT": "1", # Keep DNT
            "Origin": "https://llm.hunyuan.tencent.com", # Keep Origin
            "Referer": "https://llm.hunyuan.tencent.com/", # Keep Referer
            "Sec-Fetch-Dest": "empty", # Keep Sec-Fetch-*
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-GPC": "1", # Keep Sec-GPC
        }
        
        # Add authorization if API key is provided
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        else:
            # Default test key (may not work long-term)
            self.headers["Authorization"] = "Bearer 7auGXNATFSKl7dF"
        
        # Initialize curl_cffi Session
        self.session = Session()
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly
        self.system_message = system_prompt
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
        
        # Update headers with new fingerprint (only relevant ones)
        self.headers.update({
            "Accept-Language": self.fingerprint["accept_language"],
        })
        
        # Update session headers
        self.session.headers.update(self.headers) # Update only relevant headers
        
        return self.fingerprint

    def ask(
        self,
        prompt: str,
        stream: bool = False, # API supports streaming
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(conversation_prompt if conversationally else prompt)
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        # Generate a unique query ID for each request
        query_id = ''.join(re.findall(r'[a-z0-9]', str(uuid.uuid4())[:18]))
    

        # Payload construction
        payload = {
            "stream": True, # API seems to require stream=True based on response format
            "model": self.model,
            "query_id": query_id,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": "Always response in English\n\n" + conversation_prompt},
            ],
            "stream_moderation": True,
            "enable_enhancement": False
        }

        def for_stream():
            streaming_text = "" # Initialize outside try block
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.url, 
                    data=json.dumps(payload), 
                    stream=True, 
                    timeout=self.timeout, 
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors
                    
                # Iterate over bytes and decode manually
                for line_bytes in response.iter_lines():
                    if line_bytes:
                        try:
                            line = line_bytes.decode('utf-8').strip()
                            if line.startswith("data: "):
                                json_str = line[6:]
                                if json_str == "[DONE]":
                                    break
                                json_data = json.loads(json_str)
                                if 'choices' in json_data:
                                    choice = json_data['choices'][0]
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        content = choice['delta']['content']
                                        if content: # Ensure content is not None or empty
                                            streaming_text += content
                                            resp = dict(text=content)
                                            # Yield dict or raw string chunk
                                            yield resp if not raw else content
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue # Ignore lines that are not valid JSON or cannot be decoded
                    
                # Update history after stream finishes
                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)
                    
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {str(e)}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {str(e)} - {err_text}") from e


        def for_non_stream():
            # Aggregate the stream using the updated for_stream logic
            full_text = ""
            try:
                # Ensure raw=False so for_stream yields dicts
                for chunk_data in for_stream():
                     if isinstance(chunk_data, dict) and "text" in chunk_data:
                         full_text += chunk_data["text"]
                     # Handle raw string case if raw=True was passed
                     elif raw and isinstance(chunk_data, str):
                          full_text += chunk_data
            except Exception as e:
                 # If aggregation fails but some text was received, use it. Otherwise, re-raise.
                 if not full_text:
                     raise exceptions.FailedToGenerateResponseError(f"Failed to get non-stream response: {str(e)}") from e

            # last_response and history are updated within for_stream
            # Return the final aggregated response dict or raw string
            return full_text if raw else self.last_response


        # Since the API endpoint suggests streaming, always call the stream generator.
        # The non-stream wrapper will handle aggregation if stream=False.
        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream_chat():
            # ask() yields dicts or strings when streaming
            gen = self.ask(
                prompt, stream=True, raw=False, # Ensure ask yields dicts
                optimizer=optimizer, conversationally=conversationally
            )
            for response_dict in gen:
                yield self.get_message(response_dict) # get_message expects dict

        def for_non_stream_chat():
            # ask() returns dict or str when not streaming
            response_data = self.ask(
                prompt, stream=False, raw=False, # Ensure ask returns dict
                optimizer=optimizer, conversationally=conversationally
            )
            return self.get_message(response_data) # get_message expects dict

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in Hunyuan.AVAILABLE_MODELS:
        try:
            test_ai = Hunyuan(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            for chunk in response:
                response_text += chunk
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response_text.strip().encode('utf-8', errors='ignore').decode('utf-8')
                display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")