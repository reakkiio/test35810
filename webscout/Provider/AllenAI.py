from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import os
from uuid import uuid4
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions
from webscout.litagent import LitAgent

class AllenAI(Provider):
    """
    A class to interact with the AllenAI (Ai2 Playground) API.
    """

    AVAILABLE_MODELS = [
        'olmo-2-0325-32b-instruct',
        'tulu3-405b'
    ]

    # Default model options from JS implementation
    DEFAULT_OPTIONS = {
        "max_tokens": 2048, 
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "stop": None,
        "logprobs": None
    }
    
    # Host mapping for models - some models work best with specific hosts
    MODEL_HOST_MAP = {
        'olmo-2-0325-32b-instruct': 'modal',
        'tulu3-405b': 'inferd'
    }

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2048,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "OLMo-2-1124-13B-Instruct",
        host: str = None
    ):
        """Initializes the AllenAI API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
            
        self.url = "https://playground.allenai.org"
        self.api_endpoint = "https://olmo-api.allen.ai/v3/message/stream"
        self.whoami_endpoint = "https://olmo-api.allen.ai/v3/whoami"
        
        # Updated headers (remove those handled by impersonate)
        self.headers = {
            'Accept': '*/*',
            'Accept-Language': 'id-ID,id;q=0.9',
            'Origin': self.url,
            'Referer': f"{self.url}/",
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Priority': 'u=1, i',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'Content-Type': 'application/json'
        }
        
        # Initialize curl_cffi Session
        self.session = Session()
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies
        
        self.model = model
        
        # Auto-detect host if not provided
        if not host:
            # Use the preferred host from the model-host map, or default to modal
            self.host = self.MODEL_HOST_MAP.get(model, 'modal')
        else:
            self.host = host
            
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}       
        # Generate user ID if needed
        self.x_anonymous_user_id = None
        self.parent = None
        
        # Default options
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options["max_tokens"] = max_tokens

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

    def whoami(self):
        """Gets or creates a user ID for authentication with Allen AI API"""
        temp_id = str(uuid4())
        request_headers = self.session.headers.copy() # Use session headers as base
        request_headers.update({"x-anonymous-user-id": temp_id})
        
        try:
            # Use curl_cffi session get with impersonate
            response = self.session.get(
                self.whoami_endpoint,
                headers=request_headers, # Pass updated headers
                timeout=self.timeout,
                impersonate="chrome110" # Use a common impersonation profile
            )
            response.raise_for_status() # Check for HTTP errors
            
            data = response.json()
            self.x_anonymous_user_id = data.get("client", temp_id)
            return data
                
        except CurlError as e: # Catch CurlError
            self.x_anonymous_user_id = temp_id
            return {"client": temp_id, "error": f"CurlError: {e}"}
        except Exception as e: # Catch other potential exceptions (like HTTPError, JSONDecodeError)
            self.x_anonymous_user_id = temp_id
            err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
            return {"client": temp_id, "error": f"{type(e).__name__}: {e} - {err_text}"}
    
    @staticmethod
    def _allenai_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from AllenAI stream JSON objects."""
        if isinstance(chunk, dict):
            if chunk.get("message", "").startswith("msg_") and "content" in chunk:
                return chunk.get("content")
            elif "message" in chunk and chunk.get("content"): # Legacy handling
                return chunk.get("content")
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = False, # API supports streaming
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        host: str = None,
        private: bool = False,
        top_p: float = None,
        temperature: float = None,
        options: dict = None,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(conversation_prompt if conversationally else prompt)
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        # Ensure we have a user ID
        if not self.x_anonymous_user_id:
            self.whoami()
            # Check if whoami failed and we still don't have an ID
            if not self.x_anonymous_user_id:
                 raise exceptions.AuthenticationError("Failed to obtain anonymous user ID.")
        
        # Prepare the API request headers for this specific request
        request_headers = self.session.headers.copy()
        request_headers.update({
            "x-anonymous-user-id": self.x_anonymous_user_id,
            "Content-Type": "application/json" # Ensure Content-Type is set
        })
        
        # Create options dictionary
        opts = self.options.copy()
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p
        if options:
            opts.update(options)
        
        # Use the host param or the default host
        use_host = host or self.host
        
        # List of hosts to try - start with provided host, then try alternative hosts
        hosts_to_try = [use_host]
        if use_host == 'modal':
            hosts_to_try.append('inferd')
        else:
            hosts_to_try.append('modal')
        
        last_error = None
        
        # Try each host until one works
        for current_host in hosts_to_try:
            # Create the JSON payload as per the JS implementation
            payload = {
                "content": conversation_prompt,
                "private": private,
                "model": self.model,
                "host": current_host,
                "opts": opts
            }
            payload["host"] = current_host # Ensure host is updated in payload
                
            try:
                if stream:
                    # Pass request_headers to the stream method
                    return self._stream_request(payload, prompt, request_headers, raw)
                else:
                    # Pass request_headers to the non-stream method
                    return self._non_stream_request(payload, prompt, request_headers, raw)
            except (exceptions.FailedToGenerateResponseError, CurlError, Exception) as e:
                last_error = e
                # Log the error but continue to try other hosts
                print(f"Host '{current_host}' failed for model '{self.model}' ({type(e).__name__}), trying next host...")
                continue
        
        # If we've tried all hosts and none worked, raise the last error
        raise last_error or exceptions.FailedToGenerateResponseError("All hosts failed. Unable to complete request.")

    def _stream_request(self, payload, prompt, request_headers, raw=False):
        """Handle streaming requests with the given payload and headers"""
        streaming_text = "" # Initialize outside try block
        current_parent = None # Initialize outside try block
        try:
            # Use curl_cffi session post with impersonate
            response = self.session.post(
                self.api_endpoint,
                headers=request_headers, # Use headers passed to this method
                json=payload,
                stream=True,
                timeout=self.timeout,
                impersonate="chrome110" # Use a common impersonation profile
            )
            response.raise_for_status() # Check for HTTP errors
            
            # Use sanitize_stream
            processed_stream = sanitize_stream(
                data=response.iter_content(chunk_size=None), # Pass byte iterator
                intro_value=None, # No prefix
                to_json=True,     # Stream sends JSON lines
                content_extractor=self._allenai_extractor, # Use the specific extractor
                yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
            )

            for content_chunk in processed_stream:
                # content_chunk is the string extracted by _allenai_extractor
                if content_chunk and isinstance(content_chunk, str):
                    streaming_text += content_chunk
                    resp = dict(text=content_chunk)
                    yield resp if not raw else content_chunk

            # Try to extract parent ID from the *last* raw line (less reliable than before)
            # This part is tricky as sanitize_stream consumes the raw lines.
            # We might need to re-fetch or adjust if parent ID is critical per stream.
            # For now, we'll rely on the non-stream request to update parent ID more reliably.
            # Example placeholder logic (might not work reliably):
            try:
                last_line_data = json.loads(response.text.splitlines()[-1]) # Get last line if possible
                if last_line_data.get("id"):
                    current_parent = last_line_data.get("id")
                elif last_line_data.get("children"):
                    for child in last_line_data["children"]: # Use last_line_data here
                        if child.get("role") == "assistant":
                            current_parent = child.get("id")
                            break
                
                # Handle completion
                if last_line_data.get("final") or last_line_data.get("finish_reason") == "stop":
                    if current_parent:
                        self.parent = current_parent
                    
                    # Update conversation history only if not empty
                    if streaming_text.strip():
                        self.conversation.update_chat_history(prompt, streaming_text)
                        self.last_response = {"text": streaming_text} # Update last response here
                    return # End the generator
            except Exception as e:
                # Log the error but continue with the rest of the function
                print(f"Error processing response data: {str(e)}")
            
            # If loop finishes without returning (e.g., no final message), update history
            if current_parent:
                self.parent = current_parent
            self.conversation.update_chat_history(prompt, streaming_text)
            self.last_response = {"text": streaming_text}

        except CurlError as e: # Catch CurlError
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {str(e)}") from e
        except Exception as e: # Catch other potential exceptions (like HTTPError)
            err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
            raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {str(e)} - {err_text}") from e


    def _non_stream_request(self, payload, prompt, request_headers, raw=False):
        """Handle non-streaming requests with the given payload and headers"""
        try:
            # Use curl_cffi session post with impersonate
            response = self.session.post(
                self.api_endpoint,
                headers=request_headers, # Use headers passed to this method
                json=payload,
                stream=False, # Explicitly set stream to False
                timeout=self.timeout,
                impersonate="chrome110" # Use a common impersonation profile
            )
            response.raise_for_status() # Check for HTTP errors
            
            raw_response = response.text # Get raw text

            # Process the full text using sanitize_stream line by line
            processed_stream = sanitize_stream(
                data=raw_response.splitlines(), # Split into lines
                intro_value=None,
                to_json=True,
                content_extractor=self._allenai_extractor,
                yield_raw_on_error=False
            )
            # Aggregate the results
            parsed_response = "".join(list(processed_stream))

            # Update parent ID from the full response if possible (might need adjustment based on actual non-stream response structure)
            # This part is speculative as the non-stream structure isn't fully clear from the stream logic
            try:
                lines = raw_response.splitlines()
                if lines:
                    last_line_data = json.loads(lines[-1])
                    if last_line_data.get("id"):
                         self.parent = last_line_data.get("id")
                    elif last_line_data.get("children"):
                         for child in last_line_data["children"]:
                             if child.get("role") == "assistant":
                                 self.parent = child.get("id")
                                 break
            except (json.JSONDecodeError, IndexError):
                pass # Ignore errors parsing parent ID from non-stream

            self.conversation.update_chat_history(prompt, parsed_response)
            self.last_response = {"text": parsed_response}
            return self.last_response if not raw else parsed_response # Return dict or raw string
            
        except CurlError as e: # Catch CurlError
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {str(e)}") from e
        except Exception as e: # Catch other potential exceptions (like HTTPError, JSONDecodeError)
            err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
            raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {str(e)} - {err_text}") from e


    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        host: str = None,
        options: dict = None,
    ) -> Union[str, Generator[str, None, None]]: # Corrected return type hint
        def for_stream_chat(): # Renamed inner function
            # ask() yields dicts or strings when streaming
            gen = self.ask(
                prompt, 
                stream=True, 
                raw=False, # Ensure ask yields dicts
                optimizer=optimizer, 
                conversationally=conversationally,
                host=host,
                options=options
            )
            for response_dict in gen:
                yield self.get_message(response_dict) # get_message expects dict

        def for_non_stream_chat(): # Renamed inner function
            # ask() returns dict or str when not streaming
            response_data = self.ask(
                prompt, 
                stream=False, 
                raw=False, # Ensure ask returns dict
                optimizer=optimizer, 
                conversationally=conversationally,
                host=host,
                options=options
            )
            return self.get_message(response_data) # get_message expects dict

        return for_stream_chat() if stream else for_non_stream_chat() # Use renamed functions

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]



if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in AllenAI.AVAILABLE_MODELS:
        try:
            # Auto-detect host
            test_ai = AllenAI(model=model, timeout=60)
            # Pass the host explicitly to display accurate error messages
            response = test_ai.chat("Say 'Hello' in one word")
            response_text = response
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
                print(f"{model:<50} {status:<10} {display_text} (host: {test_ai.host})")
            else:
                status = "✗"
                display_text = "Empty or invalid response"
                print(f"{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"{model:<50} {'✗':<10} {str(e)}")