from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
from typing import Optional, Union, Any, Dict, Generator

from webscout.AIutel import Optimizers, Conversation, AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent as Lit

class IBMGranite(Provider):
    """
    A class to interact with the IBM Granite API (accessed via d18n68ssusgr7r.cloudfront.net)
    using Lit agent for the user agent.
    """

    AVAILABLE_MODELS = ["granite-3-8b-instruct", "granite-3-2-8b-instruct"]

    def __init__(
        self,
        api_key: str,
        is_conversation: bool = True,
        max_tokens: int = 600, # Note: max_tokens is not used by this API
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "granite-3-2-8b-instruct",
        system_prompt: str = "You are a helpful AI assistant.",
        thinking: bool = False,
    ):
        """Initializes the IBMGranite API client using Lit agent for the user agent."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://d18n68ssusgr7r.cloudfront.net/v1/chat/completions"
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt
        self.thinking = thinking

        # Use Lit agent (keep if needed for other headers or logic)
        self.headers = {
            "authority": "d18n68ssusgr7r.cloudfront.net", # Keep authority
            "accept": "application/json,application/jsonl", # Keep accept
            "content-type": "application/json",
            "origin": "https://www.ibm.com", # Keep origin
            "referer": "https://www.ibm.com/", # Keep referer
        }
        self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly

        self.__available_optimizers = (
            method for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )

        Conversation.intro = (
            AwesomePrompts().get_act(
                act, raise_not_found=True, default=None, case_insensitive=True
            )
            if act
            else intro or Conversation.intro
        )
        self.conversation = Conversation(is_conversation, self.max_tokens_to_sample, filepath, update_file)
        self.conversation.history_offset = history_offset

    @staticmethod
    def _granite_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from IBM Granite stream JSON lists [3, "text"]."""
        if isinstance(chunk, list) and len(chunk) == 2 and chunk[0] == 3 and isinstance(chunk[1], str):
            return chunk[1]
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = False, # API supports streaming
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator[Any, None, None]]:
        """Chat with AI
        Args:
            prompt (str): Prompt to be sent.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            raw (bool, optional): Stream back raw response as received. Defaults to False.
            optimizer (str, optional): Prompt optimizer name - `[code, shell_command]`. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.
        Returns:
            Union[Dict, Generator[Dict, None, None]]: Response generated
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt},
            ],
            "stream": True # API seems to require stream=True based on response format
        }

        def for_stream():
            streaming_text = "" # Initialize outside try block
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint, 
                    # headers are set on the session
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
                    to_json=True,     # Stream sends JSON lines (which are lists)
                    content_extractor=self._granite_extractor, # Use the specific extractor
                    yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                )

                for content_chunk in processed_stream:
                    # content_chunk is the string extracted by _granite_extractor
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        resp = dict(text=content_chunk)
                        yield resp if not raw else content_chunk
                
                # Update history after stream finishes
                self.last_response = dict(text=streaming_text)
                self.conversation.update_chat_history(prompt, streaming_text)
                
            except CurlError as e: # Catch CurlError
                raise exceptions.ProviderConnectionError(f"Request failed (CurlError): {e}") from e
            except json.JSONDecodeError as e: # Keep specific JSON error handling
                raise exceptions.InvalidResponseError(f"Failed to decode JSON response: {e}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                # Use specific exception type if available, otherwise generic
                ex_type = exceptions.FailedToGenerateResponseError if not isinstance(e, exceptions.ProviderConnectionError) else type(e)
                raise ex_type(f"An unexpected error occurred ({type(e).__name__}): {e} - {err_text}") from e


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
        """Generate response as a string using chat method"""
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
        """Retrieves message only from response"""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    from rich import print
    # Example usage: Initialize without logging.
    ai = IBMGranite(
        api_key="",  # press f12 to see the API key
        thinking=True,
    )
    response = ai.chat("write a poem about AI", stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)
