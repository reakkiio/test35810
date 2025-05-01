from typing import Any, Dict, Generator, Optional, Union
from curl_cffi.requests import Session
from curl_cffi import CurlError
import json

from webscout import exceptions
from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation, sanitize_stream # Import sanitize_stream
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider

class GPTWeb(Provider):
    """
    A class to interact with the Nexra GPTWeb API.
    """

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 600, # Note: max_tokens is not used by this API
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        # Note: system_prompt is not used by this API
    ):
        """
        Initializes the Nexra GPTWeb API with given parameters.

        Args:
            is_conversation (bool, optional): Flag for chatting conversationally. Defaults to True.
            max_tokens (int, optional): Maximum number of tokens to be generated upon completion. Defaults to 600.
            timeout (int, optional): Http request timeout. Defaults to 30.
            intro (str, optional): Conversation introductory prompt. Defaults to None.
            filepath (str, optional): Path to file containing conversation history. Defaults to None.
            update_file (bool, optional): Add new prompts and responses to the file. Defaults to True.
            proxies (dict, optional): Http request proxies. Defaults to {}.
            history_offset (int, optional): Limit conversation history to this number of last texts. Defaults to 10250.
            act (str|int, optional): Awesome prompt key or index. (Used as intro). Defaults to None.
        """
        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = 'https://nexra.aryahcr.cc/api/chat/gptweb'
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.headers = {
            "Content-Type": "application/json"
            # Remove User-Agent, Accept-Encoding, etc. - handled by impersonate
        }

        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly

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

    @staticmethod
    def _gptweb_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from GPTWeb stream JSON objects."""
        if isinstance(chunk, dict):
            return chunk.get("gpt")
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = False, # API supports streaming
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[dict, Generator[dict, None, None]]: # Corrected return type hint
        """Chat with GPTWeb

        Args:
            prompt (str): Prompt to be send.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            raw (bool, optional): Stream back raw response as received. Defaults to False.
            optimizer (str, optional): Prompt optimizer name - `[code, shell_command]`. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.
        Returns:
           dict : {}
        ```json
        {
           "text" : "How may I assist you today?"
        }
        ```
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

        data = {
            "prompt": conversation_prompt,
            "markdown": False
        }

        def for_stream():
            full_response = '' # Initialize outside try block
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint, 
                    # headers are set on the session
                    data=json.dumps(data), 
                    stream=True, 
                    timeout=self.timeout,
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors
            
                # Use sanitize_stream
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value=None, # No standard prefix, potential '_' handled by json.loads
                    to_json=True,     # Stream sends JSON lines
                    content_extractor=self._gptweb_extractor, # Use the specific extractor
                    yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                )

                for content_chunk in processed_stream:
                    # content_chunk is the full text extracted by _gptweb_extractor
                    if content_chunk and isinstance(content_chunk, str):
                        full_response = content_chunk # API sends full response each time
                        resp = dict(text=full_response)
                        yield resp if not raw else full_response
                
                # Update history after stream finishes (using the final full response)
                self.last_response = dict(text=full_response)
                self.conversation.update_chat_history(
                    prompt, full_response
                )
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"Failed to generate response ({type(e).__name__}): {e} - {err_text}") from e


        def for_non_stream():
            # Aggregate the stream using the updated for_stream logic
            # Since the stream yields the full response each time, we just need the last one.
            last_chunk = None
            try:
                for chunk in for_stream():
                    last_chunk = chunk
            except Exception as e:
                 # If aggregation fails, re-raise.
                 raise exceptions.FailedToGenerateResponseError(f"Failed to get non-stream response: {str(e)}") from e

            # last_response and history are updated within for_stream
            # Return the final aggregated response dict or raw string
            if last_chunk is None:
                 raise exceptions.FailedToGenerateResponseError("No response received from stream.")
            
            return last_chunk # last_chunk is already dict or raw string based on 'raw'


        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]: # Corrected return type hint
        """Generate response `str`
        Args:
            prompt (str): Prompt to be send.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            optimizer (str, optional): Prompt optimizer name - `[code, shell_command]`. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.
        Returns:
            str: Response generated
        """

        def for_stream_chat():
            # ask() yields dicts or strings when streaming
            gen = self.ask(
                prompt, stream=True, raw=False, # Ensure ask yields dicts
                optimizer=optimizer, conversationally=conversationally
            )
            # Since the API sends the full response each time, we only need the last one.
            # However, to maintain the streaming interface, we yield the message from each chunk.
            # This might result in repeated text if the client doesn't handle it.
            # A better approach might be to track changes, but for simplicity, yield each message.
            for response_dict in gen:
                yield self.get_message(response_dict) 

        def for_non_stream_chat():
            # ask() returns dict or str when not streaming
            response_data = self.ask(
                prompt,
                stream=False,
                raw=False, # Ensure ask returns dict
                optimizer=optimizer,
                conversationally=conversationally,
            )
            return self.get_message(response_data) # get_message expects dict

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        """Retrieves message only from response

        Args:
            response (dict): Response generated by `self.ask`

        Returns:
            str: Message extracted
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == '__main__':
    # Ensure curl_cffi is installed
    from rich import print
    ai = GPTWeb()
    response = ai.chat("tell me about Abhay koul, HelpingAI", stream=True)
    for chunk in response:
        print(chunk, end='', flush=True)