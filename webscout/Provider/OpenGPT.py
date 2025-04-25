from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
from typing import Dict, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent


class OpenGPT(Provider):
    """
    A class to interact with the Open-GPT API.
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
        app_id: str = "clf3yg8730000ih08ndbdi2v4",
    ):
        """Initializes the OpenGPT API client.
        
        Args:
            is_conversation (bool, optional): Whether to maintain conversation history. Defaults to True.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 600.
            timeout (int, optional): Http request timeout. Defaults to 30.
            intro (str, optional): Conversation introductory prompt. Defaults to None.
            filepath (str, optional): Path to file containing conversation history. Defaults to None.
            update_file (bool, optional): Add new prompts and responses to the file. Defaults to True.
            proxies (dict, optional): Http request proxies. Defaults to {}.
            history_offset (int, optional): Limit conversation history to this number of last texts. Defaults to 10250.
            act (str, optional): Awesome prompt key or index. (Used as intro). Defaults to None.
            app_id (str, optional): The OpenGPT application ID. Defaults to "clf3yg8730000ih08ndbdi2v4".
        """
        # Initialize curl_cffi Session
        self.session = Session()
        self.agent = LitAgent() # Keep for potential future use or other headers
        
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.app_id = app_id
        
        # Set up headers (remove User-Agent if using impersonate)
        self.headers = {
            "Content-Type": "application/json",
            # "User-Agent": self.agent.random(), # Removed, handled by impersonate
            "Referer": f"https://open-gpt.app/id/app/{self.app_id}",
            # Add sec-ch-ua headers if needed for impersonation consistency
        }
        
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly
        
        # Initialize optimizers
        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        
        # Setup conversation
        Conversation.intro = (
            AwesomePrompts().get_act(
                act, raise_not_found=True, default=None, case_insensitive=True
            ) if act else intro or Conversation.intro
        )
        
        self.conversation = Conversation(
            is_conversation, self.max_tokens_to_sample, filepath, update_file
        )
        self.conversation.history_offset = history_offset

    def ask(
        self,
        prompt: str,
        stream: bool = False, # Note: API does not support streaming
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict, Generator]:
        """
        Send a prompt to the OpenGPT API and get a response.
        
        Args:
            prompt: The user input/prompt for the API.
            stream: Whether to stream the response.
            raw: Whether to return the raw API response.
            optimizer: Optimizer to use on the prompt.
            conversationally: Whether to apply the optimizer on the full conversation prompt.
            
        Returns:
            A dictionary or generator with the response.
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")
                
        # Prepare the request body payload
        payload = {
            "userInput": conversation_prompt,
            "id": self.app_id,
            "userKey": ""  # Assuming userKey is meant to be empty as in the original code
        }
        
        # API does not stream, implement non-stream logic directly
        def for_non_stream():
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    "https://open-gpt.app/api/generate",
                    # headers are set on the session
                    data=json.dumps(payload), # Keep data as JSON string
                    timeout=self.timeout,
                    # proxies are set on the session
                    impersonate="chrome110" # Use a common impersonation profile
                )
                
                response.raise_for_status() # Check for HTTP errors
                
                # Use response.text which is already decoded
                response_text = response.text
                self.last_response = {"text": response_text}
                self.conversation.update_chat_history(prompt, response_text)
                
                # Return dict or raw string based on raw flag
                return {"raw": response_text} if raw else {"text": response_text}
                
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError, JSONDecodeError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e} - {err_text}") from e
        
        # This provider doesn't support streaming, so just return non-stream
        return for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False, # Keep stream param for interface consistency
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send a prompt to the OpenGPT API and get a text response.
        
        Args:
            prompt: The user input/prompt for the API.
            stream: Whether to stream the response (not supported).
            optimizer: Optimizer to use on the prompt.
            conversationally: Whether to apply the optimizer on the full conversation prompt.
            
        Returns:
            A string with the response text.
        """
        # Since ask() now handles both stream=True/False by returning the full response dict:
        response_data = self.ask(
            prompt, 
            stream=False, # Call ask in non-stream mode internally
            raw=False, # Ensure ask returns dict with 'text' key
            optimizer=optimizer, 
            conversationally=conversationally
        )
        # If stream=True was requested, simulate streaming by yielding the full message at once
        if stream:
            def stream_wrapper():
                yield self.get_message(response_data)
            return stream_wrapper()
        else:
            # If stream=False, return the full message directly
            return self.get_message(response_data)

    def get_message(self, response: dict) -> str:
        """
        Extract the message from the response dictionary.
        
        Args:
            response: Response dictionary from the ask method.
            
        Returns:
            The text response as a string.
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]


if __name__ == "__main__":
    ai = OpenGPT()
    print(ai.chat("Hello, how are you?"))