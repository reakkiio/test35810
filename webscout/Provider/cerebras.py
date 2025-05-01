
import re
import curl_cffi
from curl_cffi.requests import Session
import json
import os
from typing import Any, Dict, Optional, Generator, List, Union
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent as UserAgent

class Cerebras(Provider):
    """
    A class to interact with the Cerebras API using a cookie for authentication.
    """
    
    AVAILABLE_MODELS = [
        "llama3.1-8b",
        "llama-3.3-70b",
        "deepseek-r1-distill-llama-70b",
        "llama-4-scout-17b-16e-instruct"

    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2049,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        cookie_path: str = "cookie.json",
        model: str = "llama3.1-8b",
        system_prompt: str = "You are a helpful assistant.",
    ):
        # Validate model choice
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}"
            )

        # Initialize basic settings first
        self.timeout = timeout
        self.model = model
        self.system_prompt = system_prompt
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.last_response = {}

        self.session = Session() # Initialize curl_cffi session

        # Get API key first
        try:
            self.api_key = self.get_demo_api_key(cookie_path)
        except Exception as e:
            raise exceptions.APIConnectionError(f"Failed to initialize Cerebras client: {e}")

        # Initialize optimizers
        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )

        # Initialize conversation settings
        Conversation.intro = (
            AwesomePrompts().get_act(
                act, raise_not_found=True, default=None, case_insensitive=True
            )
            if act
            else None
        )
        self.conversation = Conversation(
            is_conversation, self.max_tokens_to_sample, filepath, update_file
        )
        self.conversation.history_offset = history_offset
        
        # Apply proxies to the session
        self.session.proxies = proxies

    # Rest of the class implementation remains the same...
    @staticmethod
    def extract_query(text: str) -> str:
        """Extracts the first code block from the given text."""
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

    @staticmethod
    def refiner(text: str) -> str:
        """Refines the input text by removing surrounding quotes."""
        return text.strip('"')

    @staticmethod
    def _cerebras_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from Cerebras stream JSON objects."""
        if isinstance(chunk, dict):
            return chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        return None

    def get_demo_api_key(self, cookie_path: str) -> str: # Keep this using requests or switch to curl_cffi
        """Retrieves the demo API key using the provided cookie."""
        try:
            with open(cookie_path, "r") as file:
                cookies = {item["name"]: item["value"] for item in json.load(file)}
        except FileNotFoundError:
            raise FileNotFoundError(f"Cookie file not found at path: {cookie_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Invalid JSON format in the cookie file.", "", 0)

        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "Origin": "https://inference.cerebras.ai",
            "Referer": "https://inference.cerebras.ai/",
            "user-agent": UserAgent().random(),
        }

        json_data = {
            "operationName": "GetMyDemoApiKey",
            "variables": {},
            "query": "query GetMyDemoApiKey {\n  GetMyDemoApiKey\n}",
        }

        try:
            # Use the initialized curl_cffi session
            response = self.session.post(
                "https://inference.cerebras.ai/api/graphql",
                cookies=cookies,
                headers=headers,
                json=json_data,
                timeout=self.timeout,
                impersonate="chrome120" # Add impersonate
            )
            response.raise_for_status()
            api_key = response.json().get("data", {}).get("GetMyDemoApiKey")
            return api_key
        except curl_cffi.CurlError as e:
            raise exceptions.APIConnectionError(f"Failed to retrieve API key: {e}")
        except KeyError:
            raise exceptions.InvalidResponseError("API key not found in response.")

    def _make_request(self, messages: List[Dict], stream: bool = False) -> Union[Dict, Generator]:
        """Make a request to the Cerebras API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": UserAgent().random(),
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }

        try:
            # Use the initialized curl_cffi session
            response = self.session.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=stream,
                timeout=self.timeout,
                impersonate="chrome120" # Add impersonate
            )
            response.raise_for_status()

            if stream:
                def generate_stream():
                    # Use sanitize_stream
                    processed_stream = sanitize_stream(
                        data=response.iter_content(chunk_size=None), # Pass byte iterator
                        intro_value="data:",
                        to_json=True,     # Stream sends JSON
                        content_extractor=self._cerebras_extractor, # Use the specific extractor
                        yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                    )
                    for content_chunk in processed_stream:
                        if content_chunk and isinstance(content_chunk, str):
                            yield content_chunk # Yield the extracted text chunk

                return generate_stream()
            else:
                response_json = response.json()
                # Extract content for non-streaming response
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content")
                return content if content else "" # Return empty string if not found

        except curl_cffi.CurlError as e:
            raise exceptions.APIConnectionError(f"Request failed (CurlError): {e}") from e
        except Exception as e: # Catch other potential errors
            raise exceptions.APIConnectionError(f"Request failed: {e}")

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False, # Add raw parameter for consistency
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict, Generator]:
        """Send a prompt to the model and get a response."""
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conversation_prompt}
        ]

        try:
            response = self._make_request(messages, stream)
            
            if stream:
                # Wrap the generator to yield dicts or raw strings
                def stream_wrapper():
                    full_text = ""
                    for chunk in response:
                        full_text += chunk
                        yield chunk if raw else {"text": chunk}
                    # Update history after stream finishes
                    self.last_response = {"text": full_text}
                    self.conversation.update_chat_history(prompt, full_text)
                return stream_wrapper()
            else:
                # Non-streaming response is already the full text string
                self.last_response = {"text": response}
                self.conversation.update_chat_history(prompt, response)
                return self.last_response if not raw else response # Return dict or raw string

        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Error during request: {e}")

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator]:
        """Chat with the model."""
        # Ask returns a generator for stream=True, dict/str for stream=False
        response_gen_or_dict = self.ask(prompt, stream, raw=False, optimizer=optimizer, conversationally=conversationally)
        
        if stream:
            # Wrap the generator from ask() to get message text
            def stream_wrapper():
                for chunk_dict in response_gen_or_dict:
                    yield self.get_message(chunk_dict)
            return stream_wrapper()
        else:
            # Non-streaming response is already a dict
            return self.get_message(response_gen_or_dict)

    def get_message(self, response: str) -> str:
        """Retrieves message from response."""
        # Updated to handle dict input from ask()
        assert isinstance(response, dict), "Response should be of dict data-type only for get_message"
        return response.get("text", "")


if __name__ == "__main__":
    from rich import print
    
    # Example usage
    cerebras = Cerebras(
        cookie_path=r'cookies.json',
        model='llama3.1-8b',
        system_prompt="You are a helpful AI assistant."
    )
    
    # Test with streaming
    response = cerebras.chat("Hello!", stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)
