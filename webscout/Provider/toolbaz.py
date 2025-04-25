import re
from curl_cffi.requests import Session
from curl_cffi import CurlError
import uuid
import base64
import json
import random
import string
import time
from datetime import datetime
from typing import Any, Dict, Optional, Generator, Union, List

from webscout import exceptions
from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider

class Toolbaz(Provider):
    """
    A class to interact with the Toolbaz API. Supports streaming responses.
    """

    AVAILABLE_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash-thinking",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gpt-4o-latest",
        "gpt-4o",
        "deepseek-r1",
        "Llama-4-Maverick",
        "Llama-4-Scout",
        "Llama-3.3-70B",
        "Qwen2.5-72B",
        "Qwen2-72B",
        "grok-2-1212",
        "grok-3-beta",
        "toolbaz_v3.5_pro",
        "toolbaz_v3",
        "mixtral_8x22b",
        "L3-70B-Euryale-v2.1",
        "midnight-rose",
        "unity",
        "unfiltered_x"
    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 600, # Note: max_tokens is not directly used by the API
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "gemini-2.0-flash",
        system_prompt: str = "You are a helpful AI assistant." # Note: system_prompt is not directly used by the API
    ):
        """
        Initializes the Toolbaz API with given parameters.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.system_prompt = system_prompt
        self.model = model
        self.proxies = proxies # Store proxies for later use in requests

        # Set up headers for the curl_cffi session
        self.session.headers.update({
            "user-agent": "Mozilla/5.0 (Linux; Android 10)", # Keep specific user-agent
            "accept": "*/*",
            "accept-language": "en-US",
            "cache-control": "no-cache",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "origin": "https://toolbaz.com",
            "pragma": "no-cache",
            "referer": "https://toolbaz.com/",
            "sec-fetch-mode": "cors"
            # Add sec-ch-ua headers if needed for impersonation consistency
        })
        # Assign proxies directly to the session
        self.session.proxies = proxies

        # Initialize conversation history
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

    def random_string(self, length):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def generate_token(self):
        payload = {
            "bR6wF": {
                "nV5kP": "Mozilla/5.0 (Linux; Android 10)",
                "lQ9jX": "en-US",
                "sD2zR": "431x958",
                "tY4hL": time.tzname[0] if time.tzname else "UTC",
                "pL8mC": "Linux armv81",
                "cQ3vD": datetime.now().year,
                "hK7jN": datetime.now().hour
            },
            "uT4bX": {
                "mM9wZ": [],
                "kP8jY": []
            },
            "tuTcS": int(time.time()),
            "tDfxy": None,
            "RtyJt": str(uuid.uuid4())
        }
        return "d8TW0v" + base64.b64encode(json.dumps(payload).encode()).decode()

    def get_auth(self):
        try:
            session_id = self.random_string(36)
            token = self.generate_token()
            data = {
                "session_id": session_id,
                "token": token
            }
            # Use curl_cffi session post WITHOUT impersonate for token request
            resp = self.session.post(
                "https://data.toolbaz.com/token.php", 
                data=data
                # Removed impersonate="chrome110" for this specific request
            )
            resp.raise_for_status() # Check for HTTP errors
            result = resp.json()
            if result.get("success"):
                return {"token": result["token"], "session_id": session_id}
            # Raise error if success is not true
            raise exceptions.FailedToGenerateResponseError(f"Authentication failed: API response indicates failure. Response: {result}")
        except CurlError as e: # Catch CurlError specifically
            # Raise a specific error indicating CurlError during auth
            raise exceptions.FailedToGenerateResponseError(f"Authentication failed due to network error (CurlError): {e}") from e
        except json.JSONDecodeError as e:
             # Raise error for JSON decoding issues
             raise exceptions.FailedToGenerateResponseError(f"Authentication failed: Could not decode JSON response. Error: {e}. Response text: {getattr(resp, 'text', 'N/A')}") from e
        except Exception as e: # Catch other potential errors (like HTTPError from raise_for_status)
            # Raise a specific error indicating a general failure during auth
            err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
            raise exceptions.FailedToGenerateResponseError(f"Authentication failed due to an unexpected error ({type(e).__name__}): {e} - {err_text}") from e

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,  # Kept for compatibility, but output is always dict/string
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        """Sends a prompt to the Toolbaz API and returns the response."""
        if optimizer and optimizer not in self.__available_optimizers:
            raise exceptions.FailedToGenerateResponseError(f"Optimizer is not one of {self.__available_optimizers}")

        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            conversation_prompt = getattr(Optimizers, optimizer)(
                conversation_prompt if conversationally else prompt
            )

        # get_auth now raises exceptions on failure
        auth = self.get_auth() 
        # No need to check if auth is None, as an exception would have been raised

        data = {
            "text": conversation_prompt,
            "capcha": auth["token"],
            "model": self.model,
            "session_id": auth["session_id"]
        }

        def for_stream():
            try:
                # Use curl_cffi session post with impersonate for the main request
                resp = self.session.post(
                    "https://data.toolbaz.com/writing.php",
                    data=data,
                    stream=True,
                    timeout=self.timeout,
                    impersonate="chrome110" # Keep impersonate here
                )
                resp.raise_for_status()

                buffer = ""
                tag_start = "[model:"
                streaming_text = ""

                # Iterate over bytes and decode manually
                for chunk_bytes in resp.iter_content(chunk_size=1024): # Read in larger chunks
                    if chunk_bytes:
                        text = chunk_bytes.decode(errors="ignore")
                        buffer += text
                        
                        processed_buffer = ""
                        last_processed_index = 0
                        # Find all complete tags and process text between them
                        for match in re.finditer(r"\[model:.*?\]", buffer):
                            # Add text before the tag
                            segment = buffer[last_processed_index:match.start()]
                            if segment:
                                processed_buffer += segment
                            last_processed_index = match.end()
                        
                        # Add remaining text after the last complete tag
                        processed_buffer += buffer[last_processed_index:]
                        
                        # Now, check for incomplete tag at the end
                        last_tag_start_index = processed_buffer.rfind(tag_start)
                        
                        if last_tag_start_index != -1:
                            # Text before the potential incomplete tag
                            text_to_yield = processed_buffer[:last_tag_start_index]
                            # Keep the potential incomplete tag start for the next iteration
                            buffer = processed_buffer[last_tag_start_index:] 
                        else:
                            # No potential incomplete tag found, yield everything processed
                            text_to_yield = processed_buffer
                            buffer = "" # Clear buffer as everything is processed

                        if text_to_yield:
                            streaming_text += text_to_yield
                            # Yield dict or raw string
                            yield {"text": text_to_yield} if not raw else text_to_yield

                # Process any remaining text in the buffer after the loop finishes
                # Remove any potential tags (complete or incomplete)
                final_text = re.sub(r"\[model:.*?\]", "", buffer)
                if final_text:
                    streaming_text += final_text
                    yield {"text": final_text} if not raw else final_text

                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)

            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Network error (CurlError): {str(e)}") from e
            except Exception as e: # Catch other exceptions
                raise exceptions.FailedToGenerateResponseError(f"Unexpected error during stream: {str(e)}") from e

        def for_non_stream():
            try:
                # Use curl_cffi session post with impersonate for the main request
                resp = self.session.post(
                    "https://data.toolbaz.com/writing.php",
                    data=data,
                    timeout=self.timeout,
                    impersonate="chrome110" # Keep impersonate here
                )
                resp.raise_for_status()

                # Use response.text which is already decoded
                text = resp.text 
                # Remove [model: ...] tags
                text = re.sub(r"\[model:.*?\]", "", text)

                self.last_response = {"text": text}
                self.conversation.update_chat_history(prompt, text)

                return self.last_response

            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Network error (CurlError): {str(e)}") from e
            except Exception as e: # Catch other exceptions
                raise exceptions.FailedToGenerateResponseError(f"Unexpected error: {str(e)}") from e

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generates a response from the Toolbaz API."""
        def for_stream_chat():
            # ask() yields dicts when raw=False
            for response_dict in self.ask(
                prompt,
                stream=True,
                raw=False, # Ensure ask yields dicts
                optimizer=optimizer,
                conversationally=conversationally
            ):
                yield self.get_message(response_dict)

        def for_non_stream_chat():
            # ask() returns a dict when stream=False
            response_dict = self.ask(
                prompt,
                stream=False,
                optimizer=optimizer,
                conversationally=conversationally,
            )
            return self.get_message(response_dict)

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: Dict[str, Any]) -> str:
        """Extract the message from the response.

        Args:
            response: Response dictionary

        Returns:
            str: Message extracted
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "")

# Example usage
if __name__ == "__main__":
    # Ensure curl_cffi is installed
    from rich import print # Use rich print if available
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)
    # Test the provider with different models
    for model in Toolbaz.AVAILABLE_MODELS:
        try:
            test_ai = Toolbaz(model=model, timeout=60)
            # Test stream first
            response_stream = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            # print(f"\r{model:<50} {'Streaming...':<10}", end="", flush=True)
            for chunk in response_stream:
                response_text += chunk
                # Optional: print chunks for visual feedback
                # print(chunk, end="", flush=True)

            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response_text.strip()
                display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            else:
                status = "✗ (Stream)"
                display_text = "Empty or invalid stream response"
            print(f"\r{model:<50} {status:<10} {display_text}")

            # Optional: Add non-stream test if needed
            # print(f"\r{model:<50} {'Non-Stream...':<10}", end="", flush=True)
            # response_non_stream = test_ai.chat("Say 'Hi' again", stream=False)
            # if not response_non_stream or len(response_non_stream.strip()) == 0:
            #      print(f"\r{model:<50} {'✗ (Non-Stream)':<10} Empty non-stream response")

        except Exception as e:
            # Print full error for debugging
            print(f"\r{model:<50} {'✗':<10} Error: {str(e)}")