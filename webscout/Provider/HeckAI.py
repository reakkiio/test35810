from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import uuid
import sys
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions
from webscout.litagent import LitAgent

class HeckAI(Provider):
    """
    A class to interact with the HeckAI API with LitAgent user-agent.
    """

    AVAILABLE_MODELS = [
        "google/gemini-2.0-flash-001",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-r1",
        "openai/gpt-4o-mini",
        "openai/gpt-4.1-mini",
        "x-ai/grok-3-mini-beta",
        "meta-llama/llama-4-scout"

    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2049, # Note: max_tokens is not used by this API
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "google/gemini-2.0-flash-001",
        language: str = "English"
    ):
        """Initializes the HeckAI API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
            
        self.url = "https://api.heckai.weight-wave.com/api/ha/v1/chat"
        self.session_id = str(uuid.uuid4())
        self.language = language
        
        # Use LitAgent (keep if needed for other headers or logic)
        self.headers = {
            'Content-Type': 'application/json',
            'Origin': 'https://heck.ai', # Keep Origin
            'Referer': 'https://heck.ai/', # Keep Referer
        }
        
        # Initialize curl_cffi Session
        self.session = Session()
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly

        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.previous_question = None
        self.previous_answer = None

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

        # Payload construction
        payload = {
            "model": self.model,
            "question": conversation_prompt,
            "language": self.language,
            "sessionId": self.session_id,
            "previousQuestion": self.previous_question,
            "previousAnswer": self.previous_answer,
            "imgUrls": []
        }
        
        # Store this message as previous for next request
        self.previous_question = conversation_prompt

        def for_stream():
            streaming_text = "" # Initialize outside try block
            in_answer = False # Initialize outside try block
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.url, 
                    # headers are set on the session
                    data=json.dumps(payload), 
                    stream=True, 
                    timeout=self.timeout,
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors
                    
                # Iterate over bytes and decode manually
                for line_bytes in response.iter_lines():
                    if not line_bytes:
                        continue
                        
                    try:
                        line = line_bytes.decode('utf-8')
                        # Remove "data: " prefix
                        if line.startswith("data: "):
                            data = line[6:]
                        else:
                            continue # Skip lines without the prefix
                        
                        # Check for control markers
                        if data == "[ANSWER_START]":
                            in_answer = True
                            continue
                            
                        if data == "[ANSWER_DONE]":
                            in_answer = False
                            continue
                            
                        if data == "[RELATE_Q_START]" or data == "[RELATE_Q_DONE]":
                            continue
                            
                        # Process content if we're in an answer section
                        if in_answer:
                            # Assuming 'data' is the text chunk here
                            streaming_text += data
                            resp = dict(text=data)
                            # Yield dict or raw string chunk
                            yield resp if not raw else data
                    except UnicodeDecodeError:
                        continue # Ignore decoding errors for specific lines
                
                # Update history and previous answer after stream finishes
                self.previous_answer = streaming_text
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
            self.last_response = {"text": full_text} # Update last_response here
            return full_text if raw else self.last_response


        return for_stream() if stream else for_non_stream()

    @staticmethod
    def fix_encoding(text):
        if isinstance(text, dict) and "text" in text:
            try:
                text["text"] = text["text"].encode("latin1").decode("utf-8")
                return text
            except (UnicodeError, AttributeError) as e:
                return text
        elif isinstance(text, str):
            try:
                return text.encode("latin1").decode("utf-8")
            except (UnicodeError, AttributeError) as e:
                return text
        return text

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]: # Corrected return type hint
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

    for model in HeckAI.AVAILABLE_MODELS:
        try:
            test_ai = HeckAI(model=model, timeout=60)
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