from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import uuid
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions
from webscout.litagent import LitAgent

class HeckAI(Provider):
    """
    Provides an interface to interact with the HeckAI API using a LitAgent user-agent.

    This class supports conversational AI interactions with multiple available models,
    manages session state, handles streaming and non-streaming responses, and integrates
    with conversation history and prompt optimizers.

    Attributes:
        AVAILABLE_MODELS (list): List of supported model identifiers.
        url (str): API endpoint URL.
        session_id (str): Unique session identifier for the conversation.
        language (str): Language for the conversation.
        headers (dict): HTTP headers used for API requests.
        session (Session): curl_cffi session for HTTP requests.
        is_conversation (bool): Whether to maintain conversation history.
        max_tokens_to_sample (int): Maximum tokens to sample (not used by API).
        timeout (int): Request timeout in seconds.
        last_response (dict): Stores the last API response.
        model (str): Model identifier in use.
        previous_question (str): Last question sent to the API.
        previous_answer (str): Last answer received from the API.
        conversation (Conversation): Conversation history manager.
    """

    AVAILABLE_MODELS = [
        "google/gemini-2.5-flash-preview",
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
        max_tokens: int = 2049,
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
        """
        Initializes the HeckAI API client.

        Args:
            is_conversation (bool): Whether to maintain conversation history.
            max_tokens (int): Maximum tokens to sample (not used by this API).
            timeout (int): Timeout for API requests in seconds.
            intro (str, optional): Introductory prompt for the conversation.
            filepath (str, optional): File path for storing conversation history.
            update_file (bool): Whether to update the conversation file.
            proxies (dict): Proxy settings for HTTP requests.
            history_offset (int): Offset for conversation history truncation.
            act (str, optional): Role or act for the conversation.
            model (str): Model identifier to use.
            language (str): Language for the conversation.

        Raises:
            ValueError: If the provided model is not in AVAILABLE_MODELS.
        """
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
            'User-Agent': LitAgent().random(), # Use random user agent
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
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        """
        Sends a prompt to the HeckAI API and returns the response.

        Args:
            prompt (str): The prompt or question to send to the API.
            stream (bool): If True, yields streaming responses as they arrive.
            raw (bool): If True, yields raw string chunks instead of dicts.
            optimizer (str, optional): Name of the optimizer to apply to the prompt.
            conversationally (bool): If True, optimizer is applied to the full conversation prompt.

        Returns:
            Union[Dict[str, Any], Generator]: If stream is False, returns a dict with the response text.
            If stream is True, yields response chunks as dicts or strings.

        Raises:
            Exception: If the optimizer is not available.
            exceptions.FailedToGenerateResponseError: On API or network errors.
        """
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
            "imgUrls": [],
            "superSmartMode": False  # Added based on API request data
        }
        
        # Store this message as previous for next request
        self.previous_question = conversation_prompt

        def for_stream():
            streaming_text = "" # Initialize outside try block
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

                # Use sanitize_stream to process the stream
                processed_stream = sanitize_stream(
                data=response.iter_content(chunk_size=1024), # Pass byte iterator
                intro_value="data: ",  # Prefix to remove (note the space)
                to_json=False,        # Content is text
                start_marker="data: [ANSWER_START]",
                end_marker="data: [ANSWER_DONE]",
                skip_markers=["data: [RELATE_Q_START]", "data: [RELATE_Q_DONE]", "data: [REASON_START]", "data: [REASON_DONE]"],
                yield_raw_on_error=True,
                strip_chars=" \n\r\t"  # Strip whitespace characters from chunks
                )

                for content_chunk in processed_stream:
                    # content_chunk is the text between ANSWER_START and ANSWER_DONE
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        yield dict(text=content_chunk) if not raw else content_chunk
                
                # Only update history if we received a valid response
                if streaming_text:
                    # Update history and previous answer after stream finishes
                    self.previous_answer = streaming_text
                    # Convert to simple text before updating conversation
                    try:
                        # Ensure content is valid before updating conversation
                        if streaming_text and isinstance(streaming_text, str):
                            # Sanitize the content to ensure it's valid
                            sanitized_text = streaming_text.strip()
                            if sanitized_text:  # Only update if we have non-empty content
                                self.conversation.update_chat_history(prompt, sanitized_text)
                    except Exception as e:
                        # If conversation update fails, log but don't crash
                        print(f"Warning: Failed to update conversation history: {str(e)}")
                    
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

            # Return the final aggregated response dict or raw string
            self.last_response = {"text": full_text} # Update last_response here
            return full_text if raw else self.last_response


        return for_stream() if stream else for_non_stream()

    @staticmethod
    def fix_encoding(text):
        """
        Fixes encoding issues in the response text.

        Args:
            text (Union[str, dict]): The text or response dict to fix encoding for.

        Returns:
            Union[str, dict]: The text or dict with encoding corrected if possible.
        """
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
    ) -> Union[str, Generator[str, None, None]]:
        """
        Sends a prompt to the HeckAI API and returns only the message text.

        Args:
            prompt (str): The prompt or question to send to the API.
            stream (bool): If True, yields streaming response text.
            optimizer (str, optional): Name of the optimizer to apply to the prompt.
            conversationally (bool): If True, optimizer is applied to the full conversation prompt.

        Returns:
            Union[str, Generator[str, None, None]]: The response text, or a generator yielding text chunks.
        """
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
        """
        Extracts the message text from the API response.

        Args:
            response (dict): The API response dictionary.

        Returns:
            str: The extracted message text. Returns an empty string if not found.

        Raises:
            TypeError: If the response is not a dictionary.
        """
        # Validate response format
        if not isinstance(response, dict):
            raise TypeError(f"Expected dict response, got {type(response).__name__}")

        # Handle missing text key gracefully
        if "text" not in response:
            return ""

        # Ensure text is a string
        text = response["text"]
        if not isinstance(text, str):
            return str(text)

        return text

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in HeckAI.AVAILABLE_MODELS:
        try:
            test_ai = HeckAI(model=model, timeout=60)
            # Use non-streaming mode first to avoid potential streaming issues
            try:
                response_text = test_ai.chat("Say 'Hello' in one word", stream=False)
                print(f"\r{model:<50} {'✓':<10} {response_text.strip()[:50]}")
            except Exception as e1:
                # Fall back to streaming if non-streaming fails
                print(f"\r{model:<50} {'Testing stream...':<10}", end="", flush=True)
                response = test_ai.chat("Say 'Hello' in one word", stream=True)
                response_text = ""
                for chunk in response:
                    if chunk and isinstance(chunk, str):
                        response_text += chunk
                
                if response_text and len(response_text.strip()) > 0:
                    status = "✓"
                    # Truncate response if too long
                    display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
                    print(f"\r{model:<50} {status:<10} {display_text}")
                else:
                    raise ValueError("Empty or invalid response")
        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
