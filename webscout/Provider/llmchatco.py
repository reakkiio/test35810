from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import uuid
import re
from typing import Union, Any, Dict, Optional, Generator, List

from webscout.AIutel import Optimizers, sanitize_stream # Import sanitize_stream
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent as Lit

class LLMChatCo(Provider):
    """
    A class to interact with the LLMChat.co API
    """

    AVAILABLE_MODELS = [
        "gemini-flash-2.0",        # Default model
        "llama-4-scout",
        "gpt-4o-mini",
        "gpt-4.1-nano",


        # "gpt-4.1",
        # "gpt-4.1-mini",
        # "o3-mini",
        # "claude-3-5-sonnet",
        # "deepseek-r1",
        # "claude-3-7-sonnet",
        # "deep", # deep research mode
        # "pro" # pro research mode

    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2048, # Note: max_tokens is not used by this API
        timeout: int = 60,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "gemini-flash-2.0",
        system_prompt: str = "You are a helpful assistant."
    ):
        """
        Initializes the LLMChat.co API with given parameters.
        """

        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://llmchat.co/api/completion"
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt
        self.thread_id = str(uuid.uuid4())  # Generate a unique thread ID for conversations

        # Create LitAgent instance (keep if needed for other headers)
        lit_agent = Lit()

        # Headers based on the provided request
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": lit_agent.random(),
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://llmchat.co",
            "Referer": f"https://llmchat.co/chat/{self.thread_id}",
            "DNT": "1",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            # Add sec-ch-ua headers if needed for impersonation consistency
        }

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
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly
        # Store message history for conversation context
        self.last_assistant_response = ""

    @staticmethod
    def _llmchatco_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts text content from LLMChat.co stream JSON objects."""
        if isinstance(chunk, dict) and "answer" in chunk:
            answer = chunk["answer"]
            # Prefer fullText if available and status is COMPLETED
            if answer.get("fullText") and answer.get("status") == "COMPLETED":
                return answer["fullText"]
            elif "text" in answer:
                return answer["text"]
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = True,  # Default to stream as the API uses SSE
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        web_search: bool = False,
    ) -> Union[Dict[str, Any], Generator[Any, None, None]]:
        """Chat with LLMChat.co with streaming capabilities"""

        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise exceptions.FailedToGenerateResponseError(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )


        # Generate a unique ID for this message
        thread_item_id = ''.join(str(uuid.uuid4()).split('-'))[:20]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        # Prepare payload for the API request based on observed request format
        payload = {
            "mode": self.model,
            "prompt": prompt,
            "threadId": self.thread_id,
            "messages": messages,
            "mcpConfig": {},
            "threadItemId": thread_item_id,
            "parentThreadItemId": "",
            "webSearch": web_search,
            "showSuggestions": True
        }

        def for_stream():
            full_response = "" # Initialize outside try block
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint,
                    json=payload,
                    # headers are set on the session
                    stream=True,
                    timeout=self.timeout,
                    # proxies are set on the session
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors

                # Use sanitize_stream
                # Note: This won't handle SSE 'event:' lines, only 'data:' lines.
                # The original code checked for event == 'answer'. We assume relevant data is JSON after 'data:'.
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value="data:",
                    to_json=True,     # Stream sends JSON
                    content_extractor=self._llmchatco_extractor, # Use the specific extractor
                    yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                )

                last_yielded_text = ""
                for current_full_text in processed_stream:
                    # current_full_text is the full text extracted by _llmchatco_extractor
                    if current_full_text and isinstance(current_full_text, str):
                        # Calculate the new part of the text
                        new_text = current_full_text[len(last_yielded_text):]
                        if new_text:
                            full_response = current_full_text # Keep track of the latest full text
                            last_yielded_text = current_full_text # Update tracker
                            resp = dict(text=new_text)
                            # Yield dict or raw string chunk
                            yield resp if not raw else new_text

                # Update history after stream finishes
                self.last_response = dict(text=full_response)
                self.last_assistant_response = full_response
                self.conversation.update_chat_history(
                    prompt, full_response
                )

            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"Unexpected error ({type(e).__name__}): {str(e)} - {err_text}") from e

        def for_non_stream():
            # Aggregate the stream using the updated for_stream logic
            full_response_text = ""
            try:
                # Ensure raw=False so for_stream yields dicts
                for chunk_data in for_stream():
                    if isinstance(chunk_data, dict) and "text" in chunk_data:
                        full_response_text += chunk_data["text"]
                    # Handle raw string case if raw=True was passed
                    elif raw and isinstance(chunk_data, str):
                         full_response_text += chunk_data

            except Exception as e:
                # If aggregation fails but some text was received, use it. Otherwise, re-raise.
                if not full_response_text:
                    raise exceptions.FailedToGenerateResponseError(f"Failed to get non-stream response: {str(e)}") from e

            # last_response and history are updated within for_stream
            # Return the final aggregated response dict or raw string
            return full_response_text if raw else self.last_response


        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        web_search: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response with streaming capabilities"""

        def for_stream_chat():
            # ask() yields dicts or strings when streaming
            gen = self.ask(
                prompt, stream=True, raw=False, # Ensure ask yields dicts
                optimizer=optimizer, conversationally=conversationally,
                web_search=web_search
            )
            for response_dict in gen:
                yield self.get_message(response_dict) # get_message expects dict

        def for_non_stream_chat():
            # ask() returns dict or str when not streaming
            response_data = self.ask(
                prompt,
                stream=False,
                raw=False, # Ensure ask returns dict
                optimizer=optimizer,
                conversationally=conversationally,
                web_search=web_search
            )
            return self.get_message(response_data) # get_message expects dict

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: Dict[str, Any]) -> str:
        """Retrieves message from response with validation"""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    # Test all available models
    working = 0
    total = len(LLMChatCo.AVAILABLE_MODELS)

    for model in LLMChatCo.AVAILABLE_MODELS:
        try:
            test_ai = LLMChatCo(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word")
            response_text = response

            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"{model:<50} {'✗':<10} {str(e)}")
