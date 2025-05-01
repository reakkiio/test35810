from curl_cffi import CurlError
from curl_cffi.requests import Session
import json
from typing import Any, Dict, Optional, Generator, List, Union
import uuid

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class ChatGLM(Provider):
    """
    A class to interact with the ChatGLM API.
    """

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 600,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        plus_model: bool = True,
    ):
        """Initializes the ChatGLM API client."""
        self.session = Session() # Use curl_cffi Session
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://chatglm.cn/chatglm/mainchat-api/guest/stream"
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.plus_model = plus_model
        self.headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'App-Name': 'chatglm',
            'Authorization': 'undefined',
            'Content-Type': 'application/json',
            'Origin': 'https://chatglm.cn',
            'User-Agent': LitAgent().random(),
            'X-App-Platform': 'pc',
            'X-App-Version': '0.0.1',
            'X-Device-Id': '', #Will be generated each time
            'Accept': 'text/event-stream',
        }
        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        self.session.headers.update(self.headers)
        Conversation.intro = ( # type: ignore
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
        self.session.proxies = proxies # Assign proxies directly

    @staticmethod
    def _chatglm_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from ChatGLM stream JSON objects."""
        if isinstance(chunk, dict):
            parts = chunk.get('parts', [])
            if parts and isinstance(parts[0].get('content'), list) and parts[0]['content']:
                return parts[0]['content'][0].get('text')
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = False,
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
                raise exceptions.FailedToGenerateResponseError(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )
        device_id = str(uuid.uuid4()).replace('-', '')
        self.session.headers.update({'X-Device-Id': device_id})
        payload = {
            "assistant_id": "65940acff94777010aa6b796",
            "conversation_id": "",
            "meta_data": {
                "if_plus_model": self.plus_model,
                "is_test": False,
                "input_question_type": "xxxx",
                "channel": "",
                "draft_id": "",
                "quote_log_id": "",
                "platform": "pc",
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": conversation_prompt}],
                }
            ],
        }

        def for_stream():
            streaming_text = "" # Initialize outside try block
            last_processed_content = "" # Track the last processed content
            try:
                response = self.session.post(
                    self.api_endpoint, json=payload, stream=True, timeout=self.timeout,
                    impersonate="chrome120" # Add impersonate
                )
                response.raise_for_status()

                # Use sanitize_stream
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value="data:",
                    to_json=True,     # Stream sends JSON
                    content_extractor=self._chatglm_extractor, # Use the specific extractor
                    yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                )

                for current_full_text in processed_stream:
                    # current_full_text is the full text extracted by _chatglm_extractor
                    if current_full_text and isinstance(current_full_text, str):
                        new_text = current_full_text[len(last_processed_content):]
                        if new_text: # Check for new content
                            streaming_text += new_text
                            last_processed_content = current_full_text # Update tracker
                            yield new_text if raw else dict(text=new_text)

            except CurlError as e:
                raise exceptions.ProviderConnectionError(f"Request failed (CurlError): {e}") from e
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}") from e
            finally:
                # Update history after stream finishes or fails
                if streaming_text:
                    self.last_response.update(dict(text=streaming_text))
                    self.conversation.update_chat_history(
                        prompt, self.get_message(self.last_response)
                    )

        def for_non_stream():
            for _ in for_stream():
                pass
            return self.last_response
        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response `str`"""

        def for_stream():
            for response in self.ask(
                prompt, True, optimizer=optimizer, conversationally=conversationally
            ):
                yield self.get_message(response)

        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt,
                    False,
                    optimizer=optimizer,
                    conversationally=conversationally,
                )
            )

        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        """Retrieves message only from response"""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]


if __name__ == "__main__":
    from rich import print
    ai = ChatGLM()
    response = ai.chat(input(">>> "), stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)