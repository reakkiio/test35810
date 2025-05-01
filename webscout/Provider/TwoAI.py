from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
from typing import Any, Dict, Optional, Generator, Union
import re  # Import re for parsing SSE

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent


class TwoAI(Provider):
    """
    A class to interact with the Two AI API (v2) with LitAgent user-agent.
    """

    AVAILABLE_MODELS = [
        "sutra-v2",
        "sutra-r0"

    ]

    def __init__(
        self,
        api_key: str = None,
        is_conversation: bool = True,
        max_tokens: int = 1024,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "sutra-v2",  # Update default model
        temperature: float = 0.6,
        system_message: str = "You are a helpful assistant."
    ):
        """Initializes the TwoAI API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
        self.url = "https://api.two.app/v2/chat/completions"  # Update API endpoint
        self.headers = {
            'User-Agent': LitAgent().random(),
            'Accept': 'application/json',  # Keep application/json for request, response is text/event-stream
            'Content-Type': 'application/json',
            'X-Session-Token': api_key,
            'Origin': 'https://chat.two.ai',
            'Referer': 'https://api.two.app/'
        }

        # Initialize curl_cffi Session
        self.session = Session()
        self.session.headers.update(self.headers)
        self.session.proxies = proxies

        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.temperature = temperature
        self.system_message = system_message

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

    @staticmethod
    def _twoai_extractor(chunk_json: Dict[str, Any]) -> Optional[str]:
        """Extracts content from TwoAI v2 stream JSON objects."""
        if not isinstance(chunk_json, dict) or "choices" not in chunk_json or not chunk_json["choices"]:
            return None

        delta = chunk_json["choices"][0].get("delta")
        if not isinstance(delta, dict):
            return None

        content = delta.get("content")
        return content if isinstance(content, str) else None

    def ask(
        self,
        prompt: str,
        stream: bool = True,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        online_search: bool = True,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(conversation_prompt if conversationally else prompt)
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        payload = {
            "messages": [
                *([{"role": "system", "content": self.system_message}] if self.system_message else []),
                {"role": "user", "content": conversation_prompt},
            ],
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens_to_sample,
            "extra_body": {
                "online_search": online_search,
            }
        }

        def for_stream():
            streaming_text = "" # Initialize outside try block
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    impersonate="chrome110"
                )

                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("error", {}).get("message", error_detail)
                    except json.JSONDecodeError:
                        pass
                    raise exceptions.FailedToGenerateResponseError(
                        f"Request failed with status code {response.status_code} - {error_detail}"
                    )

                # Use sanitize_stream for SSE processing
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value="data:",
                    to_json=True,     # Stream sends JSON
                    skip_markers=["[DONE]"],
                    content_extractor=self._twoai_extractor, # Use the specific extractor
                    yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                )

                for content_chunk in processed_stream:
                    # content_chunk is the string extracted by _twoai_extractor
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        resp = dict(text=content_chunk)
                        yield resp if not raw else content_chunk

                # If stream completes successfully, update history
                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)

            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except exceptions.FailedToGenerateResponseError:
                raise # Re-raise specific exception
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred during streaming ({type(e).__name__}): {e}") from e
            finally:
                # Ensure history is updated even if stream ends abruptly but text was received
                if streaming_text and not self.last_response: # Check if last_response wasn't set in the try block
                    self.last_response = {"text": streaming_text}
                    self.conversation.update_chat_history(prompt, streaming_text)


        def for_non_stream():
            # Non-stream still uses the stream internally and aggregates
            streaming_text = ""
            # We need to consume the generator from for_stream()
            gen = for_stream()
            try:
                for chunk_data in gen:
                    if isinstance(chunk_data, dict) and "text" in chunk_data:
                        streaming_text += chunk_data["text"]
                    elif isinstance(chunk_data, str): # Handle raw=True case
                        streaming_text += chunk_data
            except exceptions.FailedToGenerateResponseError:
                 # If the underlying stream fails, re-raise the error
                 raise
            # self.last_response and history are updated within for_stream's try/finally
            return self.last_response # Return the final aggregated dict

        effective_stream = stream if stream is not None else True
        return for_stream() if effective_stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = True,
        optimizer: str = None,
        conversationally: bool = False,
        online_search: bool = True,
    ) -> str:
        effective_stream = stream if stream is not None else True

        def for_stream_chat():
            # ask() yields dicts when raw=False (default for chat)
            gen = self.ask(
                prompt,
                stream=True,
                raw=False, # Ensure ask yields dicts
                optimizer=optimizer,
                conversationally=conversationally,
                online_search=online_search,
            )
            for response_dict in gen:
                yield self.get_message(response_dict) # get_message expects dict

        def for_non_stream_chat():
             # ask() returns a dict when stream=False
            response_dict = self.ask(
                prompt,
                stream=False, # Ensure ask returns dict
                raw=False,
                optimizer=optimizer,
                conversationally=conversationally,
                online_search=online_search,
            )
            return self.get_message(response_dict) # get_message expects dict

        return for_stream_chat() if effective_stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "") # Use .get for safety


if __name__ == "__main__":
    from rich import print
    import os

    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJzanl2OHJtZGxDZDFnQ2hQdGxzZHdxUlVteXkyIiwic291cmNlIjoiRmlyZWJhc2UiLCJpYXQiOjE3NDYxMDQ4NDIsImV4cCI6MTc0NjEwNTc0Mn0.EISWJOYVHxdcvYUbXd7jhD5ao3_yTKD0me8oGDcj5u8"

    try:
        ai = TwoAI(
            api_key=api_key,
            timeout=60,
            model="sutra-r0",
            system_message="You are an intelligent AI assistant. Be concise and helpful."
        )

        response_stream = ai.chat("write me a poem about AI", stream=True, online_search=True)
        full_stream_response = ""
        for chunk in response_stream:
            print(chunk, end="", flush=True)
            full_stream_response += chunk
        print("\n[bold green]Stream Test Complete.[/bold green]\n")

        # Optional: Test non-stream
        # print("[bold blue]Testing Non-Stream:[/bold blue]")
        # non_stream_response = ai.chat("What is the capital of France?", stream=False, online_search=False)
        # print(non_stream_response)
        # print("[bold green]Non-Stream Test Complete.[/bold green]\n")


    except exceptions.FailedToGenerateResponseError as e:
        print(f"\n[bold red]API Error:[/bold red] {e}")
    except ValueError as e:
        print(f"\n[bold red]Configuration Error:[/bold red] {e}")
    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")

