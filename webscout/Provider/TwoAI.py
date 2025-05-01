from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent
class TwoAI(Provider):
    """
    A class to interact with the Two AI API with LitAgent user-agent.
    """

    AVAILABLE_MODELS = [
        "sutra-light",
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
        model: str = "sutra-light",
        temperature: float = 0.6,
        system_message: str = "You are a helpful assistant."
    ):
        """Initializes the TwoAI API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
        self.url = "https://api.two.app/v1/sutra-light/completion"
        self.headers = {
            'User-Agent': LitAgent().random(),
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Session-Token': api_key,
            'Origin': 'https://chat.two.ai',
            'Referer': 'https://api.two.app/'
        }
        
        # Initialize curl_cffi Session
        self.session = Session()
        # Update curl_cffi session headers and proxies
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
    def _twoai_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from TwoAI stream JSON objects."""
        if isinstance(chunk, dict) and chunk.get("typeName") == "LLMChunk":
            return chunk.get("content")
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = True,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        online_search: bool = True,
        reasoning_on: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(conversation_prompt if conversationally else prompt)
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        # Payload construction
        payload = {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": conversation_prompt},
            ],
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens_to_sample,
            "reasoningOn": reasoning_on,
            "onlineSearch": online_search
        }

        def for_stream():
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.url, 
                    json=payload, 
                    stream=True, 
                    timeout=self.timeout,
                    impersonate="chrome110" # Add impersonate
                )
                
                if response.status_code != 200:
                    raise exceptions.FailedToGenerateResponseError(
                        f"Request failed with status code {response.status_code} - {response.text}"
                    )
                    
                streaming_text = ""
                # Use sanitize_stream with the custom extractor
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value=None, # No simple prefix
                    to_json=True,     # Each line is JSON
                    content_extractor=self._twoai_extractor, # Use the specific extractor
                    yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                )

                for content_chunk in processed_stream:
                    # content_chunk is the string extracted by _twoai_extractor
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        yield dict(text=content_chunk) if not raw else content_chunk
                
                # Update history and last response after stream finishes
                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)
                    
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e: # Catch other potential exceptions
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}")

        def for_non_stream():
            # Non-stream requests might not work the same way if the API expects streaming.
            # This implementation aggregates the stream.
            streaming_text = ""
            # Iterate through the generator provided by for_stream
            for chunk_data in for_stream():
                # Check if chunk_data is a dict (not raw) and has 'text'
                if isinstance(chunk_data, dict) and "text" in chunk_data:
                    streaming_text += chunk_data["text"]
                # If raw=True, chunk_data is the string content itself
                elif isinstance(chunk_data, str):
                     streaming_text += chunk_data
            # last_response and history are updated within for_stream
            return self.last_response # Return the final aggregated response

        # Ensure stream defaults to True if not provided, matching original behavior
        effective_stream = stream if stream is not None else True
        return for_stream() if effective_stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = True,
        optimizer: str = None,
        conversationally: bool = False,
        online_search: bool = True,
        reasoning_on: bool = False,
    ) -> str:
        # Ensure stream defaults to True if not provided
        effective_stream = stream if stream is not None else True

        def for_stream_chat():
            # ask() yields dicts when raw=False
            for response_dict in self.ask(
                prompt, 
                stream=True, 
                raw=False, # Ensure ask yields dicts
                optimizer=optimizer, 
                conversationally=conversationally,
                online_search=online_search,
                reasoning_on=reasoning_on
            ):
                yield self.get_message(response_dict)
        
        def for_non_stream_chat():
            # ask() returns a dict when stream=False
            response_dict = self.ask(
                prompt, 
                stream=False, 
                optimizer=optimizer, 
                conversationally=conversationally,
                online_search=online_search,
                reasoning_on=reasoning_on
            )
            return self.get_message(response_dict)
        
        return for_stream_chat() if effective_stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    from rich import print

    api_key = "" # Add your API key here or load from env
    
    try: # Add try-except block for testing
        ai = TwoAI(
            api_key=api_key,
            timeout=60,
            system_message="You are an intelligent AI assistant. Be concise and helpful."
        )
        
        print("[bold blue]Testing Stream:[/bold blue]")
        response_stream = ai.chat("666+444=?", stream=True, reasoning_on=True)
        full_stream_response = ""
        for chunk in response_stream:
            print(chunk, end="", flush=True)
            full_stream_response += chunk
        print("\n[bold green]Stream Test Complete.[/bold green]\n")

        # Optional: Test non-stream
        # print("[bold blue]Testing Non-Stream:[/bold blue]")
        # response_non_stream = ai.chat("What is the capital of France?", stream=False)
        # print(response_non_stream)
        # print("[bold green]Non-Stream Test Complete.[/bold green]")

    except exceptions.FailedToGenerateResponseError as e:
        print(f"\n[bold red]API Error:[/bold red] {e}")
    except ValueError as e:
         print(f"\n[bold red]Configuration Error:[/bold red] {e}")
    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")
