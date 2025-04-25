from curl_cffi.requests import Session
from curl_cffi import CurlError
import json

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from typing import Union, Any, AsyncGenerator, Dict
from webscout.litagent import LitAgent

class TurboSeek(Provider):
    """
    This class provides methods for interacting with the TurboSeek API.
    """
    AVAILABLE_MODELS = ["Llama 3.1 70B"]

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
        model: str = "Llama 3.1 70B" # Note: model parameter is not used by the API endpoint
    ):
        """Instantiates TurboSeek

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
        self.chat_endpoint = "https://www.turboseek.io/api/getAnswer"
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://www.turboseek.io",
            "priority": "u=1, i",
            "referer": "https://www.turboseek.io/?ref=taaft&utm_source=taaft&utm_medium=referral",
            "sec-ch-ua": '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": LitAgent().random(),
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

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> dict:
        """Chat with AI

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

        payload = {
            "question": conversation_prompt,
            "sources": []
        }

        def for_stream():
            try: # Add try block for CurlError
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.chat_endpoint, 
                    json=payload, 
                    stream=True, 
                    timeout=self.timeout,
                    impersonate="chrome120", # Try a different impersonation profile
                )
                if not response.ok:
                    raise exceptions.FailedToGenerateResponseError(
                        f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                    )
                streaming_text = ""
                # Iterate over bytes and decode manually
                for value_bytes in response.iter_lines():
                    try:
                        if value_bytes and value_bytes.startswith(b"data: "): # Check for bytes
                            # Decode bytes to string
                            line = value_bytes[6:].decode('utf-8') 
                            data = json.loads(line) 
                            if "text" in data:
                                # Decode potential unicode escapes
                                content = data["text"].encode().decode('unicode_escape') 
                                streaming_text += content
                                resp = dict(text=content)
                                self.last_response.update(resp) # Update last_response incrementally
                                # Yield raw bytes or dict based on flag
                                yield value_bytes if raw else resp 
                    except (json.decoder.JSONDecodeError, UnicodeDecodeError):
                        pass # Ignore lines that are not valid JSON or cannot be decoded
                # Update conversation history after stream finishes
                if streaming_text: # Only update if content was received
                    self.conversation.update_chat_history(
                        prompt, streaming_text # Use the fully aggregated text
                    )
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e: # Catch other potential exceptions
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}")


        def for_non_stream():
            # Aggregate the stream using the updated for_stream logic
            full_text = ""
            for chunk_data in for_stream():
                 # Ensure chunk_data is a dict (not raw) and has 'text'
                if isinstance(chunk_data, dict) and "text" in chunk_data:
                    full_text += chunk_data["text"]
                # If raw=True, chunk_data is bytes, decode and process if needed (though raw non-stream is less common)
                elif isinstance(chunk_data, bytes):
                     try:
                         if chunk_data.startswith(b"data: "):
                             line = chunk_data[6:].decode('utf-8')
                             data = json.loads(line)
                             if "text" in data:
                                 content = data["text"].encode().decode('unicode_escape')
                                 full_text += content
                     except (json.decoder.JSONDecodeError, UnicodeDecodeError):
                         pass
            # last_response and history are updated within for_stream
            # Ensure last_response reflects the complete aggregated text
            self.last_response = {"text": full_text} 
            return self.last_response

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> str:
        """Generate response `str`
        Args:
            prompt (str): Prompt to be send.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            optimizer (str, optional): Prompt optimizer name - `[code, shell_command]`. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.
        Returns:
            str: Response generated
        """

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
        """Retrieves message only from response

        Args:
            response (dict): Response generated by `self.ask`

        Returns:
            str: Message extracted
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        # Text is already decoded in ask method
        return response.get("text", "") 

if __name__ == '__main__':
    # Ensure curl_cffi is installed
    from rich import print
    try: # Add try-except block for testing
        ai = TurboSeek(timeout=60)
        print("[bold blue]Testing Stream:[/bold blue]")
        response_stream = ai.chat("hello buddy", stream=True)
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
    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")

