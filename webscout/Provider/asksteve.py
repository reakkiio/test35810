from typing import Any, Dict, Optional, Union
from curl_cffi import CurlError
from curl_cffi.requests import Session
from webscout import exceptions
from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream # Import sanitize_stream
from webscout.AIbase import Provider 
from webscout.litagent import LitAgent

class AskSteve(Provider):
    """
    A class to interact with the AskSteve API.
    """
    AVAILABLE_MODELS = ["Gemini"]
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
    ) -> None:
        """Instantiates AskSteve

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
            system_prompt (str, optional): System prompt for AskSteve. Defaults to the provided string.
        """
        self.session = Session() # Use curl_cffi Session
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://quickstart.asksteve.to/quickStartRequest"
        self.timeout = timeout
        self.last_response = {}
        self.headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "text/plain;charset=UTF-8",
            "origin": "chrome-extension://gldebcpkoojijledacjeboaehblhfbjg",
            "priority": "u=1, i",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "none",
            "sec-fetch-storage-access": "active",
            "user-agent": LitAgent().random(),
        }

        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        self.session.headers.update(self.headers)
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
        self.session.proxies = proxies # Assign proxies directly
    @staticmethod
    def _asksteve_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from AskSteve JSON response."""
        if isinstance(chunk, dict) and "candidates" in chunk and len(chunk["candidates"]) > 0:
            parts = chunk["candidates"][0].get("content", {}).get("parts", [])
            if parts and isinstance(parts[0].get("text"), str):
                return parts[0]["text"]
        return None

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
            "key": "asksteve",
            "prompt": conversation_prompt
        }


        # This API doesn't stream, so we process the full response
        try:
            response = self.session.post(
                self.api_endpoint,
                headers=self.headers,
                json=payload,
                stream=False, # API doesn't stream
                timeout=self.timeout,
                impersonate="chrome120" # Add impersonate
            )
            response.raise_for_status()
            response_text_raw = response.text # Get raw text

            # Process the full JSON text using sanitize_stream
            processed_stream = sanitize_stream(
                data=response_text_raw,
                to_json=True, # Parse the whole text as JSON
                intro_value=None,
                content_extractor=self._asksteve_extractor, # Use the specific extractor
                yield_raw_on_error=False
            )
            # Extract the single result
            text = next(processed_stream, None)
            text = text if isinstance(text, str) else "" # Ensure it's a string

            self.last_response.update(dict(text=text))
            self.conversation.update_chat_history(
                prompt, self.get_message(self.last_response)
            )
            # Always return a dict for consistency
            return {"text": text} if raw else self.last_response

        except CurlError as e:
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
        except Exception as e: # Catch other potential errors
            raise exceptions.FailedToGenerateResponseError(f"Failed to get response ({type(e).__name__}): {e}") from e

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

        response_data = self.ask(
            prompt,
            stream=False, # Always False for this API
            raw=False,    # Get the dict back
            optimizer=optimizer,
            conversationally=conversationally,
        )
        if stream:
            def stream_wrapper():
                yield self.get_message(response_data)
            return stream_wrapper()
        else:
            return self.get_message(response_data)

    def get_message(self, response) -> str:
        """Retrieves message only from response

        Args:
            response (dict or str): Response generated by `self.ask` or a string

        Returns:
            str: Message extracted
        """
        if isinstance(response, dict):
            return response.get("text", "") # Use .get for safety
        elif isinstance(response, str):
            return response
        else:
            raise TypeError(f"Unsupported response type: {type(response)}")


if __name__ == "__main__":
    from rich import print
    ai = AskSteve()
    response = ai.chat("write a short poem about AI", stream=True)
    for chunk in response:
        print(chunk, end="", flush=True) 