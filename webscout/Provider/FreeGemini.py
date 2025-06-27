#!/usr/bin/env python3
"""
FreeGemini API client for the free-gemini.vercel.app service.
Supports streaming responses from Gemini 2.0 Flash model.
"""

import json
from curl_cffi.requests import Session
from curl_cffi import CurlError
from typing import Dict, Generator, Any, Union, Optional

from webscout import exceptions
from webscout.AIutel import Optimizers, AwesomePrompts, sanitize_stream
from webscout.conversation import Conversation
from webscout.litagent import LitAgent
from webscout.AIbase import Provider


class FreeGemini(Provider):
    """
    A class to interact with the free-gemini.vercel.app API,
    which provides access to Gemini models.
    """
    AVAILABLE_MODELS = ["gemini-2.0-flash"]

    @staticmethod
    def _gemini_extractor(data: Dict) -> Optional[str]:
        """Extract text content from Gemini API response."""
        try:
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 4000,
        temperature: float = 0.5,
        top_p: float = 1.0,
        timeout: int = 120, # Default timeout for this specific API
        proxies: dict = {}, # Standard proxy support
        filepath: str = None,
        update_file: bool = True,
        history_offset: int = 10250,
        intro: str = None,
        act: str = None,
        model: str = "gemini-2.0-flash",
        system_prompt: str = "You are a helpful assistant.", # For consistency, though not directly used in payload
    ):
        """Initialize the FreeGemini client.

        Args:
            is_conversation (bool): Enable conversation history. Defaults to True.
            max_tokens (int): Maximum tokens to sample. Defaults to 4000.
            temperature (float): Sampling temperature. Defaults to 0.5.
            top_p (float): Nucleus sampling parameter. Defaults to 1.0.
            timeout (int): Request timeout in seconds. Defaults to 120.
            proxies (dict): HTTP proxies. Defaults to {}.
            filepath (str, optional): Path to save conversation history. Defaults to None.
            update_file (bool): Update conversation history file. Defaults to True.
            history_offset (int): Limit conversation history. Defaults to 10250.
            intro (str, optional): Introduction for the conversation.
            act (str, optional): Act for AwesomePrompts.
            model (str): Model to use. Defaults to "gemini-2.0-flash".
            system_prompt (str): System prompt (primarily for API consistency).
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.session = Session()
        self.model = model
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens # Consistent naming
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.last_response = {}
        self.system_prompt = system_prompt # Stored for consistency

        self.api_endpoint = "https://free-gemini.vercel.app/api/google/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"

        self.agent = LitAgent()
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": self.agent.random(),
            "Origin": "https://free-gemini.vercel.app",
            "Referer": "https://free-gemini.vercel.app/",
        }

        self.session.headers.update(self.headers)
        self.session.proxies = proxies

        self.__available_optimizers = (
            method for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        Conversation.intro = (
            AwesomePrompts().get_act(act, raise_not_found=True, default=None, case_insensitive=True)
            if act else intro or Conversation.intro
        )
        self.conversation = Conversation(
            is_conversation, self.max_tokens_to_sample, filepath, update_file
        )
        self.conversation.history_offset = history_offset
    
    def ask(
        self,
        prompt: str,
        stream: bool = False, # Default to False for consistency
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        """Sends a prompt to the FreeGemini API and returns the response.

        Args:
            prompt (str): The prompt to send to the model.
            stream (bool): Whether to stream the response. Defaults to False.
            raw (bool): Return raw response instead of parsed text. Defaults to False.
            optimizer (str, optional): Optimizer to use for the prompt.
            conversationally (bool, optional): Whether to apply optimizer conversationally.

        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: 
                The generated response as a dictionary or generator.
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        payload = {
            "contents": [{"role": "user", "parts": [{"text": conversation_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens_to_sample,
                "topP": self.top_p
            },
            "safetySettings": [
                # Default safety settings from original class
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
            ]
        }

        # Internal generator for handling API call and history update
        def _generate_content_and_update_history():
            streaming_text_accumulator = ""
            try:
                response = self.session.post(
                    self.api_endpoint,
                    json=payload,
                    stream=True, # API always streams
                    timeout=self.timeout,
                    impersonate="chrome120" 
                )
                response.raise_for_status()

                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value="data:",
                    to_json=True,
                    content_extractor=self._gemini_extractor,
                    yield_raw_on_error=False
                )
                
                for content_chunk_str in processed_stream: # yields string
                    if content_chunk_str and isinstance(content_chunk_str, str):
                        streaming_text_accumulator += content_chunk_str
                        yield content_chunk_str # Yield the raw text chunk
            
            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {str(e)}") from e
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {str(e)}") from e
            finally:
                if streaming_text_accumulator:
                    self.last_response = {"text": streaming_text_accumulator}
                    self.conversation.update_chat_history(prompt, streaming_text_accumulator)

        if stream:
            def stream_wrapper():
                for text_chunk in _generate_content_and_update_history():
                    yield {"text": text_chunk} if not raw else text_chunk
            return stream_wrapper()
        else: # Not streaming from the perspective of the caller of `ask`
            full_text_response = ""
            for text_chunk in _generate_content_and_update_history():
                full_text_response += text_chunk
            
            # self.last_response and history are updated by the generator's `finally`
            return {"text": full_text_response} if not raw else full_text_response

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generates a response from the FreeGemini API.

        Args:
            prompt (str): The prompt to send to the API.
            stream (bool): Whether to stream the response.
            optimizer (str): Optimizer to use for the prompt.
            conversationally (bool): Whether to generate the prompt conversationally.

        Returns:
            Union[str, Generator[str, None, None]]: The API response.
        """
        def for_stream_chat():
            gen = self.ask(
                prompt, stream=True, raw=False, # Ensure ask yields dicts
                optimizer=optimizer, conversationally=conversationally
            )
            for response_dict in gen:
                yield self.get_message(response_dict)

        def for_non_stream_chat():
            response_data = self.ask(
                prompt, stream=False, raw=False, # Ensure ask returns dict
                optimizer=optimizer, conversationally=conversationally
            )
            return self.get_message(response_data)

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "")

if __name__ == "__main__":
    # Example usage
    free_gemini = FreeGemini()
    response = free_gemini.chat("how many r in strawberry", stream=False)
    print(response)  # Should print the response from the API