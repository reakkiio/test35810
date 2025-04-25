from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import re
from typing import Union, Any, Dict, Optional, Generator

from webscout.AIutel import Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class JadveOpenAI(Provider):
    """
    A class to interact with the OpenAI API through jadve.com using the streaming endpoint.
    """

    AVAILABLE_MODELS = ["gpt-4o-mini"]

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
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful AI assistant." # Note: system_prompt is not used by this API
    ):
        """
        Initializes the JadveOpenAI client.

        Args:
            is_conversation (bool, optional): Enable conversational mode. Defaults to True.
            max_tokens (int, optional): Maximum tokens for generation. Defaults to 600.
            timeout (int, optional): HTTP request timeout in seconds. Defaults to 30.
            intro (str, optional): Introductory prompt text. Defaults to None.
            filepath (str, optional): Path to conversation history file. Defaults to None.
            update_file (bool, optional): Whether to update the conversation history file. Defaults to True.
            proxies (dict, optional): Proxies for HTTP requests. Defaults to {}.
            history_offset (int, optional): Limit for conversation history. Defaults to 10250.
            act (str|int, optional): Act key for AwesomePrompts. Defaults to None.
            model (str, optional): AI model to be used. Defaults to "gpt-4o-mini".
            system_prompt (str, optional): System prompt text. Defaults to "You are a helpful AI assistant."
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://openai.jadve.com/stream"
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt

        # Headers for API requests
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://jadve.com",
            "priority": "u=1, i", # Keep priority header if needed
            "referer": "https://jadve.com/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "x-authorization": "Bearer" # Keep custom headers
        }
        
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly

        self.__available_optimizers = (
            method for method in dir(Optimizers)
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
    ) -> Union[dict, Generator[dict, None, None]]:
        """
        Chat with AI.

        Args:
            prompt (str): Prompt to be sent.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            raw (bool, optional): Return raw content chunks. Defaults to False.
            optimizer (str, optional): Prompt optimizer name. Defaults to None.
            conversationally (bool, optional): Flag for conversational optimization. Defaults to False.
        Returns:
            dict or generator: A dictionary with the generated text or a generator yielding text chunks.
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(conversation_prompt if conversationally else prompt)
            else:
                raise Exception(f"Optimizer is not one of {list(self.__available_optimizers)}")

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "text", "text": conversation_prompt}]}
            ],
            "model": self.model,
            "botId": "",
            "chatId": "",
            "stream": True, # API endpoint suggests streaming is default/required
            "temperature": 0.7,
            "returnTokensUsage": True,
            "useTools": False
        }

        def for_stream():
            full_response_text = "" # Initialize outside try block
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint, 
                    # headers are set on the session
                    json=payload, 
                    stream=True, 
                    timeout=self.timeout,
                    # proxies are set on the session
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status() # Check for HTTP errors

                # Pattern to match the streaming chunks format: 0:"text"
                pattern = r'0:"(.*?)"'
                buffer = ""
                
                # Iterate over bytes and decode manually
                for line_bytes in response.iter_lines():
                    if not line_bytes:
                        continue
                    
                    try:
                        line = line_bytes.decode('utf-8')
                        buffer += line
                        
                        # Try to match chunks in the current buffer
                        matches = re.findall(pattern, buffer)
                        if matches:
                            for chunk in matches:
                                # Handle potential escape sequences like \\n
                                decoded_chunk = chunk.encode().decode('unicode_escape')
                                full_response_text += decoded_chunk
                                resp = {"text": decoded_chunk}
                                # Yield dict or raw string chunk
                                yield resp if not raw else decoded_chunk
                            
                            # Remove matched parts from the buffer
                            # Be careful with buffer modification during iteration if issues arise
                            matched_parts = [f'0:"{match}"' for match in matches]
                            for part in matched_parts:
                                buffer = buffer.replace(part, '', 1)
                        
                        # Check if we've reached the end of the response
                        if 'e:' in line or 'd:' in line:
                            break
                    except UnicodeDecodeError:
                        continue # Ignore decoding errors for specific lines

                # Update history after stream finishes
                self.last_response = {"text": full_response_text}
                self.conversation.update_chat_history(prompt, full_response_text)

            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"Failed to generate response ({type(e).__name__}): {e} - {err_text}") from e


        def for_non_stream():
            # Aggregate the stream using the updated for_stream logic
            collected_text = ""
            try:
                # Ensure raw=False so for_stream yields dicts
                for chunk_data in for_stream():
                    if isinstance(chunk_data, dict) and "text" in chunk_data:
                        collected_text += chunk_data["text"]
                    # Handle raw string case if raw=True was passed
                    elif raw and isinstance(chunk_data, str):
                         collected_text += chunk_data
            except Exception as e:
                 # If aggregation fails but some text was received, use it. Otherwise, re-raise.
                 if not collected_text:
                     raise exceptions.FailedToGenerateResponseError(f"Failed to get non-stream response: {str(e)}") from e

            # last_response and history are updated within for_stream
            # Return the final aggregated response dict or raw string
            return collected_text if raw else self.last_response


        # Since the API endpoint suggests streaming, always call the stream generator.
        # The non-stream wrapper will handle aggregation if stream=False.
        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a chat response (string). 

        Args:
            prompt (str): Prompt to be sent.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            optimizer (str, optional): Prompt optimizer name. Defaults to None.
            conversationally (bool, optional): Flag for conversational optimization. Defaults to False.
        Returns:
            str or generator: Generated response string or generator yielding response chunks.
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
        Retrieves message from the response.

        Args:
            response (dict): Response from the ask() method.
        Returns:
            str: Extracted text.
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in JadveOpenAI.AVAILABLE_MODELS:
        try:
            test_ai = JadveOpenAI(model=model, timeout=60)
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