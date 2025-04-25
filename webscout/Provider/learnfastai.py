import os
import json
from typing import Optional, Union, Generator
import uuid
from curl_cffi.requests import Session
from curl_cffi import CurlError

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions


class LearnFast(Provider):
    """
    A class to interact with the LearnFast.ai API.
    """

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 600, # Note: max_tokens is not used by this API
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        system_prompt: str = "You are a helpful AI assistant.", # Note: system_prompt is not used by this API
    ):
        """
        Initializes the LearnFast.ai API with given parameters.
        """
        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = 'https://autosite.erweima.ai/api/v1/chat'
        self.stream_chunk_size = 64
        self.timeout = timeout
        self.last_response = {}
        self.system_prompt = system_prompt
        self.headers = {
            "authority": "autosite.erweima.ai",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
            "authorization": "",  # Always empty
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://learnfast.ai",
            "priority": "u=1, i", # Keep priority header if needed
            "referer": "https://learnfast.ai/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            # uniqueid will be added dynamically in ask()
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

    def generate_unique_id(self) -> str:
        """Generate a 32-character hexadecimal unique ID."""
        return uuid.uuid4().hex

    def generate_session_id(self) -> str:
        """Generate a 32-character hexadecimal session ID."""
        return uuid.uuid4().hex

    def upload_image_to_0x0(self, image_path: str) -> str:
        """
        Uploads an image to 0x0.st and returns the public URL.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"The file '{image_path}' does not exist.")

        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            try:
                response = self.session.post(
                    "https://0x0.st", 
                    files=files,
                    # Add impersonate if using the main session
                    impersonate="chrome110" 
                )
                response.raise_for_status()
                image_url = response.text.strip()
                if not image_url.startswith("http"):
                    raise ValueError("Received an invalid URL from 0x0.st.")
                return image_url
            except CurlError as e: # Catch CurlError
                raise Exception(f"Failed to upload image to 0x0.st (CurlError): {e}") from e
            except Exception as e: # Catch other potential errors
                 raise Exception(f"Failed to upload image to 0x0.st: {e}") from e

    def create_payload(
        self,
        session_id: str,
        conversation_prompt: str,
        image_url: Optional[str] = None
    ) -> dict:
        """
        Creates the JSON payload for the request.
        """
        payload = {
            "prompt": conversation_prompt,
            "firstQuestionFlag": True,
            "sessionId": session_id,
            "attachments": []
        }
        if image_url:
            payload["attachments"] = [
                {
                    "fileType": "image/jpeg",
                    "file": {},
                    "fileContent": image_url
                }
            ]
        return payload

    def ask(
        self,
        prompt: str,
        stream: bool = False, # API supports streaming
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        image_path: Optional[str] = None,
    ) -> Union[dict, Generator[dict, None, None]]:
        """Chat with LearnFast

        Args:
            prompt (str): Prompt to be send.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            raw (bool, optional): Stream back raw response as received. Defaults to False.
            optimizer (str, optional): Prompt optimizer name - `[code, shell_command]`. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.
            image_path (Optional[str], optional): Path to the image to be uploaded.
                                                 Defaults to None.

        Returns:
           Union[dict, Generator[dict, None, None]]: Response generated
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

        # Generate unique ID and session ID
        unique_id = self.generate_unique_id()
        session_id = self.generate_session_id()

        # Update headers with the unique ID for this request
        current_headers = self.headers.copy()
        current_headers["uniqueid"] = unique_id

        # Upload image and get URL if image_path is provided
        image_url = None
        if image_path:
            try:
                image_url = self.upload_image_to_0x0(image_path)
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Error uploading image: {e}") from e

        # Create the payload
        payload = self.create_payload(session_id, conversation_prompt, image_url)

        # Convert the payload to a JSON string
        data = json.dumps(payload)

        def for_stream():
            full_response = "" # Initialize outside try block
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.api_endpoint, 
                    headers=current_headers, # Use headers with uniqueid
                    data=data, 
                    stream=True, 
                    timeout=self.timeout,
                    # proxies are set on the session
                    impersonate="chrome110" # Use a common impersonation profile
                )
                response.raise_for_status()  # Check for HTTP errors

                # Process the streamed response
                # Iterate over bytes and decode manually
                for line_bytes in response.iter_lines():
                    if line_bytes:
                        try:
                            line = line_bytes.decode('utf-8').strip()
                            if line == "[DONE]":
                                break
                            json_response = json.loads(line)
                            if json_response.get('code') == 200 and json_response.get('data'):
                                message = json_response['data'].get('message', '')
                                if message:
                                    full_response += message
                                    resp = {"text": message}
                                    # Yield dict or raw string chunk
                                    yield resp if not raw else message
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass # Ignore lines that are not valid JSON or cannot be decoded
                
                # Update history after stream finishes
                self.last_response = {"text": full_response}
                self.conversation.update_chat_history(prompt, full_response)

            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"An error occurred (CurlError): {e}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"An error occurred ({type(e).__name__}): {e} - {err_text}") from e

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
        image_path: Optional[str] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response `str`
        Args:
            prompt (str): Prompt to be send.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            optimizer (str, optional): Prompt optimizer name - `[code, shell_command]`. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.
            image_path (Optional[str], optional): Path to the image to be uploaded.
                                                 Defaults to None.
        Returns:
            Union[str, Generator[str, None, None]]: Response generated
        """
        try:
            # ask() yields dicts or strings when streaming
            response_gen = self.ask(
                prompt, stream=stream, raw=False, # Ensure ask yields dicts/dict
                optimizer=optimizer, conversationally=conversationally, 
                image_path=image_path
            )
            if stream:
                def stream_wrapper():
                    for chunk_dict in response_gen:
                        yield self.get_message(chunk_dict) # get_message expects dict
                return stream_wrapper()
            else:
                # response_gen is the final dict in non-stream mode
                return self.get_message(response_gen) # get_message expects dict
        except Exception as e:
            # Return error message directly, consider raising instead for better error handling upstream
            return f"Error: {str(e)}" 

    def get_message(self, response: dict) -> str:
        """Retrieves message only from response

        Args:
            response (dict): Response generated by `self.ask`

        Returns:
            str: Message extracted
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    from rich import print
    ai = LearnFast()
    response = ai.chat(input(">>> "), stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)