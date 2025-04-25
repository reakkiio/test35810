from curl_cffi.requests import Session
from curl_cffi import CurlError
import os
from typing import Union, List, Optional
from string import punctuation
from random import choice
import json
from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class TutorAI(Provider):
    """
    A class to interact with the TutorAI.me API.
    """
    AVAILABLE_MODELS = ["gpt-4o"]

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
    ):
        """
        Initializes the TutorAI.me API with given parameters.

        Args:
            is_conversation (bool, optional): Flag for chatting conversationally. Defaults to True.
            max_tokens (int, optional): Maximum number of tokens to be generated upon completion. Defaults to 1024.
            timeout (int, optional): Http request timeout. Defaults to 30.
            intro (str, optional): Conversation introductory prompt. Defaults to None.
            filepath (str, optional): Path to file containing conversation history. Defaults to None.
            update_file (bool, optional): Add new prompts and responses to the file. Defaults to True.
            proxies (dict, optional): Http request proxies. Defaults to {}.
            history_offset (int, optional): Limit conversation history to this number of last texts. Defaults to 10250.
            act (str|int, optional): Awesome prompt key or index. (Used as intro). Defaults to None.
            system_prompt (str, optional): System prompt for TutorAI.
                                   Defaults to "You are a helpful AI assistant.".
        """
        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://ai-tutor.ai/api/generate-homeworkify-response"
        self.stream_chunk_size = 1024
        self.timeout = timeout
        self.last_response = {}
        # Remove Cookie header, curl_cffi doesn't use it directly like this
        self.headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8",
            "DNT": "1",
            "Origin": "https://tutorai.me",
            "Priority": "u=1, i",
            "Referer": "https://tutorai.me/homeworkify?ref=taaft&utm_source=taaft&utm_medium=referral",
            "Sec-Ch-Ua": '"Microsoft Edge";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": LitAgent().random()
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
        stream: bool = False, # Note: API doesn't seem to truly stream text chunks
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        attachment_path: Optional[str] = None
    ) -> dict:
        """Chat with TutorAI

        Args:
            prompt (str): Prompt to be send.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            raw (bool, optional): Stream back raw response as received. Defaults to False.
            optimizer (str, optional): Prompt optimizer name - `[code, shell_command]`. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.
            attachment_path (str, optional): Path to attachment file. Defaults to None.

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
                conversation_prompt = getattr(Optimizers, optimizer)(conversation_prompt if conversationally else prompt)
            else:
                raise Exception(
                    f"Optimizer is not one of {self.__available_optimizers}"
                )

        form_data = {
            "inputMessage": conversation_prompt,
            "attachmentsCount": "1" if attachment_path else "0"
        }
        files = {}
        file_handle = None # To ensure file is closed
        if attachment_path:
            if not os.path.isfile(attachment_path):
                raise FileNotFoundError(f"Error: The file '{attachment_path}' does not exist.")
            try:
                # Open file handle to pass to curl_cffi
                file_handle = open(attachment_path, 'rb')
                files["attachment0"] = (os.path.basename(attachment_path), file_handle, 'image/png') # Adjust mime type if needed
            except Exception as e:
                if file_handle: file_handle.close() # Close if opened
                raise exceptions.FailedToGenerateResponseError(f"Error opening the file: {e}")

        # The API doesn't seem to support streaming text chunks based on the original code.
        # Both stream=True and stream=False resulted in processing the full response.
        # We will implement the non-stream logic for both cases.
        try:
            # Use curl_cffi session post with impersonate
            # Pass data and files for multipart/form-data
            response = self.session.post(
                self.api_endpoint, 
                # headers are set on the session
                data=form_data, 
                files=files, 
                timeout=self.timeout,
                impersonate="chrome120", # Try a different impersonation profile
            )
            response.raise_for_status() # Check for HTTP errors

            try:
                response_data = response.json()
            except json.JSONDecodeError as json_err:
                raise exceptions.FailedToGenerateResponseError(f"Error decoding JSON: {json_err} - Response text: {response.text}")

            homeworkify_html = response_data.get("homeworkifyResponse", "")
            if not homeworkify_html:
                 # Return empty if no content, consistent with original non-stream logic
                clean_text = ""
            else:
                # Assuming the response is HTML that needs cleaning/parsing
                # For now, just return the raw HTML content as text
                clean_text = homeworkify_html 

            self.last_response = {"text": clean_text}
            self.conversation.update_chat_history(prompt, clean_text)
            return self.last_response # Return the full response content

        except CurlError as e: # Catch CurlError
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
        except Exception as e: # Catch other potential exceptions
            # Include response text if available in HTTP errors
            err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
            raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e} - {err_text}")
        finally:
            if file_handle: # Ensure file is closed
                file_handle.close()


    def chat(
        self,
        prompt: str,
        stream: bool = False, # Keep stream param for interface consistency, though API might not support it
        optimizer: str = None,
        conversationally: bool = False,
        attachment_path: Optional[str] = None,
    ) -> str:
        """Generate response `str`
        Args:
            prompt (str): Prompt to be send.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            optimizer (str, optional): Prompt optimizer name - `[code, shell_command]`. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.
            attachment_path (str, optional): Path to attachment file. Defaults to None.
        Returns:
            str: Response generated
        """

        def for_stream():
            for response in self.ask(
                prompt, True, optimizer=optimizer, conversationally=conversationally, attachment_path=attachment_path,
            ):
                yield self.get_message(response)

        def for_non_stream():
            for response in self.ask(
                prompt, False, optimizer=optimizer, conversationally=conversationally, attachment_path=attachment_path,
            ):
                yield self.get_message(response)

        return for_stream() if stream else for_non_stream()

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
    from rich import print

    try: # Add try-except block for testing
        ai = TutorAI(timeout=120) # Increased timeout for potential uploads
        # Test without attachment first
        print("[bold blue]Testing Text Prompt:[/bold blue]")
        response_gen = ai.chat("hello buddy", stream=True) # Test stream interface
        full_response = ""
        for chunk in response_gen:
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n[bold green]Text Test Complete.[/bold green]\n")

        # Optional: Test with attachment (replace with a valid image path)
        # attachment_file = "path/to/your/image.png" 
        # if os.path.exists(attachment_file):
        #     print(f"[bold blue]Testing with Attachment ({attachment_file}):[/bold blue]")
        #     response_gen_attach = ai.chat("Describe this image", stream=True, attachment_path=attachment_file)
        #     full_response_attach = ""
        #     for chunk in response_gen_attach:
        #         print(chunk, end="", flush=True)
        #         full_response_attach += chunk
        #     print("\n[bold green]Attachment Test Complete.[/bold green]")
        # else:
        #      print(f"[bold yellow]Skipping attachment test: File not found at {attachment_file}[/bold yellow]")

    except exceptions.FailedToGenerateResponseError as e:
        print(f"\n[bold red]API Error:[/bold red] {e}")
    except FileNotFoundError as e:
         print(f"\n[bold red]File Error:[/bold red] {e}")
    except Exception as e:
        print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")
