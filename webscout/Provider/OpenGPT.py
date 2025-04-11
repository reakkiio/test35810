import requests
import json
from typing import Dict, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class OpenGPT(Provider):
    """
    A class to interact with the Open-GPT API.
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
        app_id: str = "clf3yg8730000ih08ndbdi2v4",
    ):
        """Initializes the OpenGPT API client.
        
        Args:
            is_conversation (bool, optional): Whether to maintain conversation history. Defaults to True.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 600.
            timeout (int, optional): Http request timeout. Defaults to 30.
            intro (str, optional): Conversation introductory prompt. Defaults to None.
            filepath (str, optional): Path to file containing conversation history. Defaults to None.
            update_file (bool, optional): Add new prompts and responses to the file. Defaults to True.
            proxies (dict, optional): Http request proxies. Defaults to {}.
            history_offset (int, optional): Limit conversation history to this number of last texts. Defaults to 10250.
            act (str, optional): Awesome prompt key or index. (Used as intro). Defaults to None.
            app_id (str, optional): The OpenGPT application ID. Defaults to "clf3yg8730000ih08ndbdi2v4".
        """
        self.session = requests.Session()
        self.agent = LitAgent()
        
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.app_id = app_id
        
        # Set up headers with dynamic user agent
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": self.agent.random(),
            "Referer": f"https://open-gpt.app/id/app/{app_id}"
        }
        
        self.session.headers.update(self.headers)
        self.session.proxies.update(proxies)
        
        # Initialize optimizers
        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        
        # Setup conversation
        Conversation.intro = (
            AwesomePrompts().get_act(
                act, raise_not_found=True, default=None, case_insensitive=True
            ) if act else intro or Conversation.intro
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
    ) -> Union[Dict, Generator]:
        """
        Send a prompt to the OpenGPT API and get a response.
        
        Args:
            prompt: The user input/prompt for the API.
            stream: Whether to stream the response.
            raw: Whether to return the raw API response.
            optimizer: Optimizer to use on the prompt.
            conversationally: Whether to apply the optimizer on the full conversation prompt.
            
        Returns:
            A dictionary or generator with the response.
        """
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")
                
        # Prepare the request body payload
        payload = {
            "userInput": conversation_prompt,
            "id": self.app_id,
            "userKey": ""  # Assuming userKey is meant to be empty as in the original code
        }
        
        def for_non_stream():
            try:
                response = self.session.post(
                    "https://open-gpt.app/api/generate",
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                
                # Raise an exception for bad status codes
                response.raise_for_status()
                
                response_text = response.text
                self.last_response = {"text": response_text}
                self.conversation.update_chat_history(prompt, response_text)
                
                return {"text": response_text} if not raw else {"raw": response_text}
                
            except requests.exceptions.RequestException as e:
                # Handle potential errors during the request
                error_msg = f"Error fetching data: {e}"
                raise exceptions.FailedToGenerateResponseError(error_msg)
            except Exception as e:
                # Catch any other unexpected errors
                error_msg = f"An unexpected error occurred: {e}"
                raise exceptions.FailedToGenerateResponseError(error_msg)
        
        # This provider doesn't support streaming, so just return non-stream
        return for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send a prompt to the OpenGPT API and get a text response.
        
        Args:
            prompt: The user input/prompt for the API.
            stream: Whether to stream the response (not supported).
            optimizer: Optimizer to use on the prompt.
            conversationally: Whether to apply the optimizer on the full conversation prompt.
            
        Returns:
            A string with the response text.
        """
        response = self.ask(
            prompt, False, optimizer=optimizer, conversationally=conversationally
        )
        return self.get_message(response)

    def get_message(self, response: dict) -> str:
        """
        Extract the message from the response dictionary.
        
        Args:
            response: Response dictionary from the ask method.
            
        Returns:
            The text response as a string.
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]


if __name__ == "__main__":
    # Test the provider
    print("-" * 80)
    print("Testing OpenGPT provider")
    print("-" * 80)
    
    try:
        test_ai = OpenGPT()
        response = test_ai.chat("Explain quantum physics simply.")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

