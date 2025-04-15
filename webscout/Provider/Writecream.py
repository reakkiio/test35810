import requests
import json
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class Writecream(Provider):
    """
    A class to interact with the Writecream API.
    """

    AVAILABLE_MODELS = ["writecream-gpt"]

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
        system_prompt: str = "You are a helpful and informative AI assistant.",
        base_url: str = "https://8pe3nv3qha.execute-api.us-east-1.amazonaws.com/default/llm_chat",
        user_agent: str = "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Mobile Safari/537.36",
        referer: str = "https://www.writecream.com/chatgpt-chat/",
        link: str = "writecream.com",
        model: str = "writecream-gpt"
    ):
        """
        Initializes the Writecream API with given parameters.
        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.session = requests.Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.base_url = base_url
        self.timeout = timeout
        self.last_response = {}
        self.system_prompt = system_prompt
        self.model = model
        self.user_agent = user_agent
        self.referer = referer
        self.link = link

        self.headers = {
            "User-Agent": self.user_agent,
            "Referer": self.referer
        }

        self.__available_optimizers = (
            method
            for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )

        self.session.headers.update(self.headers)
        self.session.proxies.update(proxies)

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
    ) -> Union[Dict[str, Any], Generator]:
        """
        Sends a message to the Writecream API and returns the response.

        Args:
            prompt (str): Prompt to be sent.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            raw (bool, optional): Stream back raw response as received. Defaults to False.
            optimizer (str, optional): Prompt optimizer name. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.

        Returns:
            Union[Dict[str, Any], Generator]: Response from the API.
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

        final_query = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conversation_prompt}
        ]

        params = {
            "query": json.dumps(final_query),
            "link": self.link
        }

        def for_non_stream():
            try:
                response = self.session.get(self.base_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                # Extract the response content
                response_content = data.get("response", data.get("response_content", ""))

                # Update conversation history
                self.last_response = {"text": response_content}
                self.conversation.update_chat_history(prompt, response_content)

                return {"text": response_content}
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Failed to get response from the chat API: {e}")

        # Currently, Writecream API doesn't support streaming, so we always return non-streaming response
        return for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generates a response from the Writecream API.

        Args:
            prompt (str): Prompt to be sent.
            stream (bool, optional): Flag for streaming response. Defaults to False.
            optimizer (str, optional): Prompt optimizer name. Defaults to None.
            conversationally (bool, optional): Chat conversationally when using optimizer. Defaults to False.

        Returns:
            Union[str, Generator[str, None, None]]: Response from the API.
        """
        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt,
                    stream=False,
                    optimizer=optimizer,
                    conversationally=conversationally,
                )
            )

        # Currently, Writecream API doesn't support streaming
        return for_non_stream()

    def get_message(self, response: dict) -> str:
        """
        Retrieves message only from response.

        Args:
            response (dict): Response generated by `self.ask`

        Returns:
            str: Message extracted
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]


if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<30} {'Status':<10} {'Response'}")
    print("-" * 80)

    try:
        test_api = Writecream(timeout=60)
        prompt = "Say 'Hello' in one word"
        response = test_api.chat(prompt)

        if response and len(response.strip()) > 0:
            status = "✓"
            # Clean and truncate response
            clean_text = response.strip().encode('utf-8', errors='ignore').decode('utf-8')
            display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
        else:
            status = "✗"
            display_text = "Empty or invalid response"

        print(f"{test_api.model:<30} {status:<10} {display_text}")
    except Exception as e:
        print(f"{Writecream.AVAILABLE_MODELS[0]:<30} {'✗':<10} {str(e)}")
