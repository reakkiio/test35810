from curl_cffi.requests import Session, RequestsError
import urllib.parse
from typing import Union, Any, Dict

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider

class AI4Chat(Provider):
    """
    A class to interact with the AI4Chat Riddle API.
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
        system_prompt: str = "You are a helpful and informative AI assistant.",
        country: str = "Asia",
        user_id: str = "usersmjb2oaz7y"
    ) -> None:
        """
        Initializes the AI4Chat API with given parameters.

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
            system_prompt (str, optional): System prompt to guide the AI's behavior. Defaults to "You are a helpful and informative AI assistant.".
            country (str, optional): Country parameter for API. Defaults to "Asia".
            user_id (str, optional): User ID for API. Defaults to "usersmjb2oaz7y".
        """
        self.session = Session(timeout=timeout, proxies=proxies)
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://yw85opafq6.execute-api.us-east-1.amazonaws.com/default/boss_mode_15aug"
        self.timeout = timeout
        self.last_response = {}
        self.country = country
        self.user_id = user_id
        self.headers = {
            "Accept": "*/*",
            "Accept-Language": "id-ID,id;q=0.9",
            "Origin": "https://www.ai4chat.co",
            "Priority": "u=1, i",
            "Referer": "https://www.ai4chat.co/",
            "Sec-CH-UA": '"Chromium";v="131", "Not_A Brand";v="24", "Microsoft Edge Simulate";v="131", "Lemur";v="131"',
            "Sec-CH-UA-Mobile": "?1",
            "Sec-CH-UA-Platform": '"Android"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36"
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
        self.system_prompt = system_prompt 

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        country: str = None,
        user_id: str = None,
    ) -> Dict[str, Any]:
        """
        Sends a prompt to the AI4Chat API and returns the response.

        Args:
            prompt: The text prompt to generate text from.
            stream (bool, optional): Not supported. Defaults to False.
            raw (bool, optional): Whether to return the raw response. Defaults to False.
            optimizer (str, optional): The name of the optimizer to use. Defaults to None.
            conversationally (bool, optional): Whether to chat conversationally. Defaults to False.
            country (str, optional): Country parameter for API. Defaults to None.
            user_id (str, optional): User ID for API. Defaults to None.

        Returns:
            dict: A dictionary containing the AI's response.
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

        country_param = country or self.country
        user_id_param = user_id or self.user_id
        
        encoded_text = urllib.parse.quote(conversation_prompt)
        encoded_country = urllib.parse.quote(country_param)
        encoded_user_id = urllib.parse.quote(user_id_param)
        
        url = f"{self.api_endpoint}?text={encoded_text}&country={encoded_country}&user_id={encoded_user_id}"
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=self.timeout)
        except RequestsError as e:
            raise Exception(f"Failed to generate response: {e}")
        if not response.ok:
            raise Exception(f"Failed to generate response: {response.status_code} - {response.reason}")
        
        response_text = response.text
        
        if response_text.startswith('"'):
            response_text = response_text[1:]
        if response_text.endswith('"'):
            response_text = response_text[:-1]
        
        self.last_response.update(dict(text=response_text))
        self.conversation.update_chat_history(prompt, response_text)
        return self.last_response

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        country: str = None,
        user_id: str = None,
    ) -> str:
        """
        Generates a response from the AI4Chat API.

        Args:
            prompt (str): The prompt to send to the AI.
            stream (bool, optional): Not supported. 
            optimizer (str, optional): The name of the optimizer to use. Defaults to None.
            conversationally (bool, optional): Whether to chat conversationally. Defaults to False.
            country (str, optional): Country parameter for API. Defaults to None.
            user_id (str, optional): User ID for API. Defaults to None.

        Returns:
            str: The response generated by the AI.
        """
        return self.get_message(
            self.ask(
                prompt,
                optimizer=optimizer,
                conversationally=conversationally,
                country=country,
                user_id=user_id,
            )
        )

    def get_message(self, response: dict) -> str:
        """Retrieves message only from response

        Args:
            response (dict): Response generated by `self.ask`

        Returns:
            str: Message extracted
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"].replace('\\n', '\n').replace('\\n\\n', '\n\n')

if __name__ == "__main__":
    from rich import print
    ai = AI4Chat() 
    response = ai.chat("Tell me something interesting")
    print(response)