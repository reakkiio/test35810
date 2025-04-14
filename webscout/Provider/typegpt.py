import requests
import json
from typing import Union, Any, Dict, Generator
import requests.exceptions

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class TypeGPT(Provider):
    """
    A class to interact with the TypeGPT.net API. Improved to match webscout standards.
    """
    AVAILABLE_MODELS = [
        # Working Models (based on testing)
        "gpt-4o-mini-2024-07-18",
        "chatgpt-4o-latest",
        "deepseek-r1",
        "deepseek-v3",
        "uncensored-r1",
        "Image-Generator",
    ]

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 4000,  # Set a reasonable default
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "gpt-4o",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.5,
        presence_penalty: int = 0,
        frequency_penalty: int = 0,
        top_p: float = 1,
    ):
        """Initializes the TypeGPT API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {', '.join(self.AVAILABLE_MODELS)}")

        self.session = requests.Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.api_endpoint = "https://chat.typegpt.net/api/openai/v1/chat/completions"
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.top_p = top_p
        self.headers = {
            "authority": "chat.typegpt.net",
            "accept": "application/json, text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://chat.typegpt.net",
            "referer": "https://chat.typegpt.net/",
            "user-agent": LitAgent().random()
        }

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
        self.conversation = Conversation(is_conversation, self.max_tokens_to_sample, filepath, update_file)
        self.conversation.history_offset = history_offset
        self.session.proxies = proxies

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator[Any, None, None]]:
        """Sends a prompt to the TypeGPT.net API and returns the response."""
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

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt}
            ],
            "stream": stream,
            "model": self.model,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens_to_sample,
        }

        def for_stream():
            try:
                response = self.session.post(
                    self.api_endpoint, headers=self.headers, json=payload, stream=True, timeout=self.timeout
                )
            except requests.exceptions.ConnectionError as ce:
                raise exceptions.FailedToGenerateResponseError(
                    f"Network connection failed. Check your firewall or antivirus settings. Original error: {ce}"
                ) from ce

            if not response.ok:
                raise exceptions.FailedToGenerateResponseError(
                    f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                )
            message_load = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                        # Skip [DONE] message
                        if line.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(line)
                            # Extract and yield only new content
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    new_content = delta['content']
                                    message_load += new_content
                                    # Yield only the new content
                                    yield dict(text=new_content) if not raw else new_content
                                    self.last_response = dict(text=message_load)
                        except json.JSONDecodeError:
                            continue
            self.conversation.update_chat_history(prompt, self.get_message(self.last_response))

        def for_non_stream():
            try:
                response = self.session.post(self.api_endpoint, headers=self.headers, json=payload, timeout=self.timeout)
            except requests.exceptions.ConnectionError as ce:
                raise exceptions.FailedToGenerateResponseError(
                    f"Network connection failed. Check your firewall or antivirus settings. Original error: {ce}"
                ) from ce

            if not response.ok:
                raise exceptions.FailedToGenerateResponseError(
                    f"Request failed - {response.status_code}: {response.text}"
                )
            self.last_response = response.json()
            self.conversation.update_chat_history(prompt, self.get_message(self.last_response))
            return self.last_response

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response string or stream."""
        if stream:
            gen = self.ask(
                prompt, stream=True, optimizer=optimizer, conversationally=conversationally
            )
            for chunk in gen:
                yield self.get_message(chunk)  # Extract text from streamed chunks
        else:
            return self.get_message(self.ask(prompt, stream=False, optimizer=optimizer, conversationally=conversationally))

    def get_message(self, response: Dict[str, Any]) -> str:
        """Retrieves message from response."""
        if isinstance(response, str):  # Handle raw responses
            return response
        elif isinstance(response, dict):
            assert isinstance(response, dict), "Response should be of dict data-type only"
            return response.get("text", "")  # Extract text from dictionary response
        else:
            raise TypeError("Invalid response type. Expected str or dict.")

if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)
    
    # Test all available models
    working = 0
    total = len(TypeGPT.AVAILABLE_MODELS)
    
    for model in TypeGPT.AVAILABLE_MODELS:
        try:
            test_ai = TypeGPT(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            for chunk in response:
                response_text += chunk
                print(f"\r{model:<50} {'Testing...':<10}", end="", flush=True)
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
    
