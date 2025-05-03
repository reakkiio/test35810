import re
import requests
import json
import uuid
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation
from webscout.AIutel import AwesomePrompts, sanitize_stream
from webscout.AIbase import Provider, AsyncProvider
from webscout import exceptions
from webscout.litagent import LitAgent

class FreeAIChat(Provider):
    """
    A class to interact with the FreeAIChat API
    """

    AVAILABLE_MODELS = [
        # OpenAI Models
        "GPT 4o",
        "GPT 4.5 Preview",
        "GPT 4o Latest",
        "GPT 4o mini",
        "GPT 4o Search Preview",
        "O1",
        "O1 Mini",
        "O3 Mini",
        "O3 Mini High",
        "O3 Mini Low",
        "O4 Mini",
        "O4 Mini High",
        "GPT 4.1",
        "o3",
        "GPT 4.1 Mini",


        # Anthropic Models
        "Claude 3.5 haiku",
        "claude 3.5 sonnet",
        "Claude 3.7 Sonnet",
        "Claude 3.7 Sonnet (Thinking)",

        # Deepseek Models
        "Deepseek R1",
        "Deepseek R1 Fast",
        "Deepseek V3",
        "Deepseek v3 0324",

        # Google Models
        "Gemini 1.5 Flash",
        "Gemini 1.5 Pro",
        "Gemini 2.0 Flash",
        "Gemini 2.0 Pro",
        "Gemini 2.5 Pro",

        # Llama Models
        "Llama 3.1 405B",
        "Llama 3.1 70B Fast",
        "Llama 3.3 70B",
        "Llama 3.2 90B Vision",
        "Llama 4 Scout",
        "Llama 4 Maverick",

        # Mistral Models
        "Mistral Large",
        "Mistral Nemo",
        "Mixtral 8x22B",

        # Qwen Models
        "Qwen Max",
        "Qwen Plus",
        "Qwen Turbo",
        "QwQ 32B",
        "QwQ Plus",

        # XAI Models
        "Grok 2",
        "Grok 3",
    ]

    def __init__(
        self,
        api_key: str,
        is_conversation: bool = True,
        max_tokens: int = 150,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "GPT 4o",
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
    ):
        """Initializes the FreeAIChat API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.url = "https://freeaichatplayground.com/api/v1/chat/completions"
        self.headers = {
            'User-Agent': LitAgent().random(),
            'Accept': '*/*',
            'Content-Type': 'application/json',
            'Origin': 'https://freeaichatplayground.com',
            'Referer': 'https://freeaichatplayground.com/',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(proxies)

        self.is_conversation = is_conversation
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.api_key = api_key

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

        self.conversation = Conversation(
            is_conversation, self.max_tokens, filepath, update_file
        )
        self.conversation.history_offset = history_offset

    @staticmethod
    def _extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from the x0gpt stream format '0:"..."'."""
        if isinstance(chunk, str):
            match = re.search(r'0:"(.*?)"', chunk)
            if match:
                # Decode potential unicode escapes like \u00e9
                content = match.group(1).encode().decode('unicode_escape')
                return content.replace('\\\\', '\\').replace('\\"', '"') # Handle escaped backslashes and quotes
        return None

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        payload = {
            "id": str(uuid.uuid4()),
            "messages": [{
                "role": "user",
                "content": conversation_prompt,
                "parts": [{"type": "text", "text": conversation_prompt}]
            }],
            "model": self.model,
            "config": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            },
            "apiKey": self.api_key
        }

        def for_stream():
            try:
                with requests.post(self.url, headers=self.headers, json=payload, stream=True, timeout=self.timeout) as response:
                    if response.status_code != 200:
                        raise exceptions.FailedToGenerateResponseError(
                            f"Request failed with status code {response.status_code}"
                        )
                    
                    streaming_response = ""
                    processed_stream = sanitize_stream(
                        data=response.iter_lines(decode_unicode=True),
                        intro_value=None,
                        to_json=False,
                        content_extractor=self._extractor,
                        skip_markers=None
                    )

                    for content_chunk in processed_stream:
                        if content_chunk and isinstance(content_chunk, str):
                            streaming_response += content_chunk
                            yield dict(text=content_chunk) if raw else dict(text=content_chunk)

                    self.last_response.update(dict(text=streaming_response))
                    self.conversation.update_chat_history(
                        prompt, self.get_message(self.last_response)
                    )

            except requests.RequestException as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")

        def for_non_stream():
            full_text = ""
            for chunk in for_stream():
                full_text += chunk["text"]
            return {"text": full_text}

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> str:
        def for_stream():
            for response in self.ask(prompt, True, optimizer=optimizer, conversationally=conversationally):
                yield self.get_message(response)
                
        def for_non_stream():
            return self.get_message(
                self.ask(prompt, False, optimizer=optimizer, conversationally=conversationally)
            )
            
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

    @staticmethod
    def fix_encoding(text):
        if isinstance(text, dict) and "text" in text:
            try:
                text["text"] = text["text"].encode("latin1").decode("utf-8")
                return text
            except (UnicodeError, AttributeError) as e:
                return text
        elif isinstance(text, str):
            try:
                return text.encode("latin1").decode("utf-8")
            except (UnicodeError, AttributeError) as e:
                return text
        return text

if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in FreeAIChat.AVAILABLE_MODELS:
        try:
            test_ai = FreeAIChat(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            for chunk in response:
                response_text += chunk
                print(f"\r{model:<50} {'Testing...':<10}", end="", flush=True)
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response_text.strip().encode('utf-8', errors='ignore').decode('utf-8')
                display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
