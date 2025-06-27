import re
import requests
import json
import uuid
import random
import string
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
        "Deepseek R1 Latest",
        "GPT 4o",
        "O4 Mini",
        "O4 Mini High",
        "QwQ Plus",
        "Llama 4 Maverick",
        "Grok 3",
        "GPT 4o mini",
        "Deepseek v3 0324",
        "Grok 3 Mini",
        "GPT 4.1",
        "GPT 4.1 Mini",
        "Claude 3.7 Sonnet (Thinking)",
        "Llama 4 Scout",
        "O3 High",
        "Gemini 2.5 Pro",
        "Magistral Medium 2506",
        "O3",
        "Gemini 2.5 Flash",
        "Qwen 3 235B A22B",
        "Claude 4 Sonnet",
        "Claude 4 Sonnet (Thinking)",
        "Claude 4 Opus",
        "Claude 4 Opus (Thinking)",
        "Google: Gemini 2.5 Pro (thinking)",
    ]

    def _auto_fetch_api_key(self, proxies=None, timeout=30):
        """
        Automatically register a new user and fetch an API key from FreeAIChat Playground.
        """
        session = requests.Session()
        if proxies:
            session.proxies.update(proxies)
        def random_email():
            user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            return f"{user}@bltiwd.com"
        email = random_email()
        payload = {"email": email, "password": email}
        headers = {
            'User-Agent': LitAgent().random(),
            'Accept': '*/*',
            'Content-Type': 'application/json',
            'Origin': 'https://freeaichatplayground.com',
            'Referer': 'https://freeaichatplayground.com/register',
        }
        try:
            resp = session.post(
                "https://freeaichatplayground.com/api/v1/auth/register",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            if resp.status_code == 201:
                data = resp.json()
                apikey = data.get("user", {}).get("apikey")
                if apikey:
                    return apikey
                else:
                    raise exceptions.FailedToGenerateResponseError("API key not found in registration response.")
            else:
                raise exceptions.FailedToGenerateResponseError(f"Registration failed: {resp.status_code} {resp.text}")
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"API key auto-fetch failed: {e}")

    def __init__(
        self,
        api_key: str = None,
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
        """Initializes the FreeAIChat API client. If api_key is not provided, auto-register and fetch one."""
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
        if not api_key:
            self.api_key = self._auto_fetch_api_key(proxies=proxies, timeout=timeout)
        else:
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
