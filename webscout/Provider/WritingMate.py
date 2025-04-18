import re
import requests, json
from typing import Union, Any, Dict, Generator, Optional
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class WritingMate(Provider):
    AVAILABLE_MODELS = [
        "claude-3-haiku-20240307",
        "gemini-1.5-flash-latest",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "google/gemini-flash-1.5-8b-exp",
        "gpt-4o-mini"
    ]
    """
    Provider for WritingMate streaming API.
    """
    api_endpoint = "https://chat.writingmate.ai/api/chat/tools-stream"

    def __init__(
        self,
        cookies_path: str = "cookies.json",
        is_conversation: bool = True,
        max_tokens: int = 4096,
        timeout: int = 60,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        act: str = None,
        system_prompt: str = "You are a friendly, helpful AI assistant.",
        model: str = "gpt-4o-mini"
    ):
        self.cookies_path = cookies_path
        self.cookies = self._load_cookies(cookies_path)
        self.session = requests.Session()
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.model = model
        if self.model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {self.model}. Choose from {self.AVAILABLE_MODELS}")
        self.last_response = {}
        self.headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8",
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": "https://chat.writingmate.ai",
            "Referer": "https://chat.writingmate.ai/chat",
            "Cookie": self.cookies,
            "DNT": "1",
            "sec-ch-ua": "\"Microsoft Edge\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-GPC": "1",
            "User-Agent": LitAgent().random()
        }
        self.session.headers.update(self.headers)
        self.__available_optimizers = (
            m for m in dir(Optimizers)
            if callable(getattr(Optimizers, m)) and not m.startswith("__")
        )
        Conversation.intro = (
            AwesomePrompts().get_act(act, raise_not_found=True, default=None, case_insensitive=True)
            if act else intro or Conversation.intro
        )
        self.conversation = Conversation(is_conversation, max_tokens, filepath, update_file)
        self.conversation.history_offset = 10250

    def _load_cookies(self, path: str) -> str:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return '; '.join(f"{c['name']}={c['value']}" for c in data)
        except (FileNotFoundError, json.JSONDecodeError):
            raise RuntimeError(f"Failed to load cookies from {path}")


    def ask(
        self,
        prompt: str,
        stream: bool = True,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False
    ) -> Union[Dict[str,Any], Generator[Any,None,None]]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise exceptions.FailedToGenerateResponseError(f"Unknown optimizer: {optimizer}")

        body = {
            "chatSettings": {
                "model": self.model,
                "prompt": self.system_prompt,
                "temperature": 0.5,
                "contextLength": 4096,
                "includeProfileContext": True,
                "includeWorkspaceInstructions": True,
                "embeddingsProvider": "openai"
            },
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": conversation_prompt}
            ],
            "selectedTools": []
        }

        def for_stream():
            response = self.session.post(self.api_endpoint, headers=self.headers, json=body, stream=True, timeout=self.timeout)
            if not response.ok:
                raise exceptions.FailedToGenerateResponseError(
                    f"Failed to generate response - ({response.status_code}, {response.reason}) - {response.text}"
                )
            streaming_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    match = re.search(r'0:"(.*?)"', line)
                    if match:
                        content = match.group(1)
                        streaming_response += content
                        yield content if raw else dict(text=content)
            self.last_response.update(dict(text=streaming_response))
            self.conversation.update_chat_history(
                prompt, self.get_message(self.last_response)
            )

        def for_non_stream():
            for _ in for_stream():
                pass
            return self.last_response

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False
    ) -> Union[str, Generator[str,None,None]]:
        if stream:
            # yield raw SSE lines
            def raw_stream():
                for line in self.ask(
                    prompt, stream=True, raw=True,
                    optimizer=optimizer, conversationally=conversationally
                ):
                    yield line
            return raw_stream()
        # non‐stream: return aggregated text
        return self.get_message(
            self.ask(
                prompt,
                False,
                raw=False,
                optimizer=optimizer,
                conversationally=conversationally,
            )
        )

    def get_message(self, response: dict) -> str:
        """
        Extracts the message from the API response.

        Args:
            response (dict): The API response.

        Returns:
            str: The message content.

        Examples:
            >>> ai = X0GPT()
            >>> response = ai.ask("Tell me a joke!")
            >>> message = ai.get_message(response)
            >>> print(message)
            'Why did the scarecrow win an award? Because he was outstanding in his field!'
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        formatted_text = response["text"].replace('\\n', '\n').replace('\\n\\n', '\n\n')
        return formatted_text
    
if __name__ == "__main__":
    from rich import print
    ai = WritingMate(cookies_path="cookies.json")
    response = ai.chat(input(">>> "), stream=True)
    for chunk in response:
        print(chunk, end="", flush=True)
