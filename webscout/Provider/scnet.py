import requests
import json
import secrets
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions

class SCNet(Provider):
    """
    Provider for SCNet chatbot API.
    """
    AVAILABLE_MODELS = [
        {"modelId": 2, "name": "Deepseek-r1-7B"},
        {"modelId": 3, "name": "Deepseek-r1-32B"},
        {"modelId": 5, "name": "Deepseek-r1-70B"},
        {"modelId": 7, "name": "QWQ-32B"},
        {"modelId": 8, "name": "minimax-text-01-456B"},
        # Add more models here as needed
    ]
    MODEL_NAME_TO_ID = {m["name"]: m["modelId"] for m in AVAILABLE_MODELS}
    MODEL_ID_TO_NAME = {m["modelId"]: m["name"] for m in AVAILABLE_MODELS}

    def __init__(
        self,
        model: str = "QWQ-32B",
        is_conversation: bool = True,
        max_tokens: int = 2048,
        timeout: int = 30,
        intro: Optional[str] = None,
        filepath: Optional[str] = None,
        update_file: bool = True,
        proxies: Optional[dict] = None,
        history_offset: int = 0,
        act: Optional[str] = None,
        system_prompt: str = (
            "You are a helpful, advanced LLM assistant. "
            "You must always answer in English, regardless of the user's language. "
            "If the user asks in another language, politely respond in English only. "
            "Be clear, concise, and helpful."
        ),
    ):
        if model not in self.MODEL_NAME_TO_ID:
            raise ValueError(f"Invalid model: {model}. Choose from: {list(self.MODEL_NAME_TO_ID.keys())}")
        self.model = model
        self.modelId = self.MODEL_NAME_TO_ID[model]
        self.system_prompt = system_prompt
        self.session = requests.Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response: Dict[str, Any] = {}
        self.proxies = proxies or {}
        self.cookies = {
            "Token": secrets.token_hex(16),
        }
        self.headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
            "referer": "https://www.scnet.cn/ui/chatbot/temp_1744712663464",
            "origin": "https://www.scnet.cn",
        }
        self.url = "https://www.scnet.cn/acx/chatbot/v1/chat/completion"
        self.__available_optimizers = (
            method for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )
        Conversation.intro = (
            AwesomePrompts().get_act(act, raise_not_found=True, default=None, case_insensitive=True)
            if act
            else intro or Conversation.intro
        )
        self.conversation = Conversation(is_conversation, max_tokens, filepath, update_file)
        self.conversation.history_offset = history_offset

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise exceptions.FailedToGenerateResponseError(f"Optimizer is not one of {list(self.__available_optimizers)}")

        payload = {
            "conversationId": "",
            "content": f"SYSTEM: {self.system_prompt} USER: {conversation_prompt}",
            "thinking": 0,
            "online": 0,
            "modelId": self.modelId,
            "textFile": [],
            "imageFile": [],
            "clusterId": ""
        }

        def for_stream():
            try:
                with self.session.post(
                    self.url,
                    headers=self.headers,
                    cookies=self.cookies,
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    proxies=self.proxies
                ) as resp:
                    streaming_text = ""
                    for line in resp.iter_lines(decode_unicode=True):
                        if line and line.startswith("data:"):
                            data = line[5:].strip()
                            if data and data != "[done]":
                                try:
                                    obj = json.loads(data)
                                    content = obj.get("content", "")
                                    streaming_text += content
                                    yield {"text": content} if raw else {"text": content}
                                except Exception:
                                    continue
                            elif data == "[done]":
                                break
                    self.last_response = {"text": streaming_text}
                    self.conversation.update_chat_history(prompt, streaming_text)
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")

        def for_non_stream():
            text = ""
            for chunk in for_stream():
                text += chunk["text"]
            return {"text": text}

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream():
            for response in self.ask(
                prompt, stream=True, optimizer=optimizer, conversationally=conversationally
            ):
                yield self.get_message(response)
        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt, stream=False, optimizer=optimizer, conversationally=conversationally
                )
            )
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    print("-" * 80)
    print(f"{'ModelId':<10} {'Model':<30} {'Status':<10} {'Response'}")
    print("-" * 80)
    for model in SCNet.AVAILABLE_MODELS:
        try:
            test_ai = SCNet(model=model["name"], timeout=60)
            response = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            for chunk in response:
                response_text += chunk
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"{model['modelId']:<10} {model['name']:<30} {status:<10} {display_text}")
        except Exception as e:
            print(f"{model['modelId']:<10} {model['name']:<30} {'✗':<10} {str(e)}")
