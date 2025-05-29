from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import secrets
from typing import Any, Dict, Optional, Generator, Union

from webscout.AIutel import Optimizers, Conversation, AwesomePrompts, sanitize_stream
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
        {"modelId": 9, "name": "Qwen3-30B-A3B"},  # Added new model
        # Add more models here as needed
    ]
    MODEL_NAME_TO_ID = {m["name"]: m["modelId"] for m in AVAILABLE_MODELS}
    MODEL_ID_TO_NAME = {m["modelId"]: m["name"] for m in AVAILABLE_MODELS}

    def __init__(
        self,
        model: str = "QWQ-32B",
        is_conversation: bool = True,
        max_tokens: int = 2048, # Note: max_tokens is not used by this API
        timeout: int = 30,
        intro: Optional[str] = ("You are a helpful, advanced LLM assistant. "
            "You must always answer in English, regardless of the user's language. "
            "If the user asks in another language, politely respond in English only. "
            "Be clear, concise, and helpful."),
        filepath: Optional[str] = None,
        update_file: bool = True,
        proxies: Optional[dict] = None,
        history_offset: int = 0, # Note: history_offset might not be fully effective due to API structure
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
        # Initialize curl_cffi Session
        self.session = Session()
        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response: Dict[str, Any] = {}
        self.proxies = proxies or {}
        self.cookies = {
            "Token": secrets.token_hex(16), # Keep cookie generation logic
        }
        self.headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
            "referer": "https://www.scnet.cn/ui/chatbot/temp_1744712663464",
            "origin": "https://www.scnet.cn",
            # Add sec-ch-ua headers if needed for impersonation consistency
        }
        self.url = "https://www.scnet.cn/acx/chatbot/v1/chat/completion"
        
        # Update curl_cffi session headers, proxies, and cookies
        self.session.headers.update(self.headers)
        self.session.proxies = self.proxies # Assign proxies directly
        # Set cookies on the session object for curl_cffi
        for name, value in self.cookies.items():
            self.session.cookies.set(name, value) 

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

    @staticmethod
    def _scnet_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from SCNet stream JSON objects."""
        if isinstance(chunk, dict):
            return chunk.get("content")
        return None

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
                conversation_prompt = getattr(Optimizers, optimizer)(conversation_prompt if conversationally else prompt)
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
                # Use curl_cffi session post with impersonate
                # Cookies are now handled by the session object
                response = self.session.post(
                    self.url,
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                    impersonate="chrome120" # Changed impersonation to chrome120
                )
                response.raise_for_status() # Check for HTTP errors

                streaming_text = ""
                # Use sanitize_stream
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value="data:",
                    to_json=True,     # Stream sends JSON
                    skip_markers=["[done]"],
                    content_extractor=self._scnet_extractor, # Use the specific extractor
                    yield_raw_on_error=False # Skip non-JSON lines or lines where extractor fails
                )

                for content_chunk in processed_stream:
                    # content_chunk is the string extracted by _scnet_extractor
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        yield {"text": content_chunk} if not raw else content_chunk
                # Update history and last response after stream finishes
                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)

            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e: # Catch other potential exceptions (like HTTPError)
                err_text = getattr(e, 'response', None) and getattr(e.response, 'text', '')
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e} - {err_text}") from e

        def for_non_stream():
            # Aggregate the stream using the updated for_stream logic
            text = ""
             # Ensure raw=False so for_stream yields dicts
            for chunk_data in for_stream():
                if isinstance(chunk_data, dict) and "text" in chunk_data:
                     text += chunk_data["text"]
                # Handle raw string case if raw=True was passed
                elif isinstance(chunk_data, str):
                     text += chunk_data
            # last_response and history are updated within for_stream
            # Return the final aggregated response dict or raw string
            return text if raw else self.last_response


        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: Optional[str] = None,
        conversationally: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream_chat():
            # ask() yields dicts or strings when streaming
            gen = self.ask(
                prompt, stream=True, raw=False, # Ensure ask yields dicts
                optimizer=optimizer, conversationally=conversationally
            )
            for response_dict in gen:
                yield self.get_message(response_dict) # get_message expects dict

        def for_non_stream_chat():
            # ask() returns dict or str when not streaming
            response_data = self.ask(
                prompt, stream=False, raw=False, # Ensure ask returns dict
                optimizer=optimizer, conversationally=conversationally
            )
            return self.get_message(response_data) # get_message expects dict

        return for_stream_chat() if stream else for_non_stream_chat()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"]

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'ModelId':<10} {'Model':<30} {'Status':<10} {'Response'}")
    print("-" * 80)
    for model in SCNet.AVAILABLE_MODELS:
        try:
            test_ai = SCNet(model=model["name"], timeout=60)
            # Test stream first
            response_stream = test_ai.chat("Say 'Hello' in one word", stream=True)
            response_text = ""
            print(f"\r{model['modelId']:<10} {model['name']:<30} {'Streaming...':<10}", end="", flush=True)
            for chunk in response_stream:
                response_text += chunk
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response_text.strip()
                display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            else:
                status = "✗ (Stream)"
                display_text = "Empty or invalid stream response"
            print(f"\r{model['modelId']:<10} {model['name']:<30} {status:<10} {display_text}")

            # Optional: Add non-stream test if needed
            # print(f"\r{model['modelId']:<10} {model['name']:<30} {'Non-Stream...':<10}", end="", flush=True)
            # response_non_stream = test_ai.chat("Say 'Hi' again", stream=False)
            # if not response_non_stream or len(response_non_stream.strip()) == 0:
            #      print(f"\r{model['modelId']:<10} {model['name']:<30} {'✗ (Non-Stream)':<10} Empty non-stream response")

        except Exception as e:
            print(f"\r{model['modelId']:<10} {model['name']:<30} {'✗':<10} {str(e)}")
