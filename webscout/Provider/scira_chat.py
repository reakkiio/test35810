from os import system
from curl_cffi import CurlError
from curl_cffi.requests import Session
import json
import uuid
import re
from typing import Any, Dict, Optional, Union, List
from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation, sanitize_stream # Import sanitize_stream
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class SciraAI(Provider):
    """
    A class to interact with the Scira AI chat API.
    """

    AVAILABLE_MODELS = {
        "scira-default": "Grok3-mini", # thinking model
        "scira-grok-3": "Grok3",
        "scira-anthropic": "Claude 4 Sonnet",
        "scira-anthropic-thinking": "Claude 4 Sonnet Thinking", # thinking model
        "scira-vision" : "Grok2-Vision", # vision model
        "scira-4o": "GPT4o",
        "scira-qwq": "QWQ-32B",
        "scira-o4-mini": "o4-mini",
        "scira-google": "gemini 2.5 flash Thinking", # thinking model
        "scira-google-pro": "gemini 2.5 pro",
        "scira-llama-4": "llama 4 Maverick",
    }

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 2049,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "scira-default",
        chat_id: str = None,
        user_id: str = None,
        browser: str = "chrome",
        system_prompt: str = "You are a helpful assistant.",
    ):
        """Initializes the Scira AI API client.

        Args:
            is_conversation (bool): Whether to maintain conversation history.
            max_tokens (int): Maximum number of tokens to generate.
            timeout (int): Request timeout in seconds.
            intro (str): Introduction text for the conversation.
            filepath (str): Path to save conversation history.
            update_file (bool): Whether to update the conversation history file.
            proxies (dict): Proxy configuration for requests.
            history_offset (int): Maximum history length in characters.
            act (str): Persona for the AI to adopt.
            model (str): Model to use, must be one of AVAILABLE_MODELS.
            chat_id (str): Unique identifier for the chat session.
            user_id (str): Unique identifier for the user.
            browser (str): Browser to emulate in requests.
            system_prompt (str): System prompt for the AI.

        """
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")

        self.url = "https://scira.ai/api/search"

        # Initialize LitAgent for user agent generation
        self.agent = LitAgent()
        # Use fingerprinting to create a consistent browser identity
        self.fingerprint = self.agent.generate_fingerprint(browser)
        self.system_prompt = system_prompt
        
        # Use the fingerprint for headers
        self.headers = {
            "Accept": self.fingerprint["accept"],
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": self.fingerprint["accept_language"],
            "Content-Type": "application/json",
            "Origin": "https://scira.ai",
            "Referer": "https://scira.ai/",
            "Sec-CH-UA": self.fingerprint["sec_ch_ua"] or '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{self.fingerprint["platform"]}"',
            "User-Agent": self.fingerprint["user_agent"],
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }

        self.session = Session() # Use curl_cffi Session
        self.session.headers.update(self.headers)
        self.session.proxies = proxies # Assign proxies directly

        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.chat_id = chat_id or str(uuid.uuid4())
        self.user_id = user_id or f"user_{str(uuid.uuid4())[:8].upper()}"

        # Always use chat mode (no web search)
        self.search_mode = "chat"

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
            is_conversation, self.max_tokens_to_sample, filepath, update_file
        )
        self.conversation.history_offset = history_offset

    def refresh_identity(self, browser: str = None):
        """
        Refreshes the browser identity fingerprint.

        Args:
            browser: Specific browser to use for the new fingerprint
        """
        browser = browser or self.fingerprint.get("browser_type", "chrome")
        self.fingerprint = self.agent.generate_fingerprint(browser)

        # Update headers with new fingerprint
        self.headers.update({
            "Accept": self.fingerprint["accept"],
            "Accept-Language": self.fingerprint["accept_language"],
            "Sec-CH-UA": self.fingerprint["sec_ch_ua"] or self.headers["Sec-CH-UA"],
            "Sec-CH-UA-Platform": f'"{self.fingerprint["platform"]}"',
            "User-Agent": self.fingerprint["user_agent"],
        })

        # Update session headers
        for header, value in self.headers.items():
            self.session.headers[header] = value

        return self.fingerprint

    @staticmethod
    def _scira_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[dict]:
        """Extracts g and 0 chunks from the Scira stream format.
        Returns a dict: {"g": [g1, g2, ...], "0": zero} if present.
        """
        if isinstance(chunk, str):
            g_matches = re.findall(r'g:"(.*?)"', chunk)
            zero_match = re.search(r'0:"(.*?)"(?=,|$)', chunk)
            result = {}
            if g_matches:
                result["g"] = [g.encode().decode('unicode_escape').replace('\\', '\\').replace('\\"', '"') for g in g_matches]
            if zero_match:
                result["0"] = zero_match.group(1).encode().decode('unicode_escape').replace('\\', '\\').replace('\\"', '"')
            if result:
                return result
        return None

    def ask(
        self,
        prompt: str,
        optimizer: str = None,
        conversationally: bool = False,
        stream: bool = True,  # Default to True, always stream
        raw: bool = False,    # Added raw parameter
    ) -> Union[Dict[str, Any], Any]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conversation_prompt, "parts": [{"type": "text", "text": conversation_prompt}]}
        ]

        # Prepare the request payload
        payload = {
            "id": self.chat_id,
            "messages": messages,
            "model": self.model,
            "group": self.search_mode,
            "user_id": self.user_id,
            "timezone": "Asia/Calcutta"
        }

        def for_stream():
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout,
                    impersonate="chrome120",
                    stream=True
                )
                if response.status_code != 200:
                    try:
                        error_content = response.text
                    except:
                        error_content = "<could not read response content>"

                    if response.status_code in [403, 429]:
                        print(f"Received status code {response.status_code}, refreshing identity...")
                        self.refresh_identity()
                        response = self.session.post(
                            self.url, json=payload, timeout=self.timeout,
                            impersonate="chrome120", stream=True
                        )
                        if not response.ok:
                            raise exceptions.FailedToGenerateResponseError(
                                f"Failed to generate response after identity refresh - ({response.status_code}, {response.reason}) - {error_content}"
                            )
                        print("Identity refreshed successfully.")
                    else:
                        raise exceptions.FailedToGenerateResponseError(
                            f"Request failed with status code {response.status_code}. Response: {error_content}"
                        )

                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None),
                    intro_value=None,
                    to_json=False,
                    content_extractor=self._scira_extractor,
                    raw=raw
                )

                streaming_response = ""
                in_think = False
                for content in processed_stream:
                    if content is None:
                        continue
                    if isinstance(content, dict):
                        # Handle g chunks
                        g_chunks = content.get("g", [])
                        zero_chunk = content.get("0")
                        if g_chunks:
                            if not in_think:
                                if raw:
                                    yield "<think>\n\n"
                                else:
                                    yield "<think>\n\n"
                                in_think = True
                            for g in g_chunks:
                                if raw:
                                    yield g
                                else:
                                    yield dict(text=g)
                        if zero_chunk is not None:
                            if in_think:
                                if raw:
                                    yield "</think>\n\n"
                                else:
                                    yield "</think>\n\n"
                                in_think = False
                            if raw:
                                yield zero_chunk
                            else:
                                streaming_response += zero_chunk
                                yield dict(text=zero_chunk)
                    else:
                        # fallback for old string/list logic
                        if raw:
                            yield content
                        else:
                            if content and isinstance(content, str):
                                streaming_response += content
                                yield dict(text=content)
                if not raw:
                    self.last_response = {"text": streaming_response}
                    self.conversation.update_chat_history(prompt, streaming_response)
            except CurlError as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
            except Exception as e:
                raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")

        def for_non_stream():
            # Always use streaming logic, but aggregate the result
            full_response = ""
            for chunk in for_stream():
                if raw:
                    if isinstance(chunk, str):
                        full_response += chunk
                else:
                    if isinstance(chunk, dict) and "text" in chunk:
                        full_response += chunk["text"]
            if not raw:
                self.last_response = {"text": full_response}
                self.conversation.update_chat_history(prompt, full_response)
                return {"text": full_response}
            else:
                return full_response

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        optimizer: str = None,
        conversationally: bool = False,
        stream: bool = True,  # Default to True, always stream
        raw: bool = False,    # Added raw parameter
    ) -> Any:
        def for_stream():
            for response in self.ask(
                prompt, optimizer=optimizer, conversationally=conversationally, stream=True, raw=raw
            ):
                if raw:
                    yield response
                else:
                    if isinstance(response, dict):
                        yield self.get_message(response)
                    else:
                        # For <think> and </think> tags (strings), yield as is
                        yield response
        def for_non_stream():
            result = self.ask(
                prompt, optimizer=optimizer, conversationally=conversationally, stream=False, raw=raw
            )
            if raw:
                return result
            else:
                if isinstance(result, dict):
                    return self.get_message(result)
                else:
                    return result
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        """
        Retrieves message only from response

        Args:
            response (dict): Response generated by `self.ask`

        Returns:
            str: Message extracted
        """
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response.get("text", "")

if __name__ == "__main__":
    ai = SciraAI()
    resp = ai.chat("What is the capital of France?", stream=True, raw=False)
    for chunk in resp:
        print(chunk, end="", flush=True)
