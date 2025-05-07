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
        "scira-anthropic": "Sonnet 3.7 thinking",
        "scira-vision" : "Grok2-Vision", # vision model
        "scira-4o": "GPT4o",
        "scira-qwq": "QWQ-32B",
        "scira-o4-mini": "o4-mini",
        "scira-google": "gemini 2.5 flash",
        "scira-google-pro": "gemini 2.5 pro",
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
    def _scira_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from the Scira stream format '0:"..."'."""
        if isinstance(chunk, str):
            match = re.search(r'0:"(.*?)"(?=,|$)', chunk) # Look for 0:"...", possibly followed by comma or end of string
            if match:
                # Decode potential unicode escapes like \u00e9 and handle escaped quotes/backslashes
                content = match.group(1).encode().decode('unicode_escape')
                return content.replace('\\\\', '\\').replace('\\"', '"')
        return None

    def ask(
        self,
        prompt: str,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> Dict[str, Any]: # Note: Stream parameter removed as API doesn't seem to support it
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

        try:
            # Use curl_cffi post with impersonate
            response = self.session.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                impersonate="chrome120" # Add impersonate
            )
            if response.status_code != 200:
                # Try to get response content for better error messages
                try: # Use try-except for reading response content
                    error_content = response.text
                except:
                    error_content = "<could not read response content>"

                if response.status_code in [403, 429]:
                    print(f"Received status code {response.status_code}, refreshing identity...")
                    self.refresh_identity()
                    response = self.session.post(
                        self.url, json=payload, timeout=self.timeout,
                        impersonate="chrome120" # Add impersonate to retry
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

            response_text_raw = response.text # Get raw response text

            # Process the text using sanitize_stream line by line
            processed_stream = sanitize_stream(
                data=response_text_raw.splitlines(), # Split into lines
                intro_value=None, # No simple prefix
                to_json=False,    # Content is not JSON
                content_extractor=self._scira_extractor # Use the specific extractor
            )

            # Aggregate the results from the generator
            full_response = ""
            for content in processed_stream:
                if content and isinstance(content, str):
                    full_response += content

            self.last_response = {"text": full_response}
            self.conversation.update_chat_history(prompt, full_response)
            return {"text": full_response}
        except CurlError as e: # Catch CurlError
            raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}") from e
        except Exception as e:
            raise exceptions.FailedToGenerateResponseError(f"Request failed: {e}")

    def chat(
        self,
        prompt: str,
        optimizer: str = None,
        conversationally: bool = False,
    ) -> str:
        return self.get_message(
            self.ask(
                prompt, optimizer=optimizer, conversationally=conversationally
            )
        )

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        # Extractor handles formatting
        return response.get("text", "").replace('\\n', '\n').replace('\\n\\n', '\n\n')

if __name__ == "__main__":
    print("-" * 100)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 100)

    test_prompt = "Say 'Hello' in one word"

    # Test each model
    for model in SciraAI.AVAILABLE_MODELS:
        print(f"\rTesting {model}...", end="")

        try:
            test_ai = SciraAI(model=model, timeout=120)  # Increased timeout
            response = test_ai.chat(test_prompt)

            if response and len(response.strip()) > 0:
                status = "✓"
                # Clean and truncate response
                clean_text = response.strip().encode('utf-8', errors='ignore').decode('utf-8')
                display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            else:
                status = "✗"
                display_text = "Empty or invalid response"

            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            error_msg = str(e)
            # Truncate very long error messages
            if len(error_msg) > 100:
                error_msg = error_msg[:97] + "..."
            print(f"\r{model:<50} {'✗':<10} Error: {error_msg}")