from curl_cffi.requests import Session
from curl_cffi import CurlError
import json
import uuid
import re
from typing import Any, Dict, Optional, Generator, Union
from webscout.AIutel import Optimizers
from webscout.AIutel import Conversation, sanitize_stream # Import sanitize_stream
from webscout.AIutel import AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent

class UncovrAI(Provider):
    """
    A class to interact with the Uncovr AI chat API.
    """

    AVAILABLE_MODELS = [
        "default",
        "gpt-4o-mini",
        "gemini-2-flash",
        "gemini-2-flash-lite",
        "groq-llama-3-1-8b",
        "o3-mini",
        "deepseek-r1-distill-qwen-32b",
        # The following models are not available in the free plan:
        # "claude-3-7-sonnet",
        # "gpt-4o",
        # "claude-3-5-sonnet-v2",
        # "deepseek-r1-distill-llama-70b",
        # "gemini-2-flash-lite-preview",
        # "qwen-qwq-32b"
    ]

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
        model: str = "default",
        chat_id: str = None,
        user_id: str = None,
        browser: str = "chrome"
    ):
        """Initializes the Uncovr AI API client."""
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model: {model}. Choose from: {self.AVAILABLE_MODELS}")
            
        self.url = "https://uncovr.app/api/workflows/chat"
        
        # Initialize LitAgent for user agent generation
        self.agent = LitAgent()
        # Use fingerprinting to create a consistent browser identity
        self.fingerprint = self.agent.generate_fingerprint(browser)
        
        # Use the fingerprint for headers
        self.headers = {
            "Accept": self.fingerprint["accept"],
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": self.fingerprint["accept_language"],
            "Content-Type": "application/json",
            "Origin": "https://uncovr.app",
            "Referer": "https://uncovr.app/",
            "Sec-CH-UA": self.fingerprint["sec_ch_ua"] or '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": f'"{self.fingerprint["platform"]}"',
            "User-Agent": self.fingerprint["user_agent"],
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }
        
        # Initialize curl_cffi Session
        self.session = Session()
        # Update curl_cffi session headers and proxies
        self.session.headers.update(self.headers)
        self.session.proxies.update(proxies)

        self.is_conversation = is_conversation
        self.max_tokens_to_sample = max_tokens
        self.timeout = timeout
        self.last_response = {}
        self.model = model
        self.chat_id = chat_id or str(uuid.uuid4())
        self.user_id = user_id or f"user_{str(uuid.uuid4())[:8].upper()}"

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

    @staticmethod
    def _uncovr_extractor(chunk: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extracts content from the UncovrAI stream format '0:"..."'."""
        if isinstance(chunk, str):
            match = re.match(r'^0:\s*"?(.*?)"?$', chunk) # Match 0: maybe optional quotes
            if match:
                # Decode potential unicode escapes like \u00e9 and handle escaped quotes/backslashes
                content = match.group(1).encode().decode('unicode_escape')
                return content.replace('\\\\', '\\').replace('\\"', '"')
        return None

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

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        raw: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        temperature: int = 32,
        creativity: str = "medium",
        selected_focus: list = ["web"],
        selected_tools: list = ["quick-cards"]
    ) -> Union[Dict[str, Any], Generator]:
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise Exception(f"Optimizer is not one of {self.__available_optimizers}")

        # Prepare the request payload
        payload = {
            "content": conversation_prompt,
            "chatId": self.chat_id,
            "userMessageId": str(uuid.uuid4()),
            "ai_config": {
                "selectedFocus": selected_focus,
                "selectedTools": selected_tools,
                "agentId": "chat",
                "modelId": self.model,
                "temperature": temperature,
                "creativity": creativity
            }
        }

        def for_stream():
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.url, 
                    json=payload, 
                    stream=True, 
                    timeout=self.timeout,
                    impersonate=self.fingerprint.get("browser_type", "chrome110") # Use fingerprint browser type
                )
                
                if response.status_code != 200:
                    # If we get a non-200 response, try refreshing our identity once
                    if response.status_code in [403, 429]:
                        self.refresh_identity()
                        # Retry with new identity using curl_cffi session
                        retry_response = self.session.post(
                            self.url, 
                            json=payload, 
                            stream=True, 
                            timeout=self.timeout,
                            impersonate=self.fingerprint.get("browser_type", "chrome110") # Use updated fingerprint
                        )
                        if not retry_response.ok:
                            raise exceptions.FailedToGenerateResponseError(
                                f"Failed to generate response after identity refresh - ({retry_response.status_code}, {retry_response.reason}) - {retry_response.text}"
                            )
                        response = retry_response # Use the successful retry response
                    else:
                        raise exceptions.FailedToGenerateResponseError(
                            f"Request failed with status code {response.status_code} - {response.text}"
                        )

                streaming_text = ""
                # Use sanitize_stream with the custom extractor
                processed_stream = sanitize_stream(
                    data=response.iter_content(chunk_size=None), # Pass byte iterator
                    intro_value=None, # No simple prefix
                    to_json=False,    # Content is not JSON
                    content_extractor=self._uncovr_extractor, # Use the specific extractor
                    yield_raw_on_error=True # Keep yielding even if extractor fails, for potential error messages? (Adjust if needed)
                )

                for content_chunk in processed_stream:
                    if content_chunk and isinstance(content_chunk, str):
                        streaming_text += content_chunk
                        yield dict(text=content_chunk) if not raw else content_chunk
                
                self.last_response = {"text": streaming_text}
                self.conversation.update_chat_history(prompt, streaming_text)
                    
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e: # Catch other potential exceptions
                raise exceptions.FailedToGenerateResponseError(f"An unexpected error occurred ({type(e).__name__}): {e}")


        def for_non_stream():
            try:
                # Use curl_cffi session post with impersonate
                response = self.session.post(
                    self.url, 
                    json=payload, 
                    timeout=self.timeout,
                    impersonate=self.fingerprint.get("browser_type", "chrome110")
                )
                
                if response.status_code != 200:
                    if response.status_code in [403, 429]:
                        self.refresh_identity()
                        # Retry with new identity using curl_cffi session
                        response = self.session.post(
                            self.url, 
                            json=payload, 
                            timeout=self.timeout,
                            impersonate=self.fingerprint.get("browser_type", "chrome110")
                        )
                        if not response.ok:
                            raise exceptions.FailedToGenerateResponseError(
                                f"Failed to generate response after identity refresh - ({response.status_code}, {response.reason}) - {response.text}"
                            )
                    else:
                        raise exceptions.FailedToGenerateResponseError(
                            f"Request failed with status code {response.status_code} - {response.text}"
                        )

                response_text = response.text # Get the full response text
                
                # Use sanitize_stream to process the non-streaming text
                # It won't parse as JSON, but will apply the extractor line by line
                processed_stream = sanitize_stream(
                    data=response_text.splitlines(), # Split into lines first
                    intro_value=None,
                    to_json=False,
                    content_extractor=self._uncovr_extractor,
                    yield_raw_on_error=True 
                )

                # Aggregate the results from the generator
                full_response = ""
                for content in processed_stream:
                    if content and isinstance(content, str):
                        full_response += content

                # Check if aggregation resulted in empty response (might indicate error not caught by extractor)
                self.last_response = {"text": full_response}
                self.conversation.update_chat_history(prompt, full_response)
                return {"text": full_response}
                
            except CurlError as e: # Catch CurlError
                raise exceptions.FailedToGenerateResponseError(f"Request failed (CurlError): {e}")
            except Exception as e: # Catch other potential exceptions
                raise exceptions.FailedToGenerateResponseError(f"Request failed ({type(e).__name__}): {e}")


        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        optimizer: str = None,
        conversationally: bool = False,
        temperature: int = 32,
        creativity: str = "medium",
        selected_focus: list = ["web"],
        selected_tools: list = []
    ) -> Union[str, Generator[str, None, None]]:
        def for_stream():
            for response in self.ask(
                prompt, True, optimizer=optimizer, conversationally=conversationally,
                temperature=temperature, creativity=creativity,
                selected_focus=selected_focus, selected_tools=selected_tools
            ):
                yield self.get_message(response)
        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt, False, optimizer=optimizer, conversationally=conversationally,
                    temperature=temperature, creativity=creativity,
                    selected_focus=selected_focus, selected_tools=selected_tools
                )
            )
        return for_stream() if stream else for_non_stream()

    def get_message(self, response: dict) -> str:
        assert isinstance(response, dict), "Response should be of dict data-type only"
        # Formatting handled by extractor
        text = response.get("text", "")
        return text.replace('\\n', '\n').replace('\\n\\n', '\n\n') # Keep newline replacement

if __name__ == "__main__":
    # Ensure curl_cffi is installed
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in UncovrAI.AVAILABLE_MODELS:
        try:
            test_ai = UncovrAI(model=model, timeout=60)
            # Test non-stream first as stream logic depends on it
            response_non_stream = test_ai.chat("Say 'Hello' in one word", stream=False) 
            
            if response_non_stream and len(response_non_stream.strip()) > 0:
                 # Now test stream
                response_stream = test_ai.chat("Say 'Hi' in one word", stream=True)
                response_text = ""
                for chunk in response_stream:
                    response_text += chunk
                
                if response_text and len(response_text.strip()) > 0:
                    status = "✓"
                    # Clean and truncate response
                    clean_text = response_text.strip().encode('utf-8', errors='ignore').decode('utf-8')
                    display_text = clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
                else:
                    status = "✗ (Stream)"
                    display_text = "Empty or invalid stream response"
            else:
                status = "✗ (Non-Stream)"
                display_text = "Empty or invalid non-stream response"
                
            print(f"\r{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"\r{model:<50} {'✗':<10} {str(e)}")
